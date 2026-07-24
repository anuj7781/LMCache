#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-shot diagnostic: is the concurrent-write corruption AMD peer-DMA coherency?

The io_uring DMA-BUF backend corrupts data under concurrent writes (serialized
writes are clean). The write/slot/index/ring logic is correct and serialized, and
the number of concurrent DMAs is the same regardless of caller threads -- so the
remaining suspect is GPU->NVMe (peer PCIe) *visibility*: for coarse-grained VRAM
(``torch.empty`` / ``hipMalloc``) a stream sync does not guarantee the GPU's writes
are visible to the NVMe's peer DMA, so a bursty write reads stale VRAM.

This drives the REAL dmabuf write path (the same ``RawBlockDevice`` the backend
uses) against two source-memory types with an identical GPU fill and read-back:

    coarse : hipMalloc                          (what the backend uses today)
    fine   : hipExtMallocWithFlags(finegrained) (system-coherent; the candidate fix)

For each: fill N chunks on the GPU (hipMemset + hipDeviceSynchronize), fire N
concurrent WRITE_FIXED to N device slots, then serially READ_FIXED each slot back
and check the bytes. Repeat, and report corruption counts per memory type.

Verdict:
    coarse corrupts, fine clean  -> peer-DMA coherency; fix = fine-grained pool.
    both clean                   -> not reproduced at this size (raise --repeat/-N).
    both corrupt                 -> not (only) coherency; deeper issue.

WRITES ARE DESTRUCTIVE to the device region [0, N*chunk) at --device-offset.
"""

from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
import argparse
import ctypes
import os
import sys

HIP_SUCCESS = 0
HIP_DEVICE_MALLOC_FINEGRAINED = 0x1


def _hip_check(lib: ctypes.CDLL, rc: int, what: str) -> None:
    if rc != HIP_SUCCESS:
        name = b""
        try:
            name = lib.hipGetErrorName(rc) or b""
        except Exception:
            pass
        raise RuntimeError(f"{what} failed: hipError {rc} {name.decode(errors='ignore')}")


class Hip:
    """Minimal HIP runtime wrapper for alloc/memset/sync used by the diagnostic."""

    def __init__(self, device_index: int) -> None:
        # First Party -- reuse the backend's loader + export driver so this drives
        # the identical export path.
        from lmcache.v1.storage_backend.iou_dmabuf_backend import (
            _HipDriver,
            _load_shared_library,
        )

        self.lib = _load_shared_library(
            ("amdhip64", "libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so")
        )
        for fn, args, ret in (
            ("hipMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
            (
                "hipExtMallocWithFlags",
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint],
                ctypes.c_int,
            ),
            ("hipFree", [ctypes.c_void_p], ctypes.c_int),
            ("hipMemset", [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t], ctypes.c_int),
            ("hipDeviceSynchronize", [], ctypes.c_int),
            ("hipGetErrorName", [ctypes.c_int], ctypes.c_char_p),
        ):
            f = getattr(self.lib, fn)
            f.argtypes = args
            f.restype = ret
        # _HipDriver runs hipInit + hipSetDevice(device_index) and exposes export.
        self.driver = _HipDriver(device_index)
        self.device_index = device_index

    def malloc(self, size: int, finegrained: bool) -> int:
        ptr = ctypes.c_void_p()
        if finegrained:
            rc = self.lib.hipExtMallocWithFlags(
                ctypes.byref(ptr), size, HIP_DEVICE_MALLOC_FINEGRAINED
            )
            _hip_check(self.lib, rc, "hipExtMallocWithFlags(finegrained)")
        else:
            rc = self.lib.hipMalloc(ctypes.byref(ptr), size)
            _hip_check(self.lib, rc, "hipMalloc")
        if not ptr.value:
            raise RuntimeError("allocation returned NULL")
        return int(ptr.value)

    def free(self, ptr: int) -> None:
        self.lib.hipFree(ctypes.c_void_p(ptr))

    def memset(self, ptr: int, value: int, size: int) -> None:
        _hip_check(self.lib, self.lib.hipMemset(ctypes.c_void_p(ptr), value, size), "hipMemset")

    def sync(self) -> None:
        _hip_check(self.lib, self.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")

    def export(self, ptr: int, size: int) -> int:
        return self.driver.export_dmabuf(ptr, size, 0)


def _aligned(ptr: int, page: int) -> int:
    return ptr + ((-ptr) % page)


def _read_dest_bytes(dest_ptr: int, length: int) -> bytes:
    """Read ``length`` bytes from a CPU-coherent (fine-grained) device pointer."""
    return ctypes.string_at(dest_ptr, length)


def run_trial(
    hip: Hip,
    dev,  # RawBlockDevice
    finegrained: bool,
    num_chunks: int,
    chunk_bytes: int,
    device_offset: int,
    concurrency: int,
    repeats: int,
    page: int,
) -> int:
    """Return the number of corrupt chunks across all repeats for one memory type."""
    label = "fine" if finegrained else "coarse"
    pool_size = num_chunks * chunk_bytes + page
    dest_size = chunk_bytes + page

    src_raw = hip.malloc(pool_size, finegrained)
    # The read-back destination is always fine-grained so the CPU can inspect it.
    dst_raw = hip.malloc(dest_size, finegrained=True)
    src = _aligned(src_raw, page)
    dst = _aligned(dst_raw, page)

    src_fd = hip.export(src, num_chunks * chunk_bytes)
    dst_fd = hip.export(dst, chunk_bytes)
    dev.register_dmabuf_buffers([src_fd, dst_fd])  # slab 0 = src, slab 1 = dst

    corrupt = 0
    try:
        for r in range(repeats):
            # Distinct byte value per (repeat, chunk) so a stale slot is detected.
            values = [((r * 131 + c * 7) % 254) + 1 for c in range(num_chunks)]

            # Fill each source chunk on the GPU, then make it visible.
            for c in range(num_chunks):
                hip.memset(src + c * chunk_bytes, values[c], chunk_bytes)
            hip.sync()

            # Fire N WRITE_FIXED concurrently: src chunk c -> device slot c.
            def write_one(c: int) -> None:
                dev.write_fixed_dmabuf(
                    0, c * chunk_bytes, chunk_bytes, device_offset + c * chunk_bytes, 16
                )

            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                list(ex.map(write_one, range(num_chunks)))

            # Serially read each slot back into the fine-grained dest and check.
            for c in range(num_chunks):
                dev.read_fixed_dmabuf(
                    1, 0, chunk_bytes, device_offset + c * chunk_bytes, 16
                )
                hip.sync()
                got = _read_dest_bytes(dst, chunk_bytes)
                want = values[c]
                # Sample start / middle / end (stale writes differ throughout).
                sample = (got[0], got[chunk_bytes // 2], got[-1])
                if any(b != want for b in sample):
                    corrupt += 1
                    if corrupt <= 12:
                        print(
                            f"  [{label}] repeat {r} chunk {c}: "
                            f"want {want:#04x} got {sample[0]:#04x}/"
                            f"{sample[1]:#04x}/{sample[2]:#04x}"
                        )
    finally:
        hip.free(src_raw)
        hip.free(dst_raw)
    total = num_chunks * repeats
    print(f"[{label}] corrupt {corrupt}/{total} chunks")
    return corrupt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device-path", required=True, help="raw NVMe namespace (DESTRUCTIVE)")
    ap.add_argument("--device-offset", type=int, default=0, help="base byte offset (aligned)")
    ap.add_argument("-n", "--num-chunks", type=int, default=64)
    ap.add_argument("--chunk-bytes", type=int, default=2 * 1024 * 1024)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=40)
    ap.add_argument("--device-index", type=int, default=0)
    args = ap.parse_args()

    try:
        from lmcache_rust_raw_block_io import RawBlockDevice
    except Exception as e:  # noqa: BLE001
        print(f"cannot import lmcache_rust_raw_block_io: {e!r}", file=sys.stderr)
        return 2

    page = os.sysconf("SC_PAGESIZE")
    if args.chunk_bytes % page or args.device_offset % page:
        print(f"--chunk-bytes and --device-offset must be multiples of {page}", file=sys.stderr)
        return 2

    hip = Hip(args.device_index)
    print(
        f"device={args.device_path} chunks={args.num_chunks} "
        f"chunk={args.chunk_bytes // 1024}KiB concurrency={args.concurrency} "
        f"repeat={args.repeat}\n"
    )

    results = {}
    for finegrained in (False, True):
        # A fresh device per trial so registrations don't overlap.
        dev = RawBlockDevice(
            args.device_path, True, True, True, False, page, "io_uring", 256
        )
        try:
            results["fine" if finegrained else "coarse"] = run_trial(
                hip,
                dev,
                finegrained,
                args.num_chunks,
                args.chunk_bytes,
                args.device_offset,
                args.concurrency,
                args.repeat,
                page,
            )
        finally:
            dev.close()
        print()

    coarse, fine = results["coarse"], results["fine"]
    print("=== verdict ===")
    if coarse > 0 and fine == 0:
        print("    coarse corrupts, fine-grained is CLEAN.")
        print("    => confirmed: AMD peer-DMA VRAM coherency. Fix = allocate the")
        print("       exported pool as fine-grained coherent memory.")
        return 0
    if coarse == 0 and fine == 0:
        print("    neither reproduced -- raise --repeat / --num-chunks / --concurrency.")
        return 0
    if coarse > 0 and fine > 0:
        print("    BOTH corrupt -> not (only) coherency; a deeper issue remains.")
        return 1
    print("    unexpected: fine corrupts but coarse does not.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
