#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end io_uring DMA-BUF probe: export -> register -> READ_FIXED / WRITE_FIXED.

Reproduces the *exact* operation that LMCache's IouDmabufBackend issues on a cold
store/retrieve -- GPU pool export, io_uring dmabuf registration against an O_DIRECT
NVMe device, and a fixed-buffer read/write -- but with NO vLLM, RawBlockCore, or
StorageManager in the path. If the backend hangs in cold iteration and this probe
also hangs, the stall is in the kernel/driver dmabuf DMA path, not the LMCache
integration. If this completes but the backend hangs, the bug is in how the backend
drives it.

A watchdog runs the dmabuf op in a daemon thread and joins with a timeout, so the
probe always prints a verdict (OK / ERROR / HUNG) instead of hanging forever.
``submit_dmabuf_fixed_io`` releases the GIL during the completion wait, so the
watchdog thread runs even while the op blocks.

Usage
-----
    # READ_FIXED only (non-destructive: NVMe DMAs *into* the GPU buffer)
    python tools/probe_iou_dmabuf_e2e.py --device-path /dev/nvme0n1

    # also test WRITE_FIXED (DESTRUCTIVE: overwrites --device-offset on the device)
    python tools/probe_iou_dmabuf_e2e.py --device-path /dev/nvme0n1 \
        --write --device-offset $((1<<30))

Exit status
-----------
0  the dmabuf op(s) completed
1  the dmabuf op errored or HUNG (see the printed step)
2  torch/GPU/native extension unavailable
"""

from __future__ import annotations

# Standard
import argparse
import os
import sys
import threading
import time

# Third Party
import torch


def _round_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _run_with_watchdog(label: str, fn, timeout: float) -> str:
    """Run ``fn()`` in a daemon thread; return 'ok' | 'error' | 'hung'."""
    result: dict[str, object] = {}

    def worker() -> None:
        start = time.monotonic()
        try:
            n = fn()
            result["ok"] = (int(n), time.monotonic() - start)
        except BaseException as e:  # noqa: BLE001 - report anything, incl. panics
            result["err"] = e

    thread = threading.Thread(target=worker, name=f"probe-{label}", daemon=True)
    print(f"\n[{label}] submitting (timeout {timeout:.0f}s) ...", flush=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print(
            f"[{label}] HUNG: no completion within {timeout:.0f}s.\n"
            "        The op was submitted but the kernel never posted a CQE.\n"
            "        => the io_uring dmabuf DMA stalled in the kernel/driver,\n"
            "           not in LMCache. Check dmesg (amdgpu/nvme/dma-fence),\n"
            "           CONFIG_PCI_P2PDMA, and GPU<->NVMe PCIe reachability.",
            flush=True,
        )
        return "hung"
    if "err" in result:
        print(f"[{label}] ERROR: {result['err']!r}", flush=True)
        return "error"
    count, elapsed = result["ok"]  # type: ignore[misc]
    print(f"[{label}] OK: transferred {count} bytes in {elapsed * 1000:.1f} ms", flush=True)
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device-path", required=True, help="raw NVMe namespace, e.g. /dev/nvme0n1")
    parser.add_argument("--gpu-device", default="cuda:0", help="torch device")
    parser.add_argument("--length", type=int, default=4096, help="transfer length in bytes")
    parser.add_argument(
        "--device-offset",
        type=int,
        default=0,
        help="byte offset on the NVMe device (must be block-aligned)",
    )
    parser.add_argument("--exporter", default="auto", help="auto | hip | cuda_pool")
    parser.add_argument("--mem-range-flags", type=int, default=0, help="export flags (HIP/CUDA)")
    parser.add_argument("--timeout", type=float, default=15.0, help="per-op watchdog seconds")
    parser.add_argument(
        "--write",
        action="store_true",
        help="also test WRITE_FIXED (DESTRUCTIVE: overwrites the device at --device-offset)",
    )
    parser.add_argument("--ring-depth", type=int, default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No GPU available to PyTorch (torch.cuda.is_available() is False)", file=sys.stderr)
        return 2

    # Reuse the backend's exact export machinery so this drives the identical path.
    try:
        # First Party
        from lmcache.v1.storage_backend import iou_dmabuf_backend as be
    except Exception as e:  # noqa: BLE001
        print(f"Could not import iou_dmabuf_backend: {e!r}", file=sys.stderr)
        return 2
    try:
        # Third Party
        from lmcache_rust_raw_block_io import RawBlockDevice
    except Exception as e:  # noqa: BLE001
        print(
            f"Could not import lmcache_rust_raw_block_io: {e!r}\n"
            "Rebuild the crate: (cd rust/raw_block && maturin develop)",
            file=sys.stderr,
        )
        return 2

    device = torch.device(args.gpu_device)
    torch.cuda.set_device(device)
    torch.zeros(1, dtype=torch.uint8, device=device)  # force context init

    page = os.sysconf("SC_PAGESIZE")
    length = _round_up(args.length, page)
    pool_bytes = _round_up(max(length, page), page)

    if args.device_offset % page != 0:
        print(f"--device-offset must be a multiple of {page}", file=sys.stderr)
        return 2

    print("=== environment ===")
    print(f"    torch            {torch.__version__}")
    print(f"    hip build        {getattr(torch.version, 'hip', None)}")
    print(f"    cuda build        {torch.version.cuda}")
    print(f"    gpu              {torch.cuda.get_device_name(device)}")
    print(f"    device-path      {args.device_path}")
    print(f"    page/length      {page} / {length}")
    print(f"    device-offset    {args.device_offset}")

    # 1. Allocate + export the GPU pool exactly as the backend does.
    print("\n[export] allocating page-aligned GPU pool ...", flush=True)
    base, pool = be._make_page_aligned_gpu_pool(pool_bytes, device, page)
    exporter = be._resolve_dmabuf_exporter(args.exporter)
    print(f"[export] exporter={exporter}; calling address-range export ...", flush=True)
    if exporter == "hip":
        driver = be._HipDriver(be._torch_cuda_device_index(device))
    else:
        driver = be._CudaDriver()
    dmabuf_fd = driver.export_dmabuf(int(pool.data_ptr()), pool_bytes, args.mem_range_flags)
    print(f"[export] OK: dmabuf_fd={dmabuf_fd}", flush=True)

    # 2. Open the NVMe device + io_uring ring, and register the dmabuf.
    print("\n[open] opening device O_DIRECT + io_uring ...", flush=True)
    dev = RawBlockDevice(
        args.device_path,
        True,  # writable
        True,  # use_odirect
        True,  # use_iouring
        False,  # use_uring_cmd
        page,  # alignment
        "io_uring",
        args.ring_depth,
    )
    print("[register] register_dmabuf_buffers([fd]) ...", flush=True)
    dev.register_dmabuf_buffers([dmabuf_fd])
    print("[register] OK", flush=True)

    # 3. Issue the dmabuf op(s) under a watchdog. READ first (non-destructive).
    max_retries = 16
    read_status = _run_with_watchdog(
        "READ_FIXED",
        lambda: dev.read_fixed_dmabuf(0, 0, length, args.device_offset, max_retries),
        args.timeout,
    )

    write_status = "skipped"
    if args.write:
        if read_status == "ok":
            print(
                "\n!! WRITE_FIXED is DESTRUCTIVE: it overwrites "
                f"{length} bytes at device offset {args.device_offset}.",
                flush=True,
            )
            write_status = _run_with_watchdog(
                "WRITE_FIXED",
                lambda: dev.write_fixed_dmabuf(0, 0, length, args.device_offset, max_retries),
                args.timeout,
            )
        else:
            print("\n[WRITE_FIXED] skipped because READ_FIXED did not complete cleanly.")

    print("\n=== verdict ===")
    print(f"    READ_FIXED : {read_status}")
    print(f"    WRITE_FIXED: {write_status}")
    if read_status == "hung" or write_status == "hung":
        print(
            "    => Reproduced the hang WITHOUT LMCache. The kernel dmabuf DMA "
            "does not complete on this GPU/NVMe/kernel. Debug the kernel path "
            "(P2PDMA, PCIe topology, exporter<->NVMe importer support)."
        )
        os._exit(1)  # a stuck worker thread would otherwise block normal exit
    if read_status != "ok" or write_status not in ("ok", "skipped"):
        return 1
    print("    => The exact backend dmabuf op completes here. If the backend still")
    print("       hangs, the bug is in how it drives this (offsets/sizing/registration).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
