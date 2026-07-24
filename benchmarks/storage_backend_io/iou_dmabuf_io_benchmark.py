# SPDX-License-Identifier: Apache-2.0

"""Benchmark and integrity-test the io_uring DMA-BUF storage backend.

This mirrors ``storage_backend_io_benchmark.py`` but targets ``IouDmabufBackend``,
which is different enough to warrant its own harness:

- It is its *own* GPU allocator. To exercise the zero-copy ``WRITE_FIXED`` fast
  path, the source memory objects must be allocated from the backend's own
  ``DmabufGPUAllocator`` (``--source gpu``, default). ``--source cpu`` allocates
  CPU-resident objects instead, exercising the ``put_many`` fallback route.
- It requires a **real NVMe namespace** opened ``O_DIRECT`` (the dmabuf token op
  is implemented by the nvme-pci driver); a regular file will fail registration.
  **Writes are destructive to that device** at the slots it uses.
- It requires a CUDA/ROCm GPU and a kernel with ``CONFIG_DMABUF_TOKEN``.

The harness drives the backend directly (no StorageManager / vLLM): it writes
per-chunk seeded patterns, reads them back into fresh GPU slabs, and verifies the
bytes round-trip. Reported throughput is the raw backend put/get rate.

Example
-------
    python benchmarks/storage_backend_io/iou_dmabuf_io_benchmark.py \
        --device-path /dev/nvme0n1 --num-ops 128 --concurrency 4 \
        --verify-integrity
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional
import argparse
import asyncio
import json
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.iou_dmabuf_backend import IouDmabufBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

# A modest KV chunk: (num_layers, kv, tokens, heads, head_size). The per-chunk
# tensor is the flat (kv, layers, tokens, hidden) view the backend stores.
DEFAULT_CHUNK_SIZE = 256
DEFAULT_KV_SHAPE = (28, 2, DEFAULT_CHUNK_SIZE, 8, 128)
DEFAULT_CHUNK_TENSOR_SHAPE = torch.Size([2, 16, DEFAULT_CHUNK_SIZE, 128])
DEFAULT_DTYPE = torch.bfloat16
DEFAULT_FMT = MemoryFormat.KV_2LTD


def _start_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="iou-bench-loop", daemon=True)
    thread.start()
    return loop, thread


def _stop_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def _build_metadata(chunk_size: int) -> LMCacheMetadata:
    kv_shape = (
        DEFAULT_KV_SHAPE[0],
        DEFAULT_KV_SHAPE[1],
        chunk_size,
        DEFAULT_KV_SHAPE[3],
        DEFAULT_KV_SHAPE[4],
    )
    return LMCacheMetadata(
        model_name="iou_bench_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=DEFAULT_DTYPE,
        kv_shape=kv_shape,
        chunk_size=chunk_size,
    )


def _fill_seeded(tensor: torch.Tensor, seed: int) -> None:
    """Fill ``tensor`` in place with a deterministic per-seed pattern.

    Uses a CPU generator so the pattern is identical regardless of device, then
    copies into the (possibly GPU) tensor's storage. Distinct per ``seed`` so a
    cross-chunk mix-up is caught, not just bulk corruption.
    """
    gen = torch.Generator().manual_seed(1009 * seed + 1)
    pattern = torch.rand(tensor.numel(), generator=gen, dtype=torch.float32)
    tensor.copy_(pattern.to(tensor.dtype).reshape(tensor.shape).to(tensor.device))


def _chunk_bytes(shape: torch.Size, dtype: torch.dtype) -> int:
    return shape.numel() * dtype.itemsize


def _round_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


class IouDmabufIOBenchmark:
    """Write/read + integrity harness for ``IouDmabufBackend``."""

    def __init__(
        self,
        device_path: str,
        num_ops: int,
        concurrency: int,
        source: str,
        alignment: int,
        chunk_size: int,
        gpu_device: str,
        exporter: str,
        mem_range_flags: int,
        gpu_pool_bytes: int,
        capacity_bytes: int,
        max_local_cpu_gb: float,
        verify_integrity: bool,
        iters: int = 1,
        target_gib: float = 0.0,
        write_concurrency: Optional[int] = None,
        read_concurrency: Optional[int] = None,
    ) -> None:
        self.device_path = device_path
        self.num_ops = num_ops
        self.concurrency = concurrency
        # Split so the put and get phases can be stressed independently, to
        # localize a concurrency bug to the write path vs. the read path.
        self.write_concurrency = (
            write_concurrency if write_concurrency is not None else concurrency
        )
        self.read_concurrency = (
            read_concurrency if read_concurrency is not None else concurrency
        )
        self.source = source
        self.alignment = alignment
        self.chunk_size = chunk_size
        self.gpu_device = gpu_device
        self.exporter = exporter
        self.mem_range_flags = mem_range_flags
        self.gpu_pool_bytes = gpu_pool_bytes
        self.capacity_bytes = capacity_bytes
        self.iters = iters
        self.target_gib = target_gib
        self.max_local_cpu_gb = max_local_cpu_gb
        self.verify_integrity = verify_integrity

        self._chunk_shape = torch.Size(
            [
                DEFAULT_CHUNK_TENSOR_SHAPE[0],
                DEFAULT_CHUNK_TENSOR_SHAPE[1],
                chunk_size,
                DEFAULT_CHUNK_TENSOR_SHAPE[3],
            ]
        )

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._local_cpu: Optional[LocalCPUBackend] = None
        self._backend: Optional[IouDmabufBackend] = None
        self._keys: list[CacheEngineKey] = []
        self._objs: list[MemoryObj] = []
        # Globally-unique fill seed base for the current iteration, so every
        # chunk across all iterations has distinct content (catches stale reads
        # from recycled slots, not just intra-iteration swaps).
        self._seed_base = 0

    # -- setup ---------------------------------------------------------------

    def _build_config(self, pool_bytes: int) -> LMCacheEngineConfig:
        extra = {
            "iou_dmabuf.enabled": True,
            "iou_dmabuf.device_path": self.device_path,
            "iou_dmabuf.gpu_pool_bytes": pool_bytes,
            "iou_dmabuf.block_align": self.alignment,
            "iou_dmabuf.header_bytes": self.alignment,
            "iou_dmabuf.exporter": self.exporter,
            "iou_dmabuf.mem_range_flags": self.mem_range_flags,
            "iou_dmabuf.load_checkpoint_on_init": False,
            "iou_dmabuf.meta_enable_periodic": False,
        }
        if self.capacity_bytes > 0:
            extra["iou_dmabuf.capacity_bytes"] = self.capacity_bytes
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=self.chunk_size,
            local_cpu=True,
            max_local_cpu_size=self.max_local_cpu_gb,
            lmcache_instance_id="iou_dmabuf_bench",
        )
        config.extra_config = extra
        return config

    def _resolve_pool_bytes(self) -> int:
        """Return a GPU pool big enough to hold every source object at once."""
        if self.gpu_pool_bytes > 0:
            return self.gpu_pool_bytes
        page = 4096
        per = _round_up(_chunk_bytes(self._chunk_shape, DEFAULT_DTYPE), self.alignment)
        # Writes allocate all num_ops objects simultaneously; reads allocate up
        # to num_ops again after writes are freed. 1.25x margin covers alignment
        # and the allocator's slab-boundary guards.
        needed = int(self.num_ops * per * 1.25)
        return max(_round_up(needed, page), _round_up(per * 4, page))

    # -- source objects ------------------------------------------------------

    def _make_source_objs(self) -> list[MemoryObj]:
        assert self._backend is not None and self._local_cpu is not None
        # Allocate through the underlying MemoryAllocatorInterface so GPU objects
        # get parent_allocator = the DmabufGPUAllocator -> owns() is True -> the
        # WRITE_FIXED fast path. CPU objects go through LocalCPU's allocator and
        # exercise the put_many fallback route in the backend.
        allocator = (
            self._backend.memory_allocator
            if self.source == "gpu"
            else self._local_cpu.memory_allocator
        )
        objs: list[MemoryObj] = []
        for i in range(self.num_ops):
            obj = allocator.allocate([self._chunk_shape], [DEFAULT_DTYPE], DEFAULT_FMT)
            if obj is None:
                raise RuntimeError(
                    f"allocation failed at op {i}/{self.num_ops} from "
                    f"{'GPU dmabuf pool' if self.source == 'gpu' else 'LocalCPU'}; "
                    "increase --gpu-pool-bytes / --max-local-cpu-gb or lower --num-ops"
                )
            assert obj.tensor is not None
            _fill_seeded(obj.tensor, self._seed_base + i)
            objs.append(obj)
        return objs

    # -- phases --------------------------------------------------------------

    def _slices(self, concurrency: int) -> list[tuple[int, int]]:
        size = max(1, self.num_ops // max(1, concurrency))
        return [
            (i, min(i + size, self.num_ops)) for i in range(0, self.num_ops, size)
        ]

    def _write_phase(self) -> float:
        assert self._backend is not None
        futures: list[Future] = []
        fut_lock = threading.Lock()

        def submit_slice(start: int, end: int) -> None:
            result = self._backend.batched_submit_put_task(
                self._keys[start:end], self._objs[start:end]
            )
            if result:
                with fut_lock:
                    futures.extend(result)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.write_concurrency) as ex:
            for lo, hi in self._slices(self.write_concurrency):
                ex.submit(submit_slice, lo, hi)
        for fut in futures:
            fut.result(timeout=300)
        return time.perf_counter() - start

    def _read_phase(
        self,
    ) -> list[tuple[CacheEngineKey, Optional[MemoryObj]]]:
        assert self._backend is not None
        results: list[tuple[CacheEngineKey, Optional[MemoryObj]]] = []
        lock = threading.Lock()

        def read_slice(start: int, end: int) -> None:
            batch = self._keys[start:end]
            loaded = self._backend.batched_get_blocking(batch)
            # batched_get_blocking returns [] on a total miss; normalize length.
            if not loaded:
                loaded = [None] * len(batch)
            with lock:
                results.extend(zip(batch, loaded, strict=False))

        with ThreadPoolExecutor(max_workers=self.read_concurrency) as ex:
            for lo, hi in self._slices(self.read_concurrency):
                ex.submit(read_slice, lo, hi)
        return results

    def _verify(
        self,
        read_results: list[tuple[CacheEngineKey, Optional[MemoryObj]]],
        reference: dict[CacheEngineKey, torch.Tensor],
    ) -> tuple[int, int, list[str]]:
        """Return (misses, mismatches, detail strings classifying each mismatch)."""
        misses = 0
        mismatches = 0
        details: list[str] = []
        ref_items = list(reference.items())
        for key, obj in read_results:
            if obj is None or obj.tensor is None:
                misses += 1
                continue
            got = obj.tensor.detach().to("cpu")
            want = reference.get(key)
            if want is not None and torch.equal(got, want):
                continue
            mismatches += 1
            # Classify: did we read another chunk's committed data (a slot/offset
            # swap), or something that matches no written chunk (corruption)?
            swap = None
            for other_key, other_ref in ref_items:
                if other_key is not key and torch.equal(got, other_ref):
                    swap = other_key
                    break
            if swap is not None:
                details.append(f"{key.to_string()} <- data of {swap.to_string()} (swap)")
            else:
                details.append(f"{key.to_string()} <- foreign/corrupt data")
        return misses, mismatches, details

    # -- driver --------------------------------------------------------------

    def _iter_keys(self, iter_idx: int) -> list[CacheEngineKey]:
        """Fresh keys for one iteration.

        Keys must be unique per iteration: a put for an already-indexed key is
        skipped by the backend (no I/O), so reusing keys would produce no traffic
        after the first pass.
        """
        base = iter_idx * self.num_ops
        return [
            CacheEngineKey("iou_bench_model", 1, 0, base + i, DEFAULT_DTYPE)
            for i in range(self.num_ops)
        ]

    def _resolve_iters(self) -> int:
        if self.target_gib > 0:
            per_iter = self.num_ops * _chunk_bytes(self._chunk_shape, DEFAULT_DTYPE) * 2
            target = int(self.target_gib * (1024**3))
            return max(1, -(-target // per_iter))  # ceil division
        return max(1, self.iters)

    def _run_iteration(
        self, iter_idx: int
    ) -> tuple[float, float, int, int, list[str]]:
        """One write+read cycle over a fresh key set; recycles slots on exit.

        Returns ``(write_sec, read_sec, misses, mismatches, mismatch_details)``.
        """
        assert self._backend is not None
        self._seed_base = iter_idx * self.num_ops
        self._keys = self._iter_keys(iter_idx)
        self._objs = self._make_source_objs()

        reference: dict[CacheEngineKey, torch.Tensor] = {}
        if self.verify_integrity:
            for key, obj in zip(self._keys, self._objs, strict=True):
                assert obj.tensor is not None
                reference[key] = obj.tensor.detach().to("cpu").clone()

        # The source tensors are filled with async GPU ops. The dmabuf WRITE_FIXED
        # reads that VRAM via a peer DMA that is NOT ordered against the GPU stream,
        # so make the fills globally visible before any write is submitted.
        torch.cuda.synchronize(torch.device(self.gpu_device))

        write_elapsed = self._write_phase()

        # Free source objects so the pool has room for read slabs.
        for obj in self._objs:
            obj.ref_count_down()
        self._objs = []

        read_start = time.perf_counter()
        read_results = self._read_phase()
        read_elapsed = time.perf_counter() - read_start

        misses = mismatches = 0
        details: list[str] = []
        if self.verify_integrity:
            misses, mismatches, details = self._verify(read_results, reference)

        for _, obj in read_results:
            if obj is not None:
                obj.ref_count_down()

        # Recycle device slots + index (untimed cleanup) so total device usage
        # stays at one working set even across many iterations: the traffic
        # accumulates, the on-device footprint does not.
        for key in self._keys:
            try:
                self._backend.remove(key, force=True)
            except Exception:
                pass
        self._keys = []
        return write_elapsed, read_elapsed, misses, mismatches, details

    def run(self) -> dict:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm GPU not available")
        if not IouDmabufBackend.is_available():
            raise RuntimeError(
                "IouDmabufBackend.is_available() is False: rebuild "
                "lmcache_rust_raw_block_io and use a kernel with CONFIG_DMABUF_TOKEN"
            )

        pool_bytes = self._resolve_pool_bytes()
        iters = self._resolve_iters()
        chunk_bytes = _chunk_bytes(self._chunk_shape, DEFAULT_DTYPE)
        self._loop, self._loop_thread = _start_loop()
        metadata = _build_metadata(self.chunk_size)
        config = self._build_config(pool_bytes)

        logger.info(
            "iou-dmabuf bench: device=%s num_ops=%d concurrency=%d source=%s "
            "gpu_pool=%.1f MiB iters=%d (~%.1f GiB total traffic)",
            self.device_path,
            self.num_ops,
            self.concurrency,
            self.source,
            pool_bytes / (1024 * 1024),
            iters,
            iters * self.num_ops * chunk_bytes * 2 / (1024**3),
        )

        self._local_cpu = LocalCPUBackend(
            config=config, metadata=metadata, dst_device="cpu"
        )
        self._backend = IouDmabufBackend(
            config, metadata, self._loop, self.gpu_device
        )

        total_write_time = 0.0
        total_read_time = 0.0
        total_misses = 0
        total_mismatches = 0
        mismatch_details: list[str] = []
        try:
            for it in range(iters):
                w_sec, r_sec, misses, mismatches, details = self._run_iteration(it)
                total_write_time += w_sec
                total_read_time += r_sec
                total_misses += misses
                total_mismatches += mismatches
                for d in details:
                    mismatch_details.append(f"iter {it}: {d}")
                if details:
                    for d in details:
                        logger.error("  MISMATCH iter %d: %s", it, d)
                if iters > 1:
                    logger.info(
                        "  iter %d/%d: write %.3fs read %.3fs%s",
                        it + 1,
                        iters,
                        w_sec,
                        r_sec,
                        f" (misses={misses} mismatches={mismatches})"
                        if self.verify_integrity
                        else "",
                    )

            total_ops = iters * self.num_ops
            write_bytes = total_ops * chunk_bytes
            read_bytes = total_ops * chunk_bytes
            gib = 1024**3
            return {
                "backend": "iou_dmabuf",
                "source": self.source,
                "device_path": self.device_path,
                "iters": iters,
                "num_ops_per_iter": self.num_ops,
                "total_ops": total_ops,
                "concurrency": self.concurrency,
                "write_concurrency": self.write_concurrency,
                "read_concurrency": self.read_concurrency,
                "gpu_pool_bytes": pool_bytes,
                "chunk_bytes": chunk_bytes,
                "write_bytes_total": write_bytes,
                "read_bytes_total": read_bytes,
                "traffic_gib_total": (write_bytes + read_bytes) / gib,
                "write_elapsed_sec": total_write_time,
                "read_elapsed_sec": total_read_time,
                "write_gib_per_sec": write_bytes / total_write_time / gib
                if total_write_time > 0
                else 0.0,
                "read_gib_per_sec": read_bytes / total_read_time / gib
                if total_read_time > 0
                else 0.0,
                "write_ops_per_sec": total_ops / total_write_time
                if total_write_time > 0
                else 0.0,
                "read_ops_per_sec": total_ops / total_read_time
                if total_read_time > 0
                else 0.0,
                "verify_integrity": self.verify_integrity,
                "read_misses": total_misses,
                "integrity_mismatches": total_mismatches,
                "mismatch_details": mismatch_details[:50],
                "integrity_passed": self.verify_integrity
                and total_misses == 0
                and total_mismatches == 0,
            }
        finally:
            for obj in self._objs:
                try:
                    obj.ref_count_down()
                except Exception:
                    pass
            if self._backend is not None:
                self._backend.close()
            if self._loop is not None and self._loop_thread is not None:
                _stop_loop(self._loop, self._loop_thread)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark + integrity test for the io_uring DMA-BUF backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device-path",
        required=True,
        help="raw NVMe namespace, e.g. /dev/nvme0n1 (WRITES ARE DESTRUCTIVE)",
    )
    parser.add_argument(
        "--num-ops",
        type=int,
        default=64,
        help="chunks per iteration (the GPU-pool working set)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="number of write+read iterations over the working set",
    )
    parser.add_argument(
        "--target-gib",
        type=float,
        default=0.0,
        help=(
            "run enough iterations to move ~this much total device traffic "
            "(writes + reads); overrides --iters. e.g. 50 for ~50 GiB"
        ),
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="submit threads for both phases"
    )
    parser.add_argument(
        "--write-concurrency",
        type=int,
        default=None,
        help="override submit threads for the write phase only (default: --concurrency)",
    )
    parser.add_argument(
        "--read-concurrency",
        type=int,
        default=None,
        help="override submit threads for the read phase only (default: --concurrency)",
    )
    parser.add_argument(
        "--source",
        choices=["gpu", "cpu"],
        default="gpu",
        help="gpu = allocator-owned WRITE_FIXED fast path; cpu = put_many fallback",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--exporter", default="auto", help="auto | hip | cuda_pool")
    parser.add_argument("--mem-range-flags", type=int, default=0)
    parser.add_argument(
        "--gpu-pool-bytes",
        type=int,
        default=0,
        help="GPU dmabuf pool size in bytes (0 = auto-size from --num-ops)",
    )
    parser.add_argument(
        "--capacity-bytes",
        type=int,
        default=0,
        help="usable device capacity in bytes (0 = whole device)",
    )
    parser.add_argument("--max-local-cpu-gb", type=float, default=1.0)
    parser.add_argument(
        "--verify-integrity",
        action="store_true",
        help="read back and byte-compare against the written pattern",
    )
    parser.add_argument("--output-json", default="", help="write result JSON to this path")
    args = parser.parse_args()

    bench = IouDmabufIOBenchmark(
        device_path=args.device_path,
        num_ops=args.num_ops,
        concurrency=args.concurrency,
        source=args.source,
        alignment=args.alignment,
        chunk_size=args.chunk_size,
        gpu_device=args.gpu_device,
        exporter=args.exporter,
        mem_range_flags=args.mem_range_flags,
        gpu_pool_bytes=args.gpu_pool_bytes,
        capacity_bytes=args.capacity_bytes,
        max_local_cpu_gb=args.max_local_cpu_gb,
        verify_integrity=args.verify_integrity,
        iters=args.iters,
        target_gib=args.target_gib,
        write_concurrency=args.write_concurrency,
        read_concurrency=args.read_concurrency,
    )
    result = bench.run()
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
    if args.verify_integrity and not result["integrity_passed"]:
        raise SystemExit(
            f"INTEGRITY FAILED: misses={result['read_misses']} "
            f"mismatches={result['integrity_mismatches']}"
        )


if __name__ == "__main__":
    main()
