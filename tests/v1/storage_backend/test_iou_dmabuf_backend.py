# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from types import SimpleNamespace
from typing import Any, Optional
import asyncio
import os
import sys
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.iou_dmabuf_backend import IouDmabufBackend
from lmcache.v1.storage_backend.raw_block import RawBlockKeySpec, RawBlockPutManyResult


class _FakeRawDevice:
    def __init__(self) -> None:
        self.registered_fds: list[int] = []
        self.reads: list[tuple[int, int, int, int, int]] = []
        self.writes: list[tuple[int, int, int, int, int]] = []

    def register_dmabuf_buffers(self, dmabuf_fds: list[int]) -> None:
        self.registered_fds = list(dmabuf_fds)

    def read_fixed_dmabuf(
        self,
        slab_idx: int,
        buf_offset: int,
        length: int,
        device_offset: int,
        max_eagain_retries: int,
    ) -> int:
        self.reads.append(
            (slab_idx, buf_offset, length, device_offset, max_eagain_retries)
        )
        return length

    def write_fixed_dmabuf(
        self,
        slab_idx: int,
        buf_offset: int,
        length: int,
        device_offset: int,
        max_eagain_retries: int,
    ) -> int:
        self.writes.append(
            (slab_idx, buf_offset, length, device_offset, max_eagain_retries)
        )
        return length


class _FakeCore:
    instances: list["_FakeCore"] = []

    def __init__(self, config: Any, *, key_namespace: str) -> None:
        self.config = config
        self.key_namespace = key_namespace
        self.header_bytes = int(config.header_bytes)
        self.block_align = int(config.block_align)
        self.rawdev = _FakeRawDevice()
        self.entries: dict[str, tuple[DiskCacheMetadata, int]] = {}
        self.puts: list[tuple[RawBlockKeySpec, MemoryObj]] = []
        self.reservations: list[tuple[RawBlockKeySpec, MemoryObj]] = []
        self.headers: list[tuple[RawBlockKeySpec, int, int]] = []
        self.commits: list[tuple[RawBlockKeySpec, int]] = []
        self.aborts: list[tuple[RawBlockKeySpec, int]] = []
        self.unlocked: list[list[str]] = []
        self.closed = False
        _FakeCore.instances.append(self)

    def raw_device(self) -> _FakeRawDevice:
        return self.rawdev

    def exists_many(self, encoded_keys: list[str], *, lock: bool = False) -> list[bool]:
        del lock
        return [encoded_key in self.entries for encoded_key in encoded_keys]

    def get_entries_many(
        self,
        encoded_keys: list[str],
        *,
        lock_refcount: bool = False,
    ) -> list[tuple[DiskCacheMetadata, int] | None]:
        del lock_refcount
        return [self.entries.get(encoded_key) for encoded_key in encoded_keys]

    def reserve_slot(self, key: RawBlockKeySpec, memory_obj: MemoryObj) -> int | None:
        self.reservations.append((key, memory_obj))
        return 8192

    def write_slot_header(
        self,
        key: RawBlockKeySpec,
        offset: int,
        payload_len: int,
    ) -> bool:
        self.headers.append((key, offset, payload_len))
        return True

    def commit_slot(self, key: RawBlockKeySpec, offset: int) -> bool:
        self.commits.append((key, offset))
        return True

    def abort_slot(self, key: RawBlockKeySpec, offset: int) -> None:
        self.aborts.append((key, offset))

    def put_many(
        self,
        keys: list[RawBlockKeySpec],
        objs: list[MemoryObj],
    ) -> RawBlockPutManyResult:
        for key, obj in zip(keys, objs, strict=True):
            self.puts.append((key, obj))
        return RawBlockPutManyResult(results=[True] * len(keys), stored_keys=[])

    def unlock_many(self, encoded_keys: list[str]) -> None:
        self.unlocked.append(list(encoded_keys))

    def delete_many(
        self,
        encoded_keys: list[str],
        *,
        force: bool = False,
    ) -> list[bool]:
        del force
        removed = []
        for encoded_key in encoded_keys:
            removed.append(encoded_key in self.entries)
            self.entries.pop(encoded_key, None)
        return removed

    def close(self) -> None:
        self.closed = True


class _FakeAllocator:
    instances: list["_FakeAllocator"] = []

    def __init__(
        self,
        pool_bytes: int,
        device: str,
        block_align: int,
        exporter: str = "auto",
    ) -> None:
        self.pool_bytes = pool_bytes
        self.device = device
        self.block_align = block_align
        self.exporter = exporter
        self.dmabuf_fds = [123]
        self.slabs = [SimpleNamespace(size=pool_bytes, dmabuf_fd=123, buf_slot=0)]
        self.closed = False
        self._next_address = 0
        self.allocated: list[MemoryObj] = []
        _FakeAllocator.instances.append(self)

    def owns(self, obj: MemoryObj) -> bool:
        return isinstance(obj, TensorMemoryObj) and obj.parent_allocator is self

    def decompose(self, abs_offset: int, total_len: int) -> tuple[int, int]:
        del total_len
        return 0, abs_offset

    def allocate(
        self,
        shapes: torch.Size | list[torch.Size],
        dtypes: torch.dtype | list[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        del allocator_type
        obj = _make_memory_obj(
            7,
            parent_allocator=self,
            address=self._next_address,
            fmt=fmt,
        )
        obj.meta.shape = shapes if isinstance(shapes, torch.Size) else shapes[0]
        obj.meta.dtype = dtypes if isinstance(dtypes, torch.dtype) else dtypes[0]
        self._next_address += 4096
        self.allocated.append(obj)
        return obj

    def batched_allocate(
        self,
        shapes: torch.Size | list[torch.Size],
        dtypes: torch.dtype | list[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[list[MemoryObj]]:
        del allocator_type
        return [self.allocate(shapes, dtypes, fmt) for _ in range(batch_size)]

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        del allocator_type
        memory_obj.invalidate()

    def batched_free(
        self,
        memory_objs: list[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        del allocator_type, update_stats
        for memory_obj in memory_objs:
            memory_obj.invalidate()

    def memcheck(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


class _FakeCudaTensor:
    device = SimpleNamespace(type="cuda")


class _ForeignCudaMemoryObj(TensorMemoryObj):
    @property
    def tensor(self) -> _FakeCudaTensor:
        return _FakeCudaTensor()

    @property
    def raw_tensor(self) -> _FakeCudaTensor:
        return _FakeCudaTensor()


def _make_key(chunk_hash: int = 1) -> CacheEngineKey:
    return CacheEngineKey("model", 1, 0, chunk_hash, torch.float16)


def _make_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(1, 1, 1, 1, 4),
        role="worker",
    )


def _make_config(extra_config: dict[str, object] | None = None) -> LMCacheEngineConfig:
    extra = {
        "iou_dmabuf.enabled": True,
        "iou_dmabuf.device_path": "/dev/nvme0n1",
        "iou_dmabuf.gpu_pool_bytes": 4096 * 8,
        "iou_dmabuf.capacity_bytes": 4096 * 32,
        "iou_dmabuf.block_align": 4096,
        "iou_dmabuf.header_bytes": 4096,
        "iou_dmabuf.slot_bytes": 8192,
        "iou_dmabuf.meta_total_bytes": 8192,
        "iou_dmabuf.meta_enable_periodic": False,
        "iou_dmabuf.load_checkpoint_on_init": False,
        "disk_io_threads": 1,
    }
    if extra_config:
        extra.update(extra_config)
    return LMCacheEngineConfig.from_defaults(
        local_cpu=False,
        max_local_cpu_size=0,
        extra_config=extra,
    )


def _make_memory_obj(
    size: int,
    *,
    parent_allocator: object | None = None,
    address: int = 0,
    fmt: MemoryFormat = MemoryFormat.KV_2LTD,
) -> TensorMemoryObj:
    raw_data = torch.zeros(size, dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.uint8,
        address=address,
        phy_size=((size + 4095) // 4096) * 4096,
        ref_count=1,
        pin_count=0,
        fmt=fmt,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=parent_allocator)


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    _FakeCore.instances.clear()
    _FakeAllocator.instances.clear()


@pytest.fixture
def patched_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # First Party
    import lmcache.v1.storage_backend.iou_dmabuf_backend as mod

    monkeypatch.setattr(IouDmabufBackend, "is_available", staticmethod(lambda: True))
    monkeypatch.setattr(mod, "RawBlockCore", _FakeCore)
    monkeypatch.setattr(mod, "DmabufGPUAllocator", _FakeAllocator)


def test_iou_dmabuf_available_requires_new_raw_device_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OldRawBlockDevice:
        @staticmethod
        def probe_dmabuf_support() -> bool:
            return True

    monkeypatch.setitem(
        sys.modules,
        "lmcache_rust_raw_block_io",
        types.SimpleNamespace(RawBlockDevice=OldRawBlockDevice),
    )

    assert IouDmabufBackend.is_available() is False


def test_iou_dmabuf_cpu_source_uses_raw_block_put_many(
    patched_backend: None,
) -> None:
    backend = IouDmabufBackend(
        _make_config(),
        _make_metadata(),
        asyncio.new_event_loop(),
    )
    key = _make_key()
    source = _make_memory_obj(3)
    completed: list[CacheEngineKey] = []

    try:
        futures = backend.batched_submit_put_task(
            [key],
            [source],
            on_complete_callback=completed.append,
        )
        assert futures is not None
        futures[0].result(timeout=5)
        core = _FakeCore.instances[-1]
        assert [item[0].encoded for item in core.puts] == [key.to_string()]
        assert core.rawdev.writes == []
        assert completed == [key]
        assert source.get_ref_count() == 1
    finally:
        backend.close()


def test_iou_dmabuf_owned_gpu_source_uses_write_fixed_dmabuf(
    patched_backend: None,
) -> None:
    backend = IouDmabufBackend(
        _make_config(),
        _make_metadata(),
        asyncio.new_event_loop(),
    )
    key = _make_key()
    source = _make_memory_obj(
        3,
        parent_allocator=backend.memory_allocator,
        address=4096,
    )
    completed: list[CacheEngineKey] = []

    try:
        futures = backend.batched_submit_put_task(
            [key],
            [source],
            on_complete_callback=completed.append,
        )
        assert futures is not None
        futures[0].result(timeout=5)
        core = _FakeCore.instances[-1]
        assert core.puts == []
        assert len(core.reservations) == 1
        assert core.headers[0][2] == 3
        assert core.commits[0][1] == 8192
        assert core.aborts == []
        assert core.rawdev.writes == [(0, 4096, 4096, 12288, 16)]
        assert completed == [key]
        assert source.get_ref_count() == 1
    finally:
        backend.close()


def test_iou_dmabuf_owned_gpu_no_slot_does_not_report_success(
    patched_backend: None,
) -> None:
    backend = IouDmabufBackend(
        _make_config(),
        _make_metadata(),
        asyncio.new_event_loop(),
    )
    key = _make_key()
    source = _make_memory_obj(
        3,
        parent_allocator=backend.memory_allocator,
        address=4096,
    )
    completed: list[CacheEngineKey] = []
    core = _FakeCore.instances[-1]
    # Simulate a full pool / duplicate key: reservation yields no slot.
    core.reserve_slot = lambda *args, **kwargs: None  # type: ignore[assignment]

    try:
        futures = backend.batched_submit_put_task(
            [key],
            [source],
            on_complete_callback=completed.append,
        )
        assert futures is not None
        futures[0].result(timeout=5)
        # Nothing stored -> the completion callback must not fire, and no
        # header/write/commit/abort should have run.
        assert completed == []
        assert core.headers == []
        assert core.commits == []
        assert core.aborts == []
        assert core.rawdev.writes == []
        assert source.get_ref_count() == 1
    finally:
        backend.close()


def test_iou_dmabuf_rejects_foreign_gpu_source(
    patched_backend: None,
) -> None:
    backend = IouDmabufBackend(
        _make_config(),
        _make_metadata(),
        asyncio.new_event_loop(),
    )
    key = _make_key()
    source = _ForeignCudaMemoryObj(
        torch.zeros(3, dtype=torch.uint8),
        MemoryObjMetadata(
            shape=torch.Size([3]),
            dtype=torch.uint8,
            address=0,
            phy_size=4096,
            ref_count=1,
            fmt=MemoryFormat.KV_2LTD,
        ),
        parent_allocator=None,
    )

    try:
        futures = backend.batched_submit_put_task([key], [source])
        assert futures is not None
        futures[0].result(timeout=5)
        core = _FakeCore.instances[-1]
        assert core.puts == []
        assert core.reservations == []
        assert core.rawdev.writes == []
        assert source.get_ref_count() == 1
    finally:
        backend.close()


def test_iou_dmabuf_read_locks_until_io_finishes(
    patched_backend: None,
) -> None:
    backend = IouDmabufBackend(
        _make_config(),
        _make_metadata(),
        asyncio.new_event_loop(),
    )
    key = _make_key()
    encoded = key.to_string()
    core = _FakeCore.instances[-1]
    core.entries[encoded] = (
        DiskCacheMetadata(
            path="/dev/nvme0n1@8192",
            size=3,
            shape=torch.Size([3]),
            dtype=torch.uint8,
            fmt=MemoryFormat.KV_2LTD,
        ),
        8192,
    )

    try:
        results = backend.batched_get_blocking([key])
        assert len(results) == 1
        assert results[0] is not None
        assert results[0].get_size() == 3
        assert core.rawdev.reads == [(0, 0, 4096, 12288, 16)]
        assert core.unlocked == [[encoded]]
    finally:
        backend.close()


def test_iou_dmabuf_non_blocking_get_releases_tail_after_hole(
    patched_backend: None,
) -> None:
    backend = IouDmabufBackend(
        _make_config(),
        _make_metadata(),
        asyncio.new_event_loop(),
    )
    first = _make_key(1)
    hole = _make_key(2)
    tail = _make_key(3)
    core = _FakeCore.instances[-1]
    for key in (first, tail):  # note: `hole` is intentionally absent
        core.entries[key.to_string()] = (
            DiskCacheMetadata(
                path="/dev/nvme0n1@8192",
                size=3,
                shape=torch.Size([3]),
                dtype=torch.uint8,
                fmt=MemoryFormat.KV_2LTD,
            ),
            8192,
        )

    loop = asyncio.new_event_loop()
    try:
        loaded = loop.run_until_complete(
            backend.batched_get_non_blocking("lookup", [first, hole, tail])
        )
        # Only the leading prefix (before the miss) is returned.
        assert len(loaded) == 1
        alloc = _FakeAllocator.instances[-1]
        # One object allocated per hit (first, tail).
        assert len(alloc.allocated) == 2
        returned_obj, tail_obj = alloc.allocated
        # The returned prefix object keeps its reference; the tail object loaded
        # after the hole must be released, not leaked.
        assert loaded[0] is returned_obj
        assert returned_obj.get_ref_count() == 1
        assert tail_obj.get_ref_count() == 0
    finally:
        loop.close()
        backend.close()


def test_iou_dmabuf_read_total_miss_returns_empty_list(
    patched_backend: None,
) -> None:
    backend = IouDmabufBackend(
        _make_config(),
        _make_metadata(),
        asyncio.new_event_loop(),
    )

    try:
        assert backend.batched_get_blocking([_make_key()]) == []
    finally:
        backend.close()


def test_create_storage_backends_rejects_overlapping_local_disk() -> None:
    config = LMCacheEngineConfig.from_defaults(
        local_cpu=False,
        local_disk="/tmp/lmcache",
        max_local_cpu_size=0,
        max_local_disk_size=1,
        extra_config=_make_config().extra_config,
    )

    with pytest.raises(ValueError, match="LocalDiskBackend"):
        CreateStorageBackends(config, _make_metadata(), asyncio.new_event_loop())


def test_create_storage_backends_requires_local_cpu_staging() -> None:
    # iou enabled, no overlapping disk backend, but no CPU staging allocator
    # (max_local_cpu_size=0) -> IouDmabufBackend must refuse to be the sole
    # allocator.
    config = LMCacheEngineConfig.from_defaults(
        local_cpu=False,
        max_local_cpu_size=0,
        extra_config=_make_config().extra_config,
    )

    with pytest.raises(ValueError, match="LocalCPUBackend"):
        CreateStorageBackends(config, _make_metadata(), asyncio.new_event_loop())


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DmabufGPUAllocator requires a CUDA device",
)
def test_dmabuf_allocator_rounded_read_does_not_clobber_neighbor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A block-rounded dmabuf transfer must stay inside its own allocation.

    The dmabuf read/write length is ``round_up(payload, block_align)``
    (``total_len``), which is larger than the logical payload for a
    non-block-multiple chunk. This verifies the pool allocator reserves the
    rounded size, so a ``total_len`` transfer into one allocation cannot
    corrupt the physically adjacent allocation.
    """
    # First Party
    from lmcache.v1.storage_backend import iou_dmabuf_backend as mod

    # Avoid the libcuda dependency: hand back closeable fds instead of
    # exporting real DMA-BUFs. The pool tensor itself is still real CUDA memory.
    class _FakeCudaDriver:
        def export_dmabuf(self, device_ptr: int, size: int) -> int:
            return os.open(os.devnull, os.O_RDONLY)

    monkeypatch.setattr(mod, "_CudaDriver", _FakeCudaDriver)

    block_align = 4096
    allocator = mod.DmabufGPUAllocator(
        pool_bytes=1 << 21,  # 2 MiB, one slab, page-aligned
        device="cuda:0",
        block_align=block_align,
        exporter="cuda_pool",
    )
    try:
        # 4097 bytes -> rounds up to 8192, so total_len > payload.
        shape = torch.Size([block_align + 1])
        first = allocator.allocate(shape, torch.uint8, MemoryFormat.BINARY)
        second = allocator.allocate(shape, torch.uint8, MemoryFormat.BINARY)
        assert first is not None and second is not None

        addr_first = int(first.metadata.address)
        addr_second = int(second.metadata.address)
        total_len_first = allocator.transfer_len(first)
        assert total_len_first > first.get_size()

        # The neighbor must begin at or beyond the rounded end of the first
        # allocation, so writing total_len bytes cannot reach it.
        assert addr_second >= addr_first + total_len_first

        pool = allocator.tensor
        neighbor_len = second.get_size()
        pool[addr_second : addr_second + neighbor_len].fill_(0)
        # Simulate the dmabuf read filling the full rounded transfer region.
        pool[addr_first : addr_first + total_len_first].fill_(0xFF)
        torch.cuda.synchronize()

        assert int(pool[addr_second : addr_second + neighbor_len].max().item()) == 0
    finally:
        allocator.close()
