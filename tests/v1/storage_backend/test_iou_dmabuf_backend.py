# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock
import asyncio
import ctypes
import logging
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
from lmcache.v1.storage_backend.storage_manager import StorageManager


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
        self.lock_counts: dict[str, int] = {}
        self.closed = False
        _FakeCore.instances.append(self)

    def raw_device(self) -> _FakeRawDevice:
        return self.rawdev

    def exists_many(self, encoded_keys: list[str], *, lock: bool = False) -> list[bool]:
        if lock:
            for encoded_key in encoded_keys:
                if encoded_key in self.entries:
                    self.lock_counts[encoded_key] = (
                        self.lock_counts.get(encoded_key, 0) + 1
                    )
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
        for encoded_key in encoded_keys:
            count = self.lock_counts.get(encoded_key, 0)
            if count <= 1:
                self.lock_counts.pop(encoded_key, None)
            else:
                self.lock_counts[encoded_key] = count - 1

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
        mem_range_flags: int = 0,
    ) -> None:
        self.pool_bytes = pool_bytes
        self.device = device
        self.block_align = block_align
        self.exporter = exporter
        self.mem_range_flags = mem_range_flags
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
        shape = shapes if isinstance(shapes, torch.Size) else shapes[0]
        dtype = dtypes if isinstance(dtypes, torch.dtype) else dtypes[0]
        # Build a shape-consistent object (raw_data sized to shape * dtype) so
        # get_size() and `.tensor` derive from the real layout, mirroring a real
        # allocator. A hardcoded mismatched size would only appear correct when
        # masked by set_used_size(), which the read path must NOT call.
        obj = _make_shaped_memory_obj(
            shape,
            dtype,
            parent_allocator=self,
            address=self._next_address,
            fmt=fmt,
        )
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


def _make_shaped_memory_obj(
    shape: torch.Size,
    dtype: torch.dtype,
    *,
    parent_allocator: object | None = None,
    address: int = 0,
    fmt: MemoryFormat = MemoryFormat.KV_2LTD,
) -> TensorMemoryObj:
    """Build a TensorMemoryObj whose raw buffer matches ``shape``/``dtype``.

    Unlike ``_make_memory_obj`` (a flat uint8 buffer), this keeps the layout
    self-consistent so ``get_size()`` and ``.tensor`` derive from the real KV
    shape, which is the shape a retrieved object must expose to the GPU
    connector.
    """
    nbytes = int(shape.numel()) * torch.empty(0, dtype=dtype).element_size()
    raw_data = torch.zeros(nbytes, dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=shape,
        dtype=dtype,
        address=address,
        phy_size=((nbytes + 4095) // 4096) * 4096,
        ref_count=1,
        pin_count=0,
        fmt=fmt,
        shapes=[shape],
        dtypes=[dtype],
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


def test_iou_dmabuf_put_rejects_mismatched_batch_before_scheduling(
    patched_backend: None,
) -> None:
    backend = IouDmabufBackend(
        _make_config(),
        _make_metadata(),
        asyncio.new_event_loop(),
    )
    first = _make_memory_obj(3)
    second = _make_memory_obj(3)
    core = _FakeCore.instances[-1]

    try:
        with pytest.raises(ValueError, match="same length"):
            backend.batched_submit_put_task(
                [_make_key(1), _make_key(2)],
                [first],
            )
        with pytest.raises(ValueError, match="same length"):
            backend.batched_submit_put_task(
                [_make_key(1)],
                [first, second],
            )

        assert core.puts == []
        assert core.reservations == []
        assert first.get_ref_count() == 1
        assert second.get_ref_count() == 1
        assert backend.exists_in_put_tasks(_make_key(1)) is False
        assert backend.exists_in_put_tasks(_make_key(2)) is False
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
        # Release the allocator-owned source so it is not GC'd with a live
        # ref_count and log a leak warning (addr 4096 in the test logs).
        source.ref_count_down()
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
        # Release the allocator-owned source so it is not GC'd with a live
        # ref_count and log a leak warning (addr 4096 in the test logs).
        source.ref_count_down()
        backend.close()


def test_iou_dmabuf_background_put_failure_is_logged(
    patched_backend: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.ERROR,
        logger="lmcache.v1.storage_backend.iou_dmabuf_backend",
    )
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
    core = _FakeCore.instances[-1]
    core.rawdev.write_fixed_dmabuf = MagicMock(
        side_effect=RuntimeError("forced dmabuf write failure")
    )
    source_released = False

    try:
        futures = backend.batched_submit_put_task([key], [source])
        assert futures is not None
        with pytest.raises(RuntimeError, match="forced dmabuf write failure"):
            futures[0].result(timeout=5)
        source.ref_count_down()
        source_released = True
        backend.close()
        assert "IouDmabufBackend: background task failed" in caplog.text
        assert "forced dmabuf write failure" in caplog.text
        assert len(core.aborts) == 1
    finally:
        if not source_released:
            source.ref_count_down()
        if not core.closed:
            backend.close()


def test_iou_dmabuf_rejects_foreign_gpu_source(
    patched_backend: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Rejecting a foreign GPU source logs an expected warning. Raise the
    # backend logger above WARNING for this test so the expected message does
    # not clutter output; the rejection itself is verified by the state
    # assertions below (nothing reserved, written, or stored).
    caplog.set_level(
        logging.ERROR,
        logger="lmcache.v1.storage_backend.iou_dmabuf_backend",
    )
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
        # Release the retrieved object; in production the caller
        # (StorageManager/cache_engine) owns this ref_count_down. Doing it here
        # avoids the "garbage collected with ref_count=1" leak warning.
        results[0].ref_count_down()
    finally:
        backend.close()


def test_iou_dmabuf_get_non_blocking_with_one_worker(
    patched_backend: None,
) -> None:
    backend = IouDmabufBackend(
        _make_config({"disk_io_threads": 1}),
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
        future = backend.get_non_blocking(key)
        assert future is not None
        result = future.result(timeout=1)
        assert result is not None
        assert core.rawdev.reads == [(0, 0, 4096, 12288, 16)]
        assert core.unlocked == [[encoded]]
        result.ref_count_down()
    finally:
        backend.close()


def test_iou_dmabuf_read_returns_reshaped_kv_tensor(
    patched_backend: None,
) -> None:
    # Regression: a retrieved object's `.tensor` must reshape to the stored
    # multi-dim KV_2LTD shape. A stray set_used_size() call in the read path
    # sets _used_size_override, which forces MemoryObj.tensor to a flat 1-D
    # uint8 view; the GPU connector then indexes dim 3 of a 1-D tensor and
    # raises "IndexError: Dimension out of range". Asserting the tensor rank
    # and shape guards that regression through the public interface.
    backend = IouDmabufBackend(
        _make_config(),
        _make_metadata(),
        asyncio.new_event_loop(),
    )
    key = _make_key()
    encoded = key.to_string()
    kv_shape = torch.Size([2, 1, 1, 4])  # KV_2LTD, 4-D
    kv_dtype = torch.float16
    kv_bytes = int(kv_shape.numel()) * torch.empty(0, dtype=kv_dtype).element_size()
    core = _FakeCore.instances[-1]
    core.entries[encoded] = (
        DiskCacheMetadata(
            path="/dev/nvme0n1@8192",
            size=kv_bytes,
            shape=kv_shape,
            dtype=kv_dtype,
            fmt=MemoryFormat.KV_2LTD,
        ),
        8192,
    )

    try:
        results = backend.batched_get_blocking([key])
        assert len(results) == 1
        obj = results[0]
        assert obj is not None
        assert obj.metadata.fmt == MemoryFormat.KV_2LTD
        tensor = obj.tensor
        assert tensor is not None
        # Must be the full multi-dim KV tensor, not a flattened 1-D view.
        assert tensor.dim() == len(kv_shape)
        assert tuple(tensor.shape) == tuple(kv_shape)
        assert tensor.dtype == kv_dtype
        # Release the retrieved object (caller-owned in production) so it is
        # not GC'd with a live ref_count, which would log a leak warning.
        obj.ref_count_down()
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
        # Release the returned prefix object (caller-owned in production) so it
        # is not GC'd with a live ref_count and log a leak warning.
        returned_obj.ref_count_down()
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


def test_iou_dmabuf_nested_pins_release_matching_core_locks(
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
        assert backend.pin(key) is True
        assert backend.pin(key) is True
        assert core.lock_counts[encoded] == 2

        assert backend.unpin(key) is True
        assert core.lock_counts[encoded] == 1
        assert backend.unpin(key) is True
        assert encoded not in core.lock_counts
        assert core.unlocked == [[encoded], [encoded]]
    finally:
        backend.close()


def test_storage_manager_does_not_write_back_iou_gpu_objects() -> None:
    manager = StorageManager.__new__(StorageManager)
    iou_backend = MagicMock()
    local_cpu_backend = MagicMock()
    key = _make_key()
    single = _make_memory_obj(3)
    batched = _make_memory_obj(3)
    manager.storage_backends = OrderedDict(
        [
            ("LocalCPUBackend", local_cpu_backend),
            ("IouDmabufBackend", iou_backend),
        ]
    )
    manager.get_active_storage_backends = MagicMock(  # type: ignore[method-assign]
        return_value=iter([("IouDmabufBackend", iou_backend)])
    )

    try:
        iou_backend.get_blocking.return_value = single
        assert manager.get(key) is single
        local_cpu_backend.submit_put_task.assert_not_called()

        active_backends = manager.get_active_storage_backends
        assert isinstance(active_backends, MagicMock)
        active_backends.return_value = iter([("IouDmabufBackend", iou_backend)])
        iou_backend.batched_get_blocking.return_value = [batched]
        assert manager.batched_get([key]) == [batched]
        local_cpu_backend.batched_submit_put_task.assert_not_called()
    finally:
        single.ref_count_down()
        batched.ref_count_down()


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


def test_create_storage_backends_rejects_adding_local_disk_when_iou_exists() -> None:
    # Dynamic recreation: IouDmabufBackend already exists and is skipped, while
    # LocalDiskBackend is being added. Strict overlap rejection must still fire
    # even though Iou is not being (re)created in this call.
    config = LMCacheEngineConfig.from_defaults(
        local_cpu=False,
        max_local_cpu_size=0,
        local_disk="/tmp/lmcache",
        max_local_disk_size=1,
        extra_config=_make_config().extra_config,
    )
    existing: OrderedDict[str, object] = OrderedDict()
    existing["IouDmabufBackend"] = object()  # sentinel; only membership is read

    with pytest.raises(ValueError, match="LocalDiskBackend"):
        CreateStorageBackends(
            config,
            _make_metadata(),
            asyncio.new_event_loop(),
            skip_backends={"IouDmabufBackend"},
            existing_backends=existing,  # type: ignore[arg-type]
        )


def test_create_storage_backends_iou_accepts_reused_local_cpu(
    patched_backend: None,
) -> None:
    # A LocalCPUBackend that already exists is reused (not re-added to the new
    # dict) when skipped. The staging guard must accept it via the reused
    # instance rather than dict membership.
    # First Party
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

    config = LMCacheEngineConfig.from_defaults(
        local_cpu=False,
        max_local_cpu_size=0,  # no new LocalCPUBackend is created
        extra_config=_make_config().extra_config,
    )
    existing: OrderedDict[str, object] = OrderedDict()
    existing["LocalCPUBackend"] = MagicMock(spec=LocalCPUBackend)

    backends = CreateStorageBackends(
        config,
        _make_metadata(),
        asyncio.new_event_loop(),
        skip_backends={"LocalCPUBackend"},
        existing_backends=existing,  # type: ignore[arg-type]
    )
    assert "IouDmabufBackend" in backends


def test_allocate_and_copy_objects_skips_present_keys_without_misaligning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A key already present in the target allocator is skipped; the returned
    # keys must be the actual keys copied, not a count-based prefix of the input
    # (which would mislabel later objects and silently drop a key downstream).
    # First Party
    from lmcache.v1.storage_backend import storage_manager as sm

    # Neutralize the device-stream context so the copy runs on plain CPU tensors.
    monkeypatch.setattr(
        sm,
        "torch_dev",
        SimpleNamespace(stream=lambda stream: nullcontext()),
    )

    present_key = _make_key(1)
    absent_key = _make_key(2)
    src_present = _make_memory_obj(4)
    src_absent = _make_memory_obj(4)

    class _StagingAllocator:
        def __init__(self) -> None:
            self.made: list[MemoryObj] = []

        def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
            return key.to_string() == present_key.to_string()

        def allocate(
            self,
            shape: torch.Size,
            dtype: torch.dtype,
            fmt: MemoryFormat = MemoryFormat.KV_2LTD,
            eviction: bool = True,
            busy_loop: bool = True,
        ) -> MemoryObj:
            obj = _make_memory_obj(4)
            self.made.append(obj)
            return obj

    allocator = _StagingAllocator()
    keys, objs = sm.allocate_and_copy_objects(
        allocator,
        [present_key, absent_key],
        [src_present, src_absent],
        stream=None,
    )

    assert [key.to_string() for key in keys] == [absent_key.to_string()]
    assert len(objs) == 1
    assert objs[0] is allocator.made[0]


def test_dmabuf_exporter_auto_prefers_hip_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    from lmcache.v1.storage_backend import iou_dmabuf_backend as mod

    monkeypatch.setattr(mod, "_torch_uses_hip", lambda: True)
    assert mod._resolve_dmabuf_exporter("auto") == "hip"
    assert mod._resolve_dmabuf_exporter("HIP") == "hip"

    monkeypatch.setattr(mod, "_torch_uses_hip", lambda: False)
    assert mod._resolve_dmabuf_exporter("auto") == "cuda_pool"
    assert mod._resolve_dmabuf_exporter("cuda_pool") == "cuda_pool"


def test_iou_dmabuf_allocator_receives_mem_range_flags(
    patched_backend: None,
) -> None:
    backend = IouDmabufBackend(
        _make_config({"iou_dmabuf.mem_range_flags": 1}),
        _make_metadata(),
        asyncio.new_event_loop(),
    )

    try:
        allocator = _FakeAllocator.instances[-1]
        assert allocator.mem_range_flags == 1
    finally:
        backend.close()


def test_iou_dmabuf_rejects_negative_mem_range_flags(
    patched_backend: None,
) -> None:
    with pytest.raises(ValueError, match="iou_dmabuf.mem_range_flags"):
        IouDmabufBackend(
            _make_config({"iou_dmabuf.mem_range_flags": -1}),
            _make_metadata(),
            asyncio.new_event_loop(),
        )


def test_iou_dmabuf_rejects_non_direct_io_config(
    patched_backend: None,
) -> None:
    with pytest.raises(ValueError, match="use_odirect=true"):
        IouDmabufBackend(
            _make_config({"iou_dmabuf.use_odirect": False}),
            _make_metadata(),
            asyncio.new_event_loop(),
        )


def test_iou_dmabuf_rejects_uring_cmd_config(
    patched_backend: None,
) -> None:
    with pytest.raises(ValueError, match="does not support io_uring_cmd"):
        IouDmabufBackend(
            _make_config({"iou_dmabuf.use_uring_cmd": True}),
            _make_metadata(),
            asyncio.new_event_loop(),
        )


def test_iou_dmabuf_rejects_unimplemented_require_p2p(
    patched_backend: None,
) -> None:
    with pytest.raises(NotImplementedError, match="require_p2p=true"):
        IouDmabufBackend(
            _make_config({"iou_dmabuf.require_p2p": True}),
            _make_metadata(),
            asyncio.new_event_loop(),
        )


def test_hip_driver_exports_dmabuf_with_address_range_handle_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First Party
    from lmcache.v1.storage_backend import iou_dmabuf_backend as mod

    class _FakeCFunc:
        def __init__(self, impl: Any) -> None:
            self._impl = impl
            self.argtypes: list[Any] | None = None
            self.restype: Any = None

        def __call__(self, *args: Any) -> int:
            return int(self._impl(*args))

    class _FakeHipLib:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []
            self.hipInit = _FakeCFunc(self._hip_init)
            self.hipSetDevice = _FakeCFunc(self._hip_set_device)
            self.hipMemGetHandleForAddressRange = _FakeCFunc(self._export)

        @staticmethod
        def _value(value: Any) -> int:
            return int(value.value if hasattr(value, "value") else value)

        def _hip_init(self, flags: Any) -> int:
            self.calls.append(("hipInit", self._value(flags)))
            return 0

        def _hip_set_device(self, device: Any) -> int:
            self.calls.append(("hipSetDevice", self._value(device)))
            return 0

        def _export(
            self,
            handle: Any,
            dptr: Any,
            size: Any,
            handle_type: Any,
            flags: Any,
        ) -> int:
            ctypes.cast(handle, ctypes.POINTER(ctypes.c_int)).contents.value = 456
            self.calls.append(
                (
                    "hipMemGetHandleForAddressRange",
                    self._value(dptr),
                    self._value(size),
                    self._value(handle_type),
                    self._value(flags),
                )
            )
            return 0

    fake_lib = _FakeHipLib()
    monkeypatch.setattr(mod, "_load_shared_library", lambda _: fake_lib)

    driver = mod._HipDriver(device_index=2)
    assert driver.export_dmabuf(0x1000, 4096, flags=1) == 456
    assert fake_lib.calls == [
        ("hipInit", 0),
        ("hipSetDevice", 2),
        ("hipMemGetHandleForAddressRange", 0x1000, 4096, 1, 1),
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DmabufGPUAllocator requires a torch CUDA/HIP device",
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
        def export_dmabuf(
            self,
            device_ptr: int,
            size: int,
            flags: int = 0,
        ) -> int:
            del device_ptr, size, flags
            return os.open(os.devnull, os.O_RDONLY)

    monkeypatch.setattr(mod, "_CudaDriver", _FakeCudaDriver)

    block_align = 4096
    allocator = mod.DmabufGPUAllocator(
        pool_bytes=1 << 21,  # 2 MiB, one slab, page-aligned
        device="cuda:0",
        block_align=block_align,
        exporter="cuda_pool",
    )
    first: Optional[MemoryObj] = None
    second: Optional[MemoryObj] = None
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
        # Release both allocations before closing so they are not GC'd with a
        # live ref_count (which logs "garbage collected with ref_count=1").
        # These are the addr 0 / addr 8192 objects seen in the test logs.
        if first is not None:
            first.ref_count_down()
        if second is not None:
            second.ref_count_down()
        allocator.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DmabufGPUAllocator requires a torch CUDA/HIP device",
)
def test_dmabuf_allocator_reuses_capacity_across_slab_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each exported slab must retain independently reusable free capacity."""
    # First Party
    from lmcache.v1.storage_backend import iou_dmabuf_backend as mod

    class _FakeCudaDriver:
        def export_dmabuf(
            self,
            device_ptr: int,
            size: int,
            flags: int = 0,
        ) -> int:
            del device_ptr, size, flags
            return os.open(os.devnull, os.O_RDONLY)

    slab_bytes = 64 * 1024
    monkeypatch.setattr(mod, "_SZ_1G", slab_bytes)
    monkeypatch.setattr(mod, "_CudaDriver", _FakeCudaDriver)
    allocator = mod.DmabufGPUAllocator(
        pool_bytes=2 * slab_bytes,
        device="cuda:0",
        block_align=4096,
        exporter="cuda_pool",
    )
    first: Optional[MemoryObj] = None
    second: Optional[MemoryObj] = None
    replacement: Optional[MemoryObj] = None
    try:
        shape = torch.Size([48 * 1024])
        first = allocator.allocate(shape, torch.uint8, MemoryFormat.BINARY)
        second = allocator.allocate(shape, torch.uint8, MemoryFormat.BINARY)
        assert first is not None and second is not None
        assert int(first.metadata.address) == 0
        assert int(second.metadata.address) == slab_bytes
        assert allocator.allocate(shape, torch.uint8, MemoryFormat.BINARY) is None

        first.ref_count_down()
        first = None
        replacement = allocator.allocate(shape, torch.uint8, MemoryFormat.BINARY)
        assert replacement is not None
        assert int(replacement.metadata.address) == 0
    finally:
        if first is not None:
            first.ref_count_down()
        if second is not None:
            second.ref_count_down()
        if replacement is not None:
            replacement.ref_count_down()
        allocator.close()
