# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from dataclasses import replace
import sys
import types

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import RawBlockCore, encode_object_key
from tests.v1.storage_backend.raw_block_test_utils import (
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)

pytest.importorskip("lmcache_rust_raw_block_io")


def test_raw_block_core_store_load_and_exists(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        keys = [make_object_key(i) for i in range(3)]
        specs = [encode_object_key(key) for key in keys]
        payloads = [
            bytes([1]) * 1024,
            bytes([2]) * 2048,
            bytes([3]) * 3072,
        ]
        objects = [make_memory_obj(payload) for payload in payloads]

        put_result = core.put_many(specs, objects)

        assert put_result.results == [True, True, True]
        assert put_result.stored_keys == [spec.encoded for spec in specs]
        assert core.exists_many([spec.encoded for spec in specs]) == [
            True,
            True,
            True,
        ]

        loaded = [make_empty_memory_obj(len(payload)) for payload in payloads]
        load_result = core.load_many_into([spec.encoded for spec in specs], loaded)

        assert load_result == [True, True, True]
        assert [memory_obj_bytes(obj) for obj in loaded] == payloads
    finally:
        core.close()


def test_raw_block_core_duplicate_put_keeps_original_payload(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        spec = encode_object_key(make_object_key(11))
        original = b"original"
        duplicate = b"mutated!"

        first_result = core.put_many([spec], [make_memory_obj(original)])
        duplicate_result = core.put_many([spec], [make_memory_obj(duplicate)])

        assert first_result.results == [True]
        assert first_result.stored_keys == [spec.encoded]
        assert duplicate_result.results == [True]
        assert duplicate_result.stored_keys == []

        loaded = make_empty_memory_obj(len(original))
        assert core.load_many_into([spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == original
    finally:
        core.close()


def test_raw_block_core_delete_and_missing_load(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        existing = encode_object_key(make_object_key(21))
        missing = encode_object_key(make_object_key(22))

        put_result = core.put_many([existing], [make_memory_obj(b"delete-me")])
        assert put_result.results == [True]
        assert core.contains_key(existing.encoded) is True

        assert core.delete_many([existing.encoded, missing.encoded]) == [True, False]
        assert core.exists_many([existing.encoded, missing.encoded]) == [False, False]

        loaded = make_empty_memory_obj(len(b"delete-me"))
        assert core.load_many_into([existing.encoded], [loaded]) == [False]
    finally:
        core.close()


def test_raw_block_core_get_entries_many_locks_hits(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        existing = encode_object_key(make_object_key(26))
        missing = encode_object_key(make_object_key(27))
        payload = b"entry-lookup"

        put_result = core.put_many([existing], [make_memory_obj(payload)])
        assert put_result.results == [True]

        entries = core.get_entries_many(
            [existing.encoded, missing.encoded],
            lock_refcount=True,
        )

        assert entries[1] is None
        assert entries[0] is not None
        meta, offset = entries[0]
        assert meta.size == len(payload)
        assert offset == core.entry_offset(existing.encoded)
        assert core.lock_refcount(existing.encoded) == 1
        assert core.delete_many([existing.encoded], force=False) == [False]

        core.unlock_many([existing.encoded])
        assert core.lock_refcount(existing.encoded) == 0
        assert core.delete_many([existing.encoded], force=False) == [True]
    finally:
        core.close()


def test_raw_block_core_force_delete_defers_slot_reuse_until_unlock(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        spec = encode_object_key(make_object_key(35))
        obj = make_memory_obj(b"locked-read")
        assert core.put_many([spec], [obj]).results == [True]
        offset = core.entry_offset(spec.encoded)
        assert offset is not None

        assert core.exists_many([spec.encoded], lock=True) == [True]
        assert core.delete_many([spec.encoded], force=True) == [True]
        assert core.contains_key(spec.encoded) is False
        assert core.report_status()["retired_key_count"] == 1

        # The key is gone, but its slot remains reserved while a DMA reader
        # may still be using the corresponding device offset.
        assert core.reserve_slot(spec, obj) is None

        core.unlock_many([spec.encoded])
        assert core.report_status()["retired_key_count"] == 0
        reused = core.reserve_slot(spec, obj)
        assert reused == offset
        core.abort_slot(spec, reused)
    finally:
        core.close()


def test_raw_block_core_slot_lifecycle_commit_external_payload(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        spec = encode_object_key(make_object_key(28))
        payload = b"external-payload"
        obj = make_memory_obj(payload)

        offset = core.reserve_slot(spec, obj)
        assert offset is not None
        assert core.reserve_slot(spec, obj) is None
        assert core.write_slot_header(spec, offset, obj.get_size()) is True
        core.raw_device().pwrite_from_buffer(
            offset + core.header_bytes,
            obj.byte_array,
            obj.get_size(),
            obj.get_size(),
        )

        assert core.commit_slot(spec, offset) is True
        assert core.commit_slot(spec, offset) is False

        loaded = make_empty_memory_obj(len(payload))
        assert core.load_many_into([spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == payload
    finally:
        core.close()


def test_raw_block_core_write_slot_header_pads_buffer_for_iouring(
    tmp_path,
    monkeypatch,
):
    calls: list[tuple[int, int, int, int]] = []

    class FakeRawBlockDevice:
        def __init__(self, path: str, **kwargs):
            del path, kwargs
            self._size = 128 * 1024 * 1024

        def size_bytes(self):
            return self._size

        def write_uring(self, offset, data, payload_len, total_len=None):
            calls.append((offset, len(data), payload_len, total_len))

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "lmcache_rust_raw_block_io",
        types.SimpleNamespace(RawBlockDevice=FakeRawBlockDevice),
    )

    path = make_raw_block_file(tmp_path)
    config = replace(
        make_raw_block_core_config(path),
        io_engine="io_uring",
        use_odirect=True,
        load_checkpoint_on_init=False,
    )
    core = RawBlockCore(config, key_namespace="object")

    try:
        spec = encode_object_key(make_object_key(34))
        obj = make_memory_obj(b"external-payload")
        offset = core.reserve_slot(spec, obj)
        assert offset is not None

        assert core.write_slot_header(spec, offset, obj.get_size()) is True
        assert calls == [(offset, 4096, 4096, 4096)]
    finally:
        core.close()


def test_raw_block_core_slot_lifecycle_abort_recycles_slot(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        first = encode_object_key(make_object_key(29))
        second = encode_object_key(make_object_key(30))
        obj = make_memory_obj(b"abort-me")

        offset = core.reserve_slot(first, obj)
        assert offset is not None
        core.abort_slot(first, offset)
        assert core.contains_key(first.encoded) is False

        reused = core.reserve_slot(second, obj)
        assert reused == offset
        core.abort_slot(second, reused)
    finally:
        core.close()


def test_raw_block_core_commit_recycles_canceled_inflight_slot(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        first = encode_object_key(make_object_key(32))
        second = encode_object_key(make_object_key(33))
        obj = make_memory_obj(b"cancel-me")

        offset = core.reserve_slot(first, obj)
        assert offset is not None
        assert core.delete_many([first.encoded], force=False) == [True]
        assert core.commit_slot(first, offset) is False
        assert core.contains_key(first.encoded) is False

        # A writer's finally-block aborts unconditionally; after a canceled
        # commit already freed the slot this must be a benign no-op, not a
        # double-free. The slot is recycled exactly once, proven by ``second``
        # reusing the same offset below.
        core.abort_slot(first, offset)

        reused = core.reserve_slot(second, obj)
        assert reused == offset
        core.abort_slot(second, reused)
    finally:
        core.close()


def test_raw_block_core_recovers_checkpoint_from_temp_file(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    spec = encode_object_key(make_object_key(31))
    payload = b"recoverable-raw-block-payload"

    core = RawBlockCore(config, key_namespace="object")
    try:
        put_result = core.put_many([spec], [make_memory_obj(payload)])
        assert put_result.results == [True]
        core.checkpoint_now()
    finally:
        core.close()

    recovered = RawBlockCore(config, key_namespace="object")
    try:
        assert recovered.contains_key(spec.encoded) is True
        loaded = make_empty_memory_obj(len(payload))
        assert recovered.load_many_into([spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == payload
    finally:
        recovered.close()
