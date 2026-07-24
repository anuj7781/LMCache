# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union
import asyncio
import ctypes
import ctypes.util
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    TensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.raw_block import (
    DEFAULT_IOURING_QUEUE_DEPTH,
    RawBlockCore,
    RawBlockCoreConfig,
    RawBlockKeySpec,
    encode_legacy_key,
    round_up,
    validate_raw_block_io_options,
)

logger = init_logger(__name__)

_DEFAULT_META_MAGIC = b"LMCIDX01"
_DEFAULT_META_VERSION = 1
_DEFAULT_META_TOTAL_BYTES = 128 * 1024 * 1024
_DEFAULT_HEADER_BYTES = 4096
_DEFAULT_BLOCK_ALIGN = 4096
_DEFAULT_DMABUF_EAGAIN_RETRIES = 16
_DEFAULT_DISK_IO_THREADS = 4
_SZ_1G = 1 << 30

_CU_SUCCESS = 0
_CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1

_HIP_SUCCESS = 0
_HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1


@dataclass(frozen=True)
class SlabDescriptor:
    """Registered GPU dmabuf slab metadata.

    Args:
        device_ptr: GPU device virtual address for the slab base.
        dmabuf_fd: Exported DMA-BUF file descriptor.
        buf_slot: io_uring registered buffer table index.
        size: Slab length in bytes.
    """

    device_ptr: int
    dmabuf_fd: int
    buf_slot: int
    size: int


def _load_shared_library(names: Sequence[str]) -> ctypes.CDLL:
    """Load the first available shared library from ``names``."""
    errors: list[str] = []
    tried: set[str] = set()
    for name in names:
        candidates: list[str] = []
        found = ctypes.util.find_library(name)
        if found is not None:
            candidates.append(found)
        candidates.append(name)
        if not name.startswith("lib") and ".so" not in name:
            candidates.append(f"lib{name}.so")

        for candidate in candidates:
            if candidate in tried:
                continue
            tried.add(candidate)
            try:
                return ctypes.CDLL(candidate)
            except OSError as e:
                errors.append(f"{candidate}: {e}")

    raise OSError("; ".join(errors) or "no shared library candidates supplied")


def _torch_uses_hip() -> bool:
    """Return whether this PyTorch build targets ROCm/HIP."""
    return bool(getattr(torch.version, "hip", None))


def _resolve_dmabuf_exporter(exporter: str) -> str:
    """Resolve ``auto`` and validate the requested dmabuf exporter."""
    normalized = exporter.strip().lower()
    if normalized == "auto":
        return "hip" if _torch_uses_hip() else "cuda_pool"
    if normalized in {"cuda_pool", "hip"}:
        return normalized
    if normalized in {"cuda_vmm", "amd_drm"}:
        raise NotImplementedError(
            f"IouDmabufBackend exporter='{normalized}' is still deferred; "
            "first cut supports exporter='auto', exporter='cuda_pool', "
            "or exporter='hip'"
        )
    raise ValueError(
        "IouDmabufBackend exporter must be one of: auto, cuda_pool, hip"
    )


def _torch_cuda_device_index(device: torch.device) -> int:
    """Return the active torch CUDA/HIP device index for driver calls."""
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


class _CudaDriver:
    """Small ctypes wrapper for CUDA dmabuf address-range export."""

    def __init__(self) -> None:
        try:
            self._lib = ctypes.CDLL("libcuda.so.1")
        except OSError:
            self._lib = ctypes.CDLL("libcuda.so")

        self._lib.cuInit.argtypes = [ctypes.c_uint]
        self._lib.cuInit.restype = ctypes.c_int
        rc = int(self._lib.cuInit(0))
        if rc != _CU_SUCCESS:
            raise RuntimeError(f"cuInit failed with CUDA error {rc}")

        try:
            export_fn = self._lib.cuMemGetHandleForAddressRange
        except AttributeError as e:
            raise RuntimeError(
                "CUDA driver does not expose cuMemGetHandleForAddressRange"
            ) from e
        export_fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_ulonglong,
        ]
        export_fn.restype = ctypes.c_int
        self._export_fn = export_fn

    def export_dmabuf(self, device_ptr: int, size: int, flags: int = 0) -> int:
        """Export a CUDA device address range as a DMA-BUF fd.

        Args:
            device_ptr: Page-aligned CUDA device pointer.
            size: Page-aligned range size in bytes.
            flags: CUDA address-range export flags.

        Returns:
            The exported DMA-BUF file descriptor.

        Raises:
            RuntimeError: If CUDA rejects the address range export.
        """
        fd = ctypes.c_int(-1)
        rc = int(
            self._export_fn(
                ctypes.byref(fd),
                ctypes.c_uint64(device_ptr),
                ctypes.c_size_t(size),
                ctypes.c_uint(_CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD),
                ctypes.c_ulonglong(flags),
            )
        )
        if rc != _CU_SUCCESS:
            raise RuntimeError(
                "cuMemGetHandleForAddressRange failed with CUDA error "
                f"{rc} for ptr={device_ptr:#x}, size={size}"
            )
        if fd.value < 0:
            raise RuntimeError(
                "cuMemGetHandleForAddressRange succeeded but returned an invalid fd"
            )
        return int(fd.value)


class _HipDriver:
    """Small ctypes wrapper for HIP dmabuf address-range export."""

    def __init__(self, device_index: int) -> None:
        # Try the ldconfig-resolved soname first (covers whichever ROCm major is
        # installed), then explicit versioned fallbacks for runtime-only installs
        # that ship only ``libamdhip64.so.<N>`` (no unversioned dev symlink).
        self._lib = _load_shared_library(
            (
                "amdhip64",
                "libamdhip64.so.6",
                "libamdhip64.so.5",
                "libamdhip64.so",
            )
        )

        try:
            init_fn = self._lib.hipInit
        except AttributeError as e:
            raise RuntimeError("HIP runtime does not expose hipInit") from e
        init_fn.argtypes = [ctypes.c_uint]
        init_fn.restype = ctypes.c_int
        rc = int(init_fn(0))
        if rc != _HIP_SUCCESS:
            raise RuntimeError(f"hipInit failed with HIP error {rc}")

        try:
            set_device_fn = self._lib.hipSetDevice
        except AttributeError as e:
            raise RuntimeError("HIP runtime does not expose hipSetDevice") from e
        set_device_fn.argtypes = [ctypes.c_int]
        set_device_fn.restype = ctypes.c_int
        rc = int(set_device_fn(ctypes.c_int(device_index)))
        if rc != _HIP_SUCCESS:
            raise RuntimeError(
                f"hipSetDevice({device_index}) failed with HIP error {rc}"
            )

        try:
            export_fn = self._lib.hipMemGetHandleForAddressRange
        except AttributeError as e:
            raise RuntimeError(
                "HIP runtime does not expose hipMemGetHandleForAddressRange"
            ) from e
        export_fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_ulonglong,
        ]
        export_fn.restype = ctypes.c_int
        self._export_fn = export_fn

    def export_dmabuf(self, device_ptr: int, size: int, flags: int = 0) -> int:
        """Export a HIP device address range as a DMA-BUF fd.

        Args:
            device_ptr: Page-aligned HIP device pointer.
            size: Page-aligned range size in bytes.
            flags: HIP address-range export flags.

        Returns:
            The exported DMA-BUF file descriptor.

        Raises:
            RuntimeError: If HIP rejects the address range export.
        """
        fd = ctypes.c_int(-1)
        rc = int(
            self._export_fn(
                ctypes.byref(fd),
                ctypes.c_uint64(device_ptr),
                ctypes.c_size_t(size),
                ctypes.c_uint(_HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD),
                ctypes.c_ulonglong(flags),
            )
        )
        if rc != _HIP_SUCCESS:
            raise RuntimeError(
                "hipMemGetHandleForAddressRange failed with HIP error "
                f"{rc} for ptr={device_ptr:#x}, size={size}"
            )
        if fd.value < 0:
            raise RuntimeError(
                "hipMemGetHandleForAddressRange succeeded but returned an invalid fd"
            )
        return int(fd.value)


def _get_extra_bool(
    extra: Mapping[str, Any],
    key: str,
    default: bool = False,
) -> bool:
    """Return a boolean value from ``extra_config``.

    Args:
        extra: Engine ``extra_config`` mapping.
        key: Extra config key to read.
        default: Value used when ``key`` is absent.

    Returns:
        Parsed boolean value.

    Raises:
        ValueError: If the configured value is not boolean-like.
    """
    value = extra.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"extra_config['{key}'] must be a boolean")


def _get_extra_int(
    extra: Mapping[str, Any],
    key: str,
    default: int | None = None,
    *,
    required: bool = False,
    positive: bool = False,
) -> int:
    """Return an integer value from ``extra_config``.

    Args:
        extra: Engine ``extra_config`` mapping.
        key: Extra config key to read.
        default: Value used when ``key`` is absent.
        required: Whether absence is an error.
        positive: Whether the parsed value must be greater than zero.

    Returns:
        Parsed integer value.

    Raises:
        ValueError: If the value is missing or invalid.
    """
    if key not in extra:
        if required or default is None:
            raise ValueError(f"extra_config['{key}'] is required")
        value = default
    else:
        value = extra[key]
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"extra_config['{key}'] must be an integer") from e
    if positive and parsed <= 0:
        raise ValueError(f"extra_config['{key}'] must be > 0")
    return parsed


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _make_page_aligned_gpu_pool(
    size: int,
    device: torch.device,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate a GPU uint8 pool with a page-aligned visible view.

    Args:
        size: Visible pool size in bytes.
        device: PyTorch CUDA/HIP device for allocation.
        page_size: Host page size required by dmabuf export.

    Returns:
        ``(base_tensor, aligned_view)``. The base tensor keeps the owning
        allocation alive; the aligned view is the exported pool.
    """
    base_tensor = torch.empty(size + page_size, dtype=torch.uint8, device=device)
    align_offset = (-base_tensor.data_ptr()) % page_size
    pool_tensor = base_tensor[align_offset : align_offset + size]
    return base_tensor, pool_tensor


class DmabufGPUAllocator(MemoryAllocatorInterface):
    """GPU memory allocator whose pool is exported as io_uring DMA-BUF slabs."""

    def __init__(
        self,
        pool_bytes: int,
        device: str,
        block_align: int,
        exporter: str = "auto",
        mem_range_flags: int = 0,
    ) -> None:
        """Initialize and export a GPU memory pool.

        Args:
            pool_bytes: Total GPU pool size in bytes.
            device: PyTorch CUDA/HIP device string, such as ``"cuda:0"``.
            block_align: O_DIRECT alignment in bytes.
            exporter: Export strategy. First cut supports ``"auto"``,
                ``"cuda_pool"``, and ``"hip"``.
            mem_range_flags: Exporter-specific address-range flags.

        Raises:
            NotImplementedError: If a deferred exporter is requested.
            RuntimeError: If GPU allocation or DMA-BUF export fails.
            ValueError: If pool geometry cannot satisfy dmabuf constraints.
        """
        resolved_exporter = _resolve_dmabuf_exporter(exporter)
        torch_device = torch.device(device)
        if torch_device.type != "cuda":
            raise ValueError(
                "IouDmabufBackend requires a PyTorch CUDA/HIP device"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda is not available for IouDmabufBackend")
        if pool_bytes <= 0:
            raise ValueError("pool_bytes must be > 0")
        if not _is_power_of_two(block_align):
            raise ValueError("block_align must be a positive power of two")
        if mem_range_flags < 0:
            raise ValueError("mem_range_flags must be >= 0")

        page_size = os.sysconf("SC_PAGESIZE")
        if pool_bytes % page_size != 0:
            raise ValueError("pool_bytes must be aligned to the host page size")
        if block_align % page_size != 0 and page_size % block_align != 0:
            raise ValueError("block_align must be compatible with host page size")

        self.pool_bytes = int(pool_bytes)
        self.block_align = int(block_align)
        self.slab_bytes = _SZ_1G
        self.device = torch_device
        self.exporter = resolved_exporter
        self.mem_range_flags = int(mem_range_flags)
        self._closed = False
        self._lock = threading.Lock()

        with torch.cuda.device(self.device):
            self._base_tensor, self.tensor = _make_page_aligned_gpu_pool(
                self.pool_bytes,
                self.device,
                page_size,
            )
        if self.tensor.data_ptr() % page_size != 0:
            raise RuntimeError("GPU pool pointer is not host-page aligned")

        self.slabs = self._export_slabs(page_size)
        self._inners: list[TensorMemoryAllocator] = []
        for slab in self.slabs:
            slab_start = slab.buf_slot * self.slab_bytes
            slab_end = slab_start + slab.size
            self._inners.append(
                TensorMemoryAllocator(
                    self.tensor[slab_start:slab_end],
                    align_bytes=self.block_align,
                )
            )

    @property
    def dmabuf_fds(self) -> list[int]:
        """Return the DMA-BUF fds backing all registered slabs."""
        return [slab.dmabuf_fd for slab in self.slabs]

    def owns(self, obj: MemoryObj) -> bool:
        """Return whether ``obj`` was allocated from this dmabuf GPU pool.

        Args:
            obj: Memory object to test.

        Returns:
            True when ``obj`` is a live ``TensorMemoryObj`` owned by this
            allocator and its pool-relative address is in range.
        """
        return (
            isinstance(obj, TensorMemoryObj)
            and obj.is_valid()
            and obj.parent_allocator is self
            and 0 <= int(obj.metadata.address) < self.pool_bytes
        )

    def decompose(self, abs_offset: int, total_len: int) -> tuple[int, int]:
        """Convert a pool offset into ``(slab_idx, dmabuf_offset)``.

        Args:
            abs_offset: Pool-relative byte offset from a memory object.
            total_len: Block-aligned transfer length.

        Returns:
            The registered buffer slot index and offset within that slab.

        Raises:
            ValueError: If the transfer is unaligned, outside the pool, or
                crosses a dmabuf slab boundary.
        """
        if abs_offset < 0:
            raise ValueError("abs_offset must be non-negative")
        if total_len <= 0:
            raise ValueError("total_len must be > 0")
        if abs_offset % self.block_align != 0:
            raise ValueError("abs_offset must be block aligned")
        if total_len % self.block_align != 0:
            raise ValueError("total_len must be block aligned")
        if abs_offset + total_len > self.pool_bytes:
            raise ValueError("transfer exceeds dmabuf pool size")

        slab_idx = abs_offset // self.slab_bytes
        buf_offset = abs_offset % self.slab_bytes
        slab = self.slabs[slab_idx]
        if buf_offset + total_len > slab.size:
            raise ValueError("transfer crosses a dmabuf slab boundary")
        return int(slab_idx), int(buf_offset)

    def transfer_len(self, obj: MemoryObj) -> int:
        """Return the O_DIRECT-rounded transfer length for ``obj``.

        Args:
            obj: Memory object whose logical payload will be transferred.

        Returns:
            The block-aligned transfer length in bytes.
        """
        return round_up(int(obj.get_size()), self.block_align)

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """Allocate a memory object from the exported GPU pool.

        Args:
            shapes: Tensor shape or group shapes.
            dtypes: Tensor dtype or group dtypes.
            fmt: LMCache memory format.
            allocator_type: Ignored compatibility argument.

        Returns:
            A ``TensorMemoryObj`` whose parent allocator is this object, or
            None when the pool is exhausted.
        """
        del allocator_type
        with self._lock:
            return self._allocate_no_crossing(shapes, dtypes, fmt)

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """Allocate multiple memory objects from the exported GPU pool.

        Args:
            shapes: Tensor shape or group shapes.
            dtypes: Tensor dtype or group dtypes.
            batch_size: Number of memory objects to allocate.
            fmt: LMCache memory format.
            allocator_type: Ignored compatibility argument.

        Returns:
            A list of memory objects, or None if the full batch cannot be
            allocated.
        """
        del allocator_type
        if batch_size <= 0:
            return []
        allocated: list[MemoryObj] = []
        with self._lock:
            for _ in range(batch_size):
                obj = self._allocate_no_crossing(shapes, dtypes, fmt)
                if obj is None:
                    self._batched_free_locked(allocated)
                    return None
                allocated.append(obj)
        return allocated

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        """Free a memory object previously allocated from this pool.

        Args:
            memory_obj: Memory object to release.
            allocator_type: Ignored compatibility argument.
        """
        del allocator_type
        with self._lock:
            self._free_locked(memory_obj)

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        """Free memory objects previously allocated from this pool.

        Args:
            memory_objs: Memory objects to release.
            allocator_type: Ignored compatibility argument.
            update_stats: Whether allocator stats should be updated.
        """
        del allocator_type
        with self._lock:
            self._batched_free_locked(memory_objs, update_stats=update_stats)

    def memcheck(self) -> bool:
        """Return whether the wrapped tensor allocator is internally consistent."""
        with self._lock:
            return all(inner.memcheck() for inner in self._inners)

    def close(self) -> None:
        """Close exported DMA-BUF fds after io_uring unregisters them."""
        if self._closed:
            return
        self._closed = True
        for slab in self.slabs:
            try:
                os.close(slab.dmabuf_fd)
            except OSError:
                logger.warning(
                    "IouDmabufBackend: failed to close dmabuf fd %d",
                    slab.dmabuf_fd,
                    exc_info=True,
                )
        self.slabs = []

    def __str__(self) -> str:
        return "DmabufGPUAllocator"

    def _allocate_no_crossing(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        shapes_list, dtypes_list = self._adapt_shapes_and_dtypes(shapes, dtypes)
        raw_size = get_size_bytes(shapes_list, dtypes_list)
        if raw_size <= 0:
            return None
        if round_up(raw_size, self.block_align) > self.slab_bytes:
            raise ValueError("single allocation exceeds dmabuf slab size")

        for slab, inner in zip(self.slabs, self._inners, strict=True):
            obj = inner.allocate(shapes_list, dtypes_list, fmt)
            if obj is None:
                continue
            obj.metadata.address += slab.buf_slot * self.slab_bytes
            obj.parent_allocator = self
            return obj
        return None

    def _batched_free_locked(
        self,
        memory_objs: list[MemoryObj],
        *,
        update_stats: bool = True,
    ) -> None:
        grouped: dict[int, list[tuple[MemoryObj, int]]] = {}
        for memory_obj in memory_objs:
            if not memory_obj.is_valid():
                continue
            slab_idx, global_address = self._locate_owned_object(memory_obj)
            grouped.setdefault(slab_idx, []).append((memory_obj, global_address))

        for slab_idx, entries in grouped.items():
            slab_base = slab_idx * self.slab_bytes
            for memory_obj, global_address in entries:
                memory_obj.metadata.address = global_address - slab_base

        try:
            for slab_idx, entries in grouped.items():
                self._inners[slab_idx].batched_free(
                    [memory_obj for memory_obj, _ in entries],
                    update_stats=update_stats,
                )
        finally:
            for entries in grouped.values():
                for memory_obj, global_address in entries:
                    memory_obj.metadata.address = global_address

    def _free_locked(self, memory_obj: MemoryObj) -> None:
        if not memory_obj.is_valid():
            return
        slab_idx, global_address = self._locate_owned_object(memory_obj)
        slab_base = slab_idx * self.slab_bytes
        memory_obj.metadata.address = global_address - slab_base
        try:
            self._inners[slab_idx].free(memory_obj)
        finally:
            memory_obj.metadata.address = global_address

    def _locate_owned_object(self, memory_obj: MemoryObj) -> tuple[int, int]:
        if not isinstance(memory_obj, TensorMemoryObj):
            raise ValueError("memory object was not allocated by this dmabuf pool")
        if memory_obj.parent_allocator is not self:
            raise ValueError("memory object was not allocated by this dmabuf pool")
        global_address = int(memory_obj.metadata.address)
        slab_idx = global_address // self.slab_bytes
        if slab_idx < 0 or slab_idx >= len(self._inners):
            raise ValueError("memory object address is outside the dmabuf pool")
        slab_offset = global_address - slab_idx * self.slab_bytes
        slab_size = int(self._inners[slab_idx].buffer.numel())
        if slab_offset + int(memory_obj.metadata.phy_size) > slab_size:
            raise ValueError("memory object crosses a dmabuf slab boundary")
        return slab_idx, global_address

    def _export_slabs(self, page_size: int) -> list[SlabDescriptor]:
        export_driver = self._make_export_driver()
        slabs: list[SlabDescriptor] = []
        base_ptr = int(self.tensor.data_ptr())
        try:
            for buf_slot, slab_start in enumerate(range(0, self.pool_bytes, _SZ_1G)):
                slab_size = min(_SZ_1G, self.pool_bytes - slab_start)
                device_ptr = base_ptr + slab_start
                if device_ptr % page_size != 0 or slab_size % page_size != 0:
                    raise RuntimeError(
                        "GPU dmabuf export requires page-aligned pointer and size"
                    )
                dmabuf_fd = export_driver.export_dmabuf(
                    device_ptr,
                    slab_size,
                    self.mem_range_flags,
                )
                slabs.append(
                    SlabDescriptor(
                        device_ptr=device_ptr,
                        dmabuf_fd=dmabuf_fd,
                        buf_slot=buf_slot,
                        size=slab_size,
                    )
                )
        except Exception:
            for slab in slabs:
                try:
                    os.close(slab.dmabuf_fd)
                except OSError:
                    pass
            raise
        return slabs

    def _make_export_driver(self) -> _CudaDriver | _HipDriver:
        if self.exporter == "cuda_pool":
            return _CudaDriver()
        if self.exporter == "hip":
            return _HipDriver(_torch_cuda_device_index(self.device))
        raise AssertionError(f"unexpected dmabuf exporter: {self.exporter}")


class IouDmabufBackend(AllocatorBackendInterface):
    """KV cache backend using io_uring DMA-BUF for GPU VRAM to NVMe I/O."""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = "cuda",
    ) -> None:
        """Initialize the io_uring dmabuf raw-block storage backend.

        Args:
            config: LMCache engine configuration.
            metadata: LMCache engine metadata.
            loop: Event loop used by the storage manager.
            dst_device: Device where retrieved memory objects should be allocated.

        Raises:
            RuntimeError: If native dmabuf support is unavailable.
            ValueError: If required ``extra_config`` keys are missing or invalid.
        """
        super().__init__(dst_device=dst_device)
        self.config = config
        self.metadata = metadata
        self.loop = loop
        self.extra = config.extra_config or {}

        if not self.is_available():
            raise RuntimeError(
                "io_uring DMA-BUF support is unavailable; rebuild "
                "lmcache_rust_raw_block_io and use a kernel with CONFIG_DMABUF_TOKEN"
            )

        self.device_path = self._resolve_device_path()
        self.block_align = _get_extra_int(
            self.extra,
            "iou_dmabuf.block_align",
            _DEFAULT_BLOCK_ALIGN,
            positive=True,
        )
        if not _is_power_of_two(self.block_align):
            raise ValueError(
                "extra_config['iou_dmabuf.block_align'] must be a power of two"
            )
        self.header_bytes = _get_extra_int(
            self.extra,
            "iou_dmabuf.header_bytes",
            _DEFAULT_HEADER_BYTES,
            positive=True,
        )
        capacity_bytes = _get_extra_int(
            self.extra,
            "iou_dmabuf.capacity_bytes",
            0,
        )
        gpu_pool_bytes = _get_extra_int(
            self.extra,
            "iou_dmabuf.gpu_pool_bytes",
            required=True,
            positive=True,
        )
        ring_depth = _get_extra_int(
            self.extra,
            "iou_dmabuf.ring_depth",
            DEFAULT_IOURING_QUEUE_DEPTH,
            positive=True,
        )
        validate_raw_block_io_options(iouring_queue_depth=ring_depth)
        disk_io_threads = _get_extra_int(
            self.extra,
            "disk_io_threads",
            _DEFAULT_DISK_IO_THREADS,
            positive=True,
        )
        self.max_eagain_retries = _get_extra_int(
            self.extra,
            "iou_dmabuf.max_eagain_retries",
            _DEFAULT_DMABUF_EAGAIN_RETRIES,
        )
        if self.max_eagain_retries < 0:
            raise ValueError(
                "extra_config['iou_dmabuf.max_eagain_retries'] must be >= 0"
            )
        if not _get_extra_bool(self.extra, "iou_dmabuf.use_odirect", True):
            raise ValueError(
                "IouDmabufBackend requires iou_dmabuf.use_odirect=true"
            )
        if _get_extra_bool(self.extra, "iou_dmabuf.use_uring_cmd", False):
            raise ValueError(
                "IouDmabufBackend does not support io_uring_cmd; use an "
                "O_DIRECT NVMe block-device path"
            )
        if _get_extra_bool(self.extra, "iou_dmabuf.require_p2p", False):
            raise NotImplementedError(
                "iou_dmabuf.require_p2p=true is not implemented in the first cut"
            )

        core: RawBlockCore | None = None
        memory_allocator: DmabufGPUAllocator | None = None
        rawdev: Any = None
        try:
            core = RawBlockCore(
                self._build_core_config(capacity_bytes, ring_depth),
                key_namespace="legacy",
            )
            memory_allocator = self.initialize_allocator(config, metadata)
            rawdev = core.raw_device()
            rawdev.register_dmabuf_buffers(memory_allocator.dmabuf_fds)
        except Exception:
            if core is not None:
                core.close()
            if memory_allocator is not None:
                memory_allocator.close()
            raise

        assert core is not None
        assert memory_allocator is not None
        self._core = core
        self.memory_allocator = memory_allocator
        self._rawdev = rawdev

        self._put_lock = threading.Lock()
        self._put_tasks: set[CacheEngineKey] = set()
        self._pin_lock = threading.Lock()
        self._pin_counts: dict[str, int] = {}
        self._future_lock = threading.Lock()
        self._pending_futures: set[Future] = set()
        self._thread_pool = ThreadPoolExecutor(
            max_workers=disk_io_threads,
            thread_name_prefix="iou-dmabuf",
        )

        logger.info(
            "Initialized IouDmabufBackend: device=%s, gpu_pool_bytes=%d, slabs=%d",
            self.device_path,
            gpu_pool_bytes,
            len(self.memory_allocator.slabs),
        )

    @staticmethod
    def is_available() -> bool:
        """Return whether the native Rust extension and kernel path are present.

        Returns:
            True when ``lmcache_rust_raw_block_io.RawBlockDevice`` exposes the
            dmabuf methods and its static kernel probe succeeds.
        """
        try:
            # Third Party
            from lmcache_rust_raw_block_io import RawBlockDevice  # type: ignore
        except ImportError:
            return False
        required = (
            "probe_dmabuf_support",
            "register_dmabuf_buffers",
            "read_fixed_dmabuf",
            "write_fixed_dmabuf",
        )
        if any(not hasattr(RawBlockDevice, name) for name in required):
            return False
        try:
            return bool(RawBlockDevice.probe_dmabuf_support())
        except Exception as e:
            logger.warning("IouDmabufBackend dmabuf probe failed: %s", e)
            return False

    def __str__(self) -> str:
        return "IouDmabufBackend"

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Return whether ``key`` is present in this backend.

        Args:
            key: Cache key to look up.
            pin: Whether to protect the slot from deletion.

        Returns:
            True when the key is indexed.
        """
        spec = encode_legacy_key(key)
        return self._pin_if_needed(spec.encoded) if pin else self._core.exists_many(
            [spec.encoded],
            lock=False,
        )[0]

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Return whether ``key`` has a scheduled put task.

        Args:
            key: Cache key to inspect.

        Returns:
            True when a write task is currently active for ``key``.
        """
        with self._put_lock:
            return key in self._put_tasks

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> list[Future] | None:
        """Submit asynchronous writes to the dmabuf raw-block backend.

        Args:
            keys: Cache keys corresponding to ``objs``.
            objs: Source memory objects.
            transfer_spec: Unused compatibility argument.
            on_complete_callback: Optional callback invoked after a key is stored.

        Returns:
            A list of futures for scheduled writes, or None if every key was
            skipped.
        """
        del transfer_spec
        futures: list[Future] = []
        for key, obj in zip(keys, objs, strict=True):
            with self._put_lock:
                if key in self._put_tasks:
                    continue
                self._put_tasks.add(key)
            obj.ref_count_up()
            try:
                fut = self._submit_tracked(
                    self._put_one,
                    key,
                    obj,
                    on_complete_callback,
                )
            except Exception:
                obj.ref_count_down()
                with self._put_lock:
                    self._put_tasks.discard(key)
                raise
            futures.append(fut)
        return futures or None

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Asynchronously submit and wait for a batch of put tasks.

        Args:
            keys: Cache keys corresponding to ``objs``.
            objs: Source memory objects.
            transfer_spec: Unused compatibility argument.
            on_complete_callback: Optional callback invoked after a key is stored.
        """
        futures = self.batched_submit_put_task(
            keys,
            objs,
            transfer_spec=transfer_spec,
            on_complete_callback=on_complete_callback,
        )
        if futures:
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
                return_exceptions=False,
            )

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Synchronously load one key into a registered GPU slab.

        Args:
            key: Cache key to load.

        Returns:
            Loaded memory object, or None on miss or read failure.
        """
        results = self.batched_get_blocking([key])
        return results[0] if results else None

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        """Submit a single-key nonblocking read.

        Args:
            key: Cache key to load.
            location: Unused compatibility argument.

        Returns:
            Future producing the loaded memory object, or None if the key is
            not present. The presence check is best-effort: if the key is
            evicted between this check and the read, the future resolves to
            None (a benign miss), never an error.
        """
        del location
        if not self.contains(key):
            return None
        return self._submit_tracked(self._get_one_direct, key)

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Synchronously load cache keys into registered GPU slabs.

        Args:
            keys: Ordered cache keys to load.

        Returns:
            A list aligned with ``keys``. Returns an empty list when there are
            no hits so later storage tiers can still be searched.
        """
        if not keys:
            return []

        specs = [encode_legacy_key(key) for key in keys]
        encoded_keys = [spec.encoded for spec in specs]
        entries = self._core.get_entries_many(encoded_keys, lock_refcount=True)
        locked_keys = [
            encoded_key
            for encoded_key, entry in zip(encoded_keys, entries, strict=True)
            if entry is not None
        ]
        if not locked_keys:
            return []

        results: list[MemoryObj | None] = [None] * len(keys)
        read_futures: list[tuple[int, Future, MemoryObj]] = []
        try:
            for idx, entry in enumerate(entries):
                if entry is None:
                    continue
                prepared = self._prepare_read(encoded_keys[idx], entry)
                if prepared is None:
                    continue
                memory_obj, slab_idx, buf_offset, total_len, device_offset = prepared
                try:
                    future = self._submit_tracked(
                        self._read_into_dmabuf,
                        slab_idx,
                        buf_offset,
                        total_len,
                        device_offset,
                    )
                    read_futures.append((idx, future, memory_obj))
                except Exception:
                    memory_obj.ref_count_down()
                    raise

            for idx, future, memory_obj in read_futures:
                try:
                    future.result()
                    entry = entries[idx]
                    assert entry is not None
                    meta, _ = entry
                    # NOTE: do NOT call memory_obj.set_used_size() here. It sets
                    # _used_size_override, which makes MemoryObj.tensor return a
                    # flat 1-D uint8 view (see memory_management.py) instead of
                    # reshaping raw_data to meta.shape. The GPU connector
                    # consumes .tensor via lmc_ops.multi_layer_kv_transfer and
                    # requires the full KV_2LTD shape; a narrowed view raises
                    # "IndexError: Dimension out of range". meta.size already
                    # equals the tensor's layout size (both derive from the
                    # per-chunk shape stored at write time), so there is nothing
                    # to narrow -- calling it only breaks the reshape.
                    memory_obj.metadata.cached_positions = meta.cached_positions
                    results[idx] = memory_obj
                except Exception as e:
                    logger.error(
                        "IouDmabufBackend: read failed for key %s: %s",
                        encoded_keys[idx],
                        e,
                    )
                    memory_obj.ref_count_down()

            return results if any(obj is not None for obj in results) else []
        except Exception:
            for _, future, memory_obj in read_futures:
                try:
                    future.result()
                except Exception:
                    pass
                memory_obj.ref_count_down()
            raise
        finally:
            self._core.unlock_many(locked_keys)

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Asynchronously load the hit prefix for ``keys``.

        Args:
            lookup_id: Lookup identifier supplied by the storage manager.
            keys: Cache keys to load.
            transfer_spec: Unused compatibility argument.

        Returns:
            Loaded memory objects up to the first miss or failed read.
        """
        del lookup_id, transfer_spec
        results = await asyncio.to_thread(self.batched_get_blocking, keys)
        loaded: list[MemoryObj] = []
        hole_seen = False
        for obj in results:
            if obj is None:
                hole_seen = True
                continue
            if hole_seen:
                # Only the leading contiguous prefix is returned. Any object
                # loaded after a miss was allocated and read but will not be
                # returned, so release its reference here to avoid a leak.
                obj.ref_count_down()
            else:
                loaded.append(obj)
        return loaded

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Return leading-prefix hit count for ``keys``.

        Args:
            lookup_id: Lookup identifier supplied by the storage manager.
            keys: Cache keys to inspect.
            pin: Whether to protect hit slots from deletion.

        Returns:
            Number of contiguous hits from the start of ``keys``.
        """
        del lookup_id
        specs = [encode_legacy_key(key) for key in keys]
        encoded_keys = [spec.encoded for spec in specs]
        results = self._core.exists_many(encoded_keys, lock=False)
        prefix_hits = 0
        for exists in results:
            if not exists:
                break
            prefix_hits += 1
        if pin:
            pinned_hits = 0
            for encoded_key in encoded_keys[:prefix_hits]:
                if not self._pin_if_needed(encoded_key):
                    break
                pinned_hits += 1
            prefix_hits = pinned_hits
        return prefix_hits

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin a key's raw-block slot.

        Args:
            key: Cache key to pin.

        Returns:
            True when the key exists and is pinned.
        """
        return self._pin_if_needed(encode_legacy_key(key).encoded)

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin a key's raw-block slot.

        Args:
            key: Cache key to unpin.

        Returns:
            True when the key exists or was unpinned.
        """
        return self._unpin_if_needed(encode_legacy_key(key).encoded)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove a key from this backend.

        Args:
            key: Cache key to remove.
            force: Whether to remove even when the entry is locked.

        Returns:
            True when an indexed entry was removed.
        """
        spec = encode_legacy_key(key)
        with self._pin_lock:
            removed = self._core.delete_many([spec.encoded], force=force)[0]
            return removed

    def get_allocator_backend(self) -> AllocatorBackendInterface:
        """Return this backend as the allocator backend for retrieved objects."""
        return self

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> DmabufGPUAllocator:
        """Create the dmabuf GPU allocator.

        Args:
            config: LMCache engine configuration.
            metadata: LMCache engine metadata.

        Returns:
            Initialized ``DmabufGPUAllocator``.
        """
        del config, metadata
        exporter = str(self.extra.get("iou_dmabuf.exporter", "auto") or "auto")
        pool_bytes = _get_extra_int(
            self.extra,
            "iou_dmabuf.gpu_pool_bytes",
            required=True,
            positive=True,
        )
        mem_range_flags = _get_extra_int(
            self.extra,
            "iou_dmabuf.mem_range_flags",
            0,
        )
        if mem_range_flags < 0:
            raise ValueError(
                "extra_config['iou_dmabuf.mem_range_flags'] must be >= 0"
            )
        return DmabufGPUAllocator(
            pool_bytes=pool_bytes,
            device=self.dst_device,
            block_align=self.block_align,
            exporter=exporter,
            mem_range_flags=mem_range_flags,
        )

    def get_memory_allocator(self) -> DmabufGPUAllocator:
        """Return the dmabuf GPU allocator."""
        return self.memory_allocator

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """Allocate a GPU memory object for a retrieval.

        Args:
            shapes: Tensor shape or group shapes.
            dtypes: Tensor dtype or group dtypes.
            fmt: LMCache memory format.
            eviction: Unused; this allocator has no storage eviction hook.
            busy_loop: Unused; allocation makes a single attempt.

        Returns:
            Allocated memory object or None.
        """
        del eviction, busy_loop
        return self.memory_allocator.allocate(shapes, dtypes, fmt)

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        """Allocate a batch of GPU memory objects for retrieval.

        Args:
            shapes: Tensor shape or group shapes.
            dtypes: Tensor dtype or group dtypes.
            batch_size: Number of memory objects.
            fmt: LMCache memory format.
            eviction: Unused; this allocator has no storage eviction hook.
            busy_loop: Unused; allocation makes a single attempt.

        Returns:
            Allocated memory objects or None.
        """
        del eviction, busy_loop
        return self.memory_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)

    def calculate_chunk_budget(self) -> int:
        """Return the number of full chunks that fit in the GPU dmabuf pool."""
        chunk_bytes = round_up(
            self._default_chunk_size_bytes(),
            self.block_align,
        )
        return sum(slab.size // chunk_bytes for slab in self.memory_allocator.slabs)

    def touch_cache(self) -> None:
        """No-op cache touch hook for storage manager compatibility."""
        return

    def close(self) -> None:
        """Drain I/O, close raw-block resources, then close dmabuf fds."""
        with self._future_lock:
            pending = list(self._pending_futures)
        deadline = time.monotonic() + 10.0
        for future in pending:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                future.result(timeout=remaining)
            except Exception:
                logger.warning(
                    "IouDmabufBackend: pending future did not finish cleanly",
                    exc_info=True,
                )
        self._thread_pool.shutdown(wait=True)
        self._core.close()
        self.memory_allocator.close()

    def _build_core_config(
        self,
        capacity_bytes: int,
        ring_depth: int,
    ) -> RawBlockCoreConfig:
        slot_bytes = _get_extra_int(
            self.extra,
            "iou_dmabuf.slot_bytes",
            round_up(
                self.header_bytes + self._default_chunk_size_bytes(),
                self.block_align,
            ),
            positive=True,
        )
        meta_magic_raw = self.extra.get("iou_dmabuf.meta_magic", _DEFAULT_META_MAGIC)
        if isinstance(meta_magic_raw, str):
            meta_magic = meta_magic_raw.encode("ascii")
        elif isinstance(meta_magic_raw, bytes):
            meta_magic = meta_magic_raw
        else:
            raise ValueError(
                "extra_config['iou_dmabuf.meta_magic'] must be str or bytes"
            )

        return RawBlockCoreConfig(
            device_path=self.device_path,
            capacity_bytes=capacity_bytes,
            block_align=self.block_align,
            header_bytes=self.header_bytes,
            slot_bytes=slot_bytes,
            use_odirect=True,
            enable_zero_copy=True,
            meta_total_bytes=_get_extra_int(
                self.extra,
                "iou_dmabuf.meta_total_bytes",
                _DEFAULT_META_TOTAL_BYTES,
                positive=True,
            ),
            meta_magic=meta_magic,
            meta_version=_get_extra_int(
                self.extra,
                "iou_dmabuf.meta_version",
                _DEFAULT_META_VERSION,
                positive=True,
            ),
            meta_checkpoint_interval_sec=_get_extra_int(
                self.extra,
                "iou_dmabuf.meta_checkpoint_interval_sec",
                60,
                positive=True,
            ),
            meta_idle_quiet_ms=_get_extra_int(
                self.extra,
                "iou_dmabuf.meta_idle_quiet_ms",
                100,
                positive=True,
            ),
            meta_enable_periodic=_get_extra_bool(
                self.extra,
                "iou_dmabuf.meta_enable_periodic",
                True,
            ),
            load_checkpoint_on_init=_get_extra_bool(
                self.extra,
                "iou_dmabuf.load_checkpoint_on_init",
                True,
            ),
            meta_verify_on_load=_get_extra_bool(
                self.extra,
                "iou_dmabuf.meta_verify_on_load",
                True,
            ),
            max_data_transfer_size=_get_extra_int(
                self.extra,
                "iou_dmabuf.max_data_transfer_size",
                0,
            ),
            io_engine="io_uring",
            iouring_queue_depth=ring_depth,
            use_uring_cmd=False,
        )

    def _default_chunk_size_bytes(self) -> int:
        """Return the byte size of a full KV chunk, used to size raw-block slots.

        Derives the size from ``metadata.kv_shape``, whose layout is
        ``(num_layers, kv_size, num_tokens, num_heads, head_size)`` -- the same
        convention ``LMCacheMetadata`` uses (``kv_size`` is the K/V dimension,
        i.e. 2). The stored ``num_tokens`` (index 2) is ignored; the token count
        comes from ``config.chunk_size`` instead, so the result is the size of a
        *full* chunk (slots are fixed at the maximum chunk size).

        Returns:
            Full-chunk size in bytes.

        Raises:
            ValueError: If metadata is unavailable (required to derive the size).
        """
        if self.metadata is None:
            raise ValueError(
                "metadata is required when iou_dmabuf.slot_bytes is not configured"
            )
        chunk_tokens = int(self.config.chunk_size)
        kv_shape = self.metadata.kv_shape
        kv_size = int(kv_shape[1])
        num_layers = int(kv_shape[0])
        num_heads = int(kv_shape[3])
        head_size = int(kv_shape[4])
        dtype_size = int(self.metadata.kv_dtype.itemsize)
        hidden_dim = num_heads * head_size
        return kv_size * num_layers * chunk_tokens * hidden_dim * dtype_size

    def _resolve_device_path(self) -> str:
        if self.metadata is not None and self.metadata.world_size > 1:
            per_tp_devices = self.extra.get("iou_dmabuf.per_tp_device_paths", {})
            if not isinstance(per_tp_devices, Mapping) or not per_tp_devices:
                raise ValueError(
                    "For TP > 1, extra_config['iou_dmabuf.per_tp_device_paths'] "
                    "is required"
                )
            values = [str(value) for value in per_tp_devices.values()]
            if len(values) != len(set(values)):
                raise ValueError(
                    "Duplicate device path configured in "
                    "iou_dmabuf.per_tp_device_paths"
                )
            rank = self.metadata.worker_id
            device_path = per_tp_devices.get(str(rank), per_tp_devices.get(rank))
            if not device_path:
                raise ValueError(f"No iou_dmabuf device path for TP rank {rank}")
            return str(device_path)

        device_path = str(self.extra.get("iou_dmabuf.device_path", "") or "")
        if not device_path:
            raise ValueError("extra_config['iou_dmabuf.device_path'] is required")
        return device_path

    def _submit_tracked(self, fn: Callable[..., Any], *args: Any) -> Future:
        future = self._thread_pool.submit(fn, *args)
        with self._future_lock:
            self._pending_futures.add(future)
        future.add_done_callback(self._discard_future)
        return future

    def _discard_future(self, future: Future) -> None:
        with self._future_lock:
            self._pending_futures.discard(future)
        if future.cancelled():
            return
        exception = future.exception()
        if exception is not None:
            logger.error(
                "IouDmabufBackend: background task failed: %s",
                exception,
                exc_info=(
                    type(exception),
                    exception,
                    exception.__traceback__,
                ),
            )

    def _get_one_direct(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Load one key into a GPU slab inline on the calling worker thread.

        Unlike ``batched_get_blocking``, the dmabuf read is issued directly
        rather than re-submitted to the thread pool, so a single-key
        ``get_non_blocking`` cannot deadlock the pool when
        ``disk_io_threads == 1``.

        Args:
            key: Cache key to load.

        Returns:
            The loaded memory object, or None on miss or read failure.
        """
        raw_key = encode_legacy_key(key)
        entries = self._core.get_entries_many(
            [raw_key.encoded],
            lock_refcount=True,
        )
        entry = entries[0]
        if entry is None:
            return None
        try:
            prepared = self._prepare_read(raw_key.encoded, entry)
            if prepared is None:
                return None
            memory_obj, slab_idx, buf_offset, total_len, device_offset = prepared
            try:
                self._read_into_dmabuf(slab_idx, buf_offset, total_len, device_offset)
            except Exception:
                memory_obj.ref_count_down()
                raise
            memory_obj.metadata.cached_positions = entry[0].cached_positions
            return memory_obj
        except Exception as e:
            logger.error(
                "IouDmabufBackend: read failed for key %s: %s",
                raw_key.encoded,
                e,
            )
            return None
        finally:
            self._core.unlock_many([raw_key.encoded])

    def _prepare_read(
        self,
        encoded_key: str,
        entry: tuple[DiskCacheMetadata, int],
    ) -> tuple[MemoryObj, int, int, int, int] | None:
        """Allocate a GPU slab and resolve dmabuf read parameters for one hit.

        Shared by the blocking and non-blocking read paths. Does not issue the
        read; the caller performs the transfer (inline or via the thread pool).
        On failure after a slab was allocated, the allocation is released before
        returning None or propagating.

        Args:
            encoded_key: Encoded raw-block key, used only for log messages.
            entry: The ``(metadata, slot_base_offset)`` pair returned by
                ``RawBlockCore.get_entries_many`` for this key.

        Returns:
            ``(memory_obj, slab_idx, buf_offset, total_len, device_offset)`` on
            success, or None when the metadata is incomplete or the GPU pool is
            exhausted (both logged).

        Raises:
            Exception: If slab-offset decomposition fails (e.g. a transfer that
                crosses a slab boundary); the allocated object is released
                first.
        """
        meta, slot_base_offset = entry
        if meta.shape is None or meta.dtype is None or meta.fmt is None:
            logger.warning(
                "IouDmabufBackend: metadata incomplete for key %s",
                encoded_key,
            )
            return None
        memory_obj = self.allocate(meta.shape, meta.dtype, meta.fmt)
        if memory_obj is None:
            logger.warning(
                "IouDmabufBackend: failed to allocate GPU slab for key %s",
                encoded_key,
            )
            return None
        try:
            total_len = round_up(int(meta.size), self.block_align)
            slab_idx, buf_offset = self.memory_allocator.decompose(
                int(memory_obj.metadata.address),
                total_len,
            )
        except Exception:
            memory_obj.ref_count_down()
            raise
        device_offset = int(slot_base_offset) + self.header_bytes
        return memory_obj, slab_idx, buf_offset, total_len, device_offset

    def _put_one(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> None:
        """Persist one source object, dispatching on where its memory lives.

        Runs on a thread-pool worker. Classifies the source three ways
        (design doc section 10): an allocator-owned GPU object takes the
        ``WRITE_FIXED`` fast path; a CPU-resident object is written via
        ``RawBlockCore.put_many``; any other (foreign GPU) source is rejected
        and not stored. The completion callback fires only when the chunk is
        actually persisted. Always releases the object's ref count and clears
        the in-flight marker on exit.

        Args:
            key: Cache key being stored.
            memory_obj: Source object; its ref count was incremented by the
                caller and is released here.
            on_complete_callback: Optional callback invoked once, only after a
                successful store.
        """
        raw_key = encode_legacy_key(key)
        try:
            if self.memory_allocator.owns(memory_obj):
                if self._put_owned_gpu(raw_key, memory_obj):
                    self._invoke_callback(on_complete_callback, key)
                else:
                    logger.debug(
                        "IouDmabufBackend: GPU put for %s stored nothing "
                        "(no free slot or duplicate key); callback not fired",
                        key,
                    )
                return
            if self._is_cpu_readable_source(memory_obj):
                result = self._core.put_many([raw_key], [memory_obj])
                if result.results and result.results[0]:
                    self._invoke_callback(on_complete_callback, key)
                else:
                    logger.warning("IouDmabufBackend: CPU put failed for key %s", key)
                return
            logger.warning(
                "IouDmabufBackend: unsupported foreign GPU source for %s; not stored",
                key,
            )
        finally:
            memory_obj.ref_count_down()
            with self._put_lock:
                self._put_tasks.discard(key)

    def _put_owned_gpu(self, raw_key: RawBlockKeySpec, memory_obj: MemoryObj) -> bool:
        """Write an allocator-owned GPU object via WRITE_FIXED.

        Args:
            raw_key: Encoded raw-block key spec.
            memory_obj: Source object owned by this backend's GPU pool.

        Returns:
            True only when the payload was written and committed to the index.
            False means no slot was reserved (pool full, or the key is already
            in-flight/indexed) and nothing was stored.

        Raises:
            RuntimeError: If a reserved slot's header write, dmabuf write, or
                commit fails; the slot is aborted before propagating.
        """
        payload_len = int(memory_obj.get_size())
        total_len = round_up(payload_len, self.block_align)
        slot_base_offset = self._core.reserve_slot(raw_key, memory_obj)
        if slot_base_offset is None:
            return False

        committed = False
        try:
            if not self._core.write_slot_header(raw_key, slot_base_offset, payload_len):
                raise RuntimeError(f"failed to write slot header for {raw_key.encoded}")
            slab_idx, buf_offset = self.memory_allocator.decompose(
                int(memory_obj.metadata.address),
                total_len,
            )
            self._rawdev.write_fixed_dmabuf(
                slab_idx,
                buf_offset,
                total_len,
                int(slot_base_offset) + self.header_bytes,
                self.max_eagain_retries,
            )
            committed = self._core.commit_slot(raw_key, slot_base_offset)
            if not committed:
                raise RuntimeError(f"failed to commit slot for {raw_key.encoded}")
        finally:
            if not committed:
                self._core.abort_slot(raw_key, slot_base_offset)
        return committed

    def _read_into_dmabuf(
        self,
        slab_idx: int,
        buf_offset: int,
        total_len: int,
        device_offset: int,
    ) -> None:
        self._rawdev.read_fixed_dmabuf(
            slab_idx,
            buf_offset,
            total_len,
            device_offset,
            self.max_eagain_retries,
        )

    def _invoke_callback(
        self,
        callback: Optional[Callable[[CacheEngineKey], None]],
        key: CacheEngineKey,
    ) -> None:
        if callback is None:
            return
        try:
            callback(key)
        except Exception as e:
            logger.warning("on_complete_callback failed for key %s: %s", key, e)

    def _is_cpu_readable_source(self, memory_obj: MemoryObj) -> bool:
        """Return whether ``memory_obj`` is host memory safe for a CPU put.

        The CPU put path (``RawBlockCore.put_many``) reads ``byte_array``, which
        interprets the object's ``data_ptr`` as host memory. A CUDA device
        pointer must therefore never be classified as CPU-readable, or put_many
        would read device memory as host memory (garbage/segfault). Only an
        object that positively confirms a CPU-resident tensor is accepted;
        anything whose backing cannot be confirmed as host memory is rejected
        and handled as an unsupported source by the caller.

        Args:
            memory_obj: Candidate source object for a put.

        Returns:
            True only when the object is a CPU-resident ``TensorMemoryObj``.
        """
        if isinstance(memory_obj, TensorMemoryObj):
            raw_tensor = memory_obj.raw_tensor
            return raw_tensor is not None and raw_tensor.device.type == "cpu"
        return False

    def _pin_if_needed(self, encoded_key: str) -> bool:
        with self._pin_lock:
            if not self._core.exists_many([encoded_key], lock=True)[0]:
                return False
            self._pin_counts[encoded_key] = self._pin_counts.get(encoded_key, 0) + 1
            return True

    def _unpin_if_needed(self, encoded_key: str) -> bool:
        with self._pin_lock:
            count = self._pin_counts.get(encoded_key, 0)
            if count > 0:
                self._core.unlock_many([encoded_key])
                if count == 1:
                    self._pin_counts.pop(encoded_key, None)
                else:
                    self._pin_counts[encoded_key] = count - 1
                return True
            return self._core.exists_many([encoded_key], lock=False)[0]
