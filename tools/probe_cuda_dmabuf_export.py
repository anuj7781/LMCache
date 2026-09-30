# SPDX-License-Identifier: Apache-2.0
"""Probe CUDA device-memory DMA-BUF export without PyTorch or LMCache."""

# Standard
from typing import Any
import argparse
import ctypes
import os
import sys

CUDA_SUCCESS = 0
CUDA_ERROR_NOT_SUPPORTED = 801
CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124
CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1


class CudaDriver:
    """Minimal CUDA Driver API wrapper for a direct ``cuMemAlloc`` probe."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")

        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDeviceGet.restype = ctypes.c_int
        self.lib.cuDeviceGetName.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cuDeviceGetName.restype = ctypes.c_int
        self.lib.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cuDeviceGetAttribute.restype = ctypes.c_int

        self.cu_ctx_create = self._versioned_function("cuCtxCreate_v2", "cuCtxCreate")
        self.cu_ctx_create.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        self.cu_ctx_create.restype = ctypes.c_int

        self.cu_ctx_destroy = self._versioned_function(
            "cuCtxDestroy_v2", "cuCtxDestroy"
        )
        self.cu_ctx_destroy.argtypes = [ctypes.c_void_p]
        self.cu_ctx_destroy.restype = ctypes.c_int

        self.cu_mem_alloc = self._versioned_function("cuMemAlloc_v2", "cuMemAlloc")
        self.cu_mem_alloc.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        self.cu_mem_alloc.restype = ctypes.c_int

        self.cu_mem_free = self._versioned_function("cuMemFree_v2", "cuMemFree")
        self.cu_mem_free.argtypes = [ctypes.c_uint64]
        self.cu_mem_free.restype = ctypes.c_int

        self.lib.cuMemGetHandleForAddressRange.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_ulonglong,
        ]
        self.lib.cuMemGetHandleForAddressRange.restype = ctypes.c_int

        self.lib.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.cuGetErrorName.restype = ctypes.c_int
        self.lib.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.cuGetErrorString.restype = ctypes.c_int

    def check(self, result: int, operation: str) -> None:
        """Raise when a CUDA Driver API operation fails."""
        if result != CUDA_SUCCESS:
            raise RuntimeError(f"{operation} failed: {self.error_text(result)}")

    def error_text(self, result: int) -> str:
        """Return a printable CUDA result name and description."""
        name = ctypes.c_char_p()
        description = ctypes.c_char_p()
        self.lib.cuGetErrorName(result, ctypes.byref(name))
        self.lib.cuGetErrorString(result, ctypes.byref(description))
        name_text = name.value.decode() if name.value else "unknown"
        description_text = (
            description.value.decode() if description.value else "unknown error"
        )
        return f"{result} ({name_text}: {description_text})"

    def _versioned_function(self, preferred: str, fallback: str) -> Any:
        function = getattr(self.lib, preferred, None)
        if function is None:
            function = getattr(self.lib, fallback)
        return function


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export a direct cuMemAlloc allocation as a DMA-BUF fd."
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--size-mib",
        type=int,
        default=2,
        help="Allocation size in MiB (default: 2)",
    )
    return parser.parse_args()


def main() -> int:
    """Allocate with ``cuMemAlloc`` and attempt DMA-BUF export."""
    args = parse_args()
    if args.size_mib <= 0:
        raise ValueError("--size-mib must be positive")

    driver = CudaDriver()
    driver.check(driver.lib.cuInit(0), "cuInit")

    device = ctypes.c_int()
    driver.check(
        driver.lib.cuDeviceGet(ctypes.byref(device), args.device), "cuDeviceGet"
    )
    name = ctypes.create_string_buffer(256)
    driver.check(
        driver.lib.cuDeviceGetName(name, len(name), device.value), "cuDeviceGetName"
    )

    supported = ctypes.c_int(-1)
    driver.check(
        driver.lib.cuDeviceGetAttribute(
            ctypes.byref(supported),
            CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
            device.value,
        ),
        "cuDeviceGetAttribute(DMA_BUF_SUPPORTED)",
    )

    context = ctypes.c_void_p()
    driver.check(
        driver.cu_ctx_create(ctypes.byref(context), 0, device.value), "cuCtxCreate"
    )

    size = args.size_mib * 1024 * 1024
    device_ptr = ctypes.c_uint64()
    dmabuf_fd = ctypes.c_int(-1)
    export_result = -1
    try:
        driver.check(driver.cu_mem_alloc(ctypes.byref(device_ptr), size), "cuMemAlloc")
        try:
            export_result = int(
                driver.lib.cuMemGetHandleForAddressRange(
                    ctypes.byref(dmabuf_fd),
                    device_ptr.value,
                    size,
                    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                    0,
                )
            )
        finally:
            if dmabuf_fd.value >= 0:
                os.close(dmabuf_fd.value)
            driver.check(driver.cu_mem_free(device_ptr.value), "cuMemFree")
    finally:
        driver.check(driver.cu_ctx_destroy(context), "cuCtxDestroy")

    print(f"device: {args.device} ({name.value.decode()})")
    print(f"CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED: {supported.value}")
    print(f"allocation: cuMemAlloc(ptr={device_ptr.value:#x}, size={size})")
    print(
        "cuMemGetHandleForAddressRange: "
        + (
            "CUDA_SUCCESS"
            if export_result == CUDA_SUCCESS
            else driver.error_text(export_result)
        )
    )

    if export_result == CUDA_SUCCESS:
        return 0
    if export_result == CUDA_ERROR_NOT_SUPPORTED:
        return 2
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
