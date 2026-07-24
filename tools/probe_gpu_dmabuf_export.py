#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe whether a PyTorch GPU pointer can be exported as a DMA-BUF fd.

This is the go/no-go gate for the io_uring DMA-BUF storage backend. The whole
approach depends on turning a ``torch.empty(device="cuda")`` allocation into a
DMA-BUF fd that an O_DIRECT NVMe device can then DMA into and out of.

On an AMD/ROCm PyTorch build this uses ``hipMemGetHandleForAddressRange`` from
``libamdhip64``; on an NVIDIA/CUDA build it uses ``cuMemGetHandleForAddressRange``
from ``libcuda``. PyTorch keeps the ``torch.cuda`` namespace and ``device="cuda"``
on ROCm, so the torch side is identical either way -- only the driver library and
the export symbol differ. The backend is auto-detected from ``torch.version.hip``.

This script tests *only* that export, in isolation: no LMCache, no io_uring, no
NVMe. Run it first, on the target GPU + driver, so the export question is answered
before wiring the full engine.

What it checks
--------------
1. The default PyTorch allocator pointer.
2. A *sub-range* of that allocation -- the backend exports <=1 GiB sub-ranges of a
   larger pool, so sub-range export must work too.
3. The CUDA/HIP PCIe-mapping flag, which a peer NVMe device needs for real P2P.

Usage
-----
    python tools/probe_gpu_dmabuf_export.py [--size-mib 8] [--device cuda:0]

Exit status
-----------
0  the GPU pointer exported successfully (Approach A viable)
1  export failed on this driver/config (see the printed remediation)
2  GPU/torch unavailable
"""

from __future__ import annotations

# Standard
from typing import Optional
import argparse
import ctypes
import os
import sys

# Third Party
import torch


class GpuDriver:
    """Common interface for the HIP/CUDA DMA-BUF address-range export call."""

    #: Human label, e.g. "HIP" or "CUDA".
    api: str = "GPU"
    #: DMA-BUF value of the range-handle-type enum (1 on both HIP and CUDA).
    handle_type_dmabuf: int = 1
    #: Success return code (0 on both hipError_t and CUresult).
    success: int = 0

    def driver_version(self) -> int:
        raise NotImplementedError

    def strerror(self, rc: int) -> str:
        raise NotImplementedError

    def export_dmabuf(self, dptr: int, size: int, flags: int) -> int:
        raise NotImplementedError

    def pcie_flag(self) -> Optional[int]:
        """Optional flags value requesting a PCIe/BAR mapping (P2P), if any."""
        return None

    def remediation(self) -> list[str]:
        raise NotImplementedError

    def _finish_export(self, rc: int, fd: "ctypes.c_int") -> int:
        if rc != self.success:
            raise OSError(f"{self.api} export: {self.strerror(rc)}")
        if fd.value < 0:
            raise OSError(f"{self.api} export reported success but returned fd < 0")
        return fd.value


class HipDriver(GpuDriver):
    """AMD/ROCm HIP runtime wrapper (``libamdhip64``)."""

    api = "HIP"
    # hipMemRangeHandleTypeDmaBufFd == 1
    handle_type_dmabuf = 1
    # hipSuccess == 0
    success = 0
    # hipMemRangeFlagDmaBufMappingTypePcie == 0x1
    _PCIE_FLAG = 1

    def __init__(self) -> None:
        lib = None
        for name in (
            "libamdhip64.so",
            "libamdhip64.so.6",
            "libamdhip64.so.5",
        ):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError(
                "could not load libamdhip64.so -- is ROCm installed and on the "
                "loader path?"
            )
        self._lib = lib

        lib.hipInit.argtypes = [ctypes.c_uint]
        lib.hipInit.restype = ctypes.c_int
        rc = lib.hipInit(0)
        if rc != self.success:
            raise RuntimeError(f"hipInit failed: {self.strerror(rc)}")

        lib.hipDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.hipDriverGetVersion.restype = ctypes.c_int

        if not hasattr(lib, "hipMemGetHandleForAddressRange"):
            raise RuntimeError(
                "libamdhip64 does not export hipMemGetHandleForAddressRange "
                "(needs ROCm >= 5.6)"
            )
        export = lib.hipMemGetHandleForAddressRange
        export.argtypes = [
            ctypes.c_void_p,   # void *handle (out)
            ctypes.c_void_p,   # hipDeviceptr_t dptr
            ctypes.c_size_t,   # size_t size
            ctypes.c_int,      # hipMemRangeHandleType
            ctypes.c_ulonglong,  # unsigned long long flags
        ]
        export.restype = ctypes.c_int
        self._export = export

        # hipGetErrorName returns const char* directly (unlike cuGetErrorName).
        try:
            lib.hipGetErrorName.argtypes = [ctypes.c_int]
            lib.hipGetErrorName.restype = ctypes.c_char_p
        except AttributeError:
            pass

    def driver_version(self) -> int:
        version = ctypes.c_int(0)
        self._lib.hipDriverGetVersion(ctypes.byref(version))
        return version.value

    def strerror(self, rc: int) -> str:
        try:
            name = self._lib.hipGetErrorName(rc)
            if name:
                return f"{name.decode()} (hipError_t {rc})"
        except Exception:
            pass
        return f"hipError_t {rc}"

    def export_dmabuf(self, dptr: int, size: int, flags: int) -> int:
        fd = ctypes.c_int(-1)
        rc = self._export(
            ctypes.byref(fd),
            ctypes.c_void_p(dptr),
            ctypes.c_size_t(size),
            ctypes.c_int(self.handle_type_dmabuf),
            ctypes.c_ulonglong(flags),
        )
        return self._finish_export(rc, fd)

    def pcie_flag(self) -> Optional[int]:
        return self._PCIE_FLAG

    def remediation(self) -> list[str]:
        lines = [
            "    -> On AMD, hipMalloc-backed pointers are normally exportable, so a",
            "       failure here usually means the ROCm version predates dmabuf",
            "       export, the amdgpu kernel module lacks DMABUF support, or the",
            "       GPU/driver combination does not support it.",
            "    -> Check `dmesg | grep amdgpu` and the ROCm version, then fall back",
            "       to the DRM GEM export path (amdgpu GEM_CREATE + PrimeHandleToFD)",
            "       as Approach B for AMD.",
        ]
        conf = os.environ.get("PYTORCH_HIP_ALLOC_CONF", "") or os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", ""
        )
        if "expandable_segments" not in conf:
            lines.insert(
                0,
                "    -> Also try PYTORCH_HIP_ALLOC_CONF=expandable_segments:True "
                "(VMM-backed pool).",
            )
        return lines


class CudaDriver(GpuDriver):
    """NVIDIA CUDA driver wrapper (``libcuda``)."""

    api = "CUDA"
    # CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD == 1
    handle_type_dmabuf = 1
    # CUDA_SUCCESS == 0
    success = 0
    # CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE == 1 (CUDA 12.2+)
    _PCIE_FLAG = 1

    def __init__(self) -> None:
        lib = None
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("could not load libcuda.so(.1) -- is the driver installed?")
        self._lib = lib

        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        rc = lib.cuInit(0)
        if rc != self.success:
            raise RuntimeError(f"cuInit failed: {self.strerror(rc)}")

        lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cuDriverGetVersion.restype = ctypes.c_int

        if not hasattr(lib, "cuMemGetHandleForAddressRange"):
            raise RuntimeError(
                "libcuda does not export cuMemGetHandleForAddressRange "
                "(needs driver >= 515)"
            )
        export = lib.cuMemGetHandleForAddressRange
        export.argtypes = [
            ctypes.c_void_p,     # void *handle (out)
            ctypes.c_ulonglong,  # CUdeviceptr dptr
            ctypes.c_size_t,     # size_t size
            ctypes.c_uint,       # CUmemRangeHandleType
            ctypes.c_ulonglong,  # unsigned long long flags
        ]
        export.restype = ctypes.c_int
        self._export = export

        try:
            lib.cuGetErrorName.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_char_p),
            ]
            lib.cuGetErrorName.restype = ctypes.c_int
        except AttributeError:
            pass

    def driver_version(self) -> int:
        version = ctypes.c_int(0)
        self._lib.cuDriverGetVersion(ctypes.byref(version))
        return version.value

    def strerror(self, rc: int) -> str:
        try:
            name = ctypes.c_char_p()
            if self._lib.cuGetErrorName(rc, ctypes.byref(name)) == 0 and name.value:
                return f"{name.value.decode()} (CUresult {rc})"
        except Exception:
            pass
        return f"CUresult {rc}"

    def export_dmabuf(self, dptr: int, size: int, flags: int) -> int:
        fd = ctypes.c_int(-1)
        rc = self._export(
            ctypes.byref(fd),
            ctypes.c_ulonglong(dptr),
            ctypes.c_size_t(size),
            ctypes.c_uint(self.handle_type_dmabuf),
            ctypes.c_ulonglong(flags),
        )
        return self._finish_export(rc, fd)

    def pcie_flag(self) -> Optional[int]:
        return self._PCIE_FLAG

    def remediation(self) -> list[str]:
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in conf:
            return [
                "    -> Re-run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                "       so PyTorch backs the pool with the CUDA VMM API (cuMemMap),",
                "       which is far more likely to be exportable.",
            ]
        return [
            "    -> expandable_segments did not help. Fall back to Approach B:",
            "       allocate slabs directly with the VMM API (cuMemCreate/cuMemMap)",
            "       and wrap them with torch.from_blob.",
        ]


def make_driver() -> GpuDriver:
    """Select the HIP or CUDA driver based on the active PyTorch build."""
    if getattr(torch.version, "hip", None):
        return HipDriver()
    return CudaDriver()


def page_aligned_gpu_buffer(
    size: int, device: torch.device, page: int
) -> tuple[torch.Tensor, int]:
    """Allocate a GPU buffer and return ``(owning_tensor, page_aligned_ptr)``.

    The export requires a page-aligned base and size; device allocators do not
    guarantee full-page alignment, so over-allocate by one page and hand back an
    aligned pointer into it. The owning tensor must outlive every exported fd.
    """
    base = torch.empty(size + page, dtype=torch.uint8, device=device)
    ptr = base.data_ptr()
    align_offset = (-ptr) % page
    return base, ptr + align_offset


def dmabuf_size(fd: int) -> int:
    """Best-effort size of a dmabuf fd via ``lseek(SEEK_END)`` (-1 if unsupported)."""
    try:
        size = os.lseek(fd, 0, os.SEEK_END)
        os.lseek(fd, 0, os.SEEK_SET)
        return size
    except OSError:
        return -1


def attempt(driver: GpuDriver, label: str, dptr: int, size: int, flags: int) -> bool:
    """Run one export attempt and print the outcome."""
    print(f"\n[{label}]")
    print(f"    ptr=0x{dptr:x} size={size} flags={flags}")
    try:
        fd = driver.export_dmabuf(dptr, size, flags)
    except OSError as e:
        print(f"    FAIL: {e}")
        return False
    reported = dmabuf_size(fd)
    print(f"    OK:   dmabuf_fd={fd}  reported_size={reported}")
    os.close(fd)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe DMA-BUF export of a PyTorch GPU pointer (HIP or CUDA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size-mib", type=int, default=8, help="probe buffer size")
    parser.add_argument("--device", default="cuda:0", help="torch device")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No GPU is available to PyTorch (torch.cuda.is_available() is False)",
              file=sys.stderr)
        return 2

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    # Force PyTorch to create its device context before we call the driver API, so
    # the export runs against the same context that owns the allocation.
    torch.zeros(1, dtype=torch.uint8, device=device)

    page = os.sysconf("SC_PAGESIZE")
    size = args.size_mib * 1024 * 1024
    size = ((size + page - 1) // page) * page

    driver = make_driver()

    print("=== environment ===")
    print(f"    api                     {driver.api}")
    print(f"    torch                   {torch.__version__}")
    print(f"    torch cuda build        {torch.version.cuda}")
    print(f"    torch hip build         {getattr(torch.version, 'hip', None)}")
    print(f"    driver version          {driver.driver_version()}")
    print(f"    gpu                     {torch.cuda.get_device_name(device)}")
    print(
        "    PYTORCH_CUDA_ALLOC_CONF "
        f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(unset)')}"
    )
    print(
        "    PYTORCH_HIP_ALLOC_CONF  "
        f"{os.environ.get('PYTORCH_HIP_ALLOC_CONF', '(unset)')}"
    )
    print(f"    page size               {page}")
    print(f"    probe size              {size} bytes ({size // (1024 * 1024)} MiB)")

    base, aligned_ptr = page_aligned_gpu_buffer(size, device, page)

    primary_ok = False
    try:
        # 1. Full allocation, driver-chosen mapping. This is the core question.
        primary_ok = attempt(
            driver, f"{driver.api} torch pointer, flags=0", aligned_ptr, size, 0
        )

        # 2. Sub-range export (the backend exports <=1 GiB sub-ranges of a pool).
        if size > 2 * page:
            attempt(
                driver,
                f"{driver.api} torch pointer sub-range, flags=0",
                aligned_ptr + page,
                size - page,
                0,
            )

        # 3. PCIe mapping flag -- what a peer NVMe device needs for real P2P.
        #    Informational: a failure here with #1 passing means export works but
        #    P2P may fall back through host memory, or the driver predates it.
        pcie = driver.pcie_flag()
        if pcie is not None:
            attempt(
                driver,
                f"{driver.api} torch pointer, PCIe mapping flag",
                aligned_ptr,
                size,
                pcie,
            )
    finally:
        del base  # keep the allocation alive across every attempt above

    print("\n=== result ===")
    if primary_ok:
        print(f"    {driver.api}: torch GPU pointer IS exportable as a DMA-BUF fd.")
        print("    -> Approach A is viable; back the backend pool with a torch tensor.")
        print("    -> Export success is NOT the same as P2P: verify true peer-to-peer")
        print("       separately (PCIe topology + kernel P2PDMA). A plain export can")
        print("       still bounce through host memory.")
        return 0

    print(f"    {driver.api}: torch GPU pointer is NOT exportable on this driver/config.")
    for line in driver.remediation():
        print(line)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
