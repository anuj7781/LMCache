// SPDX-License-Identifier: Apache-2.0
//
// io_uring DMA-BUF read/write reproducer: NVIDIA GPU VRAM vs. udmabuf.
//
// CUDA counterpart of repro_iou_dmabuf_basic.c (same structure, same checks,
// AMD/HIP calls swapped for the CUDA driver API equivalents). Registers a
// 2 MiB dma-buf with io_uring against an NVMe block device and runs 4 checks
// against it, once for a udmabuf (host-memory-backed) buffer and once for
// NVIDIA GPU VRAM:
//
//   1. STORE       - write a known pattern into the buffer, then read it
//                    straight back through the buffer's own API (no
//                    io_uring), to confirm the buffer itself is sane.
//   2. WRITE_FIXED - write the buffer to the NVMe device via io_uring, then
//                    verify with a plain pread() on the device.
//   3. READ_FIXED  - write a known pattern to the NVMe device with a plain
//                    pwrite(), zero the buffer (so a no-op read can't hide
//                    behind stale data), then read it back via io_uring and
//                    compare.
//
// Each check prints its own PASS/FAIL, so the output shows exactly which
// operation is broken.
//
// WARNING: overwrites 2 MiB at the given device offset.
//
// Build:
//   cc -O2 -o repro_iou_dmabuf_basic_nvidia repro_iou_dmabuf_basic_nvidia.c -luring -lcuda
//   # If -lcuda is not found, link the driver library directly instead:
//   #   "$(ldconfig -p | grep libcuda.so | awk '{print $NF}' | head -1)"
//
// Run:
//   ./repro_iou_dmabuf_basic_nvidia /dev/nvme0n1 <byte-offset>

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <linux/memfd.h>
#include <linux/udmabuf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define IO_SIZE (2U * 1024U * 1024U)
#define FILL_BYTE 0xAB

// Declared here so this builds with a plain C compiler against a
// runtime-only CUDA driver install (libcuda.so) -- no CUDA Toolkit dev
// headers needed. NOTE: the CUDA driver API versions several of these
// symbols (a holdover from the CUDA 3.2 32->64-bit CUdeviceptr transition):
// the real exported symbols are the _v2 names below, not the bare names
// cuda.h normally #defines them to. If linking fails with an undefined
// symbol, that's the first thing to check.
typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef struct CUctx_st *CUcontext;
enum {
    CUDA_SUCCESS = 0,
    CU_MEM_ALLOCATION_TYPE_PINNED = 0x1,
    CU_MEM_LOCATION_TYPE_DEVICE = 0x1,
    CU_MEM_HANDLE_TYPE_NONE = 0x0,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3,
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0,
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 0x1,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
};

// Struct layouts verified against the real cuda.h (CUDA 12.9,
// nvidia-cuda-runtime-cu12 wheel) -- field order/padding matters here since
// these are passed by pointer to the real driver, not just used internally.
typedef struct {
    int type;
    int id;
} CUmemLocation;

typedef struct {
    int type;
    int requestedHandleTypes;
    CUmemLocation location;
    void *win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocFlags;
} CUmemAllocationProp;

typedef struct {
    CUmemLocation location;
    int flags;
} CUmemAccessDesc;

extern CUresult cuInit(unsigned int flags);
extern CUresult cuDeviceGetCount(int *count);
extern CUresult cuDeviceGet(CUdevice *device, int ordinal);
extern CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice dev);
extern CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
extern CUresult cuCtxDestroy_v2(CUcontext ctx);
extern CUresult cuCtxSynchronize(void);
extern CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes);
extern CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytes);
extern CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t n);
extern CUresult cuMemGetHandleForAddressRange(void *handle, CUdeviceptr dptr,
                                              size_t size, int handleType,
                                              unsigned long long flags);
extern CUresult cuGetErrorString(CUresult error, const char **str);

// Virtual Memory Management (VMM) API: plain cuMemAlloc does not reliably
// support cuMemGetHandleForAddressRange (see
// docs/design/v1/storage_backend/iou_dmabuf_backend.md, "Approach B") -- VMM
// allocation is the guaranteed-exportable path, so that's what's used here
// instead of cuMemAlloc.
extern CUresult cuMemGetAllocationGranularity(size_t *granularity,
                                              const CUmemAllocationProp *prop,
                                              int option);
extern CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size,
                                    size_t alignment, CUdeviceptr addr,
                                    unsigned long long flags);
extern CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                            const CUmemAllocationProp *prop,
                            unsigned long long flags);
extern CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                         CUmemGenericAllocationHandle handle,
                         unsigned long long flags);
extern CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                               const CUmemAccessDesc *desc, size_t count);
extern CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
extern CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
extern CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);

#define DMABUF_REGBUF_TYPE 2
#define DMABUF_RSRC_UPDATE_EXTENDED (1u << 1)

struct dmabuf_regbuf_desc {
    uint32_t type;
    uint32_t flags;
    uint64_t size;
    uint64_t uaddr;
    int32_t dmabuf_fd;
    int32_t target_fd;
    uint64_t __resv[6];
};

// Exactly one of {memfd, gpu_ptr} is meaningful, depending on gpu.
// phys_handle/mapped_size are VMM-path-only (see create_nvidia()).
struct exporter {
    const char *name;
    int gpu;
    int dmabuf_fd;
    int memfd;
    CUdeviceptr gpu_ptr;
    CUcontext cuda_ctx;
    CUmemGenericAllocationHandle phys_handle;
    size_t mapped_size;
};

static void die(const char *what)
{
    fprintf(stderr, "%s: %s\n", what, strerror(errno));
    exit(2);
}

static void cuda_check(CUresult error, const char *what)
{
    const char *msg = NULL;

    if (error == CUDA_SUCCESS)
        return;
    cuGetErrorString(error, &msg);
    fprintf(stderr, "%s: CUDA error %d (%s)\n", what, error,
            msg ? msg : "unknown");
    exit(2);
}

static void require_full(ssize_t ret, const char *what)
{
    if (ret == IO_SIZE)
        return;
    if (ret < 0)
        die(what);
    fprintf(stderr, "%s: short I/O (%zd/%u)\n", what, ret, IO_SIZE);
    exit(2);
}

// Page-aligned: used for pread()/pwrite() ground truth against the
// O_DIRECT NVMe fd, which requires aligned buffers.
static unsigned char *alloc_aligned(size_t alignment, size_t size)
{
    void *ptr;
    int ret = posix_memalign(&ptr, alignment, size);

    if (ret) {
        errno = ret;
        die("posix_memalign");
    }
    return ptr;
}

static struct exporter create_udmabuf(void)
{
    struct exporter exp = {.name = "udmabuf", .dmabuf_fd = -1, .memfd = -1};
    struct udmabuf_create create = {0};
    int devfd = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);

    if (devfd < 0)
        die("open /dev/udmabuf");
    exp.memfd = memfd_create("iou-dmabuf-basic-nvidia",
                             MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (exp.memfd < 0)
        die("memfd_create");
    if (fcntl(exp.memfd, F_ADD_SEALS, F_SEAL_SHRINK) < 0)
        die("F_ADD_SEALS");
    if (ftruncate(exp.memfd, IO_SIZE) < 0)
        die("ftruncate");

    create.memfd = (uint32_t)exp.memfd;
    create.flags = UDMABUF_FLAGS_CLOEXEC;
    create.size = IO_SIZE;
    exp.dmabuf_fd = ioctl(devfd, UDMABUF_CREATE, &create);
    close(devfd);
    if (exp.dmabuf_fd < 0)
        die("UDMABUF_CREATE");
    return exp;
}

static struct exporter create_nvidia(void)
{
    struct exporter exp = {.name = "nvidia", .gpu = 1, .dmabuf_fd = -1};
    CUdevice dev;
    CUmemAllocationProp prop = {0};
    CUmemAccessDesc access = {0};
    size_t granularity = 0;
    int count = 0, vmm_supported = 0;

    cuda_check(cuInit(0), "cuInit");
    cuda_check(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (count == 0) {
        fprintf(stderr, "no CUDA devices found\n");
        exit(2);
    }
    cuda_check(cuDeviceGet(&dev, 0), "cuDeviceGet");
    cuda_check(cuCtxCreate_v2(&exp.cuda_ctx, 0, dev), "cuCtxCreate");

    cuda_check(cuDeviceGetAttribute(
                   &vmm_supported,
                   CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev),
              "cuDeviceGetAttribute(VMM)");
    if (!vmm_supported) {
        fprintf(stderr, "device does not support the VMM API\n");
        exit(2);
    }

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = (int)dev;

    cuda_check(cuMemGetAllocationGranularity(&granularity, &prop,
                                             CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "cuMemGetAllocationGranularity");
    exp.mapped_size = (IO_SIZE + granularity - 1) / granularity * granularity;

    cuda_check(cuMemAddressReserve(&exp.gpu_ptr, exp.mapped_size, granularity,
                                   0, 0),
              "cuMemAddressReserve");
    cuda_check(cuMemCreate(&exp.phys_handle, exp.mapped_size, &prop, 0),
              "cuMemCreate");
    cuda_check(cuMemMap(exp.gpu_ptr, exp.mapped_size, 0, exp.phys_handle, 0),
              "cuMemMap");

    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cuda_check(cuMemSetAccess(exp.gpu_ptr, exp.mapped_size, &access, 1),
              "cuMemSetAccess");

    if (exp.gpu_ptr % (CUdeviceptr)sysconf(_SC_PAGESIZE)) {
        fprintf(stderr, "cuMemAddressReserve returned an unaligned address\n");
        exit(2);
    }
    cuda_check(cuMemGetHandleForAddressRange(
                   &exp.dmabuf_fd, exp.gpu_ptr, exp.mapped_size,
                   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
              "cuMemGetHandleForAddressRange");
    return exp;
}

static void close_exporter(struct exporter *exp)
{
    if (exp->gpu) {
        close(exp->dmabuf_fd);
        cuda_check(cuMemUnmap(exp->gpu_ptr, exp->mapped_size), "cuMemUnmap");
        cuda_check(cuMemRelease(exp->phys_handle), "cuMemRelease");
        cuda_check(cuMemAddressFree(exp->gpu_ptr, exp->mapped_size),
                  "cuMemAddressFree");
        cuda_check(cuCtxDestroy_v2(exp->cuda_ctx), "cuCtxDestroy");
    } else {
        close(exp->dmabuf_fd);
        close(exp->memfd);
    }
}

static void store_pattern(struct exporter *exp, const unsigned char *pattern)
{
    if (exp->gpu) {
        cuda_check(cuMemcpyHtoD_v2(exp->gpu_ptr, pattern, IO_SIZE),
                  "cuMemcpyHtoD");
        cuda_check(cuCtxSynchronize(), "cuCtxSynchronize");
        return;
    }
    require_full(pwrite(exp->memfd, pattern, IO_SIZE, 0), "pwrite memfd");
}

static void clear_backing(struct exporter *exp, const unsigned char *zeros)
{
    if (exp->gpu) {
        cuda_check(cuMemsetD8_v2(exp->gpu_ptr, 0, IO_SIZE), "cuMemsetD8");
        cuda_check(cuCtxSynchronize(), "cuCtxSynchronize");
        return;
    }
    require_full(pwrite(exp->memfd, zeros, IO_SIZE, 0), "pwrite zero memfd");
}

static void load_pattern(struct exporter *exp, unsigned char *out)
{
    if (exp->gpu) {
        cuda_check(cuMemcpyDtoH_v2(out, exp->gpu_ptr, IO_SIZE),
                  "cuMemcpyDtoH");
        return;
    }
    require_full(pread(exp->memfd, out, IO_SIZE, 0), "pread memfd");
}

static int register_buffer(struct io_uring *ring, int nvme_fd, int dmabuf_fd)
{
    struct dmabuf_regbuf_desc desc = {
        .type = DMABUF_REGBUF_TYPE,
        .dmabuf_fd = dmabuf_fd,
        .target_fd = nvme_fd,
    };
    struct io_uring_rsrc_update2 update = {
        .resv = DMABUF_RSRC_UPDATE_EXTENDED,
        .data = (uint64_t)(uintptr_t)&desc,
        .nr = 1,
    };
    int ret = io_uring_register_buffers_sparse(ring, 1);

    if (ret < 0)
        return ret;
    ret = io_uring_register(ring->ring_fd, IORING_REGISTER_BUFFERS_UPDATE,
                            &update, sizeof(update));
    return ret == 1 ? 0 : (ret < 0 ? ret : -EIO);
}

static int fixed_io(struct io_uring *ring, int fd, uint64_t offset, int read)
{
    struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
    struct io_uring_cqe *cqe;
    int ret;

    if (read)
        io_uring_prep_read_fixed(sqe, fd, NULL, IO_SIZE, offset, 0);
    else
        io_uring_prep_write_fixed(sqe, fd, NULL, IO_SIZE, offset, 0);
    ret = io_uring_submit(ring);
    if (ret < 0)
        return ret;
    ret = io_uring_wait_cqe(ring, &cqe);
    if (ret < 0)
        return ret;
    ret = cqe->res;
    io_uring_cqe_seen(ring, cqe);
    return ret;
}

// Tri-state per-check verdict: -1 = harness/transport error, 0 = data
// mismatch, 1 = pass.
struct phase_result {
    int store_ok;
    int write_ok;
    int read_ok;
};

static struct phase_result run_test(struct exporter *exp, int nvme_fd,
                                    uint64_t offset,
                                    const unsigned char *pattern,
                                    const unsigned char *zeros,
                                    unsigned char *actual)
{
    struct phase_result result = {.store_ok = -1, .write_ok = -1, .read_ok = -1};
    struct io_uring ring;
    int ret = io_uring_queue_init(8, &ring, 0);

    if (ret < 0) {
        fprintf(stderr, "io_uring_queue_init: %s\n", strerror(-ret));
        exit(2);
    }
    ret = register_buffer(&ring, nvme_fd, exp->dmabuf_fd);
    if (ret < 0) {
        fprintf(stderr, "register %s dma-buf: %s\n", exp->name, strerror(-ret));
        exit(2);
    }

    store_pattern(exp, pattern);
    load_pattern(exp, actual);
    result.store_ok = memcmp(actual, pattern, IO_SIZE) == 0;
    printf("%-7s STORE      : %s\n", exp->name, result.store_ok ? "PASS" : "FAIL");

    ret = fixed_io(&ring, nvme_fd, offset, 0 /* write */);
    if (ret == IO_SIZE) {
        require_full(pread(nvme_fd, actual, IO_SIZE, offset),
                     "pread ground truth");
        result.write_ok = memcmp(actual, pattern, IO_SIZE) == 0;
        printf("%-7s WRITE_FIXED: %s\n", exp->name,
               result.write_ok ? "PASS" : "FAIL");
    } else {
        printf("%-7s WRITE_FIXED: ERROR (%s)\n", exp->name,
               ret < 0 ? strerror(-ret) : "short I/O");
    }

    require_full(pwrite(nvme_fd, pattern, IO_SIZE, offset),
                 "pwrite ground truth");
    clear_backing(exp, zeros);
    ret = fixed_io(&ring, nvme_fd, offset, 1 /* read */);
    if (ret == IO_SIZE) {
        load_pattern(exp, actual);
        result.read_ok = memcmp(actual, pattern, IO_SIZE) == 0;
        printf("%-7s READ_FIXED : %s\n", exp->name,
               result.read_ok ? "PASS" : "FAIL");
    } else {
        printf("%-7s READ_FIXED : ERROR (%s)\n", exp->name,
               ret < 0 ? strerror(-ret) : "short I/O");
    }

    io_uring_queue_exit(&ring);
    return result;
}

static int phase_has_error(const struct phase_result *r)
{
    return r->store_ok < 0 || r->write_ok < 0 || r->read_ok < 0;
}

static int phase_all_pass(const struct phase_result *r)
{
    return r->store_ok == 1 && r->write_ok == 1 && r->read_ok == 1;
}

int main(int argc, char **argv)
{
    unsigned char *pattern, *zeros, *actual;
    struct exporter udmabuf, nvidia;
    struct phase_result udma_result, gpu_result;
    char *end;
    uint64_t offset;
    long page = sysconf(_SC_PAGESIZE);
    int nvme_fd;

    if (argc != 3) {
        fprintf(stderr, "usage: %s /dev/nvmeXnY byte-offset\n", argv[0]);
        return 2;
    }
    errno = 0;
    offset = strtoull(argv[2], &end, 0);
    if (errno || *end != '\0' || page <= 0 || offset % (uint64_t)page) {
        fprintf(stderr, "offset must be page aligned\n");
        return 2;
    }
    nvme_fd = open(argv[1], O_RDWR | O_DIRECT);
    if (nvme_fd < 0)
        die("open NVMe");

    pattern = alloc_aligned((size_t)page, IO_SIZE);
    zeros = alloc_aligned((size_t)page, IO_SIZE);
    actual = alloc_aligned((size_t)page, IO_SIZE);
    memset(pattern, FILL_BYTE, IO_SIZE);
    memset(zeros, 0, IO_SIZE);

    printf("WARNING: overwriting %u bytes at %s offset %llu\n", IO_SIZE,
           argv[1], (unsigned long long)offset);

    udmabuf = create_udmabuf();
    udma_result = run_test(&udmabuf, nvme_fd, offset, pattern, zeros, actual);
    close_exporter(&udmabuf);

    nvidia = create_nvidia();
    gpu_result = run_test(&nvidia, nvme_fd, offset, pattern, zeros, actual);
    close_exporter(&nvidia);

    close(nvme_fd);
    free(actual);
    free(zeros);
    free(pattern);

    if (phase_has_error(&udma_result) || phase_has_error(&gpu_result))
        return 2;
    return phase_all_pass(&udma_result) && phase_all_pass(&gpu_result) ? 0 : 1;
}
