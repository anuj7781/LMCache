// SPDX-License-Identifier: Apache-2.0
//
// Checks whether an AMD-VRAM READ_FIXED that appears to fail (destination
// unchanged, per prior investigation) actually failed to transfer data, or
// whether the DMA landed correctly and the failure is an artifact of how
// the result was being checked.
//
// Every prior check of READ_FIXED's result used hipMemcpyDtoH -- a
// GPU-engine-mediated read. This program checks the SAME post-READ_FIXED
// buffer two ways:
//
//   1. hipMemcpyDtoH   -- as before.
//   2. CPU mmap() of the dma-buf fd, bracketed with DMA_BUF_IOCTL_SYNC
//      (DMA_BUF_SYNC_START/END) -- the dma-buf framework's own blessed
//      mechanism for cross-device coherency on CPU access, independent of
//      the GPU engine entirely.
//
// If (1) shows stale/wrong data but (2) shows the correct freshly-written
// bytes, that means the DMA transfer actually succeeded and the real bug is
// a GPU-cache-visibility gap in how ROCm/amdgpu exposes externally-written
// memory back to the GPU engine -- not a transport failure. If both show
// wrong data, that rules out the coherency explanation.
//
// WARNING: overwrites 2 MiB at the given device offset.
//
// Build:
//   cc -O2 -o amd_cache_coherency_check amd_cache_coherency_check.c -luring -L/opt/rocm/lib -lamdhip64
//
// Run:
//   ./amd_cache_coherency_check /dev/nvme0n1 <byte-offset>

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <linux/dma-buf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define IO_SIZE (2U * 1024U * 1024U)
#define PATTERN_BYTE 0x37
#define POISON_BYTE 0xAB

typedef int hipError_t;
typedef void *hipDeviceptr_t;
enum {
    hipSuccess = 0,
    hipMemcpyDeviceToHost = 2,
    hipMemRangeHandleTypeDmaBufFd = 1,
};

extern hipError_t hipSetDevice(int device);
extern hipError_t hipMalloc(void **ptr, size_t size);
extern hipError_t hipFree(void *ptr);
extern hipError_t hipMemcpy(void *dst, const void *src, size_t size, int kind);
extern hipError_t hipMemset(void *dst, int value, size_t size);
extern hipError_t hipDeviceSynchronize(void);
extern hipError_t hipMemGetHandleForAddressRange(void *handle,
                                                 hipDeviceptr_t ptr,
                                                 size_t size, int type,
                                                 unsigned long long flags);
extern const char *hipGetErrorString(hipError_t error);

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

static void die(const char *what)
{
    fprintf(stderr, "%s: %s\n", what, strerror(errno));
    exit(2);
}

static void hip_check(hipError_t error, const char *what)
{
    if (error == hipSuccess)
        return;
    fprintf(stderr, "%s: HIP error %d (%s)\n", what, error,
            hipGetErrorString(error));
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

// Verify via hipMemcpyDtoH -- the GPU-engine path used by every prior check
// in this investigation.
static int verify_via_hip(void *gpu_ptr, const unsigned char *pattern,
                          unsigned char *scratch)
{
    hip_check(hipMemcpy(scratch, gpu_ptr, IO_SIZE, hipMemcpyDeviceToHost),
              "hipMemcpy D2H");
    return memcmp(scratch, pattern, IO_SIZE) == 0;
}

// Verify via CPU mmap() of the dma-buf fd + DMA_BUF_IOCTL_SYNC, bypassing
// the GPU engine entirely. Returns -1 if mmap() itself is unsupported for
// this buffer (which is itself informative -- reported by the caller), 0/1
// for fail/pass otherwise.
static int verify_via_mmap(int dmabuf_fd, const unsigned char *pattern,
                           unsigned char *scratch)
{
    struct dma_buf_sync sync;
    void *map = mmap(NULL, IO_SIZE, PROT_READ, MAP_SHARED, dmabuf_fd, 0);

    if (map == MAP_FAILED) {
        fprintf(stderr, "mmap(dmabuf_fd): %s\n", strerror(errno));
        return -1;
    }

    sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if (ioctl(dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync) < 0) {
        fprintf(stderr, "DMA_BUF_IOCTL_SYNC START|READ: %s\n",
                strerror(errno));
        munmap(map, IO_SIZE);
        return -1;
    }
    memcpy(scratch, map, IO_SIZE);
    sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    if (ioctl(dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync) < 0)
        fprintf(stderr, "DMA_BUF_IOCTL_SYNC END|READ: %s\n", strerror(errno));

    munmap(map, IO_SIZE);
    return memcmp(scratch, pattern, IO_SIZE) == 0;
}

int main(int argc, char **argv)
{
    struct io_uring ring;
    void *gpu_ptr;
    int dmabuf_fd = -1;
    unsigned char *pattern, *zeros, *scratch_hip, *scratch_mmap;
    char *end;
    uint64_t offset;
    long page = sysconf(_SC_PAGESIZE);
    int nvme_fd, ret, hip_pass, mmap_pass;

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
    scratch_hip = alloc_aligned((size_t)page, IO_SIZE);
    scratch_mmap = alloc_aligned((size_t)page, IO_SIZE);
    memset(pattern, PATTERN_BYTE, IO_SIZE);
    memset(zeros, POISON_BYTE, IO_SIZE);

    printf("WARNING: overwriting %u bytes at %s offset %llu\n", IO_SIZE,
           argv[1], (unsigned long long)offset);

    hip_check(hipSetDevice(0), "hipSetDevice");
    hip_check(hipMalloc(&gpu_ptr, IO_SIZE), "hipMalloc");
    if ((uintptr_t)gpu_ptr % (uintptr_t)page) {
        fprintf(stderr, "hipMalloc returned an unaligned address\n");
        return 2;
    }
    hip_check(hipMemGetHandleForAddressRange(
                  &dmabuf_fd, (hipDeviceptr_t)gpu_ptr, IO_SIZE,
                  hipMemRangeHandleTypeDmaBufFd, 0),
              "hipMemGetHandleForAddressRange");

    ret = io_uring_queue_init(8, &ring, 0);
    if (ret < 0) {
        fprintf(stderr, "io_uring_queue_init: %s\n", strerror(-ret));
        return 2;
    }
    ret = register_buffer(&ring, nvme_fd, dmabuf_fd);
    if (ret < 0) {
        fprintf(stderr, "register dma-buf: %s\n", strerror(-ret));
        return 2;
    }

    // Ground truth on disk, independent of the GPU buffer.
    require_full(pwrite(nvme_fd, pattern, IO_SIZE, offset),
                 "pwrite ground truth");

    // Poison the GPU buffer so a no-op READ_FIXED can't hide behind
    // leftover data.
    hip_check(hipMemset(gpu_ptr, POISON_BYTE, IO_SIZE), "hipMemset (poison)");
    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");

    ret = fixed_io(&ring, nvme_fd, offset, 1 /* read */);
    if (ret != IO_SIZE) {
        fprintf(stderr, "READ_FIXED: ERROR (%s)\n",
                ret < 0 ? strerror(-ret) : "short I/O");
        return 2;
    }

    hip_pass = verify_via_hip(gpu_ptr, pattern, scratch_hip);
    printf("verify via hipMemcpyDtoH : %s\n", hip_pass ? "PASS" : "FAIL");

    mmap_pass = verify_via_mmap(dmabuf_fd, pattern, scratch_mmap);
    if (mmap_pass < 0)
        printf("verify via mmap+DMA_BUF_SYNC: UNSUPPORTED (see error above)\n");
    else
        printf("verify via mmap+DMA_BUF_SYNC: %s\n",
               mmap_pass ? "PASS" : "FAIL");

    if (mmap_pass < 0) {
        printf("interpretation: mmap of this dma-buf isn't usable for CPU "
               "verification on this system; can't distinguish transport "
               "failure from a coherency gap this way.\n");
    } else if (!hip_pass && mmap_pass) {
        printf("interpretation: DMA transfer succeeded -- hipMemcpyDtoH is "
               "returning stale/cached data. This points at a GPU-cache "
               "coherency gap, not a transport failure.\n");
    } else if (!hip_pass && !mmap_pass) {
        printf("interpretation: both paths see wrong data -- the DMA "
               "transfer itself did not land correctly. Coherency "
               "explanation ruled out.\n");
    } else if (hip_pass && mmap_pass) {
        printf("interpretation: both paths agree the data is correct -- no "
               "failure reproduced on this run.\n");
    } else {
        printf("interpretation: hipMemcpyDtoH passed but mmap+sync failed -- "
               "unexpected; worth double-checking the mmap path itself.\n");
    }

    io_uring_queue_exit(&ring);
    close(dmabuf_fd);
    hip_check(hipFree(gpu_ptr), "hipFree");
    close(nvme_fd);
    free(scratch_mmap);
    free(scratch_hip);
    free(zeros);
    free(pattern);

    return (hip_pass == 1 && mmap_pass == 1) ? 0 : 1;
}
