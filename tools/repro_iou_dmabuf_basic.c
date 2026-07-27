// SPDX-License-Identifier: Apache-2.0
//
// Stage-1 minimal io_uring DMA-BUF round-trip test.
//
// The simplest thing that could possibly show the bug: for each exporter
// (udmabuf, then AMD VRAM), fill a buffer with a known byte pattern,
// WRITE_FIXED it to NVMe, zero the buffer, READ_FIXED it back from NVMe,
// and compare. No poisoning classification, no independent ground-truth
// pread/pwrite side-channel, no DMA_BUF_SYNC -- see
// docs/design/v1/storage_backend/iou_dmabuf_debug_log.md ("basic" vs
// "minimal" reproducer) for why those aren't needed here. For a test that
// tells you WHICH direction (write vs read) is broken, use
// repro_iou_dmabuf_minimal.c instead -- this one only tells you THAT the
// round trip is broken.
//
// The zero-out between WRITE_FIXED and READ_FIXED is not optional: without
// it, a READ_FIXED that silently does nothing would leave the buffer's
// original (correct) contents in place and this test would falsely PASS.
//
// Like repro_iou_dmabuf_p2p.c and repro_iou_dmabuf_minimal.c, this redeclares
// its own minimal copy of the dma-buf registration ABI and calls it via the
// raw io_uring_register(2) syscall, so it builds against any stock
// liburing-dev -- no patched headers needed.
//
// WARNING: overwrites 2 MiB at the supplied raw-device offset.
//
// Build:
//   cc -O2 -o repro_iou_dmabuf_basic tools/repro_iou_dmabuf_basic.c -luring -L/opt/rocm/lib -lamdhip64
//
// Run:
//   ./repro_iou_dmabuf_basic /dev/nvme0n1 $((4<<30))

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
#include <sys/syscall.h>
#include <unistd.h>

#define IO_SIZE (2U * 1024U * 1024U)
#define FILL_BYTE 0xAB
#define RETRIES 1000

typedef int hipError_t;
typedef void *hipDeviceptr_t;
enum {
    hipSuccess = 0,
    hipMemcpyHostToDevice = 1,
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

// ---- io_uring dma-buf token ABI (patched kernel; define defensively) --------
// Own minimal copy, matching repro_iou_dmabuf_p2p.c / repro_iou_dmabuf_minimal.c.
#ifndef IORING_REGISTER_BUFFERS2
#define IORING_REGISTER_BUFFERS2 15
#endif
#ifndef IORING_REGISTER_BUFFERS_UPDATE
#define IORING_REGISTER_BUFFERS_UPDATE 16
#endif
#ifndef IORING_RSRC_REGISTER_SPARSE
#define IORING_RSRC_REGISTER_SPARSE (1u << 0)
#endif
#define REPRO_RSRC_UPDATE_EXTENDED (1u << 1)
#define REPRO_REGBUF_TYPE_DMABUF 2

struct repro_rsrc_register {
    uint32_t nr;
    uint32_t flags;
    uint64_t resv2;
    uint64_t data;
    uint64_t tags;
};

struct repro_rsrc_update2 {
    uint32_t offset;
    uint32_t flags;
    uint64_t data;
    uint64_t tags;
    uint32_t nr;
    uint32_t resv2;
};

struct repro_regbuf_desc {
    uint32_t type;
    uint32_t flags;
    uint64_t size;
    uint64_t uaddr;
    int32_t dmabuf_fd;
    int32_t target_fd;
    uint64_t __resv[6];
};

static int io_uring_register_raw(int ring_fd, unsigned opcode, void *arg,
                                 unsigned nr)
{
    long r = syscall(__NR_io_uring_register, ring_fd, opcode, arg, nr);
    return (int)r;
}

// One exporter under test: either a udmabuf (host memfd-backed) or an AMD
// VRAM allocation. Exactly one of {memfd, gpu_ptr} is meaningful per kind.
struct exporter {
    const char *name;
    int gpu;
    int dmabuf_fd;
    int memfd;
    void *gpu_ptr;
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

static struct exporter create_udmabuf(void)
{
    struct exporter exp = {.name = "udmabuf", .dmabuf_fd = -1, .memfd = -1};
    struct udmabuf_create create = {0};
    int devfd = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);

    if (devfd < 0)
        die("open /dev/udmabuf");
    exp.memfd = memfd_create("iou-dmabuf-basic", MFD_CLOEXEC | MFD_ALLOW_SEALING);
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

static struct exporter create_amdgpu(void)
{
    struct exporter exp = {.name = "amdgpu", .gpu = 1, .dmabuf_fd = -1};

    hip_check(hipSetDevice(0), "hipSetDevice");
    hip_check(hipMalloc(&exp.gpu_ptr, IO_SIZE), "hipMalloc");
    if ((uintptr_t)exp.gpu_ptr % (uintptr_t)sysconf(_SC_PAGESIZE)) {
        fprintf(stderr, "hipMalloc returned an unaligned address\n");
        exit(2);
    }
    hip_check(hipMemGetHandleForAddressRange(
                  &exp.dmabuf_fd, (hipDeviceptr_t)exp.gpu_ptr, IO_SIZE,
                  hipMemRangeHandleTypeDmaBufFd, 0),
              "hipMemGetHandleForAddressRange");
    return exp;
}

static void close_exporter(struct exporter *exp)
{
    if (exp->gpu) {
        close(exp->dmabuf_fd);
        hip_check(hipFree(exp->gpu_ptr), "hipFree");
    } else {
        close(exp->dmabuf_fd);
        close(exp->memfd);
    }
}

// Operation 1: fill the exporter's backing store with a known pattern.
static void store_pattern(struct exporter *exp, const unsigned char *pattern)
{
    if (exp->gpu) {
        hip_check(hipMemcpy(exp->gpu_ptr, pattern, IO_SIZE,
                            hipMemcpyHostToDevice),
                  "hipMemcpy H2D");
        hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");
        return;
    }
    // Plain pwrite on the memfd -- same physical pages the dmabuf wraps.
    // No mmap, no DMA_BUF_SYNC: we're not going through the dma-buf mmap
    // fop at all.
    require_full(pwrite(exp->memfd, pattern, IO_SIZE, 0), "pwrite memfd");
}

// Required reset so a no-op READ_FIXED can't hide behind leftover data.
static void clear_backing(struct exporter *exp, const unsigned char *zeros)
{
    if (exp->gpu) {
        hip_check(hipMemset(exp->gpu_ptr, 0, IO_SIZE), "hipMemset");
        hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");
        return;
    }
    require_full(pwrite(exp->memfd, zeros, IO_SIZE, 0), "pwrite zero memfd");
}

// Operation 4 (verify half): read the exporter's backing store back to host.
static void load_pattern(struct exporter *exp, unsigned char *out)
{
    if (exp->gpu) {
        hip_check(hipMemcpy(out, exp->gpu_ptr, IO_SIZE, hipMemcpyDeviceToHost),
                  "hipMemcpy D2H");
        return;
    }
    require_full(pread(exp->memfd, out, IO_SIZE, 0), "pread memfd");
}

static int register_buffer(struct io_uring *ring, int nvme_fd, int dmabuf_fd)
{
    struct repro_rsrc_register reg = {0};
    struct repro_regbuf_desc desc = {0};
    struct repro_rsrc_update2 update = {0};
    int ret;

    reg.nr = 1;
    reg.flags = IORING_RSRC_REGISTER_SPARSE;
    ret = io_uring_register_raw(ring->ring_fd, IORING_REGISTER_BUFFERS2, &reg,
                                sizeof(reg));
    if (ret < 0)
        return ret;

    desc.type = REPRO_REGBUF_TYPE_DMABUF;
    desc.dmabuf_fd = dmabuf_fd;
    desc.target_fd = nvme_fd;
    update.flags = REPRO_RSRC_UPDATE_EXTENDED;
    update.data = (uint64_t)(uintptr_t)&desc;
    update.nr = 1;
    ret = io_uring_register_raw(ring->ring_fd, IORING_REGISTER_BUFFERS_UPDATE,
                                &update, sizeof(update));
    return ret == 1 ? 0 : (ret < 0 ? ret : -EIO);
}

// EAGAIN retry is part of the API contract (dma-buf mapping invalidation can
// legitimately return it a few times), not optional test complexity.
static int fixed_io(struct io_uring *ring, int fd, uint64_t offset, int read)
{
    for (int attempt = 0; attempt < RETRIES; attempt++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
        struct io_uring_cqe *cqe;
        int ret;

        if (!sqe)
            return -ENOSPC;
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
        if (ret != -EAGAIN)
            return ret;
    }
    return -EAGAIN;
}

// Operations 2 and 3 (WRITE_FIXED, READ_FIXED) plus pass/fail verdict.
static int run_test(struct exporter *exp, int nvme_fd, uint64_t offset,
                    const unsigned char *pattern, const unsigned char *zeros,
                    unsigned char *actual)
{
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

    store_pattern(exp, pattern);          // op 1
    ret = fixed_io(&ring, nvme_fd, offset, 0 /* write */);   // op 2
    if (ret != IO_SIZE) {
        printf("%-7s WRITE_FIXED: ERROR (%s)\n", exp->name,
               ret < 0 ? strerror(-ret) : "short I/O");
        io_uring_queue_exit(&ring);
        return -1;
    }

    clear_backing(exp, zeros);
    ret = fixed_io(&ring, nvme_fd, offset, 1 /* read */);    // op 3
    if (ret != IO_SIZE) {
        printf("%-7s READ_FIXED : ERROR (%s)\n", exp->name,
               ret < 0 ? strerror(-ret) : "short I/O");
        io_uring_queue_exit(&ring);
        return -1;
    }

    load_pattern(exp, actual);            // op 4
    io_uring_queue_exit(&ring);

    int pass = memcmp(actual, pattern, IO_SIZE) == 0;
    printf("%-7s round trip: %s\n", exp->name, pass ? "PASS" : "FAIL");
    return pass;
}

int main(int argc, char **argv)
{
    unsigned char *pattern, *zeros, *actual;
    struct exporter udmabuf, amdgpu;
    char *end;
    uint64_t offset;
    long page = sysconf(_SC_PAGESIZE);
    int nvme_fd, udma_pass, gpu_pass;

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

    // No O_DIRECT alignment requirement: these host buffers only ever go
    // through hipMemcpy or plain memfd pread/pwrite, never against nvme_fd
    // directly (WRITE_FIXED/READ_FIXED address the dmabuf by fd, not a
    // host pointer).
    pattern = malloc(IO_SIZE);
    zeros = malloc(IO_SIZE);
    actual = malloc(IO_SIZE);
    if (!pattern || !zeros || !actual)
        die("malloc");
    memset(pattern, FILL_BYTE, IO_SIZE);
    memset(zeros, 0, IO_SIZE);

    printf("WARNING: overwriting %u bytes at %s offset %llu\n", IO_SIZE,
           argv[1], (unsigned long long)offset);

    udmabuf = create_udmabuf();
    udma_pass = run_test(&udmabuf, nvme_fd, offset, pattern, zeros, actual);
    close_exporter(&udmabuf);

    amdgpu = create_amdgpu();
    gpu_pass = run_test(&amdgpu, nvme_fd, offset, pattern, zeros, actual);
    close_exporter(&amdgpu);

    close(nvme_fd);
    free(actual);
    free(zeros);
    free(pattern);

    if (udma_pass < 0 || gpu_pass < 0)
        return 2;
    return udma_pass && gpu_pass ? 0 : 1;
}
