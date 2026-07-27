// SPDX-License-Identifier: Apache-2.0
//
// Stage-1 minimal io_uring DMA-BUF test.
//
// For each exporter (udmabuf, then AMD VRAM), runs 4 operations and verifies
// each one independently against its own ground truth:
//
//   1. STORE:       fill the exporter's backing store with a known pattern,
//                   then read it straight back through the exporter itself
//                   (no io_uring) to confirm the exporter's own set/get path
//                   works before layering io_uring on top of it.
//   2. WRITE_FIXED:  buffer -> NVMe, checked with a plain pread() ground
//                    truth on the device. Bypasses READ_FIXED entirely, so
//                    this result does not depend on phase 3.
//   3. READ_FIXED:   NVMe -> buffer, seeded by a plain pwrite() ground truth
//                    on the device (independent of phase 2's WRITE_FIXED)
//                    and preceded by zeroing the buffer, so a no-op
//                    READ_FIXED can't hide behind phase 1's leftover data.
//
// Each phase prints its own PASS/FAIL/ERROR, so a run tells you not just
// THAT something is broken but WHICH of the 4 operations is broken --
// without stage 2's byte-level poison classification (plain memcmp here,
// not expected/poison/other counts) and without DMA_BUF_SYNC. See
// docs/design/v1/storage_backend/iou_dmabuf_debug_log.md ("basic" vs
// "minimal" reproducer) for why those two are unneeded here.
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

// Page-aligned: these buffers are used for direct pread()/pwrite() ground
// truth against the O_DIRECT NVMe fd, which requires aligned buffers.
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

// Op 1 (write half): fill the exporter's backing store with a known pattern.
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

// Op 1 (read half) / op 4: read the exporter's backing store back to host.
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

// Tri-state per-phase verdict: -1 = harness/transport error (never reached
// this phase's data check), 0 = data mismatch, 1 = pass.
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

    // Phase 1 (op 1): store, verified by reading the exporter's own backing
    // store straight back -- no io_uring involved yet.
    store_pattern(exp, pattern);
    load_pattern(exp, actual);
    result.store_ok = memcmp(actual, pattern, IO_SIZE) == 0;
    printf("%-7s STORE      : %s\n", exp->name, result.store_ok ? "PASS" : "FAIL");

    // Phase 2 (op 2): WRITE_FIXED, verified by a plain pread() ground truth
    // on the NVMe device. Independent of phase 3 (READ_FIXED never runs).
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

    // Phase 3 (op 3): READ_FIXED, seeded by a plain pwrite() ground truth on
    // the NVMe device -- independent of phase 2's WRITE_FIXED result. The
    // buffer is zeroed first so a no-op READ_FIXED can't hide behind phase
    // 1's leftover correct data.
    require_full(pwrite(nvme_fd, pattern, IO_SIZE, offset),
                 "pwrite ground truth");
    clear_backing(exp, zeros);
    ret = fixed_io(&ring, nvme_fd, offset, 1 /* read */);
    if (ret == IO_SIZE) {
        load_pattern(exp, actual);        // op 4
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
    struct exporter udmabuf, amdgpu;
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

    amdgpu = create_amdgpu();
    gpu_result = run_test(&amdgpu, nvme_fd, offset, pattern, zeros, actual);
    close_exporter(&amdgpu);

    close(nvme_fd);
    free(actual);
    free(zeros);
    free(pattern);

    if (phase_has_error(&udma_result) || phase_has_error(&gpu_result))
        return 2;
    return phase_all_pass(&udma_result) && phase_all_pass(&gpu_result) ? 0 : 1;
}
