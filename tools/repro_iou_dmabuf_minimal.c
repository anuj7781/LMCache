// SPDX-License-Identifier: Apache-2.0
//
// Minimal io_uring DMA-BUF reproducer.
//
// For each exporter (udmabuf and AMD VRAM), issue one WRITE_FIXED and one
// READ_FIXED. Plain O_DIRECT pread/pwrite provide ground truth, so the two
// directions are independent. READ_FIXED starts from a known poison pattern.
//
// WARNING: overwrites 2 MiB at the supplied raw-device offset.
//
// DEPENDS ON THE PATCHED liburing HEADERS. Unlike repro_iou_dmabuf_p2p.c (which
// redeclares its own minimal io_uring_regbuf_desc/IO_REGBUF_TYPE_DMABUF structs
// to build against any stock liburing), this file includes <liburing.h> and
// uses those symbols directly from the header. A stock/unpatched liburing-dev
// does NOT define them and this will fail to compile with "unknown type name
// 'struct io_uring_regbuf_desc'" or similar. Point -I at the patched liburing
// checkout's src/include directory (the one with the top commit adding dma-buf
// token support -- io_uring/io_uring.h there defines these structs). No newer
// *runtime* library is required: these additions are header-level struct/macro
// definitions dispatched through the existing io_uring_register() syscall
// wrapper, so linking against any liburing.so (even a stock one, via -luring)
// is fine as long as the *headers* used at compile time are the patched ones.
//
// Build:
//   SRC=tools/repro_iou_dmabuf_minimal.c
//   URING_INC=/path/to/patched/liburing/src/include   # adjust to your checkout
//   LIBS='-luring -L/opt/rocm/lib -lamdhip64'
//   cc -O2 -I"$URING_INC" -o repro_iou_dmabuf_minimal "$SRC" $LIBS
//   # If -lamdhip64 is not found (runtime-only ROCm, no unversioned .so
//   # symlink), link the versioned soname directly instead of -lamdhip64:
//   #   "$(ls /opt/rocm*/lib/libamdhip64.so* 2>/dev/null | head -1)"
//
// Run:
//   ./repro_iou_dmabuf_minimal /dev/nvme0n1 $((4<<30))
//
// Observed signature: udmabuf PASS/PASS, amdgpu PASS/FAIL with poison unchanged.

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <linux/dma-buf.h>
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
#define POISON_XOR 0xff
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
extern hipError_t hipDeviceSynchronize(void);
extern hipError_t hipMemGetHandleForAddressRange(void *handle,
                                                 hipDeviceptr_t ptr,
                                                 size_t size, int type,
                                                 unsigned long long flags);
extern const char *hipGetErrorString(hipError_t error);

struct test_buffer {
    const char *name;
    void *ptr;
    int dmabuf_fd;
    int memfd;
    int gpu;
};

struct test_result {
    int write_ok;
    int read_ok;
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

static unsigned char *alloc_aligned(size_t alignment)
{
    void *ptr;
    int ret = posix_memalign(&ptr, alignment, IO_SIZE);

    if (ret) {
        errno = ret;
        die("posix_memalign");
    }
    return ptr;
}

static int dmabuf_sync(int fd, uint64_t flags)
{
    struct dma_buf_sync sync = {.flags = flags};

    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}

static struct test_buffer create_udmabuf(void)
{
    struct test_buffer buf = {
        .name = "udmabuf",
        .ptr = MAP_FAILED,
        .dmabuf_fd = -1,
        .memfd = -1,
    };
    struct udmabuf_create create = {0};
    int devfd = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);

    if (devfd < 0)
        die("open /dev/udmabuf");
    buf.memfd = memfd_create("iou-dmabuf-minimal",
                             MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (buf.memfd < 0)
        die("memfd_create");
    if (fcntl(buf.memfd, F_ADD_SEALS, F_SEAL_SHRINK) < 0)
        die("F_ADD_SEALS");
    if (ftruncate(buf.memfd, IO_SIZE) < 0)
        die("ftruncate");

    create.memfd = (uint32_t)buf.memfd;
    create.flags = UDMABUF_FLAGS_CLOEXEC;
    create.size = IO_SIZE;
    buf.dmabuf_fd = ioctl(devfd, UDMABUF_CREATE, &create);
    close(devfd);
    if (buf.dmabuf_fd < 0)
        die("UDMABUF_CREATE");

    buf.ptr = mmap(NULL, IO_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                   buf.dmabuf_fd, 0);
    if (buf.ptr == MAP_FAILED)
        die("mmap udmabuf");
    return buf;
}

static struct test_buffer create_amdgpu(void)
{
    struct test_buffer buf = {
        .name = "amdgpu",
        .dmabuf_fd = -1,
        .memfd = -1,
        .gpu = 1,
    };

    hip_check(hipSetDevice(0), "hipSetDevice");
    hip_check(hipMalloc(&buf.ptr, IO_SIZE), "hipMalloc");
    if ((uintptr_t)buf.ptr % (uintptr_t)sysconf(_SC_PAGESIZE)) {
        fprintf(stderr, "hipMalloc returned an unaligned address\n");
        exit(2);
    }
    hip_check(hipMemGetHandleForAddressRange(
                  &buf.dmabuf_fd, (hipDeviceptr_t)buf.ptr, IO_SIZE,
                  hipMemRangeHandleTypeDmaBufFd, 0),
              "hipMemGetHandleForAddressRange");
    return buf;
}

static void close_buffer(struct test_buffer *buf)
{
    if (buf->gpu) {
        close(buf->dmabuf_fd);
        hip_check(hipFree(buf->ptr), "hipFree");
    } else {
        munmap(buf->ptr, IO_SIZE);
        close(buf->dmabuf_fd);
        close(buf->memfd);
    }
}

static void store_buffer(struct test_buffer *buf, const void *src)
{
    if (buf->gpu) {
        hip_check(hipMemcpy(buf->ptr, src, IO_SIZE, hipMemcpyHostToDevice),
                  "hipMemcpy H2D");
        hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");
        return;
    }
    if (dmabuf_sync(buf->dmabuf_fd,
                    DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE) < 0)
        die("DMA_BUF_SYNC START|WRITE");
    memcpy(buf->ptr, src, IO_SIZE);
    if (dmabuf_sync(buf->dmabuf_fd,
                    DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE) < 0)
        die("DMA_BUF_SYNC END|WRITE");
}

static void load_buffer(struct test_buffer *buf, void *dst)
{
    if (buf->gpu) {
        hip_check(hipMemcpy(dst, buf->ptr, IO_SIZE, hipMemcpyDeviceToHost),
                  "hipMemcpy D2H");
        return;
    }
    if (dmabuf_sync(buf->dmabuf_fd,
                    DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ) < 0)
        die("DMA_BUF_SYNC START|READ");
    memcpy(dst, buf->ptr, IO_SIZE);
    if (dmabuf_sync(buf->dmabuf_fd,
                    DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ) < 0)
        die("DMA_BUF_SYNC END|READ");
}

static void fill_pattern(unsigned char *buf, uint64_t seed)
{
    uint64_t *words = (uint64_t *)buf;

    for (size_t i = 0; i < IO_SIZE / sizeof(*words); i++)
        words[i] = seed ^ (i * UINT64_C(0x9e3779b97f4a7c15));
}

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

static int register_buffer(struct io_uring *ring, int nvme_fd, int dmabuf_fd)
{
    struct io_uring_regbuf_desc desc = {
        .type = IO_REGBUF_TYPE_DMABUF,
        .dmabuf_fd = dmabuf_fd,
        .target_fd = nvme_fd,
    };
    struct io_uring_rsrc_update2 update = {
        .resv = IORING_RSRC_UPDATE_EXTENDED,
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

static struct test_result test_exporter(struct test_buffer *buf, int nvme_fd,
                                        uint64_t offset,
                                        unsigned char *expected,
                                        unsigned char *actual,
                                        unsigned char *poison)
{
    struct test_result result = {.write_ok = -1, .read_ok = -1};
    struct io_uring ring;
    int ret = io_uring_queue_init(8, &ring, 0);

    if (ret < 0) {
        fprintf(stderr, "io_uring_queue_init: %s\n", strerror(-ret));
        exit(2);
    }
    ret = register_buffer(&ring, nvme_fd, buf->dmabuf_fd);
    if (ret < 0) {
        fprintf(stderr, "register %s dma-buf: %s\n", buf->name,
                strerror(-ret));
        exit(2);
    }

    fill_pattern(expected, UINT64_C(0x1111222233334444));
    store_buffer(buf, expected);
    ret = fixed_io(&ring, nvme_fd, offset, 0);
    if (ret == IO_SIZE) {
        require_full(pread(nvme_fd, actual, IO_SIZE, offset),
                     "pread ground truth");
        result.write_ok = memcmp(expected, actual, IO_SIZE) == 0;
        printf("%-7s WRITE_FIXED: %s\n", buf->name,
               result.write_ok ? "PASS" : "FAIL");
    } else {
        printf("%-7s WRITE_FIXED: ERROR (%s)\n", buf->name,
               ret < 0 ? strerror(-ret) : "short I/O");
    }

    fill_pattern(expected, UINT64_C(0xaaaabbbbccccdddd));
    for (size_t i = 0; i < IO_SIZE; i++)
        poison[i] = expected[i] ^ POISON_XOR;
    require_full(pwrite(nvme_fd, expected, IO_SIZE, offset),
                 "pwrite ground truth");
    store_buffer(buf, poison);
    ret = fixed_io(&ring, nvme_fd, offset, 1);
    if (ret == IO_SIZE) {
        size_t expected_bytes = 0, poison_bytes = 0, other_bytes = 0;

        load_buffer(buf, actual);
        for (size_t i = 0; i < IO_SIZE; i++) {
            if (actual[i] == expected[i])
                expected_bytes++;
            else if (actual[i] == poison[i])
                poison_bytes++;
            else
                other_bytes++;
        }
        result.read_ok = expected_bytes == IO_SIZE;
        printf("%-7s READ_FIXED : %s (expected=%zu poison=%zu other=%zu)\n",
               buf->name, result.read_ok ? "PASS" : "FAIL", expected_bytes,
               poison_bytes, other_bytes);
    } else {
        printf("%-7s READ_FIXED : ERROR (%s)\n", buf->name,
               ret < 0 ? strerror(-ret) : "short I/O");
    }

    io_uring_queue_exit(&ring);
    return result;
}

int main(int argc, char **argv)
{
    struct test_buffer udmabuf, amdgpu;
    struct test_result udma_result, gpu_result;
    unsigned char *expected, *actual, *poison;
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
    expected = alloc_aligned((size_t)page);
    actual = alloc_aligned((size_t)page);
    poison = alloc_aligned((size_t)page);

    printf("WARNING: overwriting %u bytes at %s offset %llu\n", IO_SIZE,
           argv[1], (unsigned long long)offset);
    udmabuf = create_udmabuf();
    udma_result = test_exporter(&udmabuf, nvme_fd, offset, expected, actual,
                                poison);
    close_buffer(&udmabuf);

    amdgpu = create_amdgpu();
    gpu_result = test_exporter(&amdgpu, nvme_fd, offset, expected, actual,
                               poison);
    close_buffer(&amdgpu);

    if (udma_result.write_ok == 1 && udma_result.read_ok == 1 &&
        gpu_result.write_ok == 1 && gpu_result.read_ok == 0)
        printf("REPRODUCED: AMD-VRAM READ_FIXED fails; udmabuf passes\n");
    else
        printf("Result differs from the observed AMD failure signature\n");

    free(poison);
    free(actual);
    free(expected);
    close(nvme_fd);
    if (udma_result.write_ok < 0 || udma_result.read_ok < 0 ||
        gpu_result.write_ok < 0 || gpu_result.read_ok < 0)
        return 2;
    return udma_result.write_ok == 1 && udma_result.read_ok == 1 &&
                   gpu_result.write_ok == 1 && gpu_result.read_ok == 0
               ? 1
               : 0;
}
