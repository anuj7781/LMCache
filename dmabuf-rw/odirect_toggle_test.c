// SPDX-License-Identifier: Apache-2.0
//
// Tests whether io_uring's per-request dma-buf path still operates after
// O_DIRECT is cleared, via fcntl(F_SETFL), on the fd that was registered.
//
// Context: a kernel patch review raised "O_DIRECT is checked only during
// registration; we should recheck (file->f_flags & O_DIRECT) per request."
// The reply was "Nobody should be able to clear O_DIRECT, unless I missed
// something?" -- but O_DIRECT is one of the flags fcntl(2) documents as
// togglable at runtime via F_SETFL (unlike access-mode/creation flags,
// which are fixed at open() time), and f_flags lives in struct file, not
// per-fd, so this affects every holder of that same open file description,
// including whatever struct file * the registration holds internally.
//
// Sequence:
//   1. Register a udmabuf with io_uring against `nvme_fd` (opened O_DIRECT).
//   2. WRITE_FIXED while O_DIRECT is set -- baseline, must work.
//   3. fcntl(nvme_fd, F_SETFL, ~O_DIRECT) -- clear it on the registered fd.
//   4. READ_FIXED against the SAME registration, no re-registration --
//      does it error out, or complete? If it completes, is the data
//      actually correct, or does it silently misbehave (the exact failure
//      shape this investigation has been chasing all along: a completion
//      that reports success while the data movement itself is wrong)?
//
// Ground truth for both phases is read/written through a SEPARATE fd
// (`ground_fd`, opened without O_DIRECT and never toggled) to the same
// path, so the check stays valid regardless of what happens to nvme_fd's
// flags mid-test.
//
// WARNING: overwrites 2 MiB at the given device offset.
//
// Build:
//   cc -O2 -o odirect_toggle_test odirect_toggle_test.c -luring
// Run:
//   ./odirect_toggle_test /dev/nvme0n1 <byte-offset>

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
#define PATTERN_A 0xAB
#define PATTERN_B 0x37

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

static int create_udmabuf(int *memfd_out)
{
    struct udmabuf_create create = {0};
    int memfd, dmabuf_fd;
    int devfd = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);

    if (devfd < 0)
        die("open /dev/udmabuf");
    memfd = memfd_create("odirect-toggle-dmabuf",
                         MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (memfd < 0)
        die("memfd_create");
    if (fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK) < 0)
        die("F_ADD_SEALS");
    if (ftruncate(memfd, IO_SIZE) < 0)
        die("ftruncate");

    create.memfd = (uint32_t)memfd;
    create.flags = UDMABUF_FLAGS_CLOEXEC;
    create.size = IO_SIZE;
    dmabuf_fd = ioctl(devfd, UDMABUF_CREATE, &create);
    close(devfd);
    if (dmabuf_fd < 0)
        die("UDMABUF_CREATE");
    *memfd_out = memfd;
    return dmabuf_fd;
}

static int register_buffer(struct io_uring *ring, int target_fd, int dmabuf_fd)
{
    struct dmabuf_regbuf_desc desc = {
        .type = DMABUF_REGBUF_TYPE,
        .dmabuf_fd = dmabuf_fd,
        .target_fd = target_fd,
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

static void print_odirect_state(int fd, const char *label)
{
    int flags = fcntl(fd, F_GETFL);

    if (flags < 0)
        die("fcntl F_GETFL");
    printf("%s: O_DIRECT %s\n", label, (flags & O_DIRECT) ? "SET" : "CLEAR");
}

int main(int argc, char **argv)
{
    struct io_uring ring;
    unsigned char *pattern_a, *pattern_b, *zeros, *actual;
    char *end;
    uint64_t offset;
    long page = sysconf(_SC_PAGESIZE);
    int nvme_fd, ground_fd, dmabuf_fd, memfd, ret, flags;

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
        die("open NVMe (O_DIRECT)");
    ground_fd = open(argv[1], O_RDWR);
    if (ground_fd < 0)
        die("open NVMe (ground truth, no O_DIRECT)");

    pattern_a = alloc_aligned((size_t)page, IO_SIZE);
    pattern_b = alloc_aligned((size_t)page, IO_SIZE);
    zeros = alloc_aligned((size_t)page, IO_SIZE);
    actual = alloc_aligned((size_t)page, IO_SIZE);
    memset(pattern_a, PATTERN_A, IO_SIZE);
    memset(pattern_b, PATTERN_B, IO_SIZE);
    memset(zeros, 0, IO_SIZE);

    printf("WARNING: overwriting %u bytes at %s offset %llu\n", IO_SIZE,
           argv[1], (unsigned long long)offset);

    dmabuf_fd = create_udmabuf(&memfd);

    ret = io_uring_queue_init(8, &ring, 0);
    if (ret < 0) {
        fprintf(stderr, "io_uring_queue_init: %s\n", strerror(-ret));
        return 2;
    }
    print_odirect_state(nvme_fd, "before registration");
    ret = register_buffer(&ring, nvme_fd, dmabuf_fd);
    if (ret < 0) {
        fprintf(stderr, "register dma-buf: %s\n", strerror(-ret));
        return 2;
    }

    // Phase A: WRITE_FIXED while O_DIRECT is set. Baseline -- must work.
    require_full(pwrite(memfd, pattern_a, IO_SIZE, 0), "pwrite memfd (fill)");
    ret = fixed_io(&ring, nvme_fd, offset, 0 /* write */);
    if (ret != IO_SIZE) {
        printf("phase A WRITE_FIXED: ERROR (%s)\n",
               ret < 0 ? strerror(-ret) : "short I/O");
        return 2;
    }
    require_full(pread(ground_fd, actual, IO_SIZE, offset),
                 "pread ground truth (phase A)");
    printf("phase A WRITE_FIXED (O_DIRECT set): %s\n",
           memcmp(actual, pattern_a, IO_SIZE) == 0 ? "PASS" : "FAIL");

    // Clear O_DIRECT on the registered fd -- the crux of the test.
    flags = fcntl(nvme_fd, F_GETFL);
    if (flags < 0)
        die("fcntl F_GETFL");
    if (fcntl(nvme_fd, F_SETFL, flags & ~O_DIRECT) < 0)
        die("fcntl F_SETFL clear O_DIRECT");
    print_odirect_state(nvme_fd, "after fcntl(F_SETFL, ~O_DIRECT)");

    // Phase B: READ_FIXED against the SAME registration, no re-registration,
    // now that O_DIRECT is cleared on the registered target_fd.
    require_full(pwrite(ground_fd, pattern_b, IO_SIZE, offset),
                 "pwrite ground truth (phase B seed)");
    if (fsync(ground_fd) < 0)
        die("fsync ground_fd");
    require_full(pwrite(memfd, zeros, IO_SIZE, 0), "pwrite memfd (clear)");

    ret = fixed_io(&ring, nvme_fd, offset, 1 /* read */);
    if (ret != IO_SIZE) {
        printf("phase B READ_FIXED (O_DIRECT cleared): ERROR (%s) "
               "-- the per-request path rejected it\n",
               ret < 0 ? strerror(-ret) : "short I/O");
    } else {
        require_full(pread(memfd, actual, IO_SIZE, 0),
                     "pread memfd (phase B verify)");
        printf("phase B READ_FIXED (O_DIRECT cleared): COMPLETED, data %s\n",
               memcmp(actual, pattern_b, IO_SIZE) == 0
                   ? "CORRECT -- went through despite the cleared flag"
                   : "WRONG -- completed but did not actually transfer "
                     "correct data");
    }

    io_uring_queue_exit(&ring);
    close(dmabuf_fd);
    close(memfd);
    close(nvme_fd);
    close(ground_fd);
    free(actual);
    free(zeros);
    free(pattern_b);
    free(pattern_a);
    return 0;
}
