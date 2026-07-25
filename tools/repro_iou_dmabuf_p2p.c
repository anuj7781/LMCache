// SPDX-License-Identifier: Apache-2.0
//
// Standalone reproducer: io_uring DMA-BUF WRITE_FIXED corruption on AMD VRAM.
//
// No LMCache, no Python, no Rust. Pure HIP + io_uring. It reproduces the
// data corruption observed when an NVMe device does peer-to-peer DMA that READS
// exported AMD GPU VRAM (io_uring WRITE_FIXED), while the reverse direction
// (READ_FIXED, NVMe writing VRAM) is clean.
//
// What it does, per repeat:
//   1. hipMalloc (or hipExtMallocWithFlags fine-grained) a GPU source pool of N
//      chunks and a 1-chunk GPU dest buffer.
//   2. Export both as dma-buf fds via hipMemGetHandleForAddressRange.
//   3. Open the NVMe device O_DIRECT, set up io_uring, register a sparse buffer
//      table, and install the two dma-bufs with IORING_REGISTER_BUFFERS_UPDATE +
//      IO_REGBUF_TYPE_DMABUF, bound to the NVMe fd.
//   4. Fill each source chunk with a distinct byte (hipMemset) + hipDeviceSynchronize.
//   5. Issue N WRITE_FIXED (src chunk c -> device slot c), up to --concurrency in
//      flight, reissuing on -EAGAIN.
//   6. Read every slot back (READ_FIXED into the dest dma-buf), copy dest->host,
//      and check the bytes match what was written.
//   7. Report corrupt-chunk count.
//
// Expected on the affected stack: nonzero corruption, even at --concurrency 1.
//
// BUILD (host-only; no HIP dev headers or hipcc needed -- links libamdhip64):
//   cc -O2 -o repro_iou_dmabuf_p2p tools/repro_iou_dmabuf_p2p.c -luring \
//      -L/opt/rocm/lib -lamdhip64
//   # If -lamdhip64 is not found (runtime-only ROCm, no unversioned .so symlink),
//   # link the versioned soname directly:
//   cc -O2 -o repro_iou_dmabuf_p2p tools/repro_iou_dmabuf_p2p.c -luring \
//      "$(ls /opt/rocm*/lib/libamdhip64.so* 2>/dev/null | head -1)"
//   # liburing dev headers: apt/dnf install liburing-dev (or point -I/-L at it).
//   # ROCM_PATH may differ (e.g. /opt/rocm-6.x); adjust the lib path accordingly.
//
// RUN (WRITES ARE DESTRUCTIVE to the device at --device-offset; use scratch):
//   ./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --num-chunks 64 \
//       --concurrency 8 --repeat 40 --device-offset $((4<<30))
//   ./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --concurrency 1   # still corrupts
//   ./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --finegrained     # fine-grained VRAM
//
// Requires: a kernel with CONFIG_DMABUF_TOKEN and an NVMe device whose driver
// implements the dma-buf token op (nvme-pci), and ROCm >= 5.6.

#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

// ---- Minimal HIP runtime API (host-only) ------------------------------------
// Declared here so the repro builds against a runtime-only ROCm install
// (libamdhip64.so) with a plain C compiler -- no HIP dev headers, no hipcc.
// Values below are the stable public HIP ABI.
typedef int hipError_t;
typedef void *hipDeviceptr_t;
enum { hipSuccess = 0 };
enum { hipMemcpyDeviceToHost = 2 };
enum { hipMemRangeHandleTypeDmaBufFd = 1 };
#define hipDeviceMallocFinegrained 0x1

extern hipError_t hipSetDevice(int deviceId);
extern hipError_t hipMalloc(void **ptr, size_t size);
extern hipError_t hipExtMallocWithFlags(void **ptr, size_t size, unsigned int flags);
extern hipError_t hipFree(void *ptr);
extern hipError_t hipMemset(void *dst, int value, size_t sizeBytes);
extern hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, int kind);
extern hipError_t hipDeviceSynchronize(void);
extern hipError_t hipMemGetHandleForAddressRange(void *handle, hipDeviceptr_t dptr,
                                                 size_t size, int handleType,
                                                 unsigned long long flags);
extern const char *hipGetErrorString(hipError_t err);

// ---- io_uring dma-buf token ABI (patched kernel; define defensively) --------
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
    uint32_t flags;  // REPRO_RSRC_UPDATE_EXTENDED for dma-buf
    uint64_t data;   // pointer to a repro_regbuf_desc
    uint64_t tags;
    uint32_t nr;
    uint32_t resv2;
};

struct repro_regbuf_desc {
    uint32_t type;   // REPRO_REGBUF_TYPE_DMABUF
    uint32_t flags;
    uint64_t size;   // must be 0 for dma-buf
    uint64_t uaddr;  // must be 0 for dma-buf
    int32_t dmabuf_fd;
    int32_t target_fd;  // the O_DIRECT block fd
    uint64_t __resv[6];
};

#define SRC_IDX 0
#define DST_IDX 1

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t _e = (call);                                               \
        if (_e != hipSuccess) {                                               \
            fprintf(stderr, "HIP error %d (%s) at %s:%d: %s\n", _e,           \
                    hipGetErrorString(_e), __FILE__, __LINE__, #call);        \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static int io_uring_register_raw(int ring_fd, unsigned opcode, void *arg,
                                 unsigned nr) {
    long r = syscall(__NR_io_uring_register, ring_fd, opcode, arg, nr);
    return (int)r;
}

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// Export [ptr, ptr+size) as a dma-buf fd.
static int export_dmabuf(void *ptr, size_t size) {
    int fd = -1;
    hipError_t e = hipMemGetHandleForAddressRange(
        &fd, (hipDeviceptr_t)ptr, size, hipMemRangeHandleTypeDmaBufFd, 0);
    if (e != hipSuccess) {
        fprintf(stderr, "hipMemGetHandleForAddressRange failed: %d (%s)\n", e,
                hipGetErrorString(e));
        exit(1);
    }
    if (fd < 0) {
        fprintf(stderr, "export returned invalid fd\n");
        exit(1);
    }
    return fd;
}

// Allocate GPU memory over-allocated by `page` and return a page-aligned base.
static void *gpu_alloc_aligned(size_t size, int finegrained, long page,
                               void **raw_out) {
    void *raw = NULL;
    size_t total = size + (size_t)page;
    if (finegrained) {
        HIP_CHECK(hipExtMallocWithFlags(&raw, total, hipDeviceMallocFinegrained));
    } else {
        HIP_CHECK(hipMalloc(&raw, total));
    }
    *raw_out = raw;
    uint64_t base = align_up((uint64_t)raw, (uint64_t)page);
    return (void *)base;
}

struct opts {
    const char *device;
    int num_chunks;
    size_t chunk_bytes;
    int concurrency;
    int repeat;
    uint64_t device_offset;
    int finegrained;
    int device_index;
};

// Submit N WRITE_FIXED (src chunk c -> device slot c), <= concurrency in flight,
// reissuing on -EAGAIN. Returns 0 on success.
static void run_writes(struct io_uring *ring, int nvme_fd, const struct opts *o) {
    int done = 0, next = 0, inflight = 0;
    const size_t chunk = o->chunk_bytes;
    while (done < o->num_chunks) {
        while (inflight < o->concurrency && next < o->num_chunks) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
            if (!sqe) break;  // SQ full; drain first
            io_uring_prep_write_fixed(
                sqe, nvme_fd, (void *)(uintptr_t)((uint64_t)next * chunk), chunk,
                o->device_offset + (uint64_t)next * chunk, SRC_IDX);
            io_uring_sqe_set_data64(sqe, (uint64_t)next);
            next++;
            inflight++;
        }
        io_uring_submit(ring);

        struct io_uring_cqe *cqe;
        int r = io_uring_wait_cqe(ring, &cqe);
        if (r < 0) {
            fprintf(stderr, "wait_cqe(write): %s\n", strerror(-r));
            exit(1);
        }
        uint64_t idx = io_uring_cqe_get_data64(cqe);
        int res = cqe->res;
        io_uring_cqe_seen(ring, cqe);

        if (res == -EAGAIN) {
            // dma-buf mapping was invalidated; reissue the same request unchanged.
            struct io_uring_sqe *sqe;
            while (!(sqe = io_uring_get_sqe(ring))) io_uring_submit(ring);
            io_uring_prep_write_fixed(
                sqe, nvme_fd, (void *)(uintptr_t)(idx * chunk), chunk,
                o->device_offset + idx * chunk, SRC_IDX);
            io_uring_sqe_set_data64(sqe, idx);
            io_uring_submit(ring);
            // still in flight; do not advance `done`
        } else {
            if (res < 0)
                fprintf(stderr, "WRITE_FIXED chunk %llu: %s\n",
                        (unsigned long long)idx, strerror(-res));
            else if ((size_t)res != chunk)
                fprintf(stderr, "short WRITE_FIXED chunk %llu: %d/%zu\n",
                        (unsigned long long)idx, res, chunk);
            inflight--;
            done++;
        }
    }
}

// Read one slot back into the dest dma-buf, reissuing on -EAGAIN.
static void read_slot(struct io_uring *ring, int nvme_fd, uint64_t device_offset,
                      size_t chunk) {
    for (;;) {
        struct io_uring_sqe *sqe;
        while (!(sqe = io_uring_get_sqe(ring))) io_uring_submit(ring);
        io_uring_prep_read_fixed(sqe, nvme_fd, (void *)0, chunk, device_offset,
                                 DST_IDX);
        io_uring_submit(ring);
        struct io_uring_cqe *cqe;
        int r = io_uring_wait_cqe(ring, &cqe);
        if (r < 0) {
            fprintf(stderr, "wait_cqe(read): %s\n", strerror(-r));
            exit(1);
        }
        int res = cqe->res;
        io_uring_cqe_seen(ring, cqe);
        if (res == -EAGAIN) continue;
        if (res < 0) {
            fprintf(stderr, "READ_FIXED @%llu: %s\n",
                    (unsigned long long)device_offset, strerror(-res));
        }
        return;
    }
}

int main(int argc, char **argv) {
    struct opts o = {
        .device = NULL,
        .num_chunks = 64,
        .chunk_bytes = 2 * 1024 * 1024,
        .concurrency = 8,
        .repeat = 40,
        .device_offset = 0,
        .finegrained = 0,
        .device_index = 0,
    };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc)
            o.device = argv[++i];
        else if (!strcmp(argv[i], "--num-chunks") && i + 1 < argc)
            o.num_chunks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk-bytes") && i + 1 < argc)
            o.chunk_bytes = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--concurrency") && i + 1 < argc)
            o.concurrency = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc)
            o.repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device-offset") && i + 1 < argc)
            o.device_offset = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--finegrained"))
            o.finegrained = 1;
        else if (!strcmp(argv[i], "--device-index") && i + 1 < argc)
            o.device_index = atoi(argv[++i]);
        else {
            fprintf(stderr, "unknown/incomplete arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!o.device) {
        fprintf(stderr, "--device /dev/nvmeXnY is required (WRITES ARE DESTRUCTIVE)\n");
        return 2;
    }
    long page = sysconf(_SC_PAGESIZE);
    if (o.chunk_bytes % page || o.device_offset % page) {
        fprintf(stderr, "--chunk-bytes and --device-offset must be multiples of %ld\n",
                page);
        return 2;
    }
    if (o.concurrency < 1) o.concurrency = 1;

    HIP_CHECK(hipSetDevice(o.device_index));

    // Allocate + export source pool (N chunks) and a 1-chunk dest buffer.
    void *src_raw = NULL, *dst_raw = NULL;
    size_t pool = (size_t)o.num_chunks * o.chunk_bytes;
    void *src = gpu_alloc_aligned(pool, o.finegrained, page, &src_raw);
    void *dst = gpu_alloc_aligned(o.chunk_bytes, o.finegrained, page, &dst_raw);
    int src_fd = export_dmabuf(src, pool);
    int dst_fd = export_dmabuf(dst, o.chunk_bytes);

    int nvme_fd = open(o.device, O_RDWR | O_DIRECT);
    if (nvme_fd < 0) {
        fprintf(stderr, "open(%s, O_DIRECT): %s\n", o.device, strerror(errno));
        return 1;
    }

    struct io_uring ring;
    unsigned depth = 256;
    int r = io_uring_queue_init(depth, &ring, 0);
    if (r < 0) {
        fprintf(stderr, "io_uring_queue_init: %s\n", strerror(-r));
        return 1;
    }

    // Sparse buffer table with 2 slots, then install the two dma-bufs bound to
    // the NVMe fd.
    struct repro_rsrc_register reg = {0};
    reg.nr = 2;
    reg.flags = IORING_RSRC_REGISTER_SPARSE;
    if (io_uring_register_raw(ring.ring_fd, IORING_REGISTER_BUFFERS2, &reg,
                              sizeof(reg)) < 0) {
        fprintf(stderr, "register sparse table: %s\n", strerror(errno));
        return 1;
    }
    int fds[2] = {src_fd, dst_fd};
    for (int slot = 0; slot < 2; slot++) {
        struct repro_regbuf_desc desc = {0};
        desc.type = REPRO_REGBUF_TYPE_DMABUF;
        desc.dmabuf_fd = fds[slot];
        desc.target_fd = nvme_fd;
        struct repro_rsrc_update2 up = {0};
        up.offset = (uint32_t)slot;
        up.flags = REPRO_RSRC_UPDATE_EXTENDED;
        up.data = (uint64_t)(uintptr_t)&desc;
        up.nr = 1;
        int ret = io_uring_register_raw(ring.ring_fd,
                                        IORING_REGISTER_BUFFERS_UPDATE, &up,
                                        sizeof(up));
        if (ret != 1) {
            fprintf(stderr,
                    "register dma-buf slot %d failed: ret=%d errno=%s "
                    "(CONFIG_DMABUF_TOKEN? nvme-pci? O_DIRECT?)\n",
                    slot, ret, strerror(errno));
            return 1;
        }
    }

    printf("device=%s chunks=%d chunk=%zuKiB concurrency=%d repeat=%d "
           "mem=%s dev_off=%llu\n\n",
           o.device, o.num_chunks, o.chunk_bytes / 1024, o.concurrency, o.repeat,
           o.finegrained ? "fine-grained" : "coarse",
           (unsigned long long)o.device_offset);

    unsigned char *host = malloc(o.chunk_bytes);
    if (!host) { perror("malloc"); return 1; }

    long total_corrupt = 0;
    for (int rep = 0; rep < o.repeat; rep++) {
        // Distinct byte per (repeat, chunk) so stale slots are caught.
        for (int c = 0; c < o.num_chunks; c++) {
            int val = ((rep * 131 + c * 7) % 254) + 1;
            HIP_CHECK(hipMemset((char *)src + (size_t)c * o.chunk_bytes, val,
                                o.chunk_bytes));
        }
        HIP_CHECK(hipDeviceSynchronize());

        run_writes(&ring, nvme_fd, &o);

        for (int c = 0; c < o.num_chunks; c++) {
            read_slot(&ring, nvme_fd, o.device_offset + (uint64_t)c * o.chunk_bytes,
                      o.chunk_bytes);
            HIP_CHECK(hipMemcpy(host, dst, o.chunk_bytes, hipMemcpyDeviceToHost));
            int want = ((rep * 131 + c * 7) % 254) + 1;
            // Sample start / middle / end (corruption spans blocks, not 1 byte).
            unsigned char a = host[0];
            unsigned char b = host[o.chunk_bytes / 2];
            unsigned char e = host[o.chunk_bytes - 1];
            if (a != want || b != want || e != want) {
                total_corrupt++;
                if (total_corrupt <= 16)
                    printf("  rep %d chunk %d: want 0x%02x got 0x%02x/0x%02x/0x%02x\n",
                           rep, c, want, a, b, e);
            }
        }
    }

    printf("\ncorrupt %ld / %d chunks (%d chunks x %d repeats)\n", total_corrupt,
           o.num_chunks * o.repeat, o.num_chunks, o.repeat);
    if (total_corrupt > 0)
        printf("=> REPRODUCED: io_uring dma-buf WRITE_FIXED (NVMe peer-DMA reading "
               "VRAM) corrupts.\n");
    else
        printf("=> clean this run (raise --repeat / --concurrency / --num-chunks).\n");

    free(host);
    io_uring_queue_exit(&ring);
    close(nvme_fd);
    close(src_fd);
    close(dst_fd);
    hipFree(src_raw);
    hipFree(dst_raw);
    return 0;
}
