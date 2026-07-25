// SPDX-License-Identifier: Apache-2.0
//
// Standalone reproducer: io_uring DMA-BUF corruption on AMD VRAM.
//
// No LMCache, no Python, no Rust. Pure HIP + io_uring.
//
// IMPORTANT -- direction isolation. A round trip (WRITE_FIXED then READ_FIXED)
// cannot tell you which leg is broken: if the round trip returns wrong bytes, that
// is equally consistent with (a) WRITE_FIXED corrupting VRAM->NVMe, (b) READ_FIXED
// corrupting NVMe->VRAM, or (c) both. This tool runs three independent checks per
// chunk, selectable with --mode (default: all three):
//
//   roundtrip   : GPU fill -> WRITE_FIXED -> READ_FIXED -> compare.
//                 (Legacy mode. Cannot isolate the direction; kept for continuity
//                 with earlier runs and to measure --concurrency effects, which
//                 only this mode exercises.)
//   write       : GPU fill -> WRITE_FIXED -> plain O_DIRECT pread() (no io_uring,
//                 no dma-buf) reads back what actually landed on disk. Isolates
//                 whether WRITE_FIXED (NVMe peer-DMA READING VRAM) corrupts the
//                 outbound transfer, independent of READ_FIXED.
//   read        : plain O_DIRECT pwrite() (no io_uring, no dma-buf) puts known-good
//                 bytes directly on disk, bypassing WRITE_FIXED -> READ_FIXED
//                 fetches them into VRAM -> hipMemcpy to host -> compare. Isolates
//                 whether READ_FIXED (NVMe peer-DMA WRITING VRAM) corrupts the
//                 inbound transfer, independent of WRITE_FIXED.
//
// pread()/pwrite() on an O_DIRECT fd are a decades-old, well-established kernel
// path, not the new dma-buf-token mechanism under test, so they serve as ground
// truth for "what is actually on disk" / "what we actually asked to be written".
//
// The `write` and `read` isolation checks always run one operation at a time
// (concurrency does not apply to them -- concurrency was already ruled out
// separately; see the debug log). Only `roundtrip` honors --concurrency.
//
// STRICT ACCOUNTING. Every check on `write`/`read` is classified into exactly
// one bucket, never silently folded into another:
//   ok                  transport completed fully AND every byte matched
//   mismatch            transport completed fully but content was wrong
//   io_short            io_uring op completed with fewer bytes than requested
//                        (content not compared -- a short transfer proves nothing
//                        about the untransferred region)
//   io_error            io_uring op returned a negative, non-EAGAIN error
//                        (content not compared)
//   io_eagain_exhausted gave up after EAGAIN_CAP reissues (content not compared)
//   harness_error       the plain pread()/pwrite() ground-truth syscall itself
//                        failed or was short -- a test-harness problem, not
//                        evidence about the dma-buf path
// This matters because a transport-level failure and a genuine data mismatch
// are different findings; conflating them would make "N corrupt" ambiguous
// between "N times the bytes were wrong" and "N times something errored".
//
// --mem-range-flags F : passed as the final argument to
// hipMemGetHandleForAddressRange for BOTH the source and dest exports (0 =
// default/unspecified mapping; 1 = hipMemRangeHandleTypeDmaBufFd's PCIe mapping
// type, hipMemRangeFlagDmaBufMappingTypePcie, intended for peer-device access).
// All decisive runs recorded in the debug log before this option existed used
// flags=0 -- the PCIe mapping mode was never exercised. Test both.
//
// Output is a per-mode breakdown (all six buckets) plus an interpretation line
// that maps directly to the three failure hypotheses, based on ok/mismatch
// only (io_short/io_error/io_eagain_exhausted/harness_error chunks are excluded
// from that classification and reported separately so they cannot masquerade
// as either a clean result or a data mismatch):
//   write mismatch>0, read mismatch==0  -> WRITE_FIXED is broken (NVMe reading VRAM)
//   write mismatch==0, read mismatch>0  -> READ_FIXED is broken (NVMe writing VRAM)
//   both>0                              -> both directions are broken
// Exit status is nonzero if any active mode recorded a mismatch or a non-EAGAIN
// transport failure, so the tool is script/CI-usable.
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
//   # narrow to one direction (faster, and the direct answer to "which leg?"):
//   ./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --mode write --device-offset $((4<<30))
//   ./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --mode read  --device-offset $((4<<30))
//   ./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --concurrency 1   # roundtrip still corrupts
//   ./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --finegrained     # fine-grained VRAM
//   # test the untested variable: export with the PCIe P2P mapping flag
//   ./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --mode read --mem-range-flags 1 \
//       --device-offset $((4<<30))
//
// Requires: a kernel with CONFIG_DMABUF_TOKEN and an NVMe device whose driver
// implements the dma-buf token op (nvme-pci), and a ROCm version whose
// libamdhip64 exposes hipMemGetHandleForAddressRange. The minimum version
// varies by AMD documentation revision (reported anywhere from ROCm 5.6 to HIP
// 7.0 depending on source); verify against your installed ROCm's release notes
// rather than trusting either figure -- if the symbol is missing, _HipDriver's
// constructor (in the Python tools) or the direct hipMemGetHandleForAddressRange
// call here will fail loudly with a clear error either way.

#define _GNU_SOURCE  // expose O_DIRECT from <fcntl.h>
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

// Bounded -EAGAIN retries. dma-buf ops can legitimately return -EAGAIN a few
// times (mapping invalidation); an unbounded loop would spin forever if it never
// resolves. Cap it so the tool always terminates.
#define EAGAIN_CAP 100000

#define SRC_IDX 0
#define DST_IDX 1

// Strict per-check classification (see the file header). Transport-level
// outcomes (IO_SHORT/IO_ERROR/IO_EAGAIN_EXHAUSTED) and the harness's own
// ground-truth syscall failing (IO_HARNESS_ERROR) are never folded into
// IO_MISMATCH: a mismatch means "the transport completed fully and correctly
// but the bytes were wrong," which is the specific claim under test.
enum io_result {
    IO_OK = 0,
    IO_SHORT,
    IO_ERROR,
    IO_EAGAIN_EXHAUSTED,
    IO_HARNESS_ERROR,
    IO_MISMATCH,
};

struct mode_stats {
    long ok;
    long mismatch;
    long io_short;
    long io_error;
    long io_eagain_exhausted;
    long harness_error;
};

static void stats_record(struct mode_stats *s, enum io_result r) {
    switch (r) {
        case IO_OK: s->ok++; break;
        case IO_MISMATCH: s->mismatch++; break;
        case IO_SHORT: s->io_short++; break;
        case IO_ERROR: s->io_error++; break;
        case IO_EAGAIN_EXHAUSTED: s->io_eagain_exhausted++; break;
        case IO_HARNESS_ERROR: s->harness_error++; break;
    }
}

static long stats_total(const struct mode_stats *s) {
    return s->ok + s->mismatch + s->io_short + s->io_error +
           s->io_eagain_exhausted + s->harness_error;
}

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

// Export [ptr, ptr+size) as a dma-buf fd. `mem_range_flags` is passed through
// unchanged to hipMemGetHandleForAddressRange -- 0 for the default/unspecified
// mapping, 1 for hipMemRangeFlagDmaBufMappingTypePcie (the PCIe P2P mapping
// mode). All decisive runs recorded before --mem-range-flags existed used 0.
static int export_dmabuf(void *ptr, size_t size, unsigned long long mem_range_flags) {
    int fd = -1;
    hipError_t e = hipMemGetHandleForAddressRange(
        &fd, (hipDeviceptr_t)ptr, size, hipMemRangeHandleTypeDmaBufFd,
        mem_range_flags);
    if (e != hipSuccess) {
        fprintf(stderr,
                "hipMemGetHandleForAddressRange failed: %d (%s) [flags=%llu]\n",
                e, hipGetErrorString(e), mem_range_flags);
        exit(1);
    }
    if (fd < 0) {
        fprintf(stderr, "export returned invalid fd [flags=%llu]\n",
                mem_range_flags);
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
    int do_roundtrip;
    int do_write_only;
    int do_read_only;
    unsigned long long mem_range_flags;
};

// Submit N WRITE_FIXED (src chunk c -> device slot c), <= concurrency in flight,
// reissuing on -EAGAIN. Transport-level short/error outcomes are logged to
// stderr with the chunk index (io_uring completions can arrive out of
// submission order under concurrency, so there is no single "which chunks
// failed" list to report structurally here; short/error is rare enough in
// practice that a stderr line is sufficient for roundtrip mode, which is
// explicitly the non-decisive, non-strictly-accounted mode -- see the file
// header). Returns the number of chunks that never completed (0 in the
// overwhelmingly common case); nonzero means EAGAIN_CAP was exhausted for the
// whole batch and the caller should not trust any of this repeat's data.
static int run_writes(struct io_uring *ring, int nvme_fd, const struct opts *o) {
    int done = 0, next = 0, inflight = 0;
    long eagain = 0;
    const size_t chunk = o->chunk_bytes;
    while (done < o->num_chunks) {
        if (eagain > (long)o->num_chunks * EAGAIN_CAP) {
            fprintf(stderr, "run_writes: gave up after %ld EAGAINs "
                            "(done=%d/%d)\n",
                    eagain, done, o->num_chunks);
            return o->num_chunks - done;
        }
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
            eagain++;
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
    return 0;  // all num_chunks completed (short/error already logged above)
}

// Read one slot back into the dest dma-buf, reissuing on -EAGAIN. Returns the
// transport outcome (see enum io_result); does not compare content -- that is
// the caller's job once it has confirmed IO_OK.
static enum io_result read_slot(struct io_uring *ring, int nvme_fd,
                                uint64_t device_offset, size_t chunk) {
    for (int tries = 0; tries < EAGAIN_CAP; tries++) {
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
            return IO_ERROR;
        }
        if ((size_t)res != chunk) {
            fprintf(stderr, "short READ_FIXED @%llu: %d/%zu\n",
                    (unsigned long long)device_offset, res, chunk);
            return IO_SHORT;
        }
        return IO_OK;
    }
    fprintf(stderr, "READ_FIXED @%llu: gave up after %d EAGAINs\n",
            (unsigned long long)device_offset, EAGAIN_CAP);
    return IO_EAGAIN_EXHAUSTED;
}

// Submit a single WRITE_FIXED (src slot, offset dmabuf_off) and wait for it,
// reissuing on -EAGAIN. Unlike run_writes, exactly one op is ever in flight --
// used by the write-only isolation check, where concurrency is deliberately not
// exercised (concurrency was already ruled out as a factor; see the debug log).
// Returns the transport outcome; does not compare content.
static enum io_result write_fixed_one(struct io_uring *ring, int nvme_fd,
                                      uint64_t dmabuf_off, size_t chunk,
                                      uint64_t device_offset) {
    for (int tries = 0; tries < EAGAIN_CAP; tries++) {
        struct io_uring_sqe *sqe;
        while (!(sqe = io_uring_get_sqe(ring))) io_uring_submit(ring);
        io_uring_prep_write_fixed(sqe, nvme_fd, (void *)(uintptr_t)dmabuf_off, chunk,
                                  device_offset, SRC_IDX);
        io_uring_submit(ring);
        struct io_uring_cqe *cqe;
        int r = io_uring_wait_cqe(ring, &cqe);
        if (r < 0) {
            fprintf(stderr, "wait_cqe(write-only): %s\n", strerror(-r));
            exit(1);
        }
        int res = cqe->res;
        io_uring_cqe_seen(ring, cqe);
        if (res == -EAGAIN) continue;
        if (res < 0) {
            fprintf(stderr, "WRITE_FIXED @%llu: %s\n",
                    (unsigned long long)device_offset, strerror(-res));
            return IO_ERROR;
        }
        if ((size_t)res != chunk) {
            fprintf(stderr, "short WRITE_FIXED @%llu: %d/%zu\n",
                    (unsigned long long)device_offset, res, chunk);
            return IO_SHORT;
        }
        return IO_OK;
    }
    fprintf(stderr, "WRITE_FIXED @%llu: gave up after %d EAGAINs\n",
            (unsigned long long)device_offset, EAGAIN_CAP);
    return IO_EAGAIN_EXHAUSTED;
}

// Isolates WRITE_FIXED. GPU-fill chunk `c`, WRITE_FIXED it to NVMe, then verify
// with a PLAIN O_DIRECT pread() -- no io_uring, no dma-buf -- reading back what
// actually landed on disk. A mismatch means WRITE_FIXED (NVMe peer-DMA reading
// exported VRAM) corrupted the outbound transfer; READ_FIXED is not involved.
// Returns the strict classification (see enum io_result / the file header): a
// transport failure (IO_SHORT/IO_ERROR/IO_EAGAIN_EXHAUSTED) or ground-truth
// pread() failure (IO_HARNESS_ERROR) is never reported as IO_MISMATCH.
static enum io_result check_write_only(struct io_uring *ring, int nvme_fd,
                                       void *src, int c, size_t chunk,
                                       uint64_t device_offset,
                                       unsigned char *pread_buf, int val) {
    HIP_CHECK(hipMemset((char *)src + (size_t)c * chunk, val, chunk));
    HIP_CHECK(hipDeviceSynchronize());

    enum io_result wr = write_fixed_one(ring, nvme_fd, (uint64_t)c * chunk, chunk,
                                        device_offset + (uint64_t)c * chunk);
    if (wr != IO_OK) return wr;

    ssize_t n = pread(nvme_fd, pread_buf, chunk,
                      device_offset + (uint64_t)c * chunk);
    if (n != (ssize_t)chunk) {
        fprintf(stderr, "pread chunk %d: got %zd want %zu (%s)\n", c, n, chunk,
                n < 0 ? strerror(errno) : "short read");
        return IO_HARNESS_ERROR;
    }
    for (size_t i = 0; i < chunk; i++)
        if (pread_buf[i] != (unsigned char)val) return IO_MISMATCH;
    return IO_OK;
}

// Isolates READ_FIXED. A PLAIN O_DIRECT pwrite() -- no io_uring, no dma-buf --
// puts a known-good pattern directly on disk (WRITE_FIXED is not involved), then
// READ_FIXED fetches it into the dest dma-buf, which is copied to host and
// compared. A mismatch means READ_FIXED (NVMe peer-DMA writing exported VRAM)
// corrupted the inbound transfer. Returns the strict classification (see enum
// io_result / the file header): a transport failure or ground-truth pwrite()
// failure is never reported as IO_MISMATCH.
static enum io_result check_read_only(struct io_uring *ring, int nvme_fd,
                                      void *dst, size_t chunk,
                                      uint64_t device_offset,
                                      unsigned char *pwrite_buf,
                                      unsigned char *host, int val) {
    memset(pwrite_buf, (unsigned char)val, chunk);
    ssize_t n = pwrite(nvme_fd, pwrite_buf, chunk, device_offset);
    if (n != (ssize_t)chunk) {
        fprintf(stderr, "pwrite @%llu: got %zd want %zu (%s)\n",
                (unsigned long long)device_offset, n, chunk,
                n < 0 ? strerror(errno) : "short write");
        return IO_HARNESS_ERROR;
    }
    enum io_result rd = read_slot(ring, nvme_fd, device_offset, chunk);
    if (rd != IO_OK) return rd;
    HIP_CHECK(hipMemcpy(host, dst, chunk, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < chunk; i++)
        if (host[i] != (unsigned char)val) return IO_MISMATCH;
    return IO_OK;
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
        .do_roundtrip = 1,
        .do_write_only = 1,
        .do_read_only = 1,
        .mem_range_flags = 0,
    };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc)
            o.device = argv[++i];
        else if (!strcmp(argv[i], "--mem-range-flags") && i + 1 < argc)
            o.mem_range_flags = strtoull(argv[++i], NULL, 0);
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
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *m = argv[++i];
            o.do_roundtrip = o.do_write_only = o.do_read_only = 0;
            if (!strcmp(m, "all"))
                o.do_roundtrip = o.do_write_only = o.do_read_only = 1;
            else if (!strcmp(m, "roundtrip"))
                o.do_roundtrip = 1;
            else if (!strcmp(m, "write"))
                o.do_write_only = 1;
            else if (!strcmp(m, "read"))
                o.do_read_only = 1;
            else {
                fprintf(stderr, "--mode must be one of: all, roundtrip, write, read\n");
                return 2;
            }
        } else {
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
    int src_fd = export_dmabuf(src, pool, o.mem_range_flags);
    int dst_fd = export_dmabuf(dst, o.chunk_bytes, o.mem_range_flags);

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
           "mem=%s dev_off=%llu mem_range_flags=%llu mode=%s%s%s\n\n",
           o.device, o.num_chunks, o.chunk_bytes / 1024, o.concurrency, o.repeat,
           o.finegrained ? "fine-grained" : "coarse",
           (unsigned long long)o.device_offset, o.mem_range_flags,
           o.do_roundtrip ? "roundtrip," : "",
           o.do_write_only ? "write," : "",
           o.do_read_only ? "read," : "");

    // Page-aligned buffers for the plain O_DIRECT pread()/pwrite() ground-truth
    // checks (O_DIRECT requires aligned host buffers). `host` (below) does not
    // need alignment -- it is only ever a hipMemcpy destination.
    unsigned char *pread_buf = NULL, *pwrite_buf = NULL;
    if (posix_memalign((void **)&pread_buf, page, o.chunk_bytes) != 0) {
        perror("posix_memalign(pread_buf)");
        return 1;
    }
    if (posix_memalign((void **)&pwrite_buf, page, o.chunk_bytes) != 0) {
        perror("posix_memalign(pwrite_buf)");
        return 1;
    }

    unsigned char *host = malloc(o.chunk_bytes);
    if (!host) { perror("malloc"); return 1; }

    struct mode_stats roundtrip_stats = {0}, write_stats = {0}, read_stats = {0};
    long detail_printed = 0;
    const long DETAIL_CAP = 16;

    for (int rep = 0; rep < o.repeat; rep++) {
        if (o.do_roundtrip) {
            // Distinct byte per (repeat, chunk) so stale slots are caught.
            for (int c = 0; c < o.num_chunks; c++) {
                int val = ((rep * 131 + c * 7) % 254) + 1;
                HIP_CHECK(hipMemset((char *)src + (size_t)c * o.chunk_bytes, val,
                                    o.chunk_bytes));
            }
            HIP_CHECK(hipDeviceSynchronize());

            int incomplete = run_writes(&ring, nvme_fd, &o);
            if (incomplete > 0) {
                // The write phase never confirmed every chunk landed; nothing in
                // this repeat's device state is trustworthy. Do not read/compare
                // it -- that would either wrongly count a never-written chunk as
                // "mismatch", or wrongly count it as "ok" if stale data happens
                // to match.
                fprintf(stderr,
                        "[roundtrip] rep %d: %d/%d writes never completed -- "
                        "skipping read/compare for this repeat\n",
                        rep, incomplete, o.num_chunks);
                for (int i = 0; i < o.num_chunks; i++)
                    stats_record(&roundtrip_stats, IO_EAGAIN_EXHAUSTED);
            } else {
                for (int c = 0; c < o.num_chunks; c++) {
                    enum io_result rd = read_slot(
                        &ring, nvme_fd, o.device_offset + (uint64_t)c * o.chunk_bytes,
                        o.chunk_bytes);
                    if (rd != IO_OK) {
                        stats_record(&roundtrip_stats, rd);
                        if (detail_printed++ < DETAIL_CAP)
                            printf("  [roundtrip] rep %d chunk %d: transport "
                                   "failure (not a data check)\n",
                                   rep, c);
                        continue;
                    }
                    HIP_CHECK(hipMemcpy(host, dst, o.chunk_bytes, hipMemcpyDeviceToHost));
                    int want = ((rep * 131 + c * 7) % 254) + 1;
                    // Sample start/middle/end (corruption spans blocks, not 1 byte;
                    // roundtrip is the legacy, non-strict mode -- see file header).
                    unsigned char a = host[0];
                    unsigned char b = host[o.chunk_bytes / 2];
                    unsigned char e = host[o.chunk_bytes - 1];
                    if (a != want || b != want || e != want) {
                        stats_record(&roundtrip_stats, IO_MISMATCH);
                        if (detail_printed++ < DETAIL_CAP)
                            printf("  [roundtrip] rep %d chunk %d: want 0x%02x got "
                                   "0x%02x/0x%02x/0x%02x\n",
                                   rep, c, want, a, b, e);
                    } else {
                        stats_record(&roundtrip_stats, IO_OK);
                    }
                }
            }
        }

        if (o.do_write_only) {
            for (int c = 0; c < o.num_chunks; c++) {
                int val = ((rep * 131 + c * 7) % 254) + 1;
                enum io_result r = check_write_only(&ring, nvme_fd, src, c,
                                                    o.chunk_bytes, o.device_offset,
                                                    pread_buf, val);
                stats_record(&write_stats, r);
                if (r != IO_OK && detail_printed++ < DETAIL_CAP)
                    printf("  [write-only] rep %d chunk %d: %s (want 0x%02x)\n",
                           rep, c,
                           r == IO_MISMATCH ? "pread saw wrong bytes on disk"
                           : r == IO_SHORT ? "WRITE_FIXED was short"
                           : r == IO_ERROR ? "WRITE_FIXED errored"
                           : r == IO_EAGAIN_EXHAUSTED ? "WRITE_FIXED EAGAIN-exhausted"
                                                       : "pread ground-truth failed",
                           val);
            }
        }

        if (o.do_read_only) {
            for (int c = 0; c < o.num_chunks; c++) {
                int val = ((rep * 131 + c * 7 + 3) % 254) + 1;  // distinct stream
                enum io_result r = check_read_only(
                    &ring, nvme_fd, dst, o.chunk_bytes,
                    o.device_offset + (uint64_t)c * o.chunk_bytes, pwrite_buf, host,
                    val);
                stats_record(&read_stats, r);
                if (r != IO_OK && detail_printed++ < DETAIL_CAP)
                    printf("  [read-only] rep %d chunk %d: %s (want 0x%02x)\n",
                           rep, c,
                           r == IO_MISMATCH ? "VRAM saw wrong bytes after READ_FIXED"
                           : r == IO_SHORT ? "READ_FIXED was short"
                           : r == IO_ERROR ? "READ_FIXED errored"
                           : r == IO_EAGAIN_EXHAUSTED ? "READ_FIXED EAGAIN-exhausted"
                                                       : "pwrite ground-truth failed",
                           val);
            }
        }

        // Per-repeat progress (mismatch counts only) so a long, silent run does
        // not look hung. Full breakdown (including transport failures) is in the
        // final summary.
        printf("rep %d/%d done: roundtrip=%ld write-only=%ld read-only=%ld "
               "(mismatch counts; see final summary for transport failures)\n",
               rep + 1, o.repeat, roundtrip_stats.mismatch, write_stats.mismatch,
               read_stats.mismatch);
        fflush(stdout);
    }

    long n = (long)o.num_chunks * o.repeat;
    printf("\n=== results (%d chunks x %d repeats = %ld checks per mode) ===\n",
           o.num_chunks, o.repeat, n);
    if (o.do_roundtrip)
        printf("  roundtrip : ok=%ld mismatch=%ld short=%ld error=%ld "
               "eagain_exhausted=%ld harness_error=%ld (total=%ld; does not "
               "isolate direction)\n",
               roundtrip_stats.ok, roundtrip_stats.mismatch, roundtrip_stats.io_short,
               roundtrip_stats.io_error, roundtrip_stats.io_eagain_exhausted,
               roundtrip_stats.harness_error, stats_total(&roundtrip_stats));
    if (o.do_write_only)
        printf("  write-only: ok=%ld mismatch=%ld short=%ld error=%ld "
               "eagain_exhausted=%ld harness_error=%ld (total=%ld; isolates "
               "WRITE_FIXED)\n",
               write_stats.ok, write_stats.mismatch, write_stats.io_short,
               write_stats.io_error, write_stats.io_eagain_exhausted,
               write_stats.harness_error, stats_total(&write_stats));
    if (o.do_read_only)
        printf("  read-only : ok=%ld mismatch=%ld short=%ld error=%ld "
               "eagain_exhausted=%ld harness_error=%ld (total=%ld; isolates "
               "READ_FIXED)\n",
               read_stats.ok, read_stats.mismatch, read_stats.io_short,
               read_stats.io_error, read_stats.io_eagain_exhausted,
               read_stats.harness_error, stats_total(&read_stats));

    int exit_status = 0;
    if (o.do_write_only && o.do_read_only) {
        printf("\n=== interpretation (mismatch counts only) ===\n");
        if (write_stats.mismatch > 0 && read_stats.mismatch == 0)
            printf("  WRITE_FIXED is broken (NVMe peer-DMA READING VRAM). "
                   "READ_FIXED is clean.\n");
        else if (write_stats.mismatch == 0 && read_stats.mismatch > 0)
            printf("  READ_FIXED is broken (NVMe peer-DMA WRITING VRAM). "
                   "WRITE_FIXED is clean.\n");
        else if (write_stats.mismatch > 0 && read_stats.mismatch > 0)
            printf("  BOTH directions are broken.\n");
        else
            printf("  Neither isolation check reproduced a mismatch this run -- "
                   "raise --repeat/--num-chunks, or rely on --mode roundtrip.\n");
    }
    if (roundtrip_stats.mismatch || roundtrip_stats.io_short ||
        roundtrip_stats.io_error || roundtrip_stats.io_eagain_exhausted ||
        roundtrip_stats.harness_error)
        exit_status = 1;
    if (write_stats.mismatch || write_stats.io_short || write_stats.io_error ||
        write_stats.io_eagain_exhausted || write_stats.harness_error)
        exit_status = 1;
    if (read_stats.mismatch || read_stats.io_short || read_stats.io_error ||
        read_stats.io_eagain_exhausted || read_stats.harness_error)
        exit_status = 1;

    free(host);
    free(pread_buf);
    free(pwrite_buf);
    io_uring_queue_exit(&ring);
    close(nvme_fd);
    close(src_fd);
    close(dst_fd);
    hipFree(src_raw);
    hipFree(dst_raw);
    return exit_status;
}
