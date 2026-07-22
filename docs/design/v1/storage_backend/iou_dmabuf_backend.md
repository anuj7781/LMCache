# io_uring DMA-BUF Storage Backend Design

## 1. Overview

This document describes the design for a new LMCache storage backend that moves
KV-cache data between an NVMe SSD and GPU VRAM using io_uring's DMA-BUF interface,
bypassing the CPU entirely when PCIe peer-to-peer (P2P) DMA is available.

### 1.1 Motivation

The existing storage path is:

```
GPU VRAM  --cudaMemcpy-->  CPU DRAM (staging)  --cuFile/pread-->  NVMe
NVMe      --pread/cuFile--> CPU DRAM (staging) --cudaMemcpy-->  GPU VRAM
```

Even the GDS backend (`GdsBackend`) intermediates through a CUDA/ROCm middleware
layer. With the io_uring DMA-BUF kernel interface (added in this kernel tree's top
11 commits), the storage-tier transfer becomes:

```
registered GPU slab  <--DMA (P2P)-->  NVMe
```

**Scope of the "zero-copy" claim.** This removes the *CPU bounce buffer* on the
storage transfer — it is zero **CPU** copy, not zero copy end-to-end. On the read
path the KV data lands in a backend-owned GPU slab; the GPU connector
(`VLLMPagedMemGPUConnector`) then performs an intra-GPU (D2D) copy from that slab
into vLLM's paged KV cache. True end-to-end zero-copy would require vLLM's paged KV
buffers to be the dmabuf targets themselves — a much larger change that is out of
scope here (see §15.7). The win is: eliminate the host DRAM staging buffer and the
host DMA + `cudaMemcpy` round trip, replacing them with one P2P DMA plus one D2D
copy.

The improvement matters at the scale of LMCache's sequential disk retrieval
bottleneck: at 6912 tokens (54 chunks of ~16 MB each), the non-layerwise path reads
~864 MB sequentially. Eliminating the CPU bounce buffer and using parallel io_uring
SQE submission can cut this latency substantially. The benefit is concentrated on
the **read** path (TTFT); see §10 for why the write path is only advantageous for
GPU-resident sources.

### 1.2 Relationship to existing backends

| Feature               | `LocalDiskBackend`    | `GdsBackend`          | **`IouDmabufBackend`** (new)      |
|-----------------------|-----------------------|-----------------------|-----------------------------------|
| GPU vendor            | any                   | NVIDIA / AMD          | NVIDIA / AMD                      |
| Kernel requirement    | none                  | cuFile/hipFile driver | `CONFIG_DMABUF_TOKEN=y`           |
| Storage device        | any FS                | any FS                | raw NVMe namespace (`O_DIRECT`)   |
| Zero-copy P2P         | no                    | driver-dependent      | yes (when `CONFIG_PCI_P2PDMA=y`)  |
| Middleware            | none                  | cuFile / hipFile      | none (pure kernel path)           |
| Storage layer         | per-chunk files       | per-chunk files       | `RawBlockCore` slot storage       |
| Parallelism           | thread pool           | thread pool           | io_uring batched SQEs             |

---

## 2. Prerequisites

### 2.1 Kernel

- `CONFIG_DMABUF_TOKEN=y` — enables the dmabuf token / registered buffer path
- `CONFIG_PCI_P2PDMA=y` — enables true zero-copy P2P DMA (optional for correctness,
  required for the zero-copy performance benefit)
- NVMe block device (nvme-pci driver); `CONFIG_BLK_DEV_NVME=y`

Without `CONFIG_DMABUF_TOKEN` the backend must refuse to initialize and log a clear
error. Without P2P the kernel silently falls back to a system-RAM-mediated path —
correctness is preserved, but the zero-copy benefit is lost. See §13 for P2P
observability requirements.

### 2.2 Userspace

- The existing Rust `_rawdev()` extension inside `RawBlockCore` extended to support
  `IO_REGBUF_TYPE_DMABUF` registration (see §7).
- GPU driver ≥ 515 for NVIDIA (`cuMemGetHandleForAddressRange`) or ROCm 5.x for AMD.
- Raw NVMe namespace block device (e.g. `/dev/nvme0n1`) accessible with `O_DIRECT`
  read/write permissions.

**Rust extension build requirement.** The `lmcache_rust_raw_block_io` module is a
separate PyO3/maturin crate under `rust/raw_block/` (`pyproject.toml` build-backend
= `maturin`, `module-name = "lmcache_rust_raw_block_io"`). It is **not** part of the
default `pip install lmcache` path. The new dmabuf methods (§7) are added to this
crate, so first-cut acceptance requires: `rust/raw_block` is rebuilt and installed
(`maturin develop` / `maturin build && pip install`) with the dmabuf methods present
before `IouDmabufBackend.is_available()` can return True. If the crate is missing or
predates the dmabuf methods, `is_available()` returns False and the backend is
skipped (see §11.1, §13.4).

---

## 3. Architecture

Three layers cooperate:

```
┌──────────────────────────────────────────────────────────────────────┐
│  LMCache Python layer                                                │
│  IouDmabufBackend(AllocatorBackendInterface)                        │
│    RawBlockCore  — slot allocation, index, checkpoint, recovery      │
│    DmabufGPUAllocator — GPU VRAM slab ownership and dmabuf export   │
└────────────────────────┬───────────────────────────────────────────┘
                         │ Python ↔ Rust boundary (_rawdev extension)
┌────────────────────────▼───────────────────────────────────────────┐
│  Rust rawdev extension (extended for DMA-BUF)                       │
│    register_dmabuf_buffers(dmabuf_fds)  [uses self.fd as target_fd] │
│    read_fixed_dmabuf(slab_idx, dmabuf_offset, len, device_offset)   │
│    write_fixed_dmabuf(slab_idx, dmabuf_offset, len, device_offset)  │
│    probe_dmabuf_support() -> bool                                   │
│    Single io_uring ring (MVP); shard only after benchmarking        │
└────────────────────────┬───────────────────────────────────────────┘
                         │ io_uring READ_FIXED / WRITE_FIXED
┌────────────────────────▼───────────────────────────────────────────┐
│  Kernel: io_uring DMA-BUF path                                      │
│    dmabuf token (persistent DMA mapping per registered buffer slot) │
│    PCIe P2P DMA: NVMe ↔ GPU VRAM                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. GPU Memory Export (VRAM → dmabuf_fd)

The kernel constraint is `dmabuf size ≤ 1 GiB`. The total GPU pool is carved into
≤1 GiB slabs, each exported as its own `dmabuf_fd`.

**Preferred layout: one contiguous pool tensor, exported as 1 GiB sub-ranges.**
Back the whole pool with a single `torch.empty(pool_bytes, dtype=uint8, device=cuda)`
(the same mechanism `GPUMemoryAllocator` already uses — see §8/M3), then export each
1 GiB-aligned sub-range as a separate `dmabuf_fd` via
`cuMemGetHandleForAddressRange(base + slab_idx*SZ_1G, SZ_1G, …)`. This makes the
slabs contiguous in GPU VA and the integer-division offset math in §4.3 exact. (The
open validation question is whether the export API accepts a sub-range of a
`cudaMalloc`-backed pointer — see §8.1. If it does not, fall back to independent VMM
slabs, which are not contiguous but still satisfy the per-slab addressing below.)

### 4.1 NVIDIA path

There are two ways to obtain the exportable device memory, matching §8.1's Approach
A / B. Both feed the same final call, `cuMemGetHandleForAddressRange(...,
CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, ...)`:

- **Approach A (preferred, §8.1):** allocate the pool as a single PyTorch tensor
  (`GPUMemoryAllocator`, `torch.empty`) and export 1 GiB-aligned **sub-ranges** of
  its `data_ptr()` directly:
  ```
  cuMemGetHandleForAddressRange(&fd, pool_ptr + slab_idx*SZ_1G, SZ_1G,
                                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0)
  ```
  No `cuMemCreate`/`cuMemMap` needed; may require
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` so the pool is VMM-backed.
  This is the P1 go/no-go probe.

- **Approach B (fallback, below):** if Approach A's export fails on the target
  driver, allocate each slab explicitly with the VMM API, which is guaranteed
  exportable, then wrap per-chunk views with `torch.from_blob` (§8.1 Approach B).

The full VMM sequence for Approach B:

```
Step 1 — allocate physical memory as a handle:
    cuMemCreate(&phys_handle, slab_size, &prop, 0)
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_ordinal

Step 2 — reserve a virtual address range:
    cuMemAddressReserve(&va_ptr, slab_size, alignment=0, 0, 0)

Step 3 — map the physical handle into the VA range:
    cuMemMap(va_ptr, slab_size, 0, phys_handle, 0)
    cuMemSetAccess(va_ptr, slab_size, &access_desc, 1)

Step 4 — export the slab as a dmabuf fd:
    cuMemGetHandleForAddressRange(
        &dmabuf_fd,
        va_ptr,
        slab_size,                              // ≤ 1 GiB
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
        flags=0
    )
```

The returned `dmabuf_fd` is an integer file descriptor. `va_ptr` is the CUDA device
pointer for GPU kernel access.

**Why Approach B exists:** plain `cudaMalloc` does not guarantee the physical backing
that `cuMemGetHandleForAddressRange` needs, and PyTorch's caching allocator uses
`cudaMalloc` by default. So Approach A's export **may** fail unless the pool is
VMM-backed (`expandable_segments:True`). VMM allocation (Approach B) is explicitly
exportable and IOMMU-predictable, so it is the guaranteed fallback. Which one is
needed is a P1 measurement (§8.1) — do not assume Approach A fails before testing.

**Minimum driver:** 515.43.04 (Linux), CUDA 11.7+.

### 4.2 AMD path

**Primary (ROCm 5.6+):** `hipMemGetHandleForAddressRange` mirrors the NVIDIA VMM
path and is the first path to probe:

```
hipMalloc(&hip_ptr, slab_size)
hipMemGetHandleForAddressRange(
    &dmabuf_fd,
    (hipDeviceptr_t)hip_ptr,
    slab_size,
    hipMemRangeHandleTypeDmaBufFd,
    flags=0
)
```

**Fallback (older ROCm via DRM GEM):** If `hipMemGetHandleForAddressRange` is
absent or returns an error, fall back to direct DRM GEM allocation:

```
drm_fd = open("/dev/dri/renderD128", O_RDWR)
amdgpu_device_initialize(drm_fd, &major, &minor, &adev)

// Allocate VRAM GEM BO
struct drm_amdgpu_gem_create args = {
    .in.bo_size   = slab_size,
    .in.alignment = 2 * 1024 * 1024,
    .in.domains   = AMDGPU_GEM_DOMAIN_VRAM,
}
drmIoctl(drm_fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &args)

// Export as dmabuf fd
drmPrimeHandleToFD(drm_fd, args.out.handle, DRM_CLOEXEC | DRM_RDWR, &dmabuf_fd)

// Import into HIP address space
hipExternalMemoryHandleDesc ext_desc = {
    .type = hipExternalMemoryHandleTypeOpaqueFd,
    .handle.fd = dup(dmabuf_fd),   // HIP takes ownership; keep original alive
    .size = slab_size,
}
hipImportExternalMemory(&ext_mem, &ext_desc)
hipExternalMemoryGetMappedBuffer(&hip_ptr, ext_mem, &buf_desc)
```

The backend init code probes the HIP path first and falls back to DRM GEM only if
it is unavailable. The DRM GEM path should not be treated as a first-class design
concern until the HIP path is proven insufficient on a concrete target system.

### 4.3 Slab descriptor

Each slab tracks:

```python
@dataclass
class SlabDescriptor:
    device_ptr: int     # GPU VA base (CUdeviceptr / hipDeviceptr)
    dmabuf_fd: int      # exported fd registered with io_uring
    buf_slot: int       # io_uring registered buffer table index
    size: int           # ≤ SZ_1G
```

`buf_slot` equals the slab's position in the slab list. For a chunk at pool-relative
offset `abs_offset`:

```
slab_idx   = abs_offset // SZ_1G
buf_offset = abs_offset %  SZ_1G    # "address" arg in READ_FIXED / WRITE_FIXED
```

**Invariant: all non-tail slabs must be exactly `SZ_1G`.** Only the final slab may
be smaller (when `pool_bytes` is not a multiple of 1 GiB). The allocator enforces
this at construction time. This invariant makes the integer-division decomposition
above correct for all addresses. A variable-slab-size design requiring a cumulative
offset map is explicitly out of scope for this backend.

**Boundary check uses the O_DIRECT transfer length, not the payload length.** The
actual device/dmabuf transfer length is `total_len = round_up(chunk_bytes,
block_align)` (see §9/§10 and C1 below), because the NVMe device is `O_DIRECT` and
`RawBlockCore` rounds every transfer up to `block_align` (`core.py:773`, `:1370`).
A chunk must not cross a slab boundary *after* rounding. The allocator therefore
enforces, using the rounded length:

```
total_len = round_up(chunk_bytes, block_align)
assert abs_offset % SZ_1G + total_len ≤ SZ_1G
```

Allocating `chunk_bytes` but checking only `chunk_bytes` would let the aligned tail
spill past the 1 GiB boundary into the next slab's dmabuf, addressing the wrong
buffer. Reserve `total_len` in the GPU pool and in the NVMe slot payload region.

---

## 5. Storage Layer: RawBlockCore

### 5.1 Reuse instead of reinventing

LMCache already has `RawBlockCore`
(`lmcache/v1/storage_backend/raw_block/core.py`), which provides:

- Raw block device I/O (raw NVMe namespace opened `O_DIRECT`)
- Fixed slot allocation with configurable slot size and alignment
- Persistent checkpoint/recovery (magic header, version, slot validation)
- In-memory key → slot offset index with lock-refcount protection
- Existing io_uring path (`io_engine = "io_uring"`) with Rust `_rawdev()` extension
- `register_fixed_buffers_from_allocator()` for CPU-buffer fixed registration

`IouDmabufBackend` reuses `RawBlockCore`'s storage and index model with a small
public slot-lifecycle extension (§5.5). The only other extension is teaching the
Rust `_rawdev()` extension to register dmabuf fds using `IO_REGBUF_TYPE_DMABUF`
(§7), instead of the existing address-based CPU buffer registration.

### 5.2 Target device

Use a raw NVMe namespace block device as `device_path`, for example `/dev/nvme0n1`.
The kernel patches and the liburing `rw-dmabuf.c` test exercise this exact path.
Filesystem-backed files (e.g. an `O_DIRECT` flat file on XFS) may work because
`create_dmabuf_token` is at the blk-mq layer and is agnostic to the filesystem
above it — but this is an untested variant for MVP. Start with the raw block device,
which is what the kernel patches test, and validate the filesystem path separately.

### 5.3 Slot sizing

```
slot_bytes = round_up(max_chunk_bytes + header_bytes, block_align)
```

`RawBlockCore` already handles this calculation and alignment enforcement.
`max_chunk_bytes` for LLaMA 3.1 8B is ~16 MiB per KV chunk. `block_align = 4096`
satisfies both 512-byte and 4Kn NVMe sector alignment requirements.

### 5.4 Checkpoint and recovery

`RawBlockCore`'s existing checkpoint (`checkpoint_now()` / `apply_loaded_state()`)
and slot-header validation are reused as-is. No new index format is introduced.

### 5.5 RawBlockCore API extension required

#### 5.5.1 Atomic metadata + offset lookup

The read path needs both metadata (shape, dtype, fmt) and slot offset per key.
`get_metadata_many()` (core.py:505) and `entry_offset()` (core.py:593) each
acquire `self._lock` separately. A `delete_many()` or eviction between the two
calls can pair valid metadata with a `None` or changed offset. The read path must
therefore use a single combined method:

```python
def get_entries_many(
    self,
    encoded_keys: Sequence[str],
    *,
    lock_refcount: bool = False,
) -> list[tuple[DiskCacheMetadata, int] | None]:
    """Return (metadata, slot_base_offset) pairs under one lock.

    Args:
        encoded_keys: Ordered encoded raw-block keys.
        lock_refcount: If True, increment L2 lock refcounts for hits,
            matching the existing get_metadata_prefix(lock=True) behavior.

    Returns:
        A list aligned with encoded_keys. Each element is a (meta, offset)
        tuple for indexed keys, or None for missing keys.
    """
```

This is the sole new lookup method added to `RawBlockCore` (alongside the four
slot-lifecycle methods in §5.5.2).

#### 5.5.2 Slot lifecycle methods

`put_many()` (core.py:614) bundles slot reservation, payload write, and index commit
into one method. For the dmabuf backend the payload write is a `WRITE_FIXED` issued
by the Rust layer — `put_many()` cannot be used as-is. Four new public methods must
be added to `RawBlockCore` that expose the existing internal phases:

All four methods take `RawBlockKeySpec` (the same type `put_many()` already uses).
The backend converts `CacheEngineKey → RawBlockKeySpec` once before calling any of
them; all subsequent calls use either the `RawBlockKeySpec` object or its
`.encoded` string attribute (which is the dict key used by `_inflight` and `_index`
internally).

```python
def reserve_slot(
    self, key: RawBlockKeySpec, memory_obj: MemoryObj
) -> Optional[int]:
    """Allocate a free slot and record it (with its metadata) as in-flight.

    Mirrors put_many()'s reservation phase exactly (core.py:655-672): it
    calls _allocate_slot_locked(), builds a DiskCacheMetadata from
    memory_obj.metadata (shape, dtype, fmt, cached_positions) and the
    payload size, and stores _Inflight(offset=offset, meta=meta) under
    self._lock. Taking memory_obj (not a bare payload_len) is what lets
    commit_slot() publish the correct metadata without a second meta
    argument — it is already stored at reserve time, exactly as put_many
    does. The slot must have room for round_up(payload_len, block_align)
    (§4.3/C1).

    Returns the slot's base byte offset on the device, or None in the
    following cases (all treated as "skip" by callers for MVP):
    - No free slots available.
    - key.encoded already exists in _inflight: a concurrent write for this
      key is already in progress; a second reservation would corrupt the
      slot lifecycle. Callers should treat this as a transient miss.
    - key.encoded already exists in _index: the key is already committed.
      To overwrite, the caller must first delete_many([key.encoded]) then
      retry reserve_slot.

    Future: consider returning a SlotReserveResult enum so callers can
    distinguish NoSpace / AlreadyInflight / AlreadyIndexed / Allocated
    without logging ambiguity.

    The caller must eventually call commit_slot or abort_slot for every
    successful (non-None) reservation.
    """

def write_slot_header(
    self, key: RawBlockKeySpec, offset: int, payload_len: int
) -> bool:
    """Write the slot header at byte offset `offset` using normal CPU I/O.

    Encodes the magic, slot_identity, and payload_len fields (same format
    as _encode_header / _write_one) and writes them at `offset + 0` via
    the existing _write_buffers path. Does NOT use dmabuf.

    O_DIRECT alignment: this is NOT a bare 24-byte pwrite. On an O_DIRECT
    device the header must be written as a full block-aligned region. It
    writes exactly `self.header_bytes` bytes (a multiple of block_align —
    enforced at RawBlockCore construction, core.py:239) from a zero-padded,
    alignment-satisfying CPU buffer, matching _write_one's
    `hdr_total = round_up(len(header), block_align)` padding
    (core.py:1370-1379). Because header_bytes is block-aligned, the payload
    region at `offset + header_bytes` is also block-aligned — a hard
    requirement for the subsequent WRITE_FIXED / READ_FIXED device offset.
    """

def commit_slot(self, key: RawBlockKeySpec, offset: int) -> bool:
    """Move an in-flight entry to the committed index.

    Called after the dmabuf WRITE_FIXED has completed successfully. No meta
    argument: the DiskCacheMetadata was built and stored in _inflight at
    reserve_slot() time (mirroring put_many), so commit just publishes it.

    Invariants enforced under self._lock:
    - key.encoded must exist in _inflight; if not, log an error and return False.
    - _inflight[key.encoded].offset must equal the supplied offset; if not,
      log an error and return False without touching the index or free list.
      This prevents a retry/cancel bug from publishing or freeing the wrong slot.
    - On success, pops from _inflight and inserts _inflight.meta into _index
      using the stored inflight.offset (not the argument) as the canonical offset.
    """

def abort_slot(self, key: RawBlockKeySpec, offset: int) -> None:
    """Cancel an in-flight reservation and return the slot to the free list.

    Called when the dmabuf WRITE_FIXED fails or is cancelled.

    Invariants enforced under self._lock:
    - key.encoded must exist in _inflight; if not, log an error and return.
    - _inflight[key.encoded].offset must equal the supplied offset; if not,
      log an error and return without freeing anything.
    - On success, pops from _inflight and returns inflight.offset (not the
      argument) to the free list via _append_free_slot_locked. Using the
      stored offset matches put_many()'s existing behavior and prevents a
      mismatched argument from freeing an unrelated slot.
    """
```

`put_many()` is unchanged and remains the write path for all non-dmabuf callers.
These four methods expose the same lock/inflight/index logic that already exists
inside `put_many()` without duplicating it.

---

## 6. io_uring Ring Design

### 6.1 MVP: single native worker ring

Start with **one io_uring ring** owned by the Rust `_rawdev()` extension. The ring
is owned exclusively by a single Rust background worker thread — Python threads
never touch it directly. This avoids multiplying long-lived DMA-BUF token
registrations (one token per slab per ring) before any benchmarking justifies
the cost.

```
Single ring:
    depth: 256 SQEs
    flags: 0  (no SQPOLL)
    registered buffer table: n_slabs slots (one per GPU slab ≤ 1 GiB)
    bound target_fd: raw NVMe block device fd (self.fd inside RawBlockDevice)
```

**Concurrency model:** Multiple Python threads may call `read_fixed_dmabuf` /
`write_fixed_dmabuf` concurrently. Each call enqueues a request to the Rust
worker thread's internal Mutex-protected submission queue. The worker thread
drains the queue and drives the ring exclusively. No Python-level serialization
is required. This mirrors the existing `batched_read` / `batched_write` pattern
already in `RawBlockDevice`.

If benchmarks show the single ring is the throughput bottleneck, add a small fixed
number of rings and shard requests across them. Do not default to ring-per-thread
before that measurement.

### 6.2 Registration sequence

Called once at backend init by `rawdev.register_dmabuf_buffers(dmabuf_fds)`.
The Rust implementation uses `self.fd` (the internal NVMe device fd, already open
inside `RawBlockDevice`) as `target_fd` — it is not passed in from Python:

```c
// Inside RawBlockDevice::register_dmabuf_buffers (Rust, called at init)

// 1. Create sparse buffer table with n_slabs slots
io_uring_register_buffers_sparse(&ring, n_slabs);

// 2. Register each GPU slab's dmabuf_fd into a buffer slot
for (int i = 0; i < n_slabs; i++) {
    struct io_uring_regbuf_desc rd = {
        .type      = IO_REGBUF_TYPE_DMABUF,
        .dmabuf_fd = dmabuf_fds[i],
        .target_fd = self->fd,  // internal NVMe fd, same for all slabs
        .uaddr = 0, .size = 0,  // mandatory zero for DMABUF type
    };
    struct io_uring_rsrc_update2 up = {
        .data  = (unsigned long)&rd,
        .nr    = 1,
        .offset = i,
        .resv  = IORING_RSRC_UPDATE_EXTENDED,
    };
    io_uring_register(ring.ring_fd, IORING_REGISTER_BUFFERS_UPDATE,
                      &up, sizeof(up));
}
```

After registration, persistent dmabuf tokens exist for all slabs bound to the
device's internal fd. Tokens are reused for all subsequent I/O without re-registration.

**`device_offset` convention:** The `device_offset` parameter in
`read_fixed_dmabuf` / `write_fixed_dmabuf` is the absolute byte offset on the NVMe
device. Callers are responsible for adding `header_bytes` when targeting the payload
region of a slot (see §9 and §10).

### 6.3 Buffer-table exclusivity: dmabuf slabs vs. CPU fixed buffers

**Crack: the dmabuf backend must NOT register CPU fixed buffers on the same ring.**
`RawBlockCore.register_fixed_buffers_from_allocator()` (core.py:447) calls the Rust
`register_fixed_buffers()`, which uses `ring.submitter().register_buffers(&iovecs)`
— i.e. `IORING_REGISTER_BUFFERS`, a **full-table** registration that claims slots
`[0, n_bufs)` in one shot (lib.rs:1945). The dmabuf path instead uses
`io_uring_register_buffers_sparse` + `IORING_REGISTER_BUFFERS_UPDATE`
(`IO_REGBUF_TYPE_DMABUF`, §6.2). A ring has exactly **one** registered-buffer table;
these two registration styles are mutually exclusive — calling both makes the second
fail (`-EBUSY`) or clobber the first.

**Resolution (MVP):** For the `IouDmabufBackend`'s `RawBlockCore`, the fixed-buffer
table is owned **exclusively by the GPU dmabuf slabs**. The backend therefore does
**not** call `register_fixed_buffers_from_allocator()`. This is safe because all CPU
I/O on this ring runs in **non-fixed** mode:

- `read_uring` / `write_uring` consult `fixed_buffer_map` only when
  `fixed_buffers_registered` is true (lib.rs:2344-2351). With CPU fixed buffers not
  registered, they fall to the aligned-direct or bounce path using plain
  `IORING_OP_READ` / `IORING_OP_WRITE`, which ignore the registered-buffer table.
- Non-fixed CPU ops and `READ_FIXED`/`WRITE_FIXED` dmabuf ops coexist on one ring:
  only the `*_FIXED` ops index the table, and only dmabuf slabs live there
  (slots `[0, n_slabs)`, `buf_slot = slab_idx`).

Consequence: the CPU header write (§5.5.2 `write_slot_header`) and the CPU-source
`put_many` path (§10 step 2) lose the CPU-side zero-copy fixed-buffer optimization on
this backend. That is acceptable — the header is tiny and CPU-source puts are the
non-preferred path (§10/M1). If a future benchmark needs CPU fixed buffers too, the
Rust layer would have to migrate CPU registration onto the same sparse table with a
disjoint slot range `[n_slabs, …)`; out of scope for MVP.

### 6.4 Teardown ordering

`RawBlockDevice` (rawdev) is owned by `RawBlockCore` and closed inside
`RawBlockCore.close()` (core.py:957). The io_uring ring and all registered buffer
slots are torn down when `RawBlockDevice.close()` is called. The critical ordering
constraint is therefore:

```
IouDmabufBackend.close():
  1. Drain all in-flight I/O (wait for outstanding read/write futures)
  2. RawBlockCore.close()
       → internally: flush checkpoint, close RawBlockDevice (exits ring,
         unregisters all dmabuf buffer slots, closes NVMe fd)
  3. DmabufGPUAllocator.close()
       → closes dmabuf_fds, destroys GPU slabs (VMM unmap / hipFree)
```

Step 3 must not run before step 2 completes, because the io_uring ring holds
kernel references to the `dmabuf_fd`s until the buffer slots are unregistered.
Closing a `dmabuf_fd` while it is still registered is undefined behavior.

---

## 7. Native Layer Extension (Rust)

The existing Rust `_rawdev()` extension (`lmcache_rust_raw_block_io.RawBlockDevice`,
imported at `core.py:411`) already drives io_uring for CPU-buffer I/O via
`register_fixed_buffers(buffer_ptrs, buffer_sizes)` (lib.rs:1906). That method uses
`IORING_REGISTER_BUFFERS` (iovec-based CPU memory registration) and is a different
kernel path from the dmabuf `IORING_REGISTER_BUFFERS_UPDATE` + `IO_REGBUF_TYPE_DMABUF`
path needed here. Add the following new methods to `RawBlockDevice` in `lib.rs`,
rather than writing a parallel C extension:

```rust
// New methods on RawBlockDevice (lmcache_rust_raw_block_io)

// Probe whether the running kernel supports IO_REGBUF_TYPE_DMABUF.
// MUST be a module-level PyO3 function OR a #[staticmethod] — NOT an instance
// method. is_available() (§11.1) is a static gate that runs before any
// RawBlockDevice is opened, so it cannot depend on `self`. It opens its own
// throwaway ring internally.
#[staticmethod]
fn probe_dmabuf_support() -> PyResult<bool>

// Register GPU slab dmabuf fds into the ring's sparse buffer table.
// Uses self.fd (the already-open NVMe device fd) as target_fd.
// Calls IORING_REGISTER_BUFFERS_UPDATE + IO_REGBUF_TYPE_DMABUF for each fd.
fn register_dmabuf_buffers(&self, dmabuf_fds: Vec<i32>) -> PyResult<()>

// Enqueue a READ_FIXED: NVMe at device_offset → GPU slab[slab_idx] at dmabuf_offset.
// `length` MUST already be block_align-rounded (total_len) by the Python caller;
//  the kernel rejects a non-block-multiple length on an O_DIRECT fd with -EINVAL.
// device_offset is the absolute byte offset on the NVMe device (caller adds header_bytes).
// dmabuf_offset MUST also be block_align-aligned (the pool allocator guarantees this).
// Blocks until CQE arrives; retries internally on -EAGAIN up to max_retries.
// Returns Ok only on a full `length`-byte transfer; short/zero I/O is an error (§7.1).
fn read_fixed_dmabuf(&self, slab_idx: u32, dmabuf_offset: u64,
                     length: u32, device_offset: u64) -> PyResult<usize>

// Enqueue a WRITE_FIXED: GPU slab[slab_idx] at dmabuf_offset → NVMe at device_offset.
// Same alignment contract on `length` (block-rounded), `dmabuf_offset`, `device_offset`.
fn write_fixed_dmabuf(&self, slab_idx: u32, dmabuf_offset: u64,
                      length: u32, device_offset: u64) -> PyResult<usize>
```

**Length contract (C1).** `length` is the O_DIRECT transfer size and must equal
`total_len = round_up(chunk_bytes, block_align)`, computed by the Python caller
before the call. The native layer does not re-round. The extra
`total_len - chunk_bytes` bytes are padding on both the NVMe slot and the GPU slab;
the read overwrites the slab tail with padding (harmless — the logical tensor view
is `chunk_bytes`), and the write persists padding into the slot's payload region
(which is sized for `total_len`).

`register_dmabuf_buffers` must not be confused with the existing
`register_fixed_buffers` — they use different kernel registration paths and
different buffer table semantics. The existing method must not be modified.

If the Rust `io_uring` crate cannot express `io_uring_regbuf_desc` with
`IO_REGBUF_TYPE_DMABUF` cleanly via its safe API, a minimal `bindgen`-generated C
shim compiled as a build dependency of the Rust crate is the preferred solution.
A standalone C extension is explicitly out of scope.

### 7.1 -EAGAIN handling

Inside `read_fixed_dmabuf` / `write_fixed_dmabuf`, retry on `-EAGAIN` with a
bounded counter (`max_eagain_retries`, configurable, default 16):

```rust
for attempt in 0..max_retries {
    submit READ_FIXED / WRITE_FIXED sqe
    wait for CQE
    match cqe.result {
        n if n == length as i32 => return Ok(n as usize),  // full transfer
        n if n >= 0 => {
            // 0 ≤ n < length: short or zero I/O is an error for MVP.
            // A partial dmabuf transfer indicates a kernel or driver bug;
            // do not treat it as a partial success. Increment short_io_errors.
            return Err(libc::EIO)
        },
        -EAGAIN => continue,   // exporter invalidated mapping; kernel rebuilds
        e       => return Err(e),
    }
}
return Err(-EAGAIN)   // exhausted retries
```

Log a warning after 3 retries; treat exhaustion as a hard error.

### 7.2 Observability counters

The native layer exposes counters (via Python property or metrics callback):

- `eagain_retries` — total `-EAGAIN` retries since init
- `bytes_read` / `bytes_written`
- `registration_failures`
- `short_io_errors` — `cqe->res > 0` but `< requested_length`

---

## 8. GPU Memory Allocator: `DmabufGPUAllocator`

**Build on `GPUMemoryAllocator`, do not reinvent it (M3).** LMCache already has
`GPUMemoryAllocator(MemoryAllocatorInterface)` (memory_management.py:3018), which:

- allocates a single pool tensor `torch.empty(size, dtype=uint8, device=cuda)`
  (memory_management.py:3036) — this is exactly "Approach A" below;
- sub-allocates real `TensorMemoryObj` views over that tensor via
  `TensorMemoryAllocator` (backed by `AddressManager`, memory_management.py:1289);
- implements the `MemoryAllocatorInterface.allocate(shapes, dtypes, fmt)` contract
  (memory_management.py:3062) that §11's `initialize_allocator` must return.

`DmabufGPUAllocator` is therefore a thin subclass/wrapper of `GPUMemoryAllocator`
that adds only two things: (a) it constructs the pool with `align_bytes = block_align`
(4096) so every allocation is O_DIRECT-aligned; (b) it exports the pool as ≤1 GiB
dmabuf slabs and maps a `MemoryObj`'s `metadata.address` to `(slab_idx, buf_offset)`.
The tensor wrapping itself is *not* new code — it is the existing, proven
`TensorMemoryAllocator` path. This narrows the P1 gate (§8.1) to a single unknown:
whether the pool tensor's pointer is *exportable* via `cuMemGetHandleForAddressRange`.

```
DmabufGPUAllocator(GPUMemoryAllocator-based)
├── pool_tensor: torch.empty(pool_bytes, uint8, cuda)   (one contiguous allocation)
├── inner: TensorMemoryAllocator(pool_tensor, align_bytes=block_align)
│      manages [0, pool_bytes) offset space; returns real TensorMemoryObj views
└── slab_fds: List[SlabDescriptor]   (one dmabuf_fd per 1 GiB sub-range)
```

- `allocate(shapes, dtypes, fmt) → Optional[MemoryObj]`
  (MemoryAllocatorInterface signature): delegates to `inner.allocate(...)`. The
  returned `TensorMemoryObj` already has `parent_allocator = self` and
  `metadata.address = abs_offset`. Allocation size is rounded to `total_len`
  (§4.3/C1) so the O_DIRECT transfer never overflows the slot or crosses a slab.

- `free(memory_obj)`: delegates to `inner.free(...)`, returning the offset range to
  the pool. **The pool memory is recycled**, not returned to the driver — the
  `pool_tensor` and its dmabuf slabs live for the backend's lifetime. Consequence
  (m6): a `TensorMemoryObj` returned from a read is a *view into recyclable pool
  memory*. Any consumer (the GPU connector's D2D copy into paged KV) must complete
  before the object is `free`d, exactly as with `GPUMemoryAllocator` today. The
  standard `MemoryObj` refcount contract already enforces this; no extra handling is
  needed, but implementers must not `free` a slab region while a copy from it is
  still in flight.

- `owns(memory_obj) -> bool` (finding 1): `MemoryObjMetadata` has **no** `extra`
  field (memory_management.py:148), so do not stash an identity tag there. Use the
  existing hook instead:
  ```python
  def owns(self, obj: MemoryObj) -> bool:
      return (
          isinstance(obj, TensorMemoryObj)
          and obj.parent_allocator is self          # identity, not just range
          and 0 <= obj.metadata.address < self.pool_bytes
      )
  ```
  `TensorMemoryObj.parent_allocator` (memory_management.py:655) is set at allocation
  and uniquely identifies the owning allocator instance, so no serialized-metadata
  change is required.

### 8.1 GPU dmabuf **export** — the real P1 go/no-go gate

The wrapping of GPU memory into a valid `TensorMemoryObj` is **not** the open
problem — `GPUMemoryAllocator` / `TensorMemoryAllocator` already do exactly that in
production (§8/M3), returning real tensor views over a `torch.empty` pool with
`parent_allocator` set. `TensorMemoryObj.__init__` (memory_management.py:644)
requires a real `torch.Tensor`, and that requirement is already satisfied by reusing
that path.

The single unresolved unknown is **exportability**: can the pool tensor's device
pointer be turned into a dmabuf fd via `cuMemGetHandleForAddressRange`? PyTorch's
caching allocator uses `cudaMalloc` (or `cuMemMap` under
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments`), not necessarily a form the export
API accepts. **P2 (§16) must not begin until this export probe passes on the target
driver.** This is a much narrower gate than "tensor wrapping."

Two concrete approaches to test in P1, in order:

**Approach A — reuse the `GPUMemoryAllocator` pool tensor (preferred, test first):**
The pool is already `torch.empty(pool_bytes, dtype=uint8, device=cuda)`
(memory_management.py:3036). Probe whether `cuMemGetHandleForAddressRange` accepts a
1 GiB-aligned sub-range of `pool_tensor.data_ptr()`:
```python
export_ok = try_cuMemGetHandleForAddressRange(
    pool_tensor.data_ptr() + slab_idx * SZ_1G, SZ_1G
)
```
If it succeeds, the entire allocator (wrapping, sub-views, refcount, `owns()`) is the
existing code — only the export + `register_dmabuf_buffers` calls are new. Cleanest
path, and it may require `expandable_segments:True` so the pool is VMM-backed;
document the exact `PYTORCH_CUDA_ALLOC_CONF` needed if so.

**Approach B — VMM API allocation + `torch.from_blob` (fallback):**
Allocate slabs via `cuMemCreate` / `cuMemMap` (VMM API), which is known to be
exportable. Then wrap per-chunk VA ranges:
```python
tensor = torch.from_blob(
    ctypes.cast(va_ptr + buf_offset, ctypes.c_void_p),
    shape, dtype=dtype,
    device=torch.device(f"cuda:{ordinal}")
)
```
`torch.from_blob` on external CUDA device pointers is undocumented and not
guaranteed stable across PyTorch versions, but it has worked in practice.

**Decision rule:** If Approach A's `cuMemGetHandleForAddressRange` call succeeds,
use A. Otherwise, fall back to B and accept the `torch.from_blob` stability risk.
If both fail, the GPU connector interface requires a new transfer mode operating on
`(device_ptr: int, shape, dtype)` tuples — that is a larger change deferred to a
follow-on design.

---

## 9. Read Path: NVMe → GPU VRAM

```
batched_get_blocking(keys: list[CacheEngineKey])
│
├── 0. encoded_keys = [encode_legacy_key(k).encoded for k in keys]
│         (same key_namespace="legacy" encoding as the write path, §10 step 0)
│
├── 1. Resolve metadata and slot offsets, locking entries for I/O lifetime:
│     entries = RawBlockCore.get_entries_many(
│         encoded_keys, lock_refcount=True
│     )  → list[(DiskCacheMetadata, slot_base_offset) | None]
│         lock_refcount=True increments _lock_refcnt for each hit, preventing
│         delete_many() / eviction from reclaiming the slot while I/O is
│         in flight. Matches get_metadata_prefix(lock=True) behavior.
│     locked_keys = [k for k, e in zip(encoded_keys, entries) if e is not None]
│
│   [Steps 2–4 run inside try; step 5 runs in the finally block]
│
├── 2. DmabufGPUAllocator.allocate() × N
│         → TensorMemoryObj list (each reserving total_len; §4.3/C1)
│         → abs_offset → slab_idx, buf_offset per obj
│         → total_len = round_up(chunk_bytes, block_align)
│
├── 3. Fan out to thread pool (bounded by disk_io_threads):
│     Python threads enqueue to Rust worker; Rust worker drives the ring.
│     each worker calls:
│       rawdev.read_fixed_dmabuf(
│           slab_idx,
│           buf_offset,                          # offset into the GPU dmabuf (block-aligned)
│           total_len,                           # O_DIRECT-rounded length, NOT chunk_bytes (C1)
│           slot_base_offset + header_bytes      # skip the slot header on NVMe
│       )
│       -EAGAIN is retried inside Rust; other errors return negative errno.
│       The logical tensor view remains chunk_bytes; the [chunk_bytes, total_len)
│       tail holds padding read from the slot and is never exposed.
│
├── 4. On read failure: DmabufGPUAllocator.free(memory_obj); slot result = None
│
└── 5. [finally] RawBlockCore.unlock_many(locked_keys)
          Decrements _lock_refcnt for all pinned entries, allowing eviction.
          Runs whether steps 2–4 succeed or raise. Must run before returning.
    Return TensorMemoryObj list (None for failed reads)
```

`header_bytes` is taken from `RawBlockCore.header_bytes` (set at construction time,
same value used by `_write_one()` and `load_many_into()`). The thread pool
(`ThreadPoolExecutor(disk_io_threads)`) parallelizes calls to the Rust layer; the
Rust worker thread serializes SQE submission to the ring internally. If profiling
shows ring contention, shard to multiple rings at that point.

---

## 10. Write Path: GPU VRAM → NVMe

`batched_submit_put_task` (§11) receives a `MemoryObj` allocated by any backend —
**most commonly `LocalCPUBackend` (CPU DRAM)**, because that is the default global
allocator in `StorageManager`'s common path. The dmabuf `WRITE_FIXED` path is only a
*win* when the source is already registered GPU VRAM. For a CPU-resident source, a
GPU-staged dmabuf write would do `CPU → GPU slab (H2D) → NVMe (WRITE_FIXED)` — which
is strictly **slower** than a plain `CPU → NVMe` O_DIRECT write and adds a bounce
that this feature exists to remove. Therefore the write path **branches on
ownership** into three cases (it does **not** stage CPU memory into the GPU, and it
does **not** send foreign GPU memory to the CPU write path):

- **Registered GPU source** (`owns()` true): direct `WRITE_FIXED` from the source's
  own slab (the fast path this backend is for).
- **CPU source** (source tensor on `cpu`): delegate to `RawBlockCore.put_many()`,
  which reserves, writes header + payload with correct O_DIRECT alignment, and
  commits atomically. No GPU staging, no dmabuf.
- **Foreign/unregistered GPU source** (source tensor on CUDA but not owned by this
  allocator): **rejected in MVP** with a clear error. `put_many()` is *not* safe
  here: it reads `obj.byte_array` (core.py:663-665), which casts
  `raw_data.data_ptr()` through `ctypes.from_address()` (memory_management.py:875-885)
  — a valid CPU read only for host memory. For a CUDA pointer that is a device
  address interpreted as host memory (garbage/segfault). A GPU→CPU (or GPU→registered
  slab) staging path could be added later; it is out of first-cut scope.

Each slot on the NVMe device has two regions:
- **Header** at `slot_base_offset + 0` — block-aligned CPU write
- **Payload** at `slot_base_offset + header_bytes` — `WRITE_FIXED` (direct path only)

```
put_one(key: CacheEngineKey, memory_obj, on_complete_callback)   # per-key body of batched_submit_put_task
│
├── 0. raw_key: RawBlockKeySpec = encode_legacy_key(key)
│         # from lmcache.v1.storage_backend.raw_block import encode_legacy_key
│         # RawBlockCore must be constructed with key_namespace="legacy" to match.
│         # encode_legacy_key is the same function used by rust_raw_block_backend.py
│         # (line 19/344 of plugins/rust_raw_block_backend.py) — checkpoint-compatible
│         # key encoding; avoids cross-backend key collisions.
│
├── 1. ref_count_up(memory_obj)          [decremented in outer finally, step 7]
│
├── 2. Classify the source (three-way):
│     if DmabufGPUAllocator.owns(memory_obj):
│         # owns() = isinstance TensorMemoryObj AND parent_allocator is self
│         #          AND address in [0, pool_bytes)  (§8, finding 1 — no metadata.extra)
│         pass   # ---- FAST path: registered GPU VRAM → step 3 ----
│     elif memory_obj.tensor is not None and memory_obj.tensor.device.type == "cpu":
│         # ---- COMMON source: CPU DRAM, no dmabuf, no GPU staging ----
│         RawBlockCore.put_many([raw_key], [memory_obj])   # CPU O_DIRECT write; reads byte_array
│         invoke on_complete_callback; goto 7
│     else:
│         # ---- Foreign/unregistered GPU memory: REJECT in MVP ----
│         # put_many() would read a CUDA data_ptr as host memory (unsafe;
│         # memory_management.py:875-885). GPU→CPU staging is out of first-cut scope.
│         logger.warning("IouDmabufBackend: unsupported foreign GPU source for %s; "
│                        "not stored (enable a CPU or registered-GPU source)", key)
│         goto 7   # no slot reserved; ref_count_down runs in step 7
│
│   # ---- FAST path: source is registered GPU VRAM ----
├── 3. payload_len = chunk_bytes = memory_obj.get_size()   # logical byte length
│     total_len = round_up(chunk_bytes, block_align)                       (C1)
│     slot_base_offset = RawBlockCore.reserve_slot(raw_key, memory_obj)
│         # meta (shape/dtype/fmt/cached_positions) built + stored in _inflight now
│         → None if no free slot, key already in-flight, or key already indexed
│         → if None: goto 7  (no slot reserved; nothing to clean up)
│         slot_committed = False
│
│   try:   ← slot must be committed or aborted; exactly one path runs
│   ├── 4. RawBlockCore.write_slot_header(raw_key, slot_base_offset, payload_len)
│   │         → block-aligned header write at (slot_base_offset + 0); raises on failure
│   │
│   ├── 5. abs_offset = memory_obj.metadata.address
│   │     slab_idx, buf_offset = decompose(abs_offset)
│   │     rawdev.write_fixed_dmabuf(
│   │         slab_idx, buf_offset,
│   │         total_len,                         # O_DIRECT-rounded length, NOT chunk_bytes (C1)
│   │         slot_base_offset + header_bytes    # skip the header region
│   │     )
│   │
│   ├── 6. WRITE_FIXED completed without exception:
│   │     RawBlockCore.commit_slot(raw_key, slot_base_offset)   # publishes stored _inflight.meta
│   │     slot_committed = True
│   │     invoke on_complete_callback
│   │
│   finally:
│     if not slot_committed:
│       RawBlockCore.abort_slot(raw_key, slot_base_offset)   # exactly once; no double-abort
│
└── 7. [outer finally] ref_count_down(memory_obj)
          Runs on every exit: CPU-path put, no-slot early return, commit, or abort.
```

**Why no GPU staging branch (M1).** Staging a CPU source into a registered slab just
to use `WRITE_FIXED` never pays off — it adds an H2D bounce that is worse than a
plain `CPU → NVMe` write — so CPU sources take `put_many` directly. Foreign GPU
sources are a separate matter: they *cannot* use `put_many` at all (its `byte_array`
read assumes host memory), so rather than silently adding a GPU→CPU staging copy,
the MVP **rejects** them (step 2, else-branch) and surfaces a warning. A future
opt-in could stage foreign GPU memory either D2H into a CPU buffer for `put_many` or
D2D into a registered slab for `WRITE_FIXED`; both are out of first-cut scope. The
NVMe → GPU read path (§9) always targets a registered slab, so it never needs any of
this.

---

## 11. Backend Class Structure

```python
# lmcache/v1/storage_backend/iou_dmabuf_backend.py

class IouDmabufBackend(AllocatorBackendInterface):
    """KV cache backend using io_uring DMA-BUF for GPU VRAM ↔ NVMe transfers."""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        dst_device: str,
    ) -> None:
        # 1. probe_dmabuf_support() — refuse init if kernel lacks support
        # 2. Open raw NVMe device via RawBlockCore(RawBlockCoreConfig(...))
        #    RawBlockCore opens RawBlockDevice from lmcache_rust_raw_block_io
        # 3. Allocate GPU VRAM slabs via DmabufGPUAllocator
        #    (probe NVIDIA cudaMalloc+cuMemGetHandleForAddressRange →
        #     AMD HIP hipMemGetHandleForAddressRange → AMD DRM GEM)
        # 4. rawdev.register_dmabuf_buffers(dmabuf_fds)
        #    Uses self.fd (internal NVMe fd) as target_fd for all slots
        # 5. Log P2P availability diagnostic (see §13)
        # 6. Initialize ThreadPoolExecutor(disk_io_threads) for batched I/O

    # ---- StorageBackendInterface (signatures must match abstract_backend.py) ----
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool: ...
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool: ...
    def batched_submit_put_task(
        self, keys, objs, transfer_spec=None
    ) -> Optional[List[Future]]: ...                       # entry point; body = §10 put_one per key
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]: ...
    def batched_get_blocking(self, keys) -> List[Optional[MemoryObj]]: ...   # §9
    async def batched_get_non_blocking(                     # NOTE: async coroutine (abstract:161)
        self, lookup_id: str, keys, transfer_spec=None
    ) -> List[MemoryObj]: ...
    def get_allocator_backend(self) -> "AllocatorBackendInterface": ...

    # ---- AllocatorBackendInterface (abstract_backend.py:331-421) — REQUIRED ----
    def initialize_allocator(
        self, config, metadata
    ) -> MemoryAllocatorInterface: ...                      # returns the DmabufGPUAllocator
    def get_memory_allocator(self) -> MemoryAllocatorInterface: ...
    def allocate(
        self, shapes, dtypes, fmt=MemoryFormat.KV_2LTD, eviction=True, busy_loop=True
    ) -> Optional[MemoryObj]: ...                           # delegates to DmabufGPUAllocator.allocate
    def batched_allocate(
        self, shapes, dtypes, batch_size, fmt=MemoryFormat.KV_2LTD,
        eviction=True, busy_loop=True,
    ) -> Optional[list[MemoryObj]]: ...
    def calculate_chunk_budget(self) -> int: ...

    def close(self) -> None: ...  # drain I/O → RawBlockCore.close() → DmabufGPUAllocator.close()
```

**Interface compliance (M4).** `IouDmabufBackend` is an `AllocatorBackendInterface`,
so it MUST implement `initialize_allocator`, `get_memory_allocator`, `allocate`,
`batched_allocate`, and `calculate_chunk_budget` — none are optional. Mirror
`GdsBackend`: `initialize_allocator` constructs and returns the `DmabufGPUAllocator`
(a `MemoryAllocatorInterface`, §8/M3), stored as `self.memory_allocator`, and
`allocate`/`batched_allocate` delegate to it (see gds_backend.py:1089, 1121). Two
signature corrections vs. earlier drafts: there is **no** single-key `submit_put_task`
in the abstract interface (the entry point is `batched_submit_put_task`; §10's
`put_one` is its per-key body), and `batched_get_non_blocking` is an **`async def`**
returning `List[MemoryObj]` with `(lookup_id, keys, transfer_spec)` — not a
`Future`-returning sync method (abstract_backend.py:161).

`RawBlockCore` owns the on-disk index and slot lifecycle; `DmabufGPUAllocator`
owns GPU VRAM slab lifetime; `rawdev` owns the io_uring ring and dmabuf tokens.

### 11.1 Instantiation gate

```python
@staticmethod
def is_available() -> bool:
    try:
        # Import failure here == the maturin crate (§2.2) is not built/installed.
        from lmcache_rust_raw_block_io import RawBlockDevice
        # probe_dmabuf_support MUST be a #[staticmethod] or module-level function
        # (finding 4): is_available() runs before any device is opened, so it
        # cannot be an instance method that needs an open RawBlockDevice.
        return RawBlockDevice.probe_dmabuf_support()
    except (ImportError, AttributeError):
        # ImportError: crate missing. AttributeError: crate predates the dmabuf
        # methods (older build). Either way, backend is unavailable — skip it.
        return False
```

`probe_dmabuf_support()` is a **`#[staticmethod]`** (or module-level PyO3 function)
on the crate — it does not take `self`. It opens its own throwaway ring and attempts
`IORING_REGISTER_BUFFERS_UPDATE` with an `IO_REGBUF_TYPE_DMABUF` descriptor using
deliberately invalid fds (`dmabuf_fd = -1, target_fd = -1, uaddr = 0, size = 0`).

Error discrimination, based on the exact kernel path in `io_uring/rsrc.c`:

```
io_register_dmabuf():
  line 816: if (!IS_ENABLED(CONFIG_DMABUF_TOKEN)) return -EOPNOTSUPP
  line 818: if (desc->uaddr || desc->size) return -EINVAL      ← skipped (both 0)
  line 832: ret = -EBADF
  line 833: target_file = fget(desc->target_fd)               ← fget(-1) → NULL
  line 834: if (!target_file) goto err                         ← returns -EBADF
```

- `-EOPNOTSUPP`: `CONFIG_DMABUF_TOKEN=n` — return **False** (unsupported).
- `-EBADF`: `CONFIG_DMABUF_TOKEN=y`, passed `uaddr/size=0` check, reached fd
  validation — return **True** (API is present).
- `-EINVAL`: ambiguous. The kernel may return `-EINVAL` from the outer
  `IORING_REGISTER_BUFFERS_UPDATE` dispatch layer on an old kernel that does not
  recognize `IO_REGBUF_TYPE_DMABUF` or the extended update descriptor shape. It
  can also be returned on a *supported* kernel if `uaddr` or `size` are non-zero
  (line 818). Since our probe uses `uaddr=0, size=0`, a supported kernel would not
  emit `-EINVAL` here, but to be safe: treat `-EINVAL` as **inconclusive**, log a
  warning, and return False to avoid incorrectly claiming support.
- Any other error: treat conservatively as unsupported and log a warning.

A `udmabuf` + O_DIRECT device pair gives a stronger end-to-end capability check
and is better suited for integration tests (P0 gate) than for an init-time probe.

The Rust extension module is `lmcache_rust_raw_block_io`, the same module already
used by `RawBlockCore` (imported at `core.py:411`).

### 11.2 Registration in `CreateStorageBackends`

`CreateStorageBackends()` (`lmcache/v1/storage_backend/__init__.py:111`) builds an
**`OrderedDict[str, StorageBackendInterface]` named `storage_backends`**, keyed by
`str(backend)` (e.g. `storage_backends[str(gds_backend)] = gds_backend`, `__init__.py:242`).
There is no list to `append` to (finding 2). `IouDmabufBackend` must therefore
define `__str__` returning a stable name (`"IouDmabufBackend"`) and register itself
in that dict:

```python
# At the end of CreateStorageBackends(), after the GdsBackend block
if (
    config.extra_config
    and config.extra_config.get("iou_dmabuf.enabled", False)
    and "IouDmabufBackend" not in _skip
):
    # First cut: hard-require a LocalCPUBackend staging allocator (see below).
    if "LocalCPUBackend" not in storage_backends:
        raise ValueError(
            "iou_dmabuf.enabled=true requires a LocalCPUBackend staging "
            "allocator; set max_local_cpu_size > 0"
        )
    if IouDmabufBackend.is_available():
        iou_backend = IouDmabufBackend(config, metadata, loop, dst_device)
        storage_backends[str(iou_backend)] = iou_backend   # keyed by name, not appended
    else:
        logger.warning(
            "iou_dmabuf.enabled=true but the lmcache_rust_raw_block_io crate is "
            "missing/outdated or the kernel lacks CONFIG_DMABUF_TOKEN; "
            "skipping IouDmabufBackend"
        )
```

**Overlap rejection (strict, first cut).** `StorageManager.batched_get()` iterates
`get_active_storage_backends(location)` and **returns from the first backend that
yields results** (`storage_manager.py:489`). Insertion order in the `OrderedDict`
is the traversal order, and `IouDmabufBackend` is inserted after `LocalDiskBackend`
and `GdsBackend`, so an overlapping disk backend holding the key would shadow it.
Rather than rely on traversal order or ad-hoc `location=` addressing, the first cut
**rejects overlapping local disk backends at construction**: enabling
`iou_dmabuf.enabled: true` together with `local_disk`/`gds_path` (without skipping
them) raises a `ValueError` in `CreateStorageBackends`. Operators must disable the
overlapping disk backend, so `IouDmabufBackend` is the only local-NVMe backend in
`storage_backends`. (`location="IouDmabufBackend"` still works for explicit reads,
but is not required and is not the sanctioned way to resolve overlap.)

**Staging-allocator requirement.** `IouDmabufBackend` is never the *global* allocator
in the first cut: it requires a `LocalCPUBackend` (`max_local_cpu_size > 0`), enforced
above. `StorageManager.batched_put` stages every source object into each backend's
allocator via `allocate_and_copy_objects` (`storage_manager.py:425`), which for this
backend produces `IouDmabufBackend`-owned GPU copies that take the `WRITE_FIXED` fast
path (§10). Were it the sole allocator, a foreign GPU source object could reach
`_put_one` and be silently rejected (§10 rejects unowned GPU memory). Forcing CPU
staging removes that path; a first-class GPU→GPU staging copy is deferred.

`NixlStorageBackend` is unaffected (different storage tier). Promoting the dmabuf
backend ahead of `GdsBackend` in default traversal is deferred until it is stable.

---

## 12. Configuration

Use `extra_config` (the existing pattern for experimental backends) rather than
promoting fields to first-class `LMCacheEngineConfig` until the prototype is proven:

```yaml
extra_config:
  iou_dmabuf.enabled: true
  iou_dmabuf.device_path: "/dev/nvme0n1"   # raw NVMe namespace, not a mountpoint
  iou_dmabuf.capacity_bytes: 1099511627776  # 1 TiB
  iou_dmabuf.gpu_pool_bytes: 17179869184    # 16 GiB GPU VRAM pool
  iou_dmabuf.block_align: 4096
  iou_dmabuf.slot_bytes: 16781312           # 16 MiB + header room
  iou_dmabuf.ring_depth: 256
  iou_dmabuf.max_eagain_retries: 16
  iou_dmabuf.exporter: "auto"              # auto | cuda_pool | cuda_vmm | hip | amd_drm
  iou_dmabuf.require_p2p: false            # true = refuse init when P2P unverified
```

`exporter: "auto"` probes in the order established by §8.1/§4.1:

1. **`cuda_pool`** (NVIDIA, preferred): export a 1 GiB sub-range of the
   `GPUMemoryAllocator` / PyTorch pool tensor via `cuMemGetHandleForAddressRange`
   (Approach A; may need `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
2. **`cuda_vmm`** (NVIDIA fallback): dedicated VMM allocation
   (`cuMemCreate`/`cuMemMap`) + `torch.from_blob` wrapping (Approach B), used only if
   the pool export in (1) fails on the target driver.
3. **`hip`** (AMD): `hipMemGetHandleForAddressRange`.
4. **`amd_drm`** (AMD fallback): DRM GEM allocation + import.

---

## 13. P2P Observability and Fallback

P2P (PCIe peer-to-peer DMA) is an optimization, not a correctness requirement.
The kernel silently falls back to a host-memory-mediated path when P2P is
unavailable. The backend must make this observable.

### 13.1 Startup diagnostic

At init, after successful dmabuf registration, attempt to determine P2P status:

- Check `CONFIG_PCI_P2PDMA` via `/boot/config-$(uname -r)` or
  `/proc/config.gz` — its absence guarantees no P2P.
- Check PCIe topology: GPU and NVMe should share a PCIe switch (same upstream
  bridge) for P2P to be routed. Parse `sysfs` or use `lspci` to inspect
  upstream bridges.
- On NVIDIA: check `nvidia-smi` p2p matrix or driver sysfs nodes.
- Log the result at INFO level:
  `"P2P DMA: best-effort verified (GPU and NVMe appear to share a PCIe switch)"` or
  `"P2P DMA: unverified (topology check inconclusive; kernel may fall back to host-memory path)"`.
  This is a heuristic, not a kernel-confirmed signal. See §13.3.

### 13.2 `require_p2p` config key

When `iou_dmabuf.require_p2p: true`, refuse backend init if P2P cannot be
confirmed. This lets operators enforce the performance contract rather than silently
running at reduced throughput. Default is `false` (correctness-first startup).

### 13.3 Runtime metrics

Expose a `p2p_verified` boolean in `report_status()` (matching `RawBlockCore`'s
existing `report_status()` pattern), populated from the startup heuristic.
`p2p_verified=True` means the topology check passed; it is **not** a kernel-confirmed
guarantee that data actually travelled via P2P DMA on any given I/O.

### 13.4 Degradation table

| Condition                         | Behavior                                            |
|-----------------------------------|-----------------------------------------------------|
| `CONFIG_DMABUF_TOKEN=y` missing   | Refuse init; log error                              |
| NVMe driver not nvme-pci          | Registration returns `-EOPNOTSUPP`; refuse init     |
| GPU export API unavailable        | Refuse init; log error; suggest GdsBackend          |
| `CONFIG_PCI_P2PDMA=y` missing     | Init succeeds; warn; host-memory DMA path used      |
| P2P topology mismatch             | Init succeeds; warn; host-memory DMA path used      |
| `require_p2p: true` + no P2P     | Refuse init; log error                              |
| Rust rawdev extension not built   | `is_available()` returns False; fall back silently  |

---

## 14. Comparison with GDS Backend

| Dimension                 | `GdsBackend` (cuFile/hipFile)            | `IouDmabufBackend` (io_uring DMA-BUF)       |
|---------------------------|------------------------------------------|----------------------------------------------|
| Kernel requirement        | none (userspace middleware)              | `CONFIG_DMABUF_TOKEN=y`, nvme-pci            |
| Middleware requirement    | NVIDIA GDS driver or AMD ROCm GDS        | none (pure kernel path)                      |
| Storage layer             | per-chunk files                          | `RawBlockCore` raw-block slots               |
| P2P zero-copy             | driver-dependent (cuFile may bounce)     | yes (when `CONFIG_PCI_P2PDMA=y` + topology)  |
| Device compatibility      | any FS                                   | raw NVMe namespace only                      |
| -EAGAIN handling          | not needed                               | required (GPU memory migration/invalidation) |
| Per-I/O overhead          | CUDA/ROCm driver call + kernel I/O       | kernel I/O only                              |
| Maturity                  | production                               | experimental (new kernel feature)            |

**When to use `IouDmabufBackend` over `GdsBackend`:**
- NVMe-PCI storage device with `CONFIG_DMABUF_TOKEN=y` kernel
- PCIe topology where GPU and NVMe are under a common switch (P2P available)
- Latency is critical and no CUDA driver overhead is acceptable

**When to keep `GdsBackend`:**
- Non-NVMe storage (SATA, tmpfs, network)
- Kernel without `CONFIG_DMABUF_TOKEN`
- Per-chunk file layout is required for cross-backend interoperability

---

## 15. Open Questions

### 15.1 GPU dmabuf export (blocking question for P1)

Tensor *wrapping* is solved (reuse `GPUMemoryAllocator`/`TensorMemoryAllocator`, §8.1).
The blocking unknown is *exportability*: whether `cuMemGetHandleForAddressRange`
accepts a sub-range of the PyTorch pool tensor's pointer, and if it requires
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (VMM-backed) to do so. Approach A
(reuse pool tensor) and Approach B (dedicated VMM allocation + `torch.from_blob`)
must be prototyped against the target driver before P2. If both fail, the GPU
connector interface (`VLLMPagedMemGPUConnector.to_gpu()`) may need a new transfer
mode operating on `(device_ptr: int, shape, dtype)` tuples rather than
`torch.Tensor` objects — a larger, deferred change.

### 15.2 Rust io_uring crate expressiveness

Verify whether the Rust `io_uring` crate can construct `io_uring_regbuf_desc` with
`IO_REGBUF_TYPE_DMABUF` using its current API. If not, a small `bindgen`-generated
C shim is the preferred solution (not a standalone C extension).

### 15.3 `RawBlockCore` refactoring boundary

`RawBlockCore` currently couples slot/index logic with the transfer engine. For
`IouDmabufBackend` to use it cleanly, we need to determine: can it be used by
composition as-is (pass the core object into the backend), or does it need a
refactor to separate the index/slot component from the I/O engine? The composition
path is preferred for MVP to avoid scope creep.

### 15.4 P2P observability signal

Determine the best practical signal for whether a given run used P2P or host-memory
DMA. Candidates: kernel tracepoints, NVMe sysfs counters, perf counters on the PCIe
switch, or NVIDIA `nvidia-smi` topological output. Needs investigation on the target
hardware.

### 15.5 Concurrent write + read of same chunk

A concurrent read for a key that is being written will miss in `RawBlockCore`'s
index (entry added only after write completes and `put_many` commits). This matches
`GdsBackend`'s `exists_in_put_tasks` behavior and is acceptable for MVP.

### 15.6 Multi-GPU setup

Each GPU would have its own `DmabufGPUAllocator` with its own slab list. Multiple
`IouDmabufBackend` instances would share one `RawBlockCore` (same NVMe device) or
use separate NVMe namespaces. Routing in `StorageManager` would select by device.
Out of scope for MVP.

### 15.7 True end-to-end zero-copy into paged KV

This design removes the host bounce buffer but still incurs one intra-GPU (D2D) copy
from the backend slab into vLLM's paged KV cache (§1.1/M2). Eliminating that copy
would require the read to land directly in vLLM's paged KV buffers — i.e. those
buffers themselves registered as dmabuf targets, with the slot payload laid out to
match the paged block geometry. That couples the backend to vLLM's allocator and the
GPU connector's block layout, and is a substantially larger change. Deferred; the MVP
accepts the single D2D copy.

---

## 16. Implementation Phases

First cut = P0 + P1 + P2. After P2, **normal LMCache put/get works** on NVIDIA: a
CPU-allocated source (the common `StorageManager` path) stores via the `put_many`
fallback, a registered-GPU source stores via `WRITE_FIXED`, and reads land in GPU
slabs. The P2 read path already pins entries for the I/O lifetime
(`get_entries_many(lock_refcount=True)` + `unlock_many`, §9) so reads cannot race
eviction — this is part of P2, not deferred. Overlapping disk backends must be
disabled or the backend `location`-addressed (§11.2). P3+ adds only P2P diagnostics,
AMD, and tuning — none of which block basic correctness.

| Phase | Scope                                                                        | Gate                                              |
|-------|------------------------------------------------------------------------------|---------------------------------------------------|
| P0    | New Rust methods on `RawBlockDevice` (§7): `probe_dmabuf_support` (`#[staticmethod]`), `register_dmabuf_buffers`, `read_fixed_dmabuf`, `write_fixed_dmabuf`; **rebuild+install the maturin crate under `rust/raw_block` (§2.2)** | Crate builds/installs with dmabuf methods; `probe_dmabuf_support()` returns True on patched kernel; udmabuf smoke test passes with block-aligned lengths |
| P1    | GPU pool allocation reusing `GPUMemoryAllocator` (NVIDIA); dmabuf **export** proven (§8.1) | **`cuMemGetHandleForAddressRange` exports a 1 GiB sub-range of the pool tensor** (record any `PYTORCH_CUDA_ALLOC_CONF` needed); exported fd registers and a round-trip READ_FIXED/WRITE_FIXED verifies on GPU. **P2 must not begin until this passes.** |
| P2    | New `RawBlockCore` methods (`get_entries_many`, `reserve_slot`, `write_slot_header`, `commit_slot`, `abort_slot`); `DmabufGPUAllocator` + full `AllocatorBackendInterface` surface (§11/M4); full §10 write dispatch — direct GPU-source `WRITE_FIXED` + **CPU-source `put_many` fallback** + **foreign-GPU rejection**; read path (§9) **including the `lock_refcount=True` / `unlock_many` lock lifetime** (intrinsic to a correct read — unlocked reads race eviction, not optional) | Normal `StorageManager` put/get works end-to-end (CPU-allocated source stores via fallback; a registered-GPU source stores via `WRITE_FIXED`) on real NVMe + GPU, `location="IouDmabufBackend"`; O_DIRECT length rounding (C1) verified with a non-4K-multiple chunk; foreign GPU source rejected (not corrupted); a read holding a slot survives a concurrent `delete_many`/eviction |
| P3    | P2P diagnostics (§13) | P2P status logged at startup |
| P4    | AMD HIP address-range export path; AMD DRM GEM fallback                       | AMD GPU smoke test passes                         |
| P5    | Metrics counters; `require_p2p` enforcement; default-traversal promotion; batched async get | Perf benchmark vs `GdsBackend`; production readiness review |
