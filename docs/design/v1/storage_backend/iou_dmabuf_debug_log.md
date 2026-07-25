# io_uring DMA-BUF Backend — Debugging Log & Root-Cause Record

**Status:** blocked on a kernel/driver-level defect (not fixable in LMCache).
**Audience:** engineers/agents picking this up, and kernel/amdgpu maintainers.
**Companion repro:** `tools/repro_iou_dmabuf_p2p.c` (standalone C, no LMCache/Python).

---

## 1. TL;DR (conclusion first)

On this AMD ROCm + patched-kernel stack, the io_uring **DMA-BUF `WRITE_FIXED`
path corrupts data** — the NVMe device's peer-to-peer DMA that **reads** exported
AMD VRAM occasionally transfers wrong/stale bytes (~0.3–0.5% of 2 MiB chunks).

Established, with high confidence:

- **Write-only.** `WRITE_FIXED` (NVMe peer-DMA *reads* VRAM → NVMe) corrupts;
  `READ_FIXED` (NVMe peer-DMA *writes* VRAM) is clean at all concurrencies.
- **Not concurrency.** Corrupts even with the entire put path serialized
  (`disk_io_threads=1`, one op at a time, nothing interleaving on the ring).
- **Not LMCache code.** Reproduces through the raw Rust `RawBlockDevice` (no
  `RawBlockCore`, no Python backend logic), and is **confirmed in the standalone
  C program** `tools/repro_iou_dmabuf_p2p.c` (pure HIP + io_uring, no
  LMCache/Python/Rust): **1298 / 2560 chunks corrupt (~50%)** at 8-way write
  concurrency on `/dev/nvme0n1`, coarse VRAM (see §7.1).
- **Not memory-coherency type.** Coarse-grained (`hipMalloc`) and fine-grained
  (`hipExtMallocWithFlags`) source VRAM both corrupt.
- **Data signature:** "foreign" — the corrupt chunk holds stale/other data, not a
  clean swap of two valid chunks; often the first block(s) of a chunk are wrong.

The earlier belief that "concurrency causes it" was a **false negative**: a single
3200-op run at low concurrency can land on zero corruptions at a ~0.4% rate.

**Action:** the dmabuf write path is unsafe on this stack; it needs a kernel/driver
fix. Reads work. File upstream with the C repro.

---

## 2. Environment

| Component        | Value |
|------------------|-------|
| GPU              | AMD Radeon AI PRO R9700 (32 GiB), ROCm PyTorch (`torch.version.hip` set) |
| Storage          | Raw NVMe namespace `/dev/nvme0n1`, opened `O_DIRECT` |
| Kernel           | Patched with io_uring DMA-BUF token support (`CONFIG_DMABUF_TOKEN=y`); nvme-pci implements the `create_dmabuf_token` blk-mq op |
| Interface        | Register a dmabuf fd as an io_uring fixed buffer **bound to the O_DIRECT block fd**; issue `READ_FIXED`/`WRITE_FIXED` where the "buffer address" is a **byte offset into the dmabuf** |
| KV chunk         | 2 MiB (bf16, shape `[2,16,256,128]`) |
| GPU export       | `hipMemGetHandleForAddressRange(..., hipMemRangeHandleTypeDmaBufFd, 0)` |

Interface references: `docs/design/v1/storage_backend/iou_dmabuf_backend.md`,
the kernel userspace guide, and the liburing `test/rw-dmabuf.c` example.

---

## 3. Symptom timeline (how we got here)

Cold/warm KV-cache test with the backend enabled. Two distinct problems were
found in sequence; the first was fixed, the second is the subject of this record.

### Problem A — hang (FIXED)

- **Observed:** cold iteration hung; even a single prompt hung.
- **py-spy (EngineCore process):** all four `iou-dmabuf` worker threads stuck —
  one thread inside `MemoryObj.metadata` (re-acquiring the object's own lock),
  the rest blocked in `DmabufGPUAllocator.free`.
- **Root cause:** `MemoryObj.ref_count_down()/unpin()` call
  `parent_allocator.free(self)` **while holding the object's non-reentrant
  `self.lock`**. The allocator's free path read/rewrote `memory_obj.metadata.*`,
  and `.metadata` is a property doing `with self.lock: return self.meta` →
  self-deadlock.
- **Fix:** use the raw `.meta` attribute (never the `.metadata` property) in the
  allocator free path (`_free_locked` / `_batched_free_locked` /
  `_locate_owned_object`), matching how the stock allocators mutate metadata
  during free. (Commit: "deadlock in DmabufGPUAllocator free path".)

After the hang was fixed, cold iteration completed — and **data-integrity
verification then surfaced Problem B.**

### Problem B — data corruption (this record)

Everything below is the investigation of Problem B.

---

## 4. Investigation iterations

Each row: hypothesis → what we ran → result → what it proved. All runs use the
2 MiB chunk, `/dev/nvme0n1`, `source=gpu` (allocator-owned, `WRITE_FIXED` fast
path) unless noted.

| # | Hypothesis / question | Test & config | Result | Conclusion |
|---|---|---|---|---|
| 1 | Does it round-trip at small scale? | benchmark, `--num-ops 128 --concurrency 4 --verify-integrity` | `integrity_passed: true` | Passes at 128 ops — corruption is rare, needs volume. |
| 2 | Push real volume (~50 GiB). | benchmark, `--num-ops 128 --iters 25 --concurrency 4` (auto slots recycled per iter) | **3 mismatches** / 3200 | Corruption is real, rare (~0.1–0.5%). |
| 3 | Is the benchmark undercounting? | Fixed the fill seed (per-iteration index → **global** index) so stale reads from recycled slots are caught; added mismatch **classification** (swap vs foreign). | **20 mismatches**, all **"foreign/corrupt"** | Corruption is stale data, not a clean 2-chunk swap. Prior counts undercounted. |
| 4 | Is it concurrency? | benchmark `--concurrency 1` vs `--concurrency 4` | conc=1 **clean**; conc=4 **corrupt** | *Looked* like a concurrency race. (Later shown to be a false negative — see #9.) |
| 5 | Write path or read path? | split knobs: `--write-concurrency 4 --read-concurrency 1` vs `--write-concurrency 1 --read-concurrency 4` | write-conc=4 → **corrupt**; read-conc=4 → **clean** | Isolated to the **write** path. Reads are clean even concurrent. |
| 6 | GPU→NVMe peer-DMA coherency (stream sync ≠ peer-visible)? | Added `torch.cuda.synchronize()` before the write phase. | still corrupt | Stream sync doesn't help (the reference clone already synced CPU-side). Motivated a direct A/B. |
| 7 | Coarse vs fine-grained VRAM (coherency A/B). | `tools/diagnose_iou_dmabuf_coherency.py`: raw `RawBlockDevice` `write_fixed_dmabuf`/`read_fixed_dmabuf`, `hipMalloc` vs `hipExtMallocWithFlags(finegrained)`, identical `hipMemset` fill, N concurrent writes, serial read-back. | **BOTH corrupt** (coarse ~73, fine ~71 / 2560) | **Coherency ruled out.** Also: reproduces with **no LMCache logic** (no `RawBlockCore`, no backend) → bug is in the Rust `RawBlockDevice` dmabuf path or below. |
| 8 | Is serializing the dmabuf write enough? | Backend: lock around `write_fixed_dmabuf`; benchmark `--write-concurrency 4`. | still corrupt (**11**); `write_gib_per_sec` halved (lock provably active) | Serializing only the dmabuf write is **insufficient**. |
| 9 | Serialize the *entire* put path. | benchmark `--write-concurrency 4 --disk-io-threads 1` (one `_put_one` at a time: reserve→header→dmabuf-write→commit, nothing interleaving on the ring) | still corrupt (**14**) | **Not LMCache concurrency.** Corrupts fully serialized. The conc=1 "clean" in #4 was luck (one run below the ~0.4% rate). |
| — | Direction asymmetry (from #5 + #7). | — | writes corrupt, reads clean, both concurrent and serial | The failing primitive is **NVMe peer-DMA reading AMD VRAM** (`WRITE_FIXED`). |

Representative corrupt-chunk detail (iter, key→observed): the read returns bytes
matching **no** chunk written in that iteration ("foreign"); with the 3-point
sample (start/mid/end of the 2 MiB chunk) the corruption is frequently a wrong
**leading block** with the tail correct, or the whole chunk holding a different
chunk's value. In the raw diagnostic the chunk at **device offset 0** (chunk 0)
was deterministically wrong — an artifact of writing NVMe LBA 0; the LMCache
backend never writes offset 0 (slots start after the metadata region), so ignore
that specific chunk when reasoning about the backend.

---

## 5. Root-cause evidence chain

1. **Reads clean, writes corrupt** (#5) → the direction that fails is peer-DMA
   *reading* exported VRAM.
2. **Corrupts serialized** (`disk_io_threads=1`, #9) → not a concurrency/lock race.
3. **Corrupts through raw `RawBlockDevice`** (#7) → not `RawBlockCore`, not the
   Python backend, not slot/index/offset bookkeeping.
4. **Coarse == fine-grained** (#7) → not the VRAM allocation/coherency mode.
5. **Rust submission is direction-agnostic** (same `build_and_submit_sqe`, single
   worker, one ring for read and write) yet only writes corrupt → the fault is
   below the io_uring submission logic, i.e. the kernel/amdgpu P2P path.

⇒ **A kernel/amdgpu reliability defect in NVMe→(read)→AMD-VRAM peer DMA under the
io_uring dmabuf-token path.** ~0.3–0.5% of 2 MiB transfers deliver stale bytes.

### Explicitly ruled out
- LMCache Python logic (deadlock aside, which was fixed).
- `RawBlockCore` slot lifecycle / header/payload offset math.
- Concurrency / locking (serial still corrupts).
- VRAM coherence mode (coarse vs fine-grained).
- Source-fill visibility (stream-synced; CPU reads the source correctly).

### Not yet isolated (good next experiments)
- **Exporter vs importer side:** does the same NVMe P2P *write* work with a
  `udmabuf` (host memory) source instead of amdgpu VRAM? If udmabuf is clean and
  amdgpu VRAM is not → squarely the amdgpu exporter's P2P-read path. If udmabuf
  also corrupts → the nvme-pci token/importer side.
- **NVIDIA control:** same workload with a CUDA-exported buffer, to see if the
  defect is AMD-specific.
- **PCIe topology / `CONFIG_PCI_P2PDMA`:** confirm whether transfers are true P2P
  or host-bounced, and whether the switch/root-port matters.
- **`dmesg` during a corrupting run:** amdgpu / dma-fence / nvme warnings.

---

## 6. Tools built (all in-repo)

| Path | Purpose |
|---|---|
| `benchmarks/storage_backend_io/iou_dmabuf_io_benchmark.py` | End-to-end backend write/read + integrity. Knobs: `--num-ops`, `--iters`/`--target-gib`, `--concurrency`, `--write-concurrency`, `--read-concurrency`, `--disk-io-threads`, `--source {gpu,cpu}`, `--verify-integrity`. Globally-seeded per-chunk patterns; classifies each mismatch as swap vs foreign; recycles slots per iter so traffic ≫ footprint. |
| `tools/probe_gpu_dmabuf_export.py` | Confirms a torch GPU pointer can be exported as a dmabuf fd (HIP or CUDA). The P1 go/no-go. |
| `tools/probe_iou_dmabuf_e2e.py` | Single dmabuf `WRITE_FIXED`/`READ_FIXED` round-trip through the real `RawBlockDevice`, with a watchdog. |
| `tools/diagnose_iou_dmabuf_coherency.py` | Raw-`RawBlockDevice` reproducer. Trials coarse vs fine-grained VRAM, and concurrent vs serialized writes. No `RawBlockCore`/backend. This is what isolated the bug below LMCache. |
| `tools/repro_iou_dmabuf_p2p.c` | **Standalone C** repro (HIP + io_uring, no Python, no LMCache) — for kernel/driver maintainers. See §7. |

### How to reproduce the corruption (Python, fastest)
```bash
python benchmarks/storage_backend_io/iou_dmabuf_io_benchmark.py \
  --device-path /dev/nvme0n1 --num-ops 128 --iters 25 --verify-integrity \
  --write-concurrency 4 --read-concurrency 1 --disk-io-threads 1
# expect integrity_passed: false, a handful of "foreign/corrupt" mismatches
```

---

## 7. Standalone C reproducer

`tools/repro_iou_dmabuf_p2p.c` removes every LMCache/Python/Rust layer. It:

1. `hipMalloc` (or `hipExtMallocWithFlags` fine-grained) a GPU source pool and a
   GPU dest buffer;
2. exports each as a dmabuf fd via `hipMemGetHandleForAddressRange`;
3. opens the NVMe device `O_DIRECT`, sets up io_uring, registers a sparse buffer
   table and installs the dmabufs with `IORING_REGISTER_BUFFERS_UPDATE` +
   `IO_REGBUF_TYPE_DMABUF` bound to the NVMe fd;
4. fills N source chunks with distinct patterns (`hipMemset` + `hipDeviceSynchronize`);
5. issues N `WRITE_FIXED` (configurable in-flight depth, `-EAGAIN` reissue);
6. reads each slot back with `READ_FIXED`, copies dest→host, verifies the pattern;
7. repeats and reports the corruption count.

Build & run instructions are in the file header. Vary `--concurrency` (1 still
corrupts), `--finegrained`, `--repeat`, `--device-offset` (avoid LBA 0 for a
non-destructive scratch region). See §8 for the destructive-write warning.

Build (runtime-only ROCm — no HIP dev headers/hipcc needed; the program declares
the HIP functions it uses and links `libamdhip64`):
```
cc -O2 -o repro_iou_dmabuf_p2p tools/repro_iou_dmabuf_p2p.c -luring \
   -L/opt/rocm/lib -lamdhip64
```

### 7.1 Confirmed run (2026-07-25)

```
./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --concurrency 8 --repeat 40 \
    --device-offset $((4<<30))
# device=/dev/nvme0n1 chunks=64 chunk=2048KiB concurrency=8 repeat=40 mem=coarse
# ...
# corrupt 1298 / 2560 chunks (64 chunks x 40 repeats)
# => REPRODUCED: io_uring dma-buf WRITE_FIXED (NVMe peer-DMA reading VRAM) corrupts.
```

- **~50% of 2 MiB chunks corrupt** at 8-way write concurrency — a far higher rate
  than the full LMCache stack (~0.3–0.5%). Expected: the raw tool fires 8
  concurrent `WRITE_FIXED` at one dmabuf, while the backend interleaves one
  dmabuf write per chunk with header writes and slot bookkeeping, spacing the
  peer reads out.
- **Concurrent writes cross data:** distinct chunks read back the *same* wrong
  value (e.g. chunk 0 and chunk 1 both returned `0xc1/0x29/0x01`), and many
  chunks returned another chunk's value — consistent with peer-DMA reads of
  exported VRAM returning stale/other data under concurrency.
- Environment: AMD Radeon AI PRO R9700, runtime-only ROCm, coarse VRAM
  (`hipMalloc`), `/dev/nvme0n1`, `--device-offset 4 GiB` (scratch).
- **Still to capture:** a `--concurrency 1` run (LMCache data says it still
  corrupts, at a lower rate); a `--finegrained` run (coherency-mode control); and
  `dmesg` during a corrupting run. These strengthen the upstream report but the
  core defect is already reproduced without LMCache.

---

## 8. Cautions

- **`WRITE_FIXED` in these tools writes to the raw NVMe device** at the chosen
  offsets — destructive. Point them at a scratch namespace.
- Reads (`READ_FIXED`) are non-destructive.
- The C repro and `diagnose_iou_dmabuf_coherency.py` intentionally use device
  offset 0 by default for simplicity; pass a non-zero aligned `--device-offset`
  to avoid clobbering a partition table, and to sidestep the LBA-0 artifact noted
  in §4.

---

## 9. One-paragraph handoff

The LMCache io_uring DMA-BUF backend is functionally complete and correct on the
software side (export, registration, slot lifecycle, read path all verified). It
is **blocked by a kernel/amdgpu defect**: NVMe peer-to-peer DMA that *reads*
exported AMD VRAM (`WRITE_FIXED`) delivers stale bytes for ~0.3–0.5% of 2 MiB
transfers, independent of concurrency and VRAM allocation mode, while the reverse
direction (`READ_FIXED`) is reliable. Reproduce with
`tools/repro_iou_dmabuf_p2p.c` (no LMCache) or the Python command in §6. Next
best experiment: swap the amdgpu VRAM source for a `udmabuf` (host) source to
localize the defect to the amdgpu exporter vs the nvme-pci importer.
