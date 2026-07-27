# io_uring DMA-BUF Backend — Debugging Log & Root-Cause Record

**Status:** localized to the AMD GPU exporter / peer-DMA visibility path (kernel/
driver-level; not fixable in LMCache). The generic dma-buf-token/NVMe-importer
path is confirmed healthy via a `udmabuf` (host-memory) control.
**Audience:** engineers/agents picking this up, and kernel/amdgpu maintainers.
**Companion repro:** `tools/repro_iou_dmabuf_p2p.c` (standalone C, no LMCache/Python).

> **Correction (2026-07-25, same session).** Earlier revisions of this document
> concluded `WRITE_FIXED` was the broken direction and `READ_FIXED` was clean.
> That conclusion was **wrong** — it rested on an *indirect* signal (an LMCache
> full-stack benchmark where `write-concurrency=1 + read-concurrency=4` happened
> to come back clean) rather than a direct, isolated measurement of each
> direction. Once the standalone C repro was extended to test `WRITE_FIXED` and
> `READ_FIXED` **independently** (§7.3), the result reversed: `WRITE_FIXED` is
> clean (0/2560, every byte checked) and `READ_FIXED` is broken (2558/2560,
> ~99.9%). §1, §5, and §9 below reflect the corrected conclusion. §4's iteration
> table and the earlier §7.1/§7.2 sections are left as an accurate record of what
> was run and believed at the time — read them as history, not current truth —
> because the methodology mistake (round-trip tests can't isolate direction) is
> itself worth learning from.

> **External review (2026-07-25, same session).** An independent review (Codex)
> of the investigation and tooling found the "amdgpu defect" attribution
> **premature** and identified real tooling gaps. Summary (full detail in §5.1):
>
> - **AMD export flags, resolved:** the attempted
>   `hipMemGetHandleForAddressRange(..., flags=1)` call returns
>   `hipErrorInvalidValue`. Inspection of the current HIP implementation confirms
>   that it rejects every nonzero flag before exporting. ROCr's lower-level
>   `HSA_AMD_DMABUF_MAPPING_TYPE_PCIE` flag is not an alternate mapping mode: it
>   checks Large BAR availability for a discrete, non-XGMI GPU and then calls the
>   same KFD dma-buf export operation. The HIP tool must use `flags=0`; no valid
>   HIP export option was omitted.
> - **NVMe SGL vs. PRP, resolved:** the reviewer flagged that this kernel's top
>   commit adds a new DMA-BUF SGL descriptor path (selected above a ~32 KiB
>   average segment size, `drivers/nvme/host/pci.c`), which both LMCache and the
>   C repro could be exercising instead of (or in addition to) the older PRP
>   path — an unisolated second suspect. **Closed**: a `printk` added to the
>   kernel by the user confirmed the failing transfers go through the **PRP**
>   path, not the new SGL path. SGL selection is no longer a suspect.
> - **Tooling gaps, now fixed** (see the commit "iou-dmabuf tools: strict
>   accounting, PCIe export flag, exception safety"): the C repro conflated
>   transport failures (short reads/writes, `-EIO`, `-EAGAIN` exhaustion) with
>   genuine data mismatches under one counter, and always exited 0; the Python
>   benchmark's thread pools could silently swallow a worker exception via a
>   discarded `Future`, and `zip(..., strict=False)` could silently truncate a
>   short backend result. All fixed; `--mem-range-flags` added to the C repro.
> - **Confidence calibration (reviewer's estimate, at the time of the review):**
>   ~90% confident `READ_FIXED` genuinely fails under the tested configuration;
>   only ~35% confident the defect is specifically in amdgpu, pending a
>   `udmabuf` (non-GPU) control. **That control has since run (§5.1) and came
>   back clean in both the unpoisoned and poisoned cases, while the identical
>   AMD-VRAM-destination test fails ~100% of the time either way. Confidence
>   that the defect is specifically in the AMD exporter/GPU-visibility path,
>   not the generic dma-buf-token/NVMe-importer plumbing, is now high (~90%+).**
>   LMCache-side plumbing (allocator, locking, callback semantics, teardown
>   ordering, Rust fixed-buffer ABI) was independently assessed as sound, ~80%
>   confidence, with the main residual risk being completion-to-GPU-visibility
>   semantics rather than slot/index logic — consistent with §5's open question,
>   which the poison diagnosis (§5.1) now sharpens further.
>
> **This document's "unsafe on this stack" framing in §1 is accurate for
> `READ_FIXED` as tested, and the exporter is now localized (§5.1): the AMD
> GPU destination fails; an otherwise-identical `udmabuf` (host memory)
> destination through the same NVMe/io_uring/PRP path is clean. This is now
> filable upstream as an AMD GPU exporter / peer-DMA-visibility issue.**

---

## 1. TL;DR (conclusion first — corrected)

On this AMD ROCm + patched-kernel stack, the io_uring **DMA-BUF `READ_FIXED`
path corrupts data** — NVMe peer-to-peer DMA that **writes into** exported AMD
VRAM delivers wrong/stale bytes almost every time it's exercised in isolation.
`WRITE_FIXED` (NVMe peer-DMA **reading** VRAM) is clean. This is the opposite of
what earlier revisions of this document concluded (see the correction notice
above) — the original round-trip-based tests could not isolate which direction
was actually broken, and an indirect signal pointed the wrong way.

Established, with high confidence, from **direction-isolated** tests (§7.3) that
use plain `O_DIRECT` `pread()`/`pwrite()` — a well-established, non-dmabuf kernel
path — as ground truth on one side of each check, so each test exercises only
one dmabuf direction:

- **`READ_FIXED` is broken.** Plain `pwrite()` known-good bytes onto NVMe
  (bypassing `WRITE_FIXED` entirely) → `READ_FIXED` into VRAM → `hipMemcpy` →
  compare, **every byte of every chunk**: **2558 / 2560 corrupt (99.9%)** over a
  40-repeat, 64-chunk standalone C run.
- **`WRITE_FIXED` is clean.** GPU-fill → `WRITE_FIXED` → plain `pread()` reads
  back what's actually on disk (bypassing `READ_FIXED` entirely), every byte of
  every chunk: **0 / 2560 corrupt (0%)** over the same run.
- **Not concurrency.** All isolation checks run with exactly one op in flight —
  no concurrency is exercised by `--mode write`/`--mode read` at all — and the
  defect still manifests at ~100%. This also retroactively confirms the earlier
  concurrency=1 roundtrip finding (§7.2): it wasn't ruling out concurrency by
  accident, concurrency genuinely was never the factor.
- **Not LMCache code.** Reproduces in a standalone C program
  (`tools/repro_iou_dmabuf_p2p.c`, pure HIP + io_uring, no LMCache/Python/Rust).
- **Round-trip tests undercount.** The original `roundtrip` mode (GPU fill →
  `WRITE_FIXED` → `READ_FIXED` → compare **3 sampled bytes** per 2 MiB chunk)
  showed only 48.6% (1245/2560) corrupt in the *same run* where the full-byte
  `read-only` check showed 99.9%. A round trip including a `READ_FIXED` that
  fails 99.9% of the time should itself fail close to 99.9% of the time; the gap
  is explained by sparse sampling missing corruption that doesn't land on byte 0,
  the middle byte, or the last byte of the 2 MiB buffer. **Every "corruption
  rate" number produced by a 3-sample or coarse round-trip check in this
  investigation (including the earlier "~0.3–0.5%" LMCache-level figure, and the
  §7.1/§7.2 C-repro roundtrip numbers) should be treated as a lower bound, not
  the true rate** — see the open question in §5.

- **Localized to the AMD GPU exporter (§5.1, 2026-07-26).** The identical
  isolated `READ_FIXED` test, run through the exact same NVMe device, io_uring
  registration, and PRP transport path, but with the destination swapped from
  AMD VRAM to host memory exported via `/dev/udmabuf` (zero HIP/AMD
  participation), is **perfectly clean**: 0/2560 mismatches, with and without
  destination poisoning. The AMD-VRAM-destination case fails ~100% of the time
  under otherwise identical conditions. This rules out the generic
  dma-buf-token/nvme-pci importer path as the cause and points specifically at
  the AMD GPU exporter or its peer-DMA-visibility contract.
- **Poison diagnosis narrows the mechanism further.** With the AMD VRAM
  destination pre-filled with a sentinel byte before each `READ_FIXED`, 2554 of
  2559 mismatches (99.8%) show the destination **unchanged** — still pure
  sentinel, not stale data, not a partial write. `READ_FIXED` reports a
  full-length, error-free completion every time, but the peer DMA write does
  not appear to reach AMD VRAM. `stale_previous=0` rules out "it's just
  delayed/cached" (old data would leak through on some lag; it never does).

**Open question, not yet resolved:** the LMCache full-stack benchmark's
integrity check (`torch.equal` on the *entire* tensor, not sampled) showed only
~0.3–0.5% corruption, yet the isolated `READ_FIXED`-only check shows ~100% with
an equally full-byte comparison. Both are full-buffer checks, so sampling doesn't
explain this gap — something else about how the two tests exercise `READ_FIXED`
differs materially. Leading hypothesis and next experiment: §5.

**Action:** the dmabuf `READ_FIXED` path (NVMe writing into AMD VRAM) is unsafe
on this stack as tested; `WRITE_FIXED` is reliable, and the identical transport
through a `udmabuf` (non-AMD) destination is reliable. HIP export flags are
closed (nonzero values are invalid; the lower-level ROCr PCIe flag does not
select a different export mapping). **The defect is now localized to the AMD
GPU exporter / peer-DMA-visibility path** — this is ready to file upstream with
the C repro's four-case matrix (§5.1) as the reproduction. Remaining follow-ups
(not blocking a bug report, but useful supporting detail): `dmesg` during a
failing run, and the still-open LMCache-vs-isolated rate-gap question above.

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
| 10 | Direction asymmetry, believed (from #5 + #7). | — | writes corrupt, reads clean, both concurrent and serial | **Wrong conclusion**, reached from indirect round-trip evidence: "the failing primitive is `WRITE_FIXED`." Reversed by #11. |
| 11 | **Directly isolate each direction** (no round trip). | `repro_iou_dmabuf_p2p.c --mode all`: `write-only` = GPU-fill→`WRITE_FIXED`→plain `pread()` ground truth (bypasses `READ_FIXED`); `read-only` = plain `pwrite()` ground truth (bypasses `WRITE_FIXED`)→`READ_FIXED`→compare. Both check every byte. 40 repeats × 64 chunks. | `write-only` **0/2560** (0%); `read-only` **2558/2560** (99.9%); `roundtrip` (3-sample) only 1245/2560 (48.6%) in the *same run* | **Definitive, corrected result: `READ_FIXED` is broken, `WRITE_FIXED` is clean.** The prior belief (#10) was backwards — an artifact of inferring direction from round-trip tests instead of isolating it. See §7.3. |

Representative corrupt-chunk detail (iter, key→observed): the read returns bytes
matching **no** chunk written in that iteration ("foreign"); with the 3-point
sample (start/mid/end of the 2 MiB chunk) the corruption is frequently a wrong
**leading block** with the tail correct, or the whole chunk holding a different
chunk's value. In the raw diagnostic the chunk at **device offset 0** (chunk 0)
was deterministically wrong — an artifact of writing NVMe LBA 0; the LMCache
backend never writes offset 0 (slots start after the metadata region), so ignore
that specific chunk when reasoning about the backend.

---

## 5. Root-cause evidence chain (corrected)

1. **Direct isolation, not inference** (#11, §7.3): `write-only` (WRITE_FIXED
   alone, verified via plain `pread`) is 0/2560 corrupt; `read-only` (READ_FIXED
   alone, verified via plain `pwrite` ground truth) is 2558/2560 corrupt. Each
   check exercises exactly one dmabuf direction, so this is a direct
   measurement, not an inference from a round trip.
2. **`READ_FIXED` isolation checks run with exactly one op in flight** → not a
   concurrency/lock race in the read direction.
3. **Corrupts through raw `RawBlockDevice`-equivalent code** (the C repro talks
   to io_uring and HIP directly) → not `RawBlockCore`, not the Python backend,
   not slot/index/offset bookkeeping.
4. **Rust submission is direction-agnostic** (same `build_and_submit_sqe`, single
   worker, one ring for read and write) → the fault is below the io_uring
   submission logic, i.e. the kernel/amdgpu P2P path itself.

⇒ **A kernel/amdgpu reliability defect in NVMe→(write)→AMD-VRAM peer DMA
(`READ_FIXED`) under the io_uring dmabuf-token path.** In the isolated,
full-byte-verified standalone test this is ~99.9% of 2 MiB transfers — not a rare
event.

### Explicitly ruled out
- LMCache Python logic (deadlock aside, which was fixed).
- `RawBlockCore` slot lifecycle / header/payload offset math.
- Concurrency / locking — proven twice over: LMCache-level serialization (#9)
  and the C repro's isolation checks, which never have more than one op in
  flight (#11).
- VRAM coherence mode (coarse vs fine-grained) for the `WRITE_FIXED` direction
  (§7.1/§7.2 A/B). Not yet re-tested for `READ_FIXED` specifically with the
  isolation methodology — worth doing, see below.
- Source-fill visibility for `WRITE_FIXED` (stream-synced; CPU reads the source
  correctly, and `write-only` is 0% corrupt regardless).

### Open question: why does the LMCache-level full-tensor check show ~0.3–0.5%
### while the isolated `READ_FIXED` check shows ~99.9%?

Both are full-buffer comparisons (`torch.equal` on the whole tensor vs. a
byte-by-byte loop over the whole 2 MiB chunk), so sampling granularity does not
explain this gap — unlike the `roundtrip`-vs-`read-only` gap within the C repro
(§1), which *is* explained by sampling. Something else differs. Leading
hypothesis, **not yet tested**: the C repro's `read-only` mode reads every chunk
into the **same, single, reused 2 MiB VRAM destination buffer** (`dst`, always at
offset 0 in its slot) — 2560 consecutive `READ_FIXED` calls all targeting the
identical physical VRAM address, back-to-back with no delay before the
`hipMemcpy` that reads it. The real LMCache allocator (`DmabufGPUAllocator`)
instead hands out a **different pool offset per chunk**, cycling through the
whole GPU pool, and there is more Python/Rust scheduling latency between a
`READ_FIXED` completing and its bytes actually being consumed. If the defect is
sensitive to (a) reusing the exact same destination address repeatedly and/or
(b) how quickly the destination is read after the io_uring completion is
signaled (a possible fencing/coherency gap between "kernel says done" and "bytes
visible to a host-initiated `hipMemcpy`"), either would explain a much higher
observed rate in the tight, same-address C loop than in the real backend's more
varied, more spaced-out access pattern.

**Next experiment (not yet run):** modify `read-only` mode to (a) cycle through
multiple destination VRAM offsets instead of reusing one buffer, and separately
(b) insert an explicit `hipDeviceSynchronize()` (or a small delay) between
`READ_FIXED` completing and the `hipMemcpy` that reads it, to see if either
closes the gap toward the LMCache-level rate. This would also indicate whether
the practical impact on the real backend is smaller than the raw defect rate
suggests, or whether the backend is simply not exercising the failure mode as
persistently.

### Not yet isolated (good next experiments)
- **The rate-gap experiment above** (destination reuse / timing sensitivity).
- **GPU side vs. shared importer side:** does the same NVMe P2P read
  (`READ_FIXED`) corrupt with a `udmabuf` (host memory) destination instead of
  amdgpu VRAM? If `udmabuf` is clean and amdgpu VRAM is not, the difference is
  on the AMD export/GPU-visibility side. If `udmabuf` also corrupts, investigate
  the shared dma-buf-token/nvme-pci importer side.
- **NVIDIA control:** same workload with a CUDA-exported buffer, to see if the
  defect is AMD-specific.
- **PCIe topology / `CONFIG_PCI_P2PDMA`:** confirm whether transfers are true P2P
  or host-bounced, and whether the switch/root-port matters.
- **`dmesg` during a corrupting run:** amdgpu / dma-fence / nvme warnings.
- **Re-run the coarse-vs-fine-grained A/B (§7.1) using the `read-only` isolation
  check** rather than the round-trip methodology, now that isolation is
  available — confirm coherence mode truly doesn't matter for `READ_FIXED`
  specifically, not just inferred from the (undercounting) round-trip numbers.

### 5.1 External review findings and the decisive next matrix

An independent review (Codex) of the whole investigation, tooling, and LMCache
plumbing, conducted after §7.3's decisive result, raised six findings. Two are
substantive and change what can be claimed; the rest are tooling-quality issues
that are now fixed (commit "iou-dmabuf tools: strict accounting, PCIe export
flag, exception safety") or documentation cleanup.

1. **(High, resolved) No HIP PCIe export flag was omitted.** The proposed
   `--mem-range-flags 1` run fails with `hipErrorInvalidValue`. This is expected:
   the current HIP implementation of `hipMemGetHandleForAddressRange` explicitly
   rejects `flags != 0`. The similarly named ROCr v2 flag
   `HSA_AMD_DMABUF_MAPPING_TYPE_PCIE` only rejects a discrete, non-XGMI GPU that
   lacks Large BAR support; after that check, both flag values use the same
   `hsaKmtExportDMABufHandle` operation. It does not request a different PCIe
   mapping or cache-coherency mode. The C tool retains `--mem-range-flags` to
   make old invocations fail clearly, but accepts only `0`.
2. **(High, resolved by user-provided evidence) NVMe SGL vs. PRP was an
   unisolated second suspect.** This kernel's top feature commit adds a new
   DMA-BUF SGL descriptor path in `drivers/nvme/host/pci.c`, selected on
   SGL-capable controllers when average segment size exceeds a ~32 KiB
   threshold. Both LMCache and the standalone C repro could have been exercising
   that new path instead of (or in addition to) the older, better-established
   PRP path — "reproduces without LMCache" only places the bug below LMCache, it
   does not by itself distinguish amdgpu from new-kernel NVMe SGL plumbing.
   **Closed**: a `printk` added to the kernel by the user and exercised against
   the failing workload confirmed the transfers go through the **PRP** path.
   SGL selection is no longer a suspect variable.
3. **(Medium, fixed) The C reproducer conflated transport failures with data
   corruption.** `read_slot()`/`write_fixed_one()` did not propagate short reads
   or CQE errors as a distinguishable outcome, and `main()` always exited 0.
   Fixed: both now return `enum io_result`, `check_write_only`/`check_read_only`
   classify every check into exactly one of six disjoint buckets (ok / mismatch
   / short / error / eagain-exhausted / harness-error), the final summary reports
   all six per mode, and the exit code is nonzero on any mismatch or transport
   failure. The 2558/2560 §7.3 result predates this fix but is still credible —
   there were no CQE error messages on stderr during that run — the fix is about
   *future* runs being self-auditing, not a retraction of that number.
4. **(Medium, fixed) The Python benchmark could silently lose worker
   exceptions.** `ThreadPoolExecutor.submit()` return values (the outer futures
   wrapping `submit_slice`/`read_slice`) were discarded; an exception raised
   inside those functions would vanish, since executor shutdown waits for the
   thread but never re-raises a future nobody retrieved. `_read_phase` also used
   `zip(..., strict=False)`, which would silently truncate `results` if the
   backend returned fewer objects than requested keys. Fixed: outer futures are
   now captured and `.result()`-checked; `zip` is now `strict=True`. This does
   not retroactively invalidate positive mismatch findings from that tool (they
   used a full `torch.equal` reference compare, not sampling), but aggregate
   miss/error *rates* from runs before this fix should be treated as a lower
   bound, same caveat as round-trip sampling.
5. **(Medium, open — see the open question above) Completion-to-GPU-visibility
   is an unresolved synchronization contract, not confirmed amdgpu-specific.**
   After the NVMe CQE arrives, both LMCache and the C repro immediately expose
   the destination VRAM (LMCache to the GPU connector; the C repro via
   `hipMemcpy`). Neither establishes an explicit system-scope acquire for an
   external, third-party PCIe write. Linux's DMA-BUF documentation notes that
   DMA completion and fences do not, by themselves, guarantee all cache
   preparation is complete; AMD's position is that the *importing* API
   determines the final consistency model. This may turn out to be an
   amdgpu/exporter limitation, but today it is more accurately described as an
   unresolved NVMe/DMA-BUF/ROCm synchronization contract question — which lines
   up with this section's own rate-gap open question (same-address reuse and/or
   completion-to-visibility timing as the leading hypothesis for why the
   isolated rate is so much higher than the full-stack rate).
6. **(Low, fixed) Stale documentation.** `tools/diagnose_iou_dmabuf_coherency.py`
   still framed the problem as concurrent-write corruption with `WRITE_FIXED` as
   the broken direction; corrected. The C repro's "ROCm >= 5.6" requirement claim
   was inconsistent with AMD's own documentation (which places
   `hipMemGetHandleForAddressRange` differently across revisions, up to HIP 7.0
   in some); softened to point at the installed ROCm's own release notes instead
   of asserting a specific number.

**LMCache-side assessment (independent, from the same review):** the allocator
deadlock diagnosis and fix are technically sound; reference ownership, slot
locking, callback semantics, staging-key alignment, async tail cleanup, teardown
ordering, and the Rust fixed-buffer ABI all look coherent, and no LMCache-side
defect was found that explains the isolated `READ_FIXED` corruption. Confidence
in the LMCache first-cut plumbing: ~80%, with the main residual risk being
GPU-visible completion semantics (finding 5 above) rather than slot/index logic.
Confidence that `READ_FIXED` genuinely fails under the tested
`flags=0`/current-kernel configuration: ~90%. Confidence that this is
*specifically* an amdgpu defect (as opposed to an NVMe/DMA-BUF synchronization
contract issue or something importer-side): ~35%, pending the matrix below.

**Decisive next matrix**, in priority order:

1. ~~Add strict CQE accounting to the C reproducer.~~ *(Done.)*
2. ~~Check the proposed nonzero HIP export flag.~~ *(Done; invalid by API and
   rejected by the implementation.)*
3. ~~Distinguish PRP from SGL.~~ *(Done; the user's `printk` confirmed PRP.)*
4. ~~Run isolated `READ_FIXED` with `--read-dest gpu` and `--read-dest
   udmabuf`, first without and then with `--poison-before-read`.~~ *(Done,
   all four cases, 2026-07-26.)*
5. ~~If GPU fails while `udmabuf` passes, investigate AMD's exporter and
   completion-to-GPU-visibility contract.~~ **That is the observed result —
   see below. AMD's exporter / peer-DMA-visibility contract is the
   locus, not the generic dma-buf-token/nvme-pci importer path.**
6. If the result remains timing-sensitive, print raw/aligned/exported ranges and
   vary destination reuse. A future pattern upgrade should vary bytes within a
   chunk; the current full-buffer checker uses one distinct byte value per
   `(repeat, chunk)`. *(Superseded for the headline finding by
   `tools/repro_iou_dmabuf_minimal.c`'s per-offset-varying payload — see §7.4 —
   but still relevant for further AMD-side investigation.)*

**Full four-case matrix (2026-07-26):**

```
--read-dest gpu (no poison):
  read-only : ok=0 mismatch=2560 short=0 error=0 eagain_exhausted=0
              harness_error=0 (total=2560; isolates READ_FIXED)

--read-dest gpu --poison-before-read:
  read-only : ok=1 mismatch=2559 short=0 error=0 eagain_exhausted=0
              harness_error=0 (total=2560; isolates READ_FIXED)
  read mismatch diagnosis: unchanged_poison=2554 stale_previous=0
                            partial_expected=5 mixed_or_other=0

--read-dest udmabuf (no poison):
  read-only : ok=2560 mismatch=0 short=0 error=0 eagain_exhausted=0
              harness_error=0 (total=2560; isolates READ_FIXED)

--read-dest udmabuf --poison-before-read:
  read-only : ok=2560 mismatch=0 short=0 error=0 eagain_exhausted=0
              harness_error=0 (total=2560; isolates READ_FIXED)
```

**Decisive.** Same NVMe device, same io_uring registration, same PRP transport,
same poison-then-verify methodology, only the destination exporter changes:

- **AMD VRAM destination: fails ~100% of the time**, with or without poisoning.
  Zero transport failures in either run — every `READ_FIXED` reports a
  full-length, error-free CQE. With poisoning, **99.8% of the mismatches
  (2554/2559) are `unchanged_poison`**: the destination VRAM is *exactly* the
  pre-write sentinel byte across the full 2 MiB chunk (sample detail lines show
  `poison(0xa5)=2097152` — the entire buffer, not a partial region) — i.e.
  **the peer DMA write never became visible at the destination at all**,
  despite io_uring reporting success. `stale_previous=0` (not even one
  instance) rules out "the write is just delayed/cached" — if the write
  eventually landed on some lag, old data would leak through occasionally as
  it does; it never does here. `partial_expected=5/2560` is noise-level, ruling
  out a systematic misalignment/partial-transfer bug as the dominant mechanism.
- **`udmabuf` (host-memory) destination: perfectly clean, both with and
  without poisoning.** 0/2560 mismatches in both runs — the poison sentinel is
  reliably and completely overwritten by the correct data every single time.

Since everything else in the transport (NVMe device, io_uring dma-buf-token
registration, PRP command construction, kernel completion path) is identical
between the two runs, this rules out the generic dma-buf-token/nvme-pci
importer path as the cause and **localizes the defect specifically to the AMD
GPU exporter or its peer-DMA-visibility contract**: `READ_FIXED` completion is
disconnected from whether the peer write actually reached the destination when
that destination is AMD VRAM, but not when it is host memory through the same
kernel machinery.

Poisoning here is a diagnostic sentinel, not fault injection. Before each
`READ_FIXED`, the tool fills the destination with a byte value different from
both the expected disk byte and the previous read's byte. It then classifies
every destination byte after a successful full-length CQE:

- all poison: the peer write did not become visible at the destination;
- all previous: stale contents remained;
- expected plus poison/previous/other: only part of the destination changed;
- neither expected, poison, nor previous: wrong-address or other corruption.

The `udmabuf` case wraps CPU writes and reads in `DMA_BUF_IOCTL_SYNC`; the GPU
case uses `hipMemset`/`hipDeviceSynchronize` and synchronous device-to-host
`hipMemcpy`. Poisoning can itself perturb cache state or timing, so compare
poisoned and unpoisoned runs. A poisoned run becoming clean is evidence that the
extra synchronization or destination touch changes visibility, not proof that
the underlying path is correct.

---

## 6. Tools built (all in-repo)

| Path | Purpose |
|---|---|
| `benchmarks/storage_backend_io/iou_dmabuf_io_benchmark.py` | End-to-end backend write/read + integrity. Knobs: `--num-ops`, `--iters`/`--target-gib`, `--concurrency`, `--write-concurrency`, `--read-concurrency`, `--disk-io-threads`, `--source {gpu,cpu}`, `--verify-integrity`. Globally-seeded per-chunk patterns; classifies each mismatch as swap vs foreign; recycles slots per iter so traffic ≫ footprint. |
| `tools/probe_gpu_dmabuf_export.py` | Confirms a torch GPU pointer can be exported as a dmabuf fd (HIP or CUDA). The P1 go/no-go. |
| `tools/probe_iou_dmabuf_e2e.py` | Single dmabuf `WRITE_FIXED`/`READ_FIXED` round-trip through the real `RawBlockDevice`, with a watchdog. |
| `tools/diagnose_iou_dmabuf_coherency.py` | Raw-`RawBlockDevice` reproducer. Trials coarse vs fine-grained VRAM, and concurrent vs serialized writes. No `RawBlockCore`/backend. This is what isolated the bug below LMCache. |
| `tools/repro_iou_dmabuf_p2p.c` | **Standalone C** repro (HIP + io_uring, no Python, no LMCache) — for kernel/driver maintainers. See §7. Supports strict per-check accounting, an AMD VRAM vs. `udmabuf` read-destination control, and optional destination poisoning with full-byte classification. |
| `tools/repro_iou_dmabuf_minimal.c` | **Minimal standalone C** repro (400 lines, one op per direction per exporter) — for pasting directly into a kernel/NVMe/AMD bug report. See §7.4. |
| `tools/repro_iou_dmabuf_basic.c` | **Stage-1 standalone C** repro — 4 operations (store, `WRITE_FIXED`, `READ_FIXED`, load), each checked against its own independent ground truth. No byte-level poison classification, no `DMA_BUF_SYNC`. Says WHICH of the 4 operations is broken. See §7.5. |

### How to reproduce the corruption

Smallest that still isolates direction ("stage 1", see §7.5) — 4 operations,
each independently verified:
```bash
./repro_iou_dmabuf_basic /dev/nvme0n1 $((4<<30))
# expect: udmabuf STORE/WRITE_FIXED/READ_FIXED all PASS;
# amdgpu STORE/WRITE_FIXED PASS, READ_FIXED FAIL
```

Smallest reproducer that also isolates direction (see §7.4) — one
`WRITE_FIXED` and one `READ_FIXED` per exporter, no repeats needed:
```bash
./repro_iou_dmabuf_minimal /dev/nvme0n1 $((4<<30))
# expect: udmabuf WRITE_FIXED/READ_FIXED PASS/PASS; amdgpu WRITE_FIXED PASS,
# READ_FIXED FAIL with poison largely/completely unchanged
```

Isolated, decisive, and configurable (see §7.3). Run this four-case matrix
against a scratch range on the NVMe namespace:
```bash
./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --mode read --device-offset $((4<<30))

./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --mode read \
  --poison-before-read --device-offset $((4<<30))

./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --mode read \
  --read-dest udmabuf --device-offset $((4<<30))

./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --mode read \
  --read-dest udmabuf --poison-before-read --device-offset $((4<<30))
```

Use the same `--num-chunks`, `--repeat`, `--chunk-bytes`, and device offset in
all four runs. **Already run (§5.1): GPU fails, `udmabuf` is clean, both with
and without poisoning** — localized to AMD export/GPU visibility, not shared
io_uring DMA-BUF-token or NVMe importer plumbing. See §5.1 for the full result
and poison interpretation.

Through the full LMCache backend (round-trip; undercounts the true rate, §7.3,
but shows the defect is reachable end-to-end):
```bash
python benchmarks/storage_backend_io/iou_dmabuf_io_benchmark.py \
  --device-path /dev/nvme0n1 --num-ops 128 --iters 25 --verify-integrity \
  --write-concurrency 4 --read-concurrency 1 --disk-io-threads 1
# expect integrity_passed: false, a handful of "foreign/corrupt" mismatches
```

---

## 7. Standalone C reproducer

`tools/repro_iou_dmabuf_p2p.c` removes every LMCache/Python/Rust layer. It:

1. allocates and exports GPU source/destination buffers for the selected modes,
   or creates a host-memory dma-buf through `/dev/udmabuf` for
   `--mode read --read-dest udmabuf`;
2. exports GPU allocations via `hipMemGetHandleForAddressRange(..., flags=0)`;
3. opens the NVMe device `O_DIRECT`, sets up io_uring, registers a sparse buffer
   table and installs the dmabufs with `IORING_REGISTER_BUFFERS_UPDATE` +
   `IO_REGBUF_TYPE_DMABUF` bound to the NVMe fd;
4. fills N source chunks with distinct patterns (`hipMemset` + `hipDeviceSynchronize`);
5. issues N `WRITE_FIXED` (configurable in-flight depth, `-EAGAIN` reissue);
6. reads each slot back with `READ_FIXED`, copies the destination to host, and
   verifies every byte;
7. optionally poisons the read destination before each operation and reports
   whether mismatches contain expected, poison, previous, or other bytes.

Build & run instructions are in the file header. Vary `--concurrency` (1 still
corrupts), `--finegrained`, `--repeat`, `--device-offset` (avoid LBA 0 for a
non-destructive scratch region). See §8 for the destructive-write warning.

Build (runtime-only ROCm — no HIP dev headers/hipcc needed; the program declares
the HIP functions it uses and links `libamdhip64`):
```
cc -O2 -o repro_iou_dmabuf_p2p tools/repro_iou_dmabuf_p2p.c -luring \
   -L/opt/rocm/lib -lamdhip64
```

> §7.1 and §7.2 below use the original round-trip methodology (WRITE_FIXED then
> READ_FIXED, sampled or later full-byte comparison) and predate the direction
> isolation added for §7.3. Their raw corruption-rate numbers are real
> measurements but their **direction attribution ("WRITE_FIXED is broken") is
> superseded and wrong** — see the correction notice at the top of this
> document and §7.3.

### 7.1 Confirmed run (2026-07-25, superseded direction attribution)

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
- **Still to capture:** `dmesg` during a corrupting run. See §7.2 for the
  concurrency=1 and fine-grained confirmations, captured the same session.

### 7.2 Concurrency=1 and fine-grained confirmations (2026-07-25, superseded direction attribution)

```
./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --concurrency 1 --repeat 40 \
    --device-offset $((4<<30))
# mem=coarse
# corrupt 1351 / 2560 chunks (64 chunks x 40 repeats)   -- 52.8%
# => REPRODUCED

./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --concurrency 8 --repeat 40 \
    --finegrained
# mem=fine-grained, dev_off=0
# corrupt 1298 / 2560 chunks (64 chunks x 40 repeats)   -- 50.7%
# => REPRODUCED
```

**`--concurrency 1` is the decisive result.** At concurrency 1, `run_writes`
submits exactly one `WRITE_FIXED`, waits for its completion, and only then
submits the next — there is never more than one write in flight, no
interleaving, nothing for a software race to corrupt. It still corrupts
**52.8%** of chunks. This conclusively rules out concurrency (LMCache's,
the Rust worker's, or the raw io_uring submission pattern's) as a factor:

> A single, standalone `IORING_OP_WRITE_FIXED` against a registered
> amdgpu-exported dma-buf, issued one at a time with no other I/O in flight,
> corrupts roughly half the time on this kernel/ROCm/GPU stack.

`--finegrained` at concurrency 8 (50.7%) statistically matches the earlier
coarse-memory concurrency-8 run (1298/2560 — the same count), confirming
coherency mode (coarse vs fine-grained VRAM) does not affect the defect
either, consistent with the earlier `diagnose_iou_dmabuf_coherency.py` finding.

Combined with §7.1, every independent variable that could plausibly explain the
corruption *except direction* — concurrency, VRAM coherency mode, LMCache
software, RawBlockCore, the Rust worker — was tested and ruled out at this
point. What §7.2 did **not** do (and what made its direction conclusion wrong)
is test `WRITE_FIXED` and `READ_FIXED` **independently of each other** — it only
ever ran round trips, so "writes corrupt, reads clean" was an inference from
which knob (`--write-concurrency` vs `--read-concurrency`) changed the round-trip
outcome, not a direct measurement of either leg alone. §7.3 fixes that.

### 7.3 Direction isolation — the corrected, decisive result (2026-07-25)

Extended `repro_iou_dmabuf_p2p.c` with two checks that each exercise **only one**
dmabuf direction, using plain `O_DIRECT` `pread()`/`pwrite()` (an established,
non-dmabuf kernel path) as ground truth on the side not under test:

- **`--mode write`**: GPU-fill a chunk → `WRITE_FIXED` (VRAM→NVMe) → verify with
  a **plain `pread()`** reading back what actually landed on disk. `READ_FIXED`
  never runs. Checks every byte of the chunk (not sampled).
- **`--mode read`**: a **plain `pwrite()`** puts a known-good pattern directly on
  NVMe (`WRITE_FIXED` never runs) → `READ_FIXED` (NVMe→VRAM) → `hipMemcpy` →
  compare every byte.
- Both isolation checks always run one operation at a time; concurrency is not
  exercised (already ruled out separately, §4 #9).

```
./repro_iou_dmabuf_p2p --device /dev/nvme0n1 --device-offset $((4<<30))
# mode=roundtrip,write,read,  (all three modes; default)
# 64 chunks x 40 repeats = 2560 checks per mode

=== results (64 chunks x 40 repeats = 2560 checks per mode) ===
  roundtrip  corrupt:  1245 / 2560  (WRITE_FIXED then READ_FIXED; does not isolate direction)
  write-only corrupt:     0 / 2560  (WRITE_FIXED verified via plain pread -- isolates WRITE_FIXED)
  read-only  corrupt:  2558 / 2560  (plain pwrite verified via READ_FIXED -- isolates READ_FIXED)

=== interpretation ===
  READ_FIXED is broken (NVMe peer-DMA WRITING VRAM). WRITE_FIXED is clean.
```

**This is the decisive, corrected result.** `write-only` is 0/2560 across a
full-byte-verified, 2560-check run — as clean as a result can be.  `read-only` is
2558/2560 (99.9%) — as broken as a result can be. Neither leaves room for
ambiguity, and because each check bypasses the *other* direction entirely (using
the well-established plain `pread`/`pwrite` path as ground truth), there is no
way for a bug in one direction to be misattributed to the other.

**Why `roundtrip` (1245/2560, 48.6%) doesn't match `read-only` (99.9%) in the
same run:** `roundtrip` only compares 3 sampled bytes (start/middle/end) per 2
MiB chunk, inherited from the original tool design. A round trip that includes a
`READ_FIXED` broken 99.9% of the time should itself fail close to 99.9% of the
time — the fact that it only catches 48.6% means the corruption frequently does
not land on one of the 3 sampled positions. **Every corruption-rate figure
produced by a sampled or round-trip check in this investigation — including the
original "~0.3–0.5%" LMCache figure and the §7.1/§7.2 numbers — is a lower bound
on the true rate, not the true rate itself**, though the LMCache figure uses a
full-tensor `torch.equal` check, so sampling alone does not explain why it is so
much lower than the isolated `read-only` rate; see the open question in §5.

### 7.4 Minimal reproducer for sharing with maintainers

`tools/repro_iou_dmabuf_minimal.c` (400 lines) is a compact, purpose-built
version of the §7.3/§5.1 isolation methodology, intended for pasting into a
kernel/NVMe/AMD bug report rather than for further investigation — it does not
need 40 repeats or 64 chunks to show the failure. Per exporter (`udmabuf` then
AMD VRAM), it runs exactly one isolated `WRITE_FIXED` and one isolated
`READ_FIXED`:

- **`WRITE_FIXED`**: fill the exporter's buffer with a pattern, `WRITE_FIXED`
  it to NVMe, then verify with a plain `pread()` (ground truth, bypasses
  `READ_FIXED`).
- **`READ_FIXED`**: plain `pwrite()` a known pattern directly to NVMe
  (bypasses `WRITE_FIXED`), pre-fill the destination with a **bitwise
  complement** of that pattern (`expected[i] ^ 0xFF`, not a single sentinel
  byte — guarantees every byte differs from `expected` at every bit position,
  a stronger poison than a fixed sentinel value), issue `READ_FIXED`, then
  classify every byte as `expected` / `poison` (unchanged) / `other`.
- **Pattern**: unlike §7.1–§7.3's one-byte-per-chunk fill, the payload varies
  at every 8-byte word (`seed ^ (i * 0x9e3779b97f4a7c15)`, a golden-ratio
  multiplicative hash) across the full 2 MiB buffer. This additionally detects
  page-granularity reordering (e.g. a PRP page-list construction bug that
  moves whole 4 KiB pages without changing their contents) that a
  uniform-byte-per-chunk pattern cannot distinguish from a correct transfer.
  Not indicated by the evidence so far (the AMD failure is "nothing arrived,"
  not "arrived reordered"), but a reproducer meant for kernel/NVMe maintainers
  should rule it out independently rather than rely on the larger tool's
  weaker pattern.

**Build note:** originally this file included `<liburing.h>` and used
`io_uring_regbuf_desc`/`IO_REGBUF_TYPE_DMABUF` directly from the header,
which required `-I` pointed at the patched liburing checkout. It was changed
to match `repro_iou_dmabuf_p2p.c` (§7.1–§7.3): it now redeclares its own
minimal, `repro_`-prefixed copies of the registration structs
(`repro_regbuf_desc`, `repro_rsrc_register`, `repro_rsrc_update2`,
`REPRO_REGBUF_TYPE_DMABUF`, `REPRO_RSRC_UPDATE_EXTENDED`) and issues the
`IORING_REGISTER_BUFFERS2`/`IORING_REGISTER_BUFFERS_UPDATE` calls via a raw
`syscall(__NR_io_uring_register, ...)` instead of relying on the patched
header's types or the `io_uring_register_buffers_sparse()` liburing helper.
Everything else (`io_uring_queue_init`, `io_uring_get_sqe`,
`io_uring_prep_read_fixed`/`prep_write_fixed`, `io_uring_submit`,
`io_uring_wait_cqe`, `io_uring_queue_exit`) is stock liburing API. Net effect:
this file now builds against **any ordinary, unpatched `liburing-dev`** — no
`-I` override needed. The kernel under test still needs
`CONFIG_DMABUF_TOKEN=y`; only the build-time header dependency was removed.
Verified by compiling against a real pre-dma-buf-patch liburing header tree
(checked out at the commit immediately before `a1f1f8a1` in the local
liburing clone) with zero errors.

Exit code is 1 only if the exact observed signature reproduces (`udmabuf`
write+read PASS, AMD VRAM write PASS + read FAIL), 2 on any harness/transport
error, 0 otherwise (including "everything passed" or an unrelated failure
pattern) — usable as a regression check once this is filed and eventually
fixed. Verified: strict-warning build (`-Wall -Wextra -Wswitch-enum
-Wformat=2`) and `gcc -fanalyzer`, both zero warnings, against both the
patched and a genuinely stock liburing header tree.

### 7.5 Basic ("stage 1") per-phase reproducer

`tools/repro_iou_dmabuf_basic.c` (~330 lines) is smaller than §7.4's minimal
repro and comes first in the investigation narrative. Per exporter (`udmabuf`
then AMD VRAM), it runs 4 operations, each checked against its own
independent ground truth (not just an overall round-trip `memcmp`):

1. **STORE**: fill the buffer with a known repeated-byte pattern (`0xAB`),
   then read it straight back through the exporter itself (no io_uring) —
   confirms the exporter's own set/get path works before layering io_uring
   on top of it.
2. **`WRITE_FIXED`**: buffer → NVMe, checked with a plain `pread()` ground
   truth on the device. Independent of phase 3 — `READ_FIXED` never runs
   here, so this result can't be an artifact of the read direction.
3. **`READ_FIXED`**: NVMe → buffer, seeded by a plain `pwrite()` ground truth
   on the device (independent of phase 2's `WRITE_FIXED` result) and
   preceded by zeroing the buffer, so a no-op `READ_FIXED` can't hide behind
   phase 1's leftover correct data.

Each phase prints its own `STORE`/`WRITE_FIXED`/`READ_FIXED` PASS/FAIL/ERROR
line, so unlike an earlier round-trip-only version of this file, a single run
now tells you not just THAT something is broken but WHICH of the 4 operations
is broken — this is exactly the "how do we know if read failed or write
failed" question from §5.1's investigation, answered at the smallest possible
scale.

**Deliberately dropped versus §7.4:** byte-level poison classification (plain
`memcmp` per phase here, not `expected`/`poison`/`other` counts), and
`DMA_BUF_SYNC`.

**Why no `DMA_BUF_SYNC`:** that ioctl only matters for CPU access to a dmabuf
through its own `mmap()` — a different consumption path than what this test
exercises. `udmabuf`'s backing store is an ordinary `memfd`; this repro reads
and writes it with plain `pread()`/`pwrite()` on the memfd directly (same
physical pages, no mmap involved), so the mmap coherency contract never comes
up. AMD VRAM verification goes through `hipMemcpy`, which doesn't touch the
dmabuf mmap path either. The `WRITE_FIXED`/`READ_FIXED` DMA itself is
mediated by the kernel dma-buf attachment machinery — no userspace ioctl is
needed for that in either exporter. `DMA_BUF_SYNC` only appears in §7.4
because that repro chooses to verify udmabuf contents via `mmap()` instead;
it does not test the WRITE_FIXED/READ_FIXED path any harder.

Because phases 2 and 3 now go directly against the `O_DIRECT` NVMe fd for
their ground-truth `pread`/`pwrite`, the host pattern/zero/actual buffers
must be page-aligned (`posix_memalign`), unlike the very first version of
this file, which only ever touched `hipMemcpy` or plain `memfd` I/O and had
no such requirement.

**Prepared for sharing with kernel/NVMe/AMD maintainers**, so it was cleaned
up: no mention of this debug log or the sibling `repro_iou_dmabuf_p2p.c` /
`repro_iou_dmabuf_minimal.c` tools, phase-by-phase inline comments condensed
into the single top-of-file summary, and the EAGAIN retry loop dropped from
`fixed_io()` (a single submit + wait is enough for this file's purpose; the
retry loop matters for `repro_iou_dmabuf_p2p.c`'s longer stress runs, not
here).

**Reverted a reversal:** an intermediate version of this file switched to
using `<liburing.h>`'s real `io_uring_regbuf_desc`/`IO_REGBUF_TYPE_DMABUF`
directly, reasoning that maintainers implementing this feature would already
have the patched liburing checked out. Wrong assumption — the intended
audience is broader than the people actively writing the patch (NVMe/AMD
folks reviewing or triaging it may have nothing but a stock liburing, or
nothing at all installed), and requiring the exact in-review branch
(`isilence/liburing.git` @ `rw-dmabuf-tests-v3`, as of this writing — several
*other* branches on that same remote, e.g. `dmabuf-rw`/`regbuf-import`/
`regbuf-import2`/`zcrx-dmabuf`, are earlier iterations with incompatible
struct layouts) would block them from building it at all.

Checked what's actually new here versus what's been in liburing for years:
`struct io_uring_rsrc_update2`, `IORING_REGISTER_BUFFERS_UPDATE`,
`io_uring_register()`, and `io_uring_register_buffers_sparse()` all predate
this patch series by several years (confirmed present in the pre-dma-buf-patch
liburing snapshot used earlier in this doc) — only `io_uring_regbuf_desc`,
`IO_REGBUF_TYPE_DMABUF`, and `IORING_RSRC_UPDATE_EXTENDED` are genuinely new
and unmerged. So the fix isn't "redeclare the whole ABI" (§7.1/§7.4's
approach) — it's "redeclare just that one struct and its two constants,
under different names so they can't collide with whatever the reader's
liburing does or doesn't already define, and use real liburing calls for
everything else." Net result: `repro_iou_dmabuf_basic.c` now builds against
*any* liburing new enough to have fixed-buffer registration support (years
old, essentially universal) — not just the specific in-review branch.
Verified: strict-warning build (`-Wall -Wextra -Wswitch-enum -Wformat=2`) and
`gcc -fanalyzer`, both zero warnings, against both a genuinely stock
(pre-dma-buf-patch) liburing header tree and the patched
`rw-dmabuf-tests-v3` tree.

Expected output on this hardware:
```
udmabuf STORE      : PASS
udmabuf WRITE_FIXED: PASS
udmabuf READ_FIXED : PASS
amdgpu  STORE      : PASS
amdgpu  WRITE_FIXED: PASS
amdgpu  READ_FIXED : FAIL
```

---

## 8. Cautions

- **`WRITE_FIXED` in these tools writes to the raw NVMe device** at the chosen
  offsets — destructive. Point them at a scratch namespace.
- `READ_FIXED` itself does not modify storage, but `--mode read` first uses a
  plain `pwrite()` to place ground-truth data on NVMe. Every mode in this tool
  therefore writes to the selected raw-device range.
- The C repro and `diagnose_iou_dmabuf_coherency.py` intentionally use device
  offset 0 by default for simplicity; pass a non-zero aligned `--device-offset`
  to avoid clobbering a partition table, and to sidestep the LBA-0 artifact noted
  in §4.

---

## 9. One-paragraph handoff

The LMCache io_uring DMA-BUF backend is functionally complete on the software
side (export, registration, slot lifecycle, write path, and teardown reviewed at
~80% confidence, §5.1) and is **not the cause of the corruption**. A standalone
C program, with no LMCache/Python/Rust, isolates a `READ_FIXED` reliability
failure — NVMe peer DMA writing into exported AMD VRAM — and **localizes it
specifically to the AMD GPU exporter, not shared kernel/NVMe plumbing**: the
identical isolated `READ_FIXED` test through the same NVMe device, io_uring
registration, and PRP transport, with the destination swapped from AMD VRAM to
host memory via `/dev/udmabuf`, is perfectly clean (0/2560 mismatches, with and
without destination poisoning), while the AMD-VRAM case fails ~100% of the time
either way (§5.1, 2026-07-26). Poisoning the destination before each read shows
**99.8% of AMD-VRAM mismatches are `unchanged_poison`** — the destination is
still exactly the pre-write sentinel, not stale data, not a partial write —
meaning `READ_FIXED` reports a full-length, error-free completion while the
peer DMA write does not appear to reach AMD VRAM. Concurrency is not involved
(isolation checks never have more than one op in flight), a kernel `printk`
confirms the failing transfer uses PRP rather than SGL, and the HIP export flag
`hipMemRangeFlagDmaBufMappingTypePcie=1` is confirmed (by reading the actual
upstream ROCm CLR/ROCr source) to be rejected outright by current HIP and to
select no alternate mapping mode at the ROCr level even where accepted — neither
is a live variable. **This is now filable upstream as an AMD GPU exporter /
peer-DMA-visibility issue**, with `tools/repro_iou_dmabuf_minimal.c` (§7.4) as a
compact, one-op-per-direction reproducer suitable for pasting directly into a
bug report. The one remaining open, non-blocking question is why the full
LMCache integrity benchmark reports only ~0.3–0.5% mismatches versus ~100% in
the isolated read test despite both checking full buffers; destination reuse
and completion-to-visibility timing remain the leading hypothesis, untested.
