# io_uring DMA-BUF backend: AMD build and validation quick reference

This is the exact recipe used to build and validate the
`feat/iou-dmabuf-backend-dev` branch (the io_uring DMA-BUF storage backend,
ported onto upstream `dev` at `a2fd93e1`) on an AMD ROCm machine, and the
results. Design and contracts are in
[iou_dmabuf_backend.md](iou_dmabuf_backend.md).

**Summary:** the branch builds from scratch on ROCm 7.14, all new unit tests
pass, the existing upstream raw-block / storage-manager / local-CPU /
local-disk tests show no regressions, and the real GPU-export and
NVMe io_uring DMA-BUF paths work on hardware. The only failures observed are
(1) an upstream test whose timeout is too short for this machine (not caused
by this branch) and (2) the known, pre-existing AMD exporter data-integrity
issue under sustained load (see [Results](#results)).

## Machine

| Item | Value |
|---|---|
| GPU | AMD Radeon AI PRO R9700 (RDNA4, `gfx1201`) |
| ROCm (system, `/opt/rocm`) | 7.14.0 (`/opt/rocm/.info/version`); HIP `7.13.99004` (`hipcc --version`) |
| PyTorch | `2.11.0+rocm7.13.0` (`torch.version.hip` = `7.13.99004`), gfx120X wheel index |
| Python | 3.12.3 (venv) |
| Kernel | `7.3.0-rc3+`, patched with the io_uring DMA-BUF series (`CONFIG_DMABUF_TOKEN`) |
| NVMe | `/dev/nvme0n1` (block) / `/dev/ng0n1` (char), 512-byte LBA |
| LMCache build | `0.5.3.dev427` (editable) |

## Prerequisites

- A kernel with the io_uring DMA-BUF patches (`CONFIG_DMABUF_TOKEN=y`) booted.
- The in-tree `amdgpu` driver of that kernel, not an `amdgpu-dkms` package
  installed by ROCm (which would replace it):

  ```bash
  uname -r
  grep CONFIG_DMABUF_TOKEN /boot/config-$(uname -r)
  modinfo -n amdgpu      # must NOT be under .../updates/dkms/
  ```

- ROCm installed under `/opt/rocm` with development files (the build links
  against `/opt/rocm/lib/libamdhip64.so`).
- Rust toolchain (`cargo`) for the `rust/raw_block` crate.
- Root (or `disk` group) to open `/dev/nvme*` / `/dev/ng*`.

## 1. Virtual environment

```bash
python3 -m venv ~/venv-rocm-new
source ~/venv-rocm-new/bin/activate
pip install -U pip
cd <LMCache checkout>          # on feat/iou-dmabuf-backend-dev
pip install -r requirements/build.txt
```

## 2. PyTorch for ROCm (gfx120X)

ROCm 7.10+ ships its GPU libraries as per-architecture pip packages. Use the
**gfx120X family index**; the generic index pulls device libraries for every
AMD architecture (tens of large downloads).

```bash
pip install --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ torch torchvision
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.11.0+rocm7.13.0 7.13.99004 True AMD Radeon AI PRO R9700
```

Do **not** use `--no-deps` here: the wheel depends on the ROCm runtime
packages it installs into the venv.

## 3. Check the toolchain matches torch

```bash
cat /opt/rocm/.info/version            # 7.14.0
/opt/rocm/bin/hipcc --version | head -3   # HIP version: 7.13.99004 (same as torch.version.hip)
```

## 4. Build LMCache (C++/HIP extensions)

Install runtime dependencies without letting pip touch torch, then build with
the **system** `hipcc`:

```bash
grep -v '^torch' requirements/common.txt > /tmp/req-notorch.txt
pip install -r /tmp/req-notorch.txt -r requirements/rocm_core.txt
python -c "import torch; print(torch.__version__)"   # must still be +rocm7.13.0

rm -rf build && find lmcache -name '*.so' -delete
PYTORCH_ROCM_ARCH=gfx1201 TORCH_DONT_CHECK_COMPILER_ABI=1 \
  CXX=/opt/rocm/bin/hipcc ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm BUILD_WITH_HIP=1 \
  pip install -e . --no-build-isolation --no-deps

python -c "import lmcache.c_ops; print('c_ops ok')"
```

Notes:

- `CXX` must be `/opt/rocm/bin/hipcc`. A bare `hipcc` resolves to the venv's
  copy installed by the torch wheel, whose runtime-only package has
  `libamdhip64.so.7` but not the unversioned `libamdhip64.so`, so linking fails
  with `ld.lld: cannot open .../_rocm_sdk_core/lib/libamdhip64.so`.
- `--no-deps` on the LMCache install prevents pip from replacing the ROCm
  torch with a PyPI build.
- `PYTORCH_ROCM_ARCH=gfx1201` limits the HIP build to this GPU; the build
  takes several minutes (single GPU kernel files take minutes each at `-O3`).

## 5. Rust raw-block extension

```bash
pip install maturin
(cd rust/raw_block && maturin develop --release)
python -c "from lmcache_rust_raw_block_io import RawBlockDevice as R; print(R.probe_dmabuf_support())"   # True
python -c "import torch; print(torch.__version__)"   # still +rocm7.13.0
```

`probe_dmabuf_support()` returns `True` only on a kernel with
`CONFIG_DMABUF_TOKEN`; the backend is silently skipped otherwise.

Rust unit tests (as upstream CI runs them):

```bash
cargo test --manifest-path rust/raw_block/Cargo.toml --no-default-features
```

## 6. Unit and regression tests

Run from the repository root.

```bash
pip install -r requirements/test.txt

# New tests for this branch
pytest -q tests/v1/storage_backend/test_iou_dmabuf_backend.py \
          tests/v1/storage_backend/test_raw_block_core.py

# Existing upstream tests for code this branch modifies
pytest -q tests/v1/storage_backend/test_raw_block_device.py \
          tests/v1/storage_backend/test_raw_block_key_codec.py \
          tests/v1/storage_backend/test_raw_block_uring_cmd.py \
          tests/v1/storage_backend/test_rust_raw_block_backend.py \
          tests/v1/storage_backend/test_storage_manager.py \
          tests/v1/storage_backend/test_local_cpu_backend.py \
          tests/v1/storage_backend/test_local_disk_backend.py
```

The unit tests use fake NVMe devices and GPU allocators; they need neither
vLLM nor a real block device.

## 7. Hardware checks

`--write` and the benchmark **overwrite data** on the target device at the
offsets they use; point them at a scratch namespace.

```bash
# GPU memory -> DMA-BUF export, exactly as the backend allocates it
python tools/probe_gpu_dmabuf_export.py --size-mib 960

# io_uring DMA-BUF registration + READ_FIXED, then WRITE_FIXED
python tools/probe_iou_dmabuf_e2e.py --device-path /dev/nvme0n1
python tools/probe_iou_dmabuf_e2e.py --device-path /dev/nvme0n1 --write

# Full backend put/get with byte-for-byte verification
python benchmarks/storage_backend_io/iou_dmabuf_io_benchmark.py \
  --device-path /dev/nvme0n1 --num-ops 128 --concurrency 4 --verify-integrity

# Sustained load (~50 GiB of traffic)
python benchmarks/storage_backend_io/iou_dmabuf_io_benchmark.py \
  --device-path /dev/nvme0n1 --num-ops 128 --concurrency 8 --target-gib 50 --verify-integrity
```

## Results

Validated 2026-09-28 on the machine above.

| Check | Result |
|---|---|
| LMCache HIP build, `import lmcache.c_ops` | OK |
| Rust crate build, `probe_dmabuf_support()` | OK, `True` |
| `test_iou_dmabuf_backend.py` | 28 passed |
| `test_raw_block_core.py` | all passed |
| Upstream regression files (step 6, second command) | 140 passed, 2 skipped, 1 failed (see below) |
| `probe_gpu_dmabuf_export.py` | OK |
| `probe_iou_dmabuf_e2e.py` (READ_FIXED and WRITE_FIXED) | OK |
| Integrity benchmark | Occasional mismatches, same as before this port (known AMD issue, below) |

**`test_uring_cmd_middle_build_error_does_not_hang` fails on this machine,
independent of this branch.** The test runs its probe in a `spawn` child
process with a 10 s timeout, and the child re-imports the test module, which
imports `torch` and `lmcache`. On this ROCm machine that import alone takes
~18.4 s, so the child is killed before the probe starts (0/3 runs passed).
Run standalone, the same probe passes against both this branch's Rust
extension and upstream `dev`'s (`results=[True, False, True]`, with the
expected LBA-alignment error on the middle read).

**Integrity mismatches (known AMD issue, not caused by this branch).** A
small fraction of chunks can read back with foreign/corrupt data; short runs
often verify cleanly, and the failures show up under sustained load. On this
machine with ROCm 7.14, one sustained run on the pre-port branch saw 5 of
4,992 chunks (about 0.1%) over 39 iterations at concurrency 8, and this branch
shows the same behavior. The symptom predates this work: it was first seen on
ROCm 7.2, where a standalone C reproducer that bypasses LMCache also showed
it, and that reproducer does not fail with an NVIDIA A100 exporter. It is
tracked as an AMD GPU exporter / kernel issue.
