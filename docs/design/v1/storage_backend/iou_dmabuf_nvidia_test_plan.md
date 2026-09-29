# io_uring DMA-BUF backend: NVIDIA (RTX 5090) test plan

Step-by-step plan for testing the io_uring DMA-BUF storage backend on a shared
NVIDIA machine (RTX 5090, Blackwell, compute capability 12.0) without touching
the host's existing LMCache build or Python environment. Design and contracts
are in [iou_dmabuf_backend.md](iou_dmabuf_backend.md); the AMD equivalent is
[iou_dmabuf_amd_validation.md](iou_dmabuf_amd_validation.md) on the AMD
validation branch.

Everything that can be isolated lives under one removable directory
(`$HOME/anuj-iou-dmabuf-test`): the clone, the venv, the Rust toolchain and all
caches. The kernel, NVIDIA driver, GPU and NVMe devices are host-wide and
cannot be isolated by a venv.

**Do not run any NVMe test (including the regression tests' device-backed
cases), NVMe write or benchmark until the machine owner identifies a
disposable namespace.**

## What the backend needs on NVIDIA

- `exporter: auto` selects `cuda_pool`: the pool is allocated with
  `torch.empty(..., device="cuda")`, page-aligned, split into slabs of at most
  1 GiB, and each slab is exported with `cuMemGetHandleForAddressRange()`.
  Only `cuda_pool` is implemented; the `cuda_vmm` fallback is deferred, so if
  the PyTorch allocation cannot be exported the backend fails to initialize.
- NVIDIA requires `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` to be true before
  requesting a DMA-BUF fd, and page-aligned pointer and size.
- The NVIDIA exporter maps VRAM through BAR1. There is no generic
  system-memory fallback in the io_uring/NVMe importer: if the NVMe cannot reach
  the GPU peer-to-peer, expect attach, mapping, registration or I/O to fail.
- RTX 5090 needs CUDA 12.8+, a PyTorch build with `sm_120`, and NVIDIA's open
  kernel modules.
- The NVMe path needs a kernel with the io_uring DMA-BUF series
  (`CONFIG_DMABUF_TOKEN`).

## Stage 0: isolated workspace

```bash
export TEST_ROOT="$HOME/anuj-iou-dmabuf-test"

mkdir -p "$TEST_ROOT"
python3 -m venv "$TEST_ROOT/venv"
```

Create the reusable activation file:

```bash
cat > "$TEST_ROOT/activate-test-env" <<'EOF'
export TEST_ROOT="$HOME/anuj-iou-dmabuf-test"

source "$TEST_ROOT/venv/bin/activate"

export RUSTUP_HOME="$TEST_ROOT/rustup"
export CARGO_HOME="$TEST_ROOT/cargo"
export PATH="$CARGO_HOME/bin:$PATH"

export XDG_CACHE_HOME="$TEST_ROOT/cache"
export PIP_CACHE_DIR="$TEST_ROOT/cache/pip"
export CUDA_CACHE_PATH="$TEST_ROOT/cache/cuda"
export TORCH_EXTENSIONS_DIR="$TEST_ROOT/cache/torch-extensions"
export TRITON_CACHE_DIR="$TEST_ROOT/cache/triton"

# Keep device-backed upstream tests away from real disks unless explicitly
# pointed at an approved namespace (they default to /dev/nvme0n1 and
# /dev/ng0n1 and skip when the path does not exist).
export LMCACHE_TEST_BLOCK_DEVICE=/nonexistent
export LMCACHE_TEST_CHAR_DEVICE=/nonexistent

mkdir -p "$TEST_ROOT/cache"
EOF
```

Activate it in every shell (use the absolute path; `TEST_ROOT` is not set in a
fresh shell):

```bash
source "$HOME/anuj-iou-dmabuf-test/activate-test-env"
```

Confirm isolation; all paths must be under `$HOME/anuj-iou-dmabuf-test/venv`:

```bash
which python
which pip
python -c "import sys; print(sys.executable); print(sys.prefix)"

python -m pip install --upgrade pip setuptools wheel
```

Do not use `sudo pip`, Conda base, or the host's existing LMCache environment.

## Stage 1: record the host environment (read-only)

```bash
mkdir -p "$TEST_ROOT/results"
set -o pipefail

{
  date
  hostname
  uname -a
  python3 --version

  echo "===== NVIDIA SMI ====="
  nvidia-smi

  echo "===== CUDA COMPILER ====="
  nvcc --version || true

  echo "===== NVIDIA MODULE ====="
  cat /proc/driver/nvidia/version || true
  modinfo nvidia | grep -E '^(filename|version|license):' || true

  echo "===== KERNEL CONFIG ====="
  if [ -r "/boot/config-$(uname -r)" ]; then
    grep CONFIG_DMABUF_TOKEN "/boot/config-$(uname -r)" || true
    grep CONFIG_PCI_P2PDMA "/boot/config-$(uname -r)" || true
  elif [ -r /proc/config.gz ]; then
    zgrep CONFIG_DMABUF_TOKEN /proc/config.gz || true
    zgrep CONFIG_PCI_P2PDMA /proc/config.gz || true
  else
    echo "Kernel configuration unavailable"
  fi

  echo "===== PCI DEVICES ====="
  lspci -nn | grep -Ei 'NVIDIA|Non-Volatile memory' || true

  echo "===== BLOCK DEVICES ====="
  lsblk -o NAME,PATH,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL
} 2>&1 | tee "$TEST_ROOT/results/host-environment.log"
```

`/proc/driver/nvidia/version` should mention the open kernel module
(`modinfo` license `Dual MIT/GPL`). A missing `CONFIG_DMABUF_TOKEN` does not
prevent the CUDA export tests (stages 3 to 5), but it does prevent all
io_uring/NVMe stages.

## Stage 2: private LMCache checkout

```bash
cd "$TEST_ROOT"

git clone https://github.com/anuj7781/LMCache.git
cd LMCache

git switch feat/iou-dmabuf-backend-dev-nvidia-test-plan
git status --short
git log -5 --oneline

git rev-parse HEAD | tee "$TEST_ROOT/results/lmcache-commit.txt"
```

This branch is `feat/iou-dmabuf-backend-dev` plus the corrected design doc and
this plan; the code is identical. The checkout must be separate from the
host's existing LMCache checkout.

## Stage 3: Blackwell-compatible PyTorch

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

python - <<'PY' 2>&1 | tee "$TEST_ROOT/results/pytorch-environment.log"
import sys
import torch

print("Python:", sys.version)
print("Python executable:", sys.executable)
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

Required: `CUDA available: True` and `capability: (12, 0)`. Stop if CUDA is
unavailable, the GPU is not the expected RTX 5090, or PyTorch reports that no
compatible kernel image exists.

## Stage 4: NVIDIA DMA-BUF capability

Only needs `libcuda.so.1`. Attribute values are from the CUDA 12.9 `cuda.h`.

```bash
python - <<'PY' 2>&1 | tee "$TEST_ROOT/results/cuda-dmabuf-attributes.log"
import ctypes

CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116
CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124

cu = ctypes.CDLL("libcuda.so.1")

cu.cuInit.argtypes = [ctypes.c_uint]
cu.cuInit.restype = ctypes.c_int
cu.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cu.cuDeviceGet.restype = ctypes.c_int
cu.cuDeviceGetAttribute.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]
cu.cuDeviceGetAttribute.restype = ctypes.c_int

def check(rc: int, operation: str) -> None:
    if rc != CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed with CUDA error {rc}")

check(cu.cuInit(0), "cuInit")

device = ctypes.c_int()
check(cu.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

for name, attribute in (
    ("DMA_BUF_SUPPORTED", CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED),
    ("GPU_DIRECT_RDMA_SUPPORTED", CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED),
):
    value = ctypes.c_int(-1)
    check(
        cu.cuDeviceGetAttribute(ctypes.byref(value), attribute, device.value),
        f"cuDeviceGetAttribute({name})",
    )
    print(name, value.value)
PY
```

- `DMA_BUF_SUPPORTED 0`: stop; the NVIDIA exporter cannot work.
- `DMA_BUF_SUPPORTED 1`: continue.
- `GPU_DIRECT_RDMA_SUPPORTED` is informational (NVIDIA's older peer-memory
  APIs), not a stop condition for this Linux DMA-BUF path.

## Stage 5: non-destructive GPU export probe

Uses only PyTorch and the CUDA driver; no io_uring, no NVMe, no LMCache build.

```bash
python tools/probe_gpu_dmabuf_export.py \
  --device cuda:0 \
  --size-mib 960 \
  2>&1 | tee "$TEST_ROOT/results/gpu-dmabuf-export.log"

echo "export probe exit=${PIPESTATUS[0]}"
```

It tries three exports: the whole PyTorch allocation, an aligned sub-range (as
the backend slices slabs), and the PCIe/BAR1 flag.

| Result | Meaning |
|---|---|
| Whole allocation fails | Current backend cannot run |
| Whole passes, sub-range fails | Backend slab layout cannot run |
| Whole and sub-range pass, PCIe flag fails | DMA-BUF export works, but a direct PCIe mapping is not established |
| All three pass | Proceed to the build and kernel tests |

If `CONFIG_DMABUF_TOKEN` is missing, stop here and send the results.

## Stage 6: CUDA build toolchain

```bash
which nvcc
nvcc --version
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"
```

`nvcc` must be CUDA 12.8 or newer for native Blackwell compilation. If it is
missing or older, stop and coordinate with the machine owner; do not install or
replace a system CUDA toolkit.

```bash
export LMCACHE_CUDA_MAJOR=12
export TORCH_CUDA_ARCH_LIST=12.0

# Only if CUDA lives in /usr/local/cuda and nvcc is not already on PATH:
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

which nvcc
nvcc --version
```

## Stage 7: LMCache dependencies

`requirements/cuda12_core.txt` supplies `cupy-cuda12x` and
`cuda-python>=12,<13`; it is needed because LMCache is installed with
`--no-deps`. The `torch` line of `common.txt` is filtered out so pip cannot
replace the chosen PyTorch.

```bash
python -m pip install -r requirements/build.txt

REQ_NO_TORCH=$(mktemp)
grep -v '^torch' requirements/common.txt > "$REQ_NO_TORCH"
python -m pip install -r "$REQ_NO_TORCH" -r requirements/cuda12_core.txt
rm -f "$REQ_NO_TORCH"

python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

## Stage 8: build LMCache

```bash
cd "$TEST_ROOT/LMCache"

rm -rf build
find lmcache -name '*.so' -delete

LMCACHE_CUDA_MAJOR=12 \
TORCH_CUDA_ARCH_LIST=12.0 \
python -m pip install -e . --no-build-isolation --no-deps \
  2>&1 | tee "$TEST_ROOT/results/lmcache-build.log"

echo "LMCache build exit=${PIPESTATUS[0]}"

python - <<'PY' 2>&1 | tee "$TEST_ROOT/results/lmcache-import.log"
import lmcache
import lmcache.c_ops
import torch

print("LMCache:", lmcache.__file__)
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("c_ops: OK")
PY
```

`lmcache.__file__` must point into `$HOME/anuj-iou-dmabuf-test/LMCache/`.

## Stage 9: Rust toolchain

```bash
command -v cargo || true
cargo --version || true
rustc --version || true
```

Use the existing `cargo` if present. Otherwise, only if the machine owner
permits a user-local download, install rustup under `$TEST_ROOT` (the
activation file already set `RUSTUP_HOME`/`CARGO_HOME`; `--no-modify-path`
keeps shell profiles untouched):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
  sh -s -- -y --profile minimal --no-modify-path

which cargo
which rustc
cargo --version
rustc --version
```

## Stage 10: Rust raw-block extension

```bash
python -m pip install maturin

(
  cd rust/raw_block
  maturin develop --release
) 2>&1 | tee "$TEST_ROOT/results/rust-extension-build.log"

echo "Rust extension build exit=${PIPESTATUS[0]}"

python - <<'PY'
import lmcache_rust_raw_block_io

print(lmcache_rust_raw_block_io.__file__)
from lmcache_rust_raw_block_io import RawBlockDevice

print("DMA-BUF kernel support:", RawBlockDevice.probe_dmabuf_support())
PY
```

Required: `DMA-BUF kernel support: True`. If it prints `False`, stop before all
NVMe stages.

```bash
cargo test \
  --manifest-path rust/raw_block/Cargo.toml \
  --no-default-features \
  2>&1 | tee "$TEST_ROOT/results/rust-tests.log"

echo "Rust tests exit=${PIPESTATUS[0]}"
```

## Stage 11: Python unit and regression tests

Run from the repository root with the activation file sourced, so
`LMCACHE_TEST_BLOCK_DEVICE` / `LMCACHE_TEST_CHAR_DEVICE` point at
`/nonexistent`. `test_raw_block_uring_cmd.py` otherwise opens `/dev/nvme0n1`
and `/dev/ng0n1` writable and issues I/O to them; with the variables set, those
cases are skipped. The other files use temporary files or fake devices.

```bash
python -m pip install -r requirements/test.txt
echo "$LMCACHE_TEST_BLOCK_DEVICE $LMCACHE_TEST_CHAR_DEVICE"   # /nonexistent /nonexistent

pytest -q \
  tests/v1/storage_backend/test_iou_dmabuf_backend.py \
  tests/v1/storage_backend/test_raw_block_core.py \
  2>&1 | tee "$TEST_ROOT/results/new-unit-tests.log"

echo "New tests exit=${PIPESTATUS[0]}"

pytest -q \
  tests/v1/storage_backend/test_raw_block_device.py \
  tests/v1/storage_backend/test_raw_block_key_codec.py \
  tests/v1/storage_backend/test_raw_block_uring_cmd.py \
  tests/v1/storage_backend/test_rust_raw_block_backend.py \
  tests/v1/storage_backend/test_storage_manager.py \
  tests/v1/storage_backend/test_local_cpu_backend.py \
  tests/v1/storage_backend/test_local_disk_backend.py \
  2>&1 | tee "$TEST_ROOT/results/regression-tests.log"

echo "Regression tests exit=${PIPESTATUS[0]}"
```

Known upstream issue: `test_uring_cmd_middle_build_error_does_not_hang` runs
its probe in a `spawn` child with a 10 s timeout, and the child re-imports
`torch` and `lmcache`. If that import alone exceeds 10 s the test fails without
any backend fault (it took ~18 s on the AMD machine). With the device variables
set to `/nonexistent` it is skipped anyway.

## Stage 12: NVMe authorization

Send the machine owner:

```bash
lsblk -o NAME,PATH,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL
```

and ask for:

- the exact block namespace (e.g. `/dev/nvmeXnY`);
- confirmation that reads are permitted;
- an unused, aligned offset for the write probe;
- confirmation that the **entire namespace is disposable** before running the
  benchmark.

Do not use `/dev/nvme0n1` just because it appears in examples. Set the approved
device only after confirmation:

```bash
export SCRATCH_NVME=/dev/nvmeXnY
export SAFE_TEST_OFFSET=$((4 << 30))

lsblk -o NAME,PATH,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL "$SCRATCH_NVME"
```

Device access needs root or the `disk` group. Plain `sudo python` bypasses the
venv; if running as root is preferred over group access, use the venv's
interpreter explicitly: `sudo "$VIRTUAL_ENV/bin/python" ...`.

Optionally, rerun the device-backed upstream tests against the approved
namespace:

```bash
LMCACHE_TEST_BLOCK_DEVICE="$SCRATCH_NVME" \
LMCACHE_TEST_CHAR_DEVICE=/dev/ngXnY \
pytest -q tests/v1/storage_backend/test_raw_block_uring_cmd.py
```

## Stage 13: read-only io_uring/NVMe probe

```bash
python tools/probe_iou_dmabuf_e2e.py \
  --device-path "$SCRATCH_NVME" \
  --device-offset "$SAFE_TEST_OFFSET" \
  2>&1 | tee "$TEST_ROOT/results/e2e-read.log"

echo "E2E read exit=${PIPESTATUS[0]}"
```

Tests NVMe -> READ_FIXED -> NVIDIA DMA-BUF -> GPU VRAM.

## Stage 14: explicit-offset WRITE_FIXED probe (destructive at the offset)

```bash
python tools/probe_iou_dmabuf_e2e.py \
  --device-path "$SCRATCH_NVME" \
  --device-offset "$SAFE_TEST_OFFSET" \
  --write \
  2>&1 | tee "$TEST_ROOT/results/e2e-write.log"

echo "E2E write exit=${PIPESTATUS[0]}"
```

## Stage 15: integrity benchmark (entire namespace disposable)

The benchmark has no base-offset option: `RawBlockCore` places its metadata
region at the start of the device and data slots after it, so it is not
confined to `SAFE_TEST_OFFSET`.

```bash
python benchmarks/storage_backend_io/iou_dmabuf_io_benchmark.py \
  --device-path "$SCRATCH_NVME" \
  --num-ops 128 \
  --concurrency 4 \
  --verify-integrity \
  --output-json "$TEST_ROOT/results/nvidia-short.json" \
  2>&1 | tee "$TEST_ROOT/results/nvidia-short.log"

echo "Short benchmark exit=${PIPESTATUS[0]}"

# If clean, sustained load (~50 GiB of traffic):
python benchmarks/storage_backend_io/iou_dmabuf_io_benchmark.py \
  --device-path "$SCRATCH_NVME" \
  --num-ops 128 \
  --concurrency 8 \
  --target-gib 50 \
  --verify-integrity \
  --output-json "$TEST_ROOT/results/nvidia-50g.json" \
  2>&1 | tee "$TEST_ROOT/results/nvidia-50g.log"

echo "Sustained benchmark exit=${PIPESTATUS[0]}"
```

## What to send back first

Before building LMCache, send:

1. `host-environment.log`
2. `cuda-dmabuf-attributes.log`
3. `gpu-dmabuf-export.log`

The first decision is based on: capability `(12, 0)`, `DMA_BUF_SUPPORTED = 1`,
whole-allocation export, sub-range export, PCIe/BAR1 export, and
`CONFIG_DMABUF_TOKEN`.

The full gate chain:

```text
PyTorch recognizes the RTX 5090
  -> DMA_BUF_SUPPORTED = 1
  -> whole / sub-range export
  -> patched kernel (CONFIG_DMABUF_TOKEN)
  -> Rust registration (probe_dmabuf_support)
  -> NVMe read
  -> explicit-offset NVMe write
  -> disposable-namespace integrity benchmark
```

## Cleanup

```bash
deactivate
rm -rf "$HOME/anuj-iou-dmabuf-test"
```

This removes the clone, venv, Rust toolchain and the pip, CUDA, PyTorch and
Triton caches. Host-level kernel, driver, device state, shell history and
system logs remain.
