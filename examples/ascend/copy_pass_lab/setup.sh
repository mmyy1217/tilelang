#!/usr/bin/env bash
set -euo pipefail

lab_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$lab_dir/../../.." && pwd)"
lab_artifacts="$repo_root/../work/experiments/ascend-copy-pass-lab"
base_python="${COPY_PASS_BASE_PYTHON:-python3.11}"
mkdir -p "$lab_artifacts"
if [[ ! -x "$lab_artifacts/venv/bin/python" ]]; then
    "$base_python" -m venv --system-site-packages "$lab_artifacts/venv"
fi
lab_python="$lab_artifacts/venv/bin/python"
"$lab_python" -c 'import importlib.util; assert importlib.util.find_spec("torch") is not None, "Copy pass lab needs a base Python with PyTorch; set COPY_PASS_BASE_PYTHON to that interpreter"'
"$lab_python" -m pip install 'apache-tvm-ffi>=0.1.11,<0.1.13' \
    'z3-solver>=4.13.0,<4.15.5' scikit-build-core torch-c-dlpack-ext cython patchelf
"$lab_python" -c 'import torch, z3, tvm_ffi'
git -C "$repo_root" submodule update --init --depth 1 3rdparty/tvm
git -C "$repo_root/3rdparty/tvm" submodule update --init --depth 1 3rdparty/tvm-ffi
git -C "$repo_root/3rdparty/tvm/3rdparty/tvm-ffi" submodule update --init --depth 1 \
    3rdparty/dlpack 3rdparty/libbacktrace
mkdir -p "$lab_artifacts/build"
if [[ ! -e "$repo_root/build" ]]; then
    ln -s "$lab_artifacts/build" "$repo_root/build"
fi
if [[ "$(realpath "$repo_root/build")" != "$(realpath "$lab_artifacts/build")" ]]; then
    printf 'Expected build link to %s; found another build directory.\n' "$lab_artifacts/build" >&2
    exit 1
fi
export PATH="$lab_artifacts/venv/bin:$PATH"
export CCACHE_DIR="$lab_artifacts/ccache"
unset TVM_ROOT TVM_LIBRARY_PATH
export GIT_CONFIG_COUNT=4
export GIT_CONFIG_KEY_0=submodule.3rdparty/cutlass.update
export GIT_CONFIG_KEY_1=submodule.3rdparty/OpenCL-Headers.update
export GIT_CONFIG_KEY_2=submodule.3rdparty/cutlass_fpA_intB_gemm.update
export GIT_CONFIG_KEY_3=submodule.3rdparty/libflash_attn.update
export GIT_CONFIG_VALUE_0=none GIT_CONFIG_VALUE_1=none GIT_CONFIG_VALUE_2=none GIT_CONFIG_VALUE_3=none
if ! cmake -S "$repo_root" -B "$lab_artifacts/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ '-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG' \
    -DPython_EXECUTABLE="$lab_python" -DPython3_EXECUTABLE="$lab_python" \
    -DUSE_ASCEND=ON -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_LLVM=OFF \
    -DUSE_METAL=OFF -DTILELANG_USE_ASCEND_STUBS=ON \
    > "$lab_artifacts/configure.log" 2>&1; then
    tail -80 "$lab_artifacts/configure.log" >&2
    exit 1
fi
if ! cmake --build "$lab_artifacts/build" --parallel "${COPY_PASS_JOBS:-4}" \
    > "$lab_artifacts/build.log" 2>&1; then
    tail -80 "$lab_artifacts/build.log" >&2
    exit 1
fi
"$lab_dir/run.sh" --list > "$lab_artifacts/import-smoke.log" 2>&1
printf 'Build complete. Run %s/run.sh --case dma\n' "$lab_dir"
