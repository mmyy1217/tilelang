#!/usr/bin/env bash
set -euo pipefail

lab_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$lab_dir/../../.." && pwd)"
lab_python="${COPY_PASS_PYTHON:-$repo_root/../work/experiments/ascend-copy-pass-lab/venv/bin/python}"
export PYTHONPATH="$repo_root:$repo_root/3rdparty/tvm/python${PYTHONPATH:+:$PYTHONPATH}"
export TVM_LIBRARY_PATH="$repo_root/build/lib:$repo_root/build/tvm"
export TVM_IMPORT_PYTHON_PATH="$repo_root/3rdparty/tvm/python"
export TILELANG_CACHE_DIR="$repo_root/../work/experiments/ascend-copy-pass-lab/cache"
export TILELANG_PASS_DIFF=0
export TILELANG_PASS_PROFILE=0
exec "$lab_python" "$lab_dir/dump_passes.py" "$@"
