# Ascend examples

These examples demonstrate complete kernels on Ascend 950. Use the explicit
`tilelang.ascend.language` dialect and an Ascend-enabled build; see the
[backend guide](../../tilelang/ascend/README.md) for setup.
Inputs, reference computations and correctness checks run on the NPU.

| Example | Purpose |
| --- | --- |
| [GEMM](example_gemm.py) | Persistent matrix multiplication with swizzling, precision modes and a mixed-core epilogue |
| [Split-K GEMM](example_gemm_splitk.py) | Atomic or deterministic reduction across K partitions |
| [Vector addition](example_vecadd.py) | SIMT and explicit SIMD with pipelined data movement |
| [RMSNorm](example_rmsnorm.py) | SIMT fragments, reduction and pipelined row processing |
| [FP8 quantization](example_per_token_cast_to_fp8.py) | The same group-wise quantization in SIMT and explicit SIMD |
| [FP8 ND→NZ GEMM](example_fp8_nd2nz_gemm.py) | Automatic E4M3 ND-to-NZ conversion from UB to L1, with regular and dual copies |

For advanced GEMM implementations and tuning, see [deepgemm](deepgemm).
Attention kernels are in [flash_attention](flash_attention/README.md).

Run an example directly to check correctness and report device performance.
GEMM and attention report latency and TFLOPS; vector addition, RMSNorm and FP8
quantization report latency and effective bandwidth in GB/s.
Bandwidth counts each logical input and output tensor's bytes once, including
auxiliary outputs such as FP8 scales and RMSNorm RSTD.
GEMM throughput uses `2 * M * N * K` operations, including for Split-K.
Use `--no-bench` on a top-level example to run only correctness checks.

Examples use the configured default target. Set `TILELANG_DEFAULT_TARGET=ascend`
to select AscendC or `TILELANG_DEFAULT_TARGET=pto` to select PTO.

GEMM runs FP8, BF16, FP32, HF32, BF16 output and both accumulation cases.
It uses `M = K = N = 8192`.
Vector addition and FP8 quantization run both SIMT and SIMD by default;
use `--mode simt` or `--mode simd` to select one. Each selected case is timed by default.

```bash
python examples/ascend/example_gemm.py
python examples/ascend/example_gemm_splitk.py --deterministic
python examples/ascend/example_vecadd.py
python examples/ascend/example_per_token_cast_to_fp8.py
python examples/ascend/example_per_token_cast_to_fp8.py --mode simd
python examples/ascend/example_rmsnorm.py --no-bench
python -m pytest examples/ascend/test_rmsnorm.py -q
python examples/ascend/example_fp8_nd2nz_gemm.py --no-bench
python examples/ascend/example_fp8_nd2nz_gemm.py --dual-copy
python -m pytest examples/ascend/test_fp8_nd2nz_gemm.py -q
```

The FP8 ND-to-NZ example targets AscendC and uses a single `32×288` tile to show the layout conversion,
data movement, and synchronization inserted for `T.copy`.
PTO does not support this conversion, so its CI run skips the example tests.
The reported latency is for a single-tile GEMM and does not measure large-matrix throughput or peak DMA bandwidth.
Regression coverage for E4M3 tails, explicit NZ layouts, and A/B operands is in
`testing/ascend/language/test_tilelang_ascend_nd2nz_scatter.py`
and `testing/ascend/layout/test_tilelang_ascend_fp8_nd2nz_ub2l1.py`.

Each top-level `example_<name>.py` has one `test_<name>.py`. Correctness tests use
smaller inputs than performance runs. Feature-level and compiler regressions
live in [testing/ascend](../../testing/ascend/README.md), independently of examples.
For copy padding, buffer annotations and scheduling APIs, consult the
[Ascend programming skill](../../.agents/skills/tilelang-ascend/SKILL.md).

The [performance regression driver](../../maint/scripts/ascend_perf_regression.py)
benchmarks these applications, including both vector programming modes, DeepGEMM and attention.
Language-feature tests are not performance entries.
