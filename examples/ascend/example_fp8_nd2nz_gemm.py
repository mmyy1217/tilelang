"""E4M3 FP8 GEMM on Ascend950 with automatic ND-to-NZ conversion between UB and L1."""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def fp8_nd2nz_gemm(dual_copy=False):
    """Compute C = A @ B.T for two 32-by-288 E4M3 matrices, with FP32 output."""
    dtype = "float8_e4m3fn"
    rows, cols = 32, 288
    sids = 2 if dual_copy else 1

    @T.prim_func
    def main(A: T.Tensor((rows, cols), dtype), B: T.Tensor((rows, cols), dtype), C: T.Tensor((rows, rows), "float32")):
        with T.MixedKernel(1, sids=sids):
            a_ub = T.alloc_shared((rows // sids, cols), dtype)
            a_l1 = T.alloc_l1((rows, cols), dtype)
            b_l1 = T.alloc_l1((rows, cols), dtype)
            a_l0 = T.alloc_l0a((rows, cols), dtype)
            b_l0 = T.alloc_l0b((rows, cols), dtype)
            c_l0 = T.alloc_l0c((rows, rows), "float32")
            for _ in T.Pipelined(1):
                T.copy(B, b_l1)
                # The UB-to-L1 copy inserts ND-to-compact-NZ scatter automatically.
                # With dual_copy, each vector subblock supplies half of A's rows.
                if dual_copy:
                    T.dual_copy(A, a_ub)
                    T.dual_copy(a_ub, a_l1)
                else:
                    T.copy(A, a_ub)
                    T.copy(a_ub, a_l1)
                T.copy(a_l1, a_l0)
                T.copy(b_l1, b_l0)
                T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=True)
                T.copy(c_l0, C)

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-copy", action="store_true", help="split A's rows between two vector subblocks")
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True, help="benchmark after correctness checks")
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(0)
    a = torch.randint(-4, 5, (32, 288), generator=generator).to(torch.float8_e4m3fn)
    b = torch.randint(-4, 5, (32, 288), generator=generator).to(torch.float8_e4m3fn)
    kernel = tilelang.compile(fp8_nd2nz_gemm(args.dual_copy), target="ascend", out_idx=-1, execution_backend="tvm_ffi")
    a_npu, b_npu = a.npu(), b.npu()
    result = kernel(a_npu, b_npu).cpu()
    torch.testing.assert_close(result, a.float() @ b.float().T, rtol=0, atol=0)
    print(f"FP8 ND-to-NZ GEMM (E4M3, dual_copy={args.dual_copy}): correctness passed")
    if args.bench:
        latency = do_bench(lambda: kernel(a_npu, b_npu), backend="msprof", _n_warmup=30, _n_repeat=50)
        print(f"Single-tile GEMM latency: {latency * 1000:.2f} us")
