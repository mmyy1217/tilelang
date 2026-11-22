"""1215 L0C→UB 硬件双目标自动分半写入（dual_copy：M / N 分半，含 ND 与 NZ、FP32 与 INT32）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_l0c_layout


pytestmark = tilelang.testing.requires_ascend.marks()


@T.macro
def _gemm_32x64(a_gm, b_gm, acc):
    a = T.alloc_l1(a_gm.shape, a_gm.dtype)
    b = T.alloc_l1(b_gm.shape, b_gm.dtype)
    T.copy(a_gm, a, transpose=False)
    T.copy(b_gm, b, transpose=False)
    T.gemm(a, b, acc, transpose_B=True, clear_accum=True)


def _gemm_inputs(in_dtype, acc_dtype):
    rng = torch.Generator().manual_seed(1215)
    if in_dtype == "int8":
        a = torch.randint(-8, 9, (32, 32), generator=rng).to(torch.int8)
        b = torch.randint(-8, 9, (64, 32), generator=rng).to(torch.int8)
        return a, b, a.int() @ b.int().T
    a = (torch.randint(-100, 100, (32, 16), generator=rng) / 64).half()
    b = (torch.randint(-100, 100, (64, 16), generator=rng) / 64).half()
    return a, b, (a.float() @ b.float().T).to(getattr(torch, acc_dtype))


@pytest.mark.parametrize("in_dtype,acc_dtype", [("float16", "float32"), ("int8", "int32")])
@pytest.mark.parametrize("split", ["m", "n"])
def test_l0c_to_ub_dual_nd(in_dtype, acc_dtype, split):
    """硬件双写按 M (16×64) 或 N (32×32) 分给两个 AIV 的 ND UB，再拼回完整矩阵。"""
    k = 32 if in_dtype == "int8" else 16
    ub_shape = (16, 64) if split == "m" else (32, 32)

    @T.prim_func
    def main(a: T.Tensor((32, k), in_dtype), b: T.Tensor((64, k), in_dtype), out: T.Tensor((32, 64), acc_dtype)):
        with T.MixedKernel(1):
            acc = T.alloc_l0c((32, 64), acc_dtype)
            _gemm_32x64(a, b, acc)
            ub = T.alloc_shared(ub_shape, acc_dtype)
            T.dual_copy(acc, ub)
            T.dual_copy(ub, out)

    a, b, expected = _gemm_inputs(in_dtype, acc_dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("split", ["m", "n"])
def test_l0c_to_ub_dual_nz(split):
    """硬件双写保留 16×16 NZ 分形按 M 或 N 分给两个 AIV，SIMT 按逻辑坐标读回。"""
    rows, cols = (16, 64) if split == "m" else (32, 32)

    @T.prim_func
    def main(a: T.Tensor((32, 16), "float16"), b: T.Tensor((64, 16), "float16"), out: T.Tensor((32, 64), "float32")):
        with T.MixedKernel(1) as (_, sid):
            acc = T.alloc_l0c((32, 64), "float32")
            _gemm_32x64(a, b, acc)
            ub = T.alloc_shared((rows, cols), "float32")
            T.annotate_layout({ub: make_ascend_l0c_layout(ub)})
            T.dual_copy(acc, ub)
            with T.SimtVF(threads=128):
                for r, c in T.Parallel(rows, cols):
                    out[r + (sid * rows if split == "m" else 0), c + (sid * cols if split == "n" else 0)] = ub[r, c]

    a, b, expected = _gemm_inputs("float16", "float32")
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
