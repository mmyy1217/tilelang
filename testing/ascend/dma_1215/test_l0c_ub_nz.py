"""1215 L0C→UB 保留 16×16 NZ 分形搬入 UB（NZ→NZ，含整块与 16 列对齐子区域）。"""

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
def test_l0c_to_ub_nz(in_dtype, acc_dtype):
    """保留 16×16 NZ 分形搬入 UB（FP32 / INT32），SIMT 按逻辑坐标读回逐元素比较。"""
    k = 32 if in_dtype == "int8" else 16

    @T.prim_func
    def main(a: T.Tensor((32, k), in_dtype), b: T.Tensor((64, k), in_dtype), out: T.Tensor((32, 64), acc_dtype)):
        with T.MixedKernel(1, sids=1):
            acc = T.alloc_l0c((32, 64), acc_dtype)
            _gemm_32x64(a, b, acc)
            ub = T.alloc_shared((32, 64), acc_dtype)
            T.annotate_layout({ub: make_ascend_l0c_layout(ub)})
            T.copy(acc, ub, sub_blockid=0)
            with T.SimtVF(threads=128):
                for row, col in T.Parallel(32, 64):
                    out[row, col] = ub[row, col]

    a, b, expected = _gemm_inputs(in_dtype, acc_dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_l0c_to_ub_nz_aligned_subregion():
    """向更大 NZ UB (48, 80) 的 16 对齐偏移子区域 [16:48, 16:80] 写入，按逻辑坐标读回。"""

    @T.prim_func
    def main(a: T.Tensor((32, 16), "float16"), b: T.Tensor((64, 16), "float16"), out: T.Tensor((32, 64), "float32")):
        with T.MixedKernel(1, sids=1):
            acc = T.alloc_l0c((32, 64), "float32")
            _gemm_32x64(a, b, acc)
            ub = T.alloc_shared((48, 80), "float32")
            T.annotate_layout({ub: make_ascend_l0c_layout(ub)})
            T.copy(acc, ub[16:48, 16:80], sub_blockid=0)
            with T.SimtVF(threads=128):
                for row, col in T.Parallel(32, 64):
                    out[row, col] = ub[16 + row, 16 + col]

    a, b, expected = _gemm_inputs("float16", "float32")
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
