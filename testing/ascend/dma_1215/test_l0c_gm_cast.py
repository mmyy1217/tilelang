"""1215 L0C→GM 无系数降精度写出（FP32→FP16 / BF16，含非对齐 GM 步长与 5D 动态写回）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()


@T.macro
def _gemm_32x64(a_gm, b_gm, acc):
    a = T.alloc_l1(a_gm.shape, a_gm.dtype)
    b = T.alloc_l1(b_gm.shape, b_gm.dtype)
    T.copy(a_gm, a, transpose=False)
    T.copy(b_gm, b, transpose=False)
    T.gemm(a, b, acc, transpose_B=True, clear_accum=True)


def _gemm_inputs(dtype):
    rng = torch.Generator().manual_seed(1215)
    a = (torch.randint(-100, 100, (32, 16), generator=rng) / 64).half()
    b = (torch.randint(-100, 100, (64, 16), generator=rng) / 64).half()
    expected = (a.float() @ b.float().T).to(getattr(torch, dtype))
    return a, b, expected


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_l0c_to_gm_cast(dtype):
    """FIX 将 32×64 FP32 累加结果降精度为 FP16 / BF16 写回 GM。"""

    @T.prim_func
    def main(a: T.Tensor((32, 16), "float16"), b: T.Tensor((64, 16), "float16"), out: T.Tensor((32, 64), dtype)):
        with T.Kernel(1):
            acc = T.alloc_l0c((32, 64), "float32")
            _gemm_32x64(a, b, acc)
            T.copy(acc, out)

    a, b, expected = _gemm_inputs(dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_l0c_to_gm_cast_strided_subregion(dtype):
    """降精度写回非 32B 对齐行步长的 (40, 70) GM 子区域 [3:35, 5:69]，检查四周保护区。"""

    @T.prim_func
    def main(a: T.Tensor((32, 16), "float16"), b: T.Tensor((64, 16), "float16"), out: T.Tensor((40, 70), dtype)):
        with T.Kernel(1):
            acc = T.alloc_l0c((32, 64), "float32")
            _gemm_32x64(a, b, acc)
            T.copy(acc, out[3:35, 5:69])

    a, b, gemm_ref = _gemm_inputs(dtype)
    expected = torch.full((40, 70), -7.0, dtype=gemm_ref.dtype)
    expected[3:35, 5:69] = gemm_ref
    out = torch.full_like(expected, -7.0).npu()
    tilelang.compile(main, target="ascend")(a.npu(), b.npu(), out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_l0c_to_gm_cast_5d_and_dynamic(dtype):
    """同一已编译 kernel 结合 T.dynamic 维度与标量偏移降精度写回 5D GM 单平面子区域。"""
    M = T.dynamic("M")
    N = T.dynamic("N")

    @T.prim_func
    def main(
        a: T.Tensor((32, 16), "float16"),
        b: T.Tensor((64, 16), "float16"),
        row_off: T.int32,
        col_off: T.int32,
        out: T.Tensor((2, 2, 2, M, N), dtype),
    ):
        with T.Kernel(1):
            T.assume(M >= 40)
            T.assume(N >= 80)
            T.assume(row_off >= 0)
            T.assume(col_off >= 0)
            acc = T.alloc_l0c((32, 64), "float32")
            _gemm_32x64(a, b, acc)
            T.copy(acc, out[1, 0, 1, row_off : row_off + 32, col_off : col_off + 64])

    a, b, gemm_ref = _gemm_inputs(dtype)
    kernel = tilelang.compile(main, target="ascend")
    for shape, row_off, col_off in [((42, 83), 3, 5), ((50, 91), 7, 9)]:
        expected = torch.full((2, 2, 2, *shape), -7.0, dtype=gemm_ref.dtype)
        expected[1, 0, 1, row_off : row_off + 32, col_off : col_off + 64] = gemm_ref
        out = torch.full_like(expected, -7.0).npu()
        kernel(a.npu(), b.npu(), row_off, col_off, out)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

