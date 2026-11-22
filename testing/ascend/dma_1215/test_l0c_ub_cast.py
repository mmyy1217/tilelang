"""1215 L0C→UB 无系数降精度搬运（FP32→FP16 / BF16，单目标 UB）。"""

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
def test_l0c_to_ub_cast(dtype):
    """FIX 将 32×64 FP32 累加结果降精度为 FP16 / BF16 搬入 AIV0 的 UB。"""

    @T.prim_func
    def main(a: T.Tensor((32, 16), "float16"), b: T.Tensor((64, 16), "float16"), out: T.Tensor((32, 64), dtype)):
        with T.MixedKernel(1) as (_, sid):
            acc = T.alloc_l0c((32, 64), "float32")
            _gemm_32x64(a, b, acc)
            ub = T.alloc_shared((32, 64), dtype)
            T.copy(acc, ub, sub_blockid=0)
            if sid == 0:
                T.copy(ub, out)

    a, b, expected = _gemm_inputs(dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_l0c_to_ub_cast_padded_ub_stride(dtype):
    """降精度写入 32B 对齐宽行距 UB (32, 80) 的前 64 列。"""

    @T.prim_func
    def main(a: T.Tensor((32, 16), "float16"), b: T.Tensor((64, 16), "float16"), out: T.Tensor((32, 64), dtype)):
        with T.MixedKernel(1) as (_, sid):
            acc = T.alloc_l0c((32, 64), "float32")
            _gemm_32x64(a, b, acc)
            ub = T.alloc_shared((32, 80), dtype)
            T.copy(acc, ub[:, :64], sub_blockid=0)
            if sid == 0:
                T.copy(ub[:, :64], out)

    a, b, expected = _gemm_inputs(dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
