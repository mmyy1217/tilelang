"""1215 L0C→UB 同类型搬出并转为行优先 ND（NZ→ND，含 FP32 / INT32）。"""

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
def test_l0c_to_ub_nd(in_dtype, acc_dtype):
    """FIX 将 32×64 NZ 累加结果（FP32 / INT32）转为 ND 搬入 AIV0 的 UB，再写回 GM 校验。"""
    k = 32 if in_dtype == "int8" else 16

    @T.prim_func
    def main(a: T.Tensor((32, k), in_dtype), b: T.Tensor((64, k), in_dtype), out: T.Tensor((32, 64), acc_dtype)):
        with T.MixedKernel(1) as (_, sid):
            acc = T.alloc_l0c((32, 64), acc_dtype)
            _gemm_32x64(a, b, acc)
            ub = T.alloc_shared((32, 64), acc_dtype)
            T.copy(acc, ub, sub_blockid=0)
            if sid == 0:
                T.copy(ub, out)

    a, b, expected = _gemm_inputs(in_dtype, acc_dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
