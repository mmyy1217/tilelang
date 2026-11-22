"""1215 UB→GM 双 AIV 软件分半写回（dual_copy：M / N 分半）。"""

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


@pytest.mark.parametrize("split", ["m", "n"])
def test_ub_to_gm_dual_copy(split):
    """两个 AIV 分别持有 M 分半 (16×64) 或 N 分半 (32×32) 的 UB，经 T.dual_copy 写回完整 GM。"""
    ub_shape = (16, 64) if split == "m" else (32, 32)

    @T.prim_func
    def main(a: T.Tensor((32, 16), "float16"), b: T.Tensor((64, 16), "float16"), out: T.Tensor((32, 64), "float32")):
        with T.MixedKernel(1):
            acc = T.alloc_l0c((32, 64), "float32")
            _gemm_32x64(a, b, acc)
            ub = T.alloc_shared(ub_shape, "float32")
            T.dual_copy(acc, ub)
            T.dual_copy(ub, out)

    rng = torch.Generator().manual_seed(1215)
    a = (torch.randint(-100, 100, (32, 16), generator=rng) / 64).half()
    b = (torch.randint(-100, 100, (64, 16), generator=rng) / 64).half()
    expected = a.float() @ b.float().T
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
