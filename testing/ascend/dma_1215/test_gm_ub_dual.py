"""1215 GM→UB 双 AIV 软件分半搬入（dual_copy：M / N 分半）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("split", ["m", "n"])
def test_gm_to_ub_dual_copy(dtype, split):
    """两个 AIV 按 M (16×64) 或 N (32×32) 各自从 GM 搬入一半数据进各自 UB，再拼回输出校验。"""
    ub_shape = (16, 64) if split == "m" else (32, 32)

    @T.prim_func
    def main(
        src: T.Tensor((32, 64), dtype),
        eye: T.Tensor((16, 16), "float16"),
        out: T.Tensor((32, 64), dtype),
    ):
        with T.MixedKernel(1):
            # MixedKernel 需包含 Cube 侧任务以满足双 AIV 混合核契约。
            a_l1 = T.alloc_l1((16, 16), "float16")
            b_l1 = T.alloc_l1((16, 16), "float16")
            acc = T.alloc_l0c((16, 16), "float32")
            T.copy(eye, a_l1, transpose=False)
            T.copy(eye, b_l1, transpose=False)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)

            ub = T.alloc_shared(ub_shape, dtype)
            T.dual_copy(src, ub)
            T.dual_copy(ub, out)

    data = (torch.arange(32 * 64).reshape(32, 64) % 61 - 30).to(getattr(torch, dtype))
    eye = torch.eye(16, dtype=torch.float16)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(data.npu(), eye.npu())
    torch.testing.assert_close(actual.cpu(), data, rtol=0, atol=0)
