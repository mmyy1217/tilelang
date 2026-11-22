"""1215 L0C→UB 指定写入单个 AIV 的 UB（sub_blockid=0 或 1）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()


@pytest.mark.parametrize("target", [0, 1], ids=["aiv0", "aiv1"])
def test_single_aiv_preserves_other_ub(target):
    """将 GEMM 结果只送到指定 AIV，再分别读回两个 UB，检查另一侧的保护值 −7 未被覆写。"""

    @T.prim_func
    def main(A: T.Tensor((16, 16), "float16"), B: T.Tensor((16, 16), "float16"), C: T.Tensor((2, 16, 16), "float32")):
        with T.MixedKernel(1) as (_, sid):
            a = T.alloc_l1((16, 16), "float16")
            b = T.alloc_l1((16, 16), "float16")
            acc = T.alloc_l0c((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float32")
            T.fill(ub, -7.0)
            T.copy(A, a, transpose=False)
            T.copy(B, b, transpose=False)
            T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, ub, sub_blockid=target)
            T.copy(ub, C[sid, :, :])

    a = torch.arange(256).reshape(16, 16).half()
    expected = torch.full((2, 16, 16), -7.0)
    expected[target] = a
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    torch.testing.assert_close(kernel(a.npu(), torch.eye(16).half().npu()).cpu(), expected, rtol=0, atol=0)
