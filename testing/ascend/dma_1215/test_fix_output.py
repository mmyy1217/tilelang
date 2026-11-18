import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()


@T.macro
def _gemm_32x64(a_gm, b_gm, acc):
    a = T.alloc_l1((32, 16), "float16")
    b = T.alloc_l1((64, 16), "float16")
    T.copy(a_gm, a, transpose=False)
    T.copy(b_gm, b, transpose=False)
    T.gemm(a, b, acc, transpose_B=True, clear_accum=True)


def _check_output(main, dtype):
    rng = torch.Generator().manual_seed(1215)
    a = (torch.randint(-100, 100, (32, 16), generator=rng) / 64).half()
    b = (torch.randint(-100, 100, (64, 16), generator=rng) / 64).half()
    expected = (a.float() @ b.float().T).to(getattr(torch, dtype))
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_l0c_to_gm_nd(dtype):
    """FIX 将完整 NZ 结果转成 ND 写回 GM，含 FP32 和两种降精度。"""

    @T.prim_func
    def main(a: T.Tensor((32, 16), "float16"), b: T.Tensor((64, 16), "float16"), out: T.Tensor((32, 64), dtype)):
        with T.Kernel(1):
            acc = T.alloc_l0c((32, 64), "float32")
            _gemm_32x64(a, b, acc)
            T.copy(acc, out)

    _check_output(main, dtype)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_l0c_to_ub_nd(dtype):
    """FIX 将 NZ 转为 ND，只读回收到结果的 AIV0。"""

    @T.prim_func
    def main(a: T.Tensor((32, 16), "float16"), b: T.Tensor((64, 16), "float16"), out: T.Tensor((32, 64), dtype)):
        with T.MixedKernel(1) as (_, sid):
            acc = T.alloc_l0c((32, 64), "float32")
            _gemm_32x64(a, b, acc)
            ub = T.alloc_shared((32, 64), dtype)
            T.copy(acc, ub, sub_blockid=0)
            if sid == 0:
                T.copy(ub, out)

    _check_output(main, dtype)


def test_l0c_to_ub_dual_m():
    """目标 UB 高度减半，按 M 分给两个 AIV，再写回完整矩阵。"""
    dtype = "float32"

    @T.prim_func
    def main(a: T.Tensor((32, 16), "float16"), b: T.Tensor((64, 16), "float16"), out: T.Tensor((32, 64), dtype)):
        with T.MixedKernel(1):
            acc = T.alloc_l0c((32, 64), "float32")
            _gemm_32x64(a, b, acc)
            ub = T.alloc_shared((16, 64), "float32")
            T.dual_copy(acc, ub)
            T.dual_copy(ub, out)

    _check_output(main, dtype)


def test_l0c_to_ub_dual_n():
    """目标 UB 宽度减半，按 N 分给两个 AIV，再写回完整矩阵。"""
    dtype = "float32"

    @T.prim_func
    def main(a: T.Tensor((32, 16), "float16"), b: T.Tensor((64, 16), "float16"), out: T.Tensor((32, 64), dtype)):
        with T.MixedKernel(1):
            acc = T.alloc_l0c((32, 64), "float32")
            _gemm_32x64(a, b, acc)
            ub = T.alloc_shared((32, 32), "float32")
            T.dual_copy(acc, ub)
            T.dual_copy(ub, out)

    _check_output(main, dtype)


@pytest.mark.parametrize("target", [0, 1], ids=["aiv0", "aiv1"])
def test_single_aiv_preserves_other_ub(target):
    """将 GEMM 结果只送到指定 AIV，再分别读回两个 UB，检查另一侧的保护值。"""

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
