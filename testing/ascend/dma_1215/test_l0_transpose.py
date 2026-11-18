import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()


def _check_gemm(main, a, b, expected):
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_l1_to_l0_matrix(dtype):
    """分别把两个 16×32 完整块加载到 L0A 和 L0B。"""

    @T.prim_func
    def main(a: T.Tensor((16, 32), dtype), b: T.Tensor((16, 32), dtype), out: T.Tensor((16, 16), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((16, 32), dtype)
            b_l1 = T.alloc_l1((16, 32), dtype)
            a_l0 = T.alloc_l0a((16, 32), dtype)
            b_l0 = T.alloc_l0b((16, 32), dtype)
            acc = T.alloc_l0c((16, 16), "float32")
            T.copy(a, a_l1, transpose=False)
            T.copy(b, b_l1, transpose=False)
            T.copy(a_l1, a_l0, transpose=False)
            T.copy(b_l1, b_l0, transpose=False)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    rng = torch.Generator().manual_seed(1215)
    a = torch.randint(-4, 5, (16, 32), generator=rng).to(getattr(torch, dtype))
    b = torch.randint(-4, 5, (16, 32), generator=rng).to(getattr(torch, dtype))
    _check_gemm(main, a, b, a.float() @ b.float().T)


@pytest.mark.parametrize("dtype,rows", [("bfloat16", 24), ("float32", 25)])
def test_l1_to_l0a_transpose(dtype, rows):
    """选取 L1 中非整分形的列切片，再转置加载到 L0A。"""

    @T.prim_func
    def main(a: T.Tensor((32, 32), dtype), b: T.Tensor((64, 32), dtype), out: T.Tensor((rows, 64), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((32, 32), dtype)
            b_l1 = T.alloc_l1((64, 32), dtype)
            a_l0 = T.alloc_l0a((rows, 32), dtype)
            b_l0 = T.alloc_l0b((64, 32), dtype)
            acc = T.alloc_l0c((rows, 64), "float32")
            T.copy(a, a_l1, transpose=False)
            T.copy(b, b_l1, transpose=False)
            T.copy(a_l1[:, :rows], a_l0, transpose=True)
            T.copy(b_l1, b_l0, transpose=False)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    rng = torch.Generator().manual_seed(1215)
    a = torch.randint(-4, 5, (32, 32), generator=rng).to(getattr(torch, dtype)) / 8
    b = torch.randint(-4, 5, (64, 32), generator=rng).to(getattr(torch, dtype)) / 8
    _check_gemm(main, a, b, a[:, :rows].float().T @ b.float().T)
