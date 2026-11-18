import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()


DTYPES = ["float32", "float16", "bfloat16", "int8", "int16", "int32"]


def _check_atomic(main, dtype, reduce):
    lane = torch.arange(32) % 9 - 4
    values = torch.stack((lane, 1 - lane)).to(getattr(torch, dtype))
    initial = (torch.arange(32) % 13 - 6).to(values.dtype)
    expected = reduce(torch.cat((initial[None, :], values)), dim=0).to(values.dtype)
    out = initial.npu()
    tilelang.compile(main, target="ascend")(values.npu(), out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ub_to_gm_atomic_add(dtype):
    """两核原子累加到同一行，检查 GM 原值也参与求和。"""

    @T.prim_func
    def main(src: T.Tensor((2, 32), dtype), dst: T.Tensor((32,), dtype)):
        with T.Kernel(2) as block:
            ub = T.alloc_shared((32,), dtype)
            T.copy(src[block, :], ub)
            T.set_atomic("add", dtype)
            T.copy(ub, dst)
            T.set_atomic_none()

    _check_atomic(main, dtype, torch.sum)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ub_to_gm_atomic_max(dtype):
    """两核原子取最大值，检查 GM 原值和正负贡献值。"""

    @T.prim_func
    def main(src: T.Tensor((2, 32), dtype), dst: T.Tensor((32,), dtype)):
        with T.Kernel(2) as block:
            ub = T.alloc_shared((32,), dtype)
            T.copy(src[block, :], ub)
            T.set_atomic("max", dtype)
            T.copy(ub, dst)
            T.set_atomic_none()

    _check_atomic(main, dtype, torch.amax)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ub_to_gm_atomic_min(dtype):
    """两核原子取最小值，检查 GM 原值和正负贡献值。"""

    @T.prim_func
    def main(src: T.Tensor((2, 32), dtype), dst: T.Tensor((32,), dtype)):
        with T.Kernel(2) as block:
            ub = T.alloc_shared((32,), dtype)
            T.copy(src[block, :], ub)
            T.set_atomic("min", dtype)
            T.copy(ub, dst)
            T.set_atomic_none()

    _check_atomic(main, dtype, torch.amin)


def test_l0c_atomic_add():
    """两个 Cube 各贡献一个矩阵，GM 原有的非零值必须保留在结果中。"""

    @T.prim_func
    def main(a: T.Tensor((32, 16), "float16"), b: T.Tensor((16, 16), "float16"), out: T.Tensor((16, 16), "float32")):
        with T.Kernel(2) as block:
            x = T.alloc_l1((16, 16), "float16")
            y = T.alloc_l1((16, 16), "float16")
            acc = T.alloc_l0c((16, 16), "float32")
            T.copy(a[block * 16 : (block + 1) * 16, :], x, transpose=False)
            T.copy(b, y, transpose=False)
            T.gemm(x, y, acc, transpose_B=True, clear_accum=True)
            T.set_atomic("add", "float32")
            T.copy(acc, out)
            T.set_atomic_none()

    a = (torch.arange(512).reshape(32, 16) % 13 - 6).half()
    initial = torch.full((16, 16), 7.0)
    out = initial.npu()
    tilelang.compile(main, target="ascend")(a.npu(), torch.eye(16).half().npu(), out)
    torch.testing.assert_close(out.cpu(), initial + a[:16] + a[16:], rtol=0, atol=0)
