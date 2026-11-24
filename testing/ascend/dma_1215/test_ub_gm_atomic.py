"""1215 UB→GM 写回时硬件原子更新（Atomic Add / Max / Min，覆盖 6 种白名单标量 dtype）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()

DTYPES = ["float32", "float16", "bfloat16", "int8", "int16", "int32"]


def _check_atomic_minmax(main, dtype, reduce):
    lane = torch.arange(32) % 9 - 4
    values = torch.stack((lane, 1 - lane)).to(getattr(torch, dtype))
    initial = (torch.arange(32) % 13 - 6).to(values.dtype)
    expected = reduce(torch.cat((initial[None, :], values)), dim=0).to(values.dtype)
    out = initial.npu()
    tilelang.compile(main, target="ascend")(values.npu(), out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ub_to_gm_atomic_add(dtype):
    """两核原子累加到同一行，检查 GM 非零初值也参与求和。"""

    @T.prim_func
    def main(src: T.Tensor((2, 32), dtype), dst: T.Tensor((32,), dtype)):
        with T.Kernel(2) as block:
            ub = T.alloc_shared((32,), dtype)
            T.copy(src[block, :], ub)
            T.set_atomic("add", dtype)
            T.copy(ub, dst)
            T.set_atomic_none()

    lane = torch.arange(32) % 9 - 4
    values = torch.stack((lane, 1 - lane)).to(getattr(torch, dtype))
    initial = (torch.arange(32) % 13 - 6).to(values.dtype)
    expected = (initial + values[0] + values[1]).to(values.dtype)
    out = initial.npu()
    tilelang.compile(main, target="ascend")(values.npu(), out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int32"])
def test_ub_to_gm_atomic_add_strided_subregion(dtype):
    """两核向 2D GM 子区域原子累加，验证四周保护区未被修改。"""

    @T.prim_func
    def main(src: T.Tensor((2, 4, 16), dtype), dst: T.Tensor((6, 32), dtype)):
        with T.Kernel(2) as block:
            ub = T.alloc_shared((4, 32), dtype)
            T.copy(src[block, :, :], ub[:, :16])
            T.set_atomic("add", dtype)
            T.copy(ub[:, :16], dst[1:5, 8:24])
            T.set_atomic_none()

    torch_dtype = getattr(torch, dtype)
    values = (torch.arange(2 * 4 * 16).reshape(2, 4, 16) % 7 - 3).to(torch_dtype)
    expected = torch.full((6, 32), 5, dtype=torch_dtype)
    expected[1:5, 8:24] = expected[1:5, 8:24] + values[0] + values[1]
    out = torch.full((6, 32), 5, dtype=torch_dtype).npu()
    tilelang.compile(main, target="ascend")(values.npu(), out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


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

    _check_atomic_minmax(main, dtype, torch.amax)


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

    _check_atomic_minmax(main, dtype, torch.amin)
