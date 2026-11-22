"""1215 GM→UB 行尾不足 32B 自动右补齐（pad_value / data_select，覆盖 8/16/32-bit 标量）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()

PAD_DTYPES = ["int8", "int16", "int32", "float16", "bfloat16", "float32"]


def _check_right_padding(main, dtype, cols, width):
    torch_dtype = getattr(torch, dtype)
    source = (torch.arange(2 * cols).reshape(2, cols) % 31).to(torch_dtype)
    expected = torch.full((2, width), -1, dtype=torch_dtype)
    expected[:, :cols] = source
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(source.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", PAD_DTYPES)
@pytest.mark.parametrize("cols", [3, 30])
def test_gm_to_ub_pad_value(dtype, cols):
    """用 pad_value 将不足 32B 的行尾自动补齐为 −1。"""
    elems_per_32b = 32 // torch.empty((), dtype=getattr(torch, dtype)).element_size()
    width = (cols + elems_per_32b - 1) // elems_per_32b * elems_per_32b
    pad_val = -1.0 if dtype.startswith(("float", "bfloat")) else -1

    @T.prim_func
    def main(src: T.Tensor((2, cols), dtype), out: T.Tensor((2, width), dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((2, width), dtype)
            T.copy(src, ub[:, :cols], pad_value=pad_val)
            T.copy(ub, out)

    _check_right_padding(main, dtype, cols, width)


@pytest.mark.parametrize("dtype", PAD_DTYPES)
@pytest.mark.parametrize("cols", [3, 30])
def test_gm_to_ub_data_select(dtype, cols):
    """先设置填充寄存器，搬入时由 data_select 复用 −1 补齐到 32B 边界。"""
    elems_per_32b = 32 // torch.empty((), dtype=getattr(torch, dtype)).element_size()
    width = (cols + elems_per_32b - 1) // elems_per_32b * elems_per_32b
    pad_val = -1.0 if dtype.startswith(("float", "bfloat")) else -1

    @T.prim_func
    def main(src: T.Tensor((2, cols), dtype), out: T.Tensor((2, width), dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((2, width), dtype)
            T.ascend_set_copy_pad_value(pad_val, dtype=dtype)
            T.copy(src, ub[:, :cols], data_select=True)
            T.copy(ub, out)

    _check_right_padding(main, dtype, cols, width)
