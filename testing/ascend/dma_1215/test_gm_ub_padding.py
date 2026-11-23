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


@pytest.mark.parametrize("dtype", ["int8", "bfloat16", "float32"])
def test_gm_to_ub_padding_dynamic_row_offset(dtype):
    """最内维长度固定（cols=15），结合 T.dynamic 行维与动态行起点做右侧补齐。"""
    cols = 15
    elems_per_32b = 32 // torch.empty((), dtype=getattr(torch, dtype)).element_size()
    width = (cols + elems_per_32b - 1) // elems_per_32b * elems_per_32b
    pad_val = -1.0 if dtype.startswith(("float", "bfloat")) else -1
    M = T.dynamic("M")

    @T.prim_func
    def main(src: T.Tensor((M, 23), dtype), row_off: T.int32, out: T.Tensor((2, width), dtype)):
        with T.Kernel(1):
            T.assume(M >= 16)
            T.assume(row_off >= 0)
            ub = T.alloc_shared((2, width), dtype)
            T.copy(src[row_off : row_off + 2, :cols], ub[:, :cols], pad_value=pad_val)
            T.copy(ub, out)

    torch_dtype = getattr(torch, dtype)
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    for m_rows, row_off in [(18, 3), (25, 7)]:
        source = ((torch.arange(m_rows * 23).reshape(m_rows, 23) % 31) + 1).to(torch_dtype)
        expected = torch.full((2, width), -1, dtype=torch_dtype)
        expected[:, :cols] = source[row_off : row_off + 2, :cols]
        actual = kernel(source.npu(), row_off)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_gm_to_ub_padding_rejects_dynamic_inner_cols():
    """开启 pad_value 时最内维搬运长度必须为编译期静态常量，动态列长在编译期被拒绝。"""
    K = T.dynamic("K")

    @T.prim_func
    def main(src: T.Tensor((2, K), "float32"), valid_k: T.int32):
        with T.Kernel(1):
            T.assume(K >= 32)
            T.assume(valid_k >= 1)
            T.assume(valid_k <= 30)
            ub = T.alloc_shared((2, 32), "float32")
            T.copy(src[:, :valid_k], ub[:, :valid_k], pad_value=-1.0)

    with pytest.raises(Exception):
        tilelang.lower(main, target="ascend")
