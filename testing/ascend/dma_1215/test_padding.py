import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()


def _check_right_padding(main, cols, width):
    source = torch.arange(2 * cols, dtype=torch.float32).reshape(2, cols)
    expected = torch.full((2, width), -1.0)
    expected[:, :cols] = source
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(source.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("cols", [3, 30])
def test_gm_to_ub_pad_value(cols):
    """用 pad_value 请求行尾补 −1。"""
    width = (cols + 7) // 8 * 8

    @T.prim_func
    def main(src: T.Tensor((2, cols), "float32"), out: T.Tensor((2, width), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((2, width), "float32")
            T.copy(src, ub[:, :cols], pad_value=-1.0)
            T.copy(ub, out)

    _check_right_padding(main, cols, width)


@pytest.mark.parametrize("cols", [3, 30])
def test_gm_to_ub_data_select(cols):
    """先设置填充寄存器，搬入时由 data_select 复用 −1。"""
    width = (cols + 7) // 8 * 8

    @T.prim_func
    def main(src: T.Tensor((2, cols), "float32"), out: T.Tensor((2, width), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((2, width), "float32")
            T.ascend_set_copy_pad_value(-1.0, dtype="float32")
            T.copy(src, ub[:, :cols], data_select=True)
            T.copy(ub, out)

    _check_right_padding(main, cols, width)
