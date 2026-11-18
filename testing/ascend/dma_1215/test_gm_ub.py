"""1215 GM/UB byte-length and row-stride copies, including untouched output guards."""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()


DTYPES = ["int8", "int16", "int32", "float16", "bfloat16", "float32"]
SHAPES = [(1, 1, 0), (4, 15, 7), (4, 16, 0), (4, 17, 7)]


def _matrix(rows, cols, dtype):
    return (torch.arange(rows * cols).reshape(rows, cols) % 61 - 30).to(getattr(torch, dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,cols,gap", SHAPES)
def test_gm_to_ub(dtype, rows, cols, gap):
    """只搬有效列，源 GM 行间隔由 gap 控制，读回时比较每个元素。"""

    @T.prim_func
    def main(src: T.Tensor((rows, cols + gap), dtype), out: T.Tensor((rows, cols), dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((rows, 64), dtype)
            T.copy(src[:, :cols], ub[:, :cols])
            T.copy(ub[:, :cols], out)

    source = _matrix(rows, cols + gap, dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(source.npu())
    torch.testing.assert_close(actual.cpu(), source[:, :cols], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,cols", [(1, 1), (4, 15), (4, 16), (4, 17)])
def test_ub_to_gm_preserves_guards(dtype, rows, cols):
    """从 64 元素行距的 UB 写回 GM，逐元素核对有效数据和四周保护区。"""

    @T.prim_func
    def main(src: T.Tensor((rows, cols), dtype), dst: T.Tensor((rows + 2, cols + 16), dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((rows, 64), dtype)
            T.copy(src, ub[:, :cols])
            T.copy(ub[:, :cols], dst[1 : rows + 1, 8 : 8 + cols])

    source = _matrix(rows, cols, dtype)
    expected = torch.full((rows + 2, cols + 16), -77, dtype=source.dtype)
    expected[1 : rows + 1, 8 : 8 + cols] = source
    output = torch.full_like(expected, -77).npu()
    tilelang.compile(main, target="ascend")(source.npu(), output)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
