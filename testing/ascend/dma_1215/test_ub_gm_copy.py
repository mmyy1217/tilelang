"""1215 UB→GM 连续与跨步写回（Normal 跨步写回保护区验证 + Compact 紧凑写回）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()

DTYPES = ["int8", "int16", "int32", "float16", "bfloat16", "float32"]


def _matrix(rows, cols, dtype):
    return (torch.arange(rows * cols).reshape(rows, cols) % 61 - 30).to(getattr(torch, dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,cols", [(1, 1), (4, 15), (4, 16), (4, 17)])
def test_ub_to_gm_preserves_guards(dtype, rows, cols):
    """Normal 模式：从 64 元素行距（32B 对齐）的 UB 写回 GM，核对有效数据和四周保护区 −77。"""

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


@pytest.mark.parametrize("dtype", ["int8", "float16", "float32"])
@pytest.mark.parametrize("rows,cols", [(1, 1), (4, 15), (4, 17)])
def test_ub_to_gm_compact(dtype, rows, cols):
    """Compact 模式：紧凑 UB 整块写回连续 GM。"""

    @T.prim_func
    def main(src: T.Tensor((rows, cols), dtype), out: T.Tensor((rows, cols), dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((rows, cols), dtype)
            T.copy(src, ub)
            T.copy(ub, out)

    source = _matrix(rows, cols, dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(source.npu())
    torch.testing.assert_close(actual.cpu(), source, rtol=0, atol=0)
