"""1215 GM→UB 连续与跨步搬运（Compact 紧凑搬运 + Normal 跨步搬运 + 打包 FP4）。"""

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
def test_gm_to_ub_strided(dtype, rows, cols, gap):
    """Normal 模式：只搬有效列，源 GM 行间隔由 gap 控制，UB 行宽 64（32B 对齐）。"""

    @T.prim_func
    def main(src: T.Tensor((rows, cols + gap), dtype), out: T.Tensor((rows, cols), dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((rows, 64), dtype)
            T.copy(src[:, :cols], ub[:, :cols])
            T.copy(ub[:, :cols], out)

    source = _matrix(rows, cols + gap, dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(source.npu())
    torch.testing.assert_close(actual.cpu(), source[:, :cols], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["int8", "float16", "float32"])
@pytest.mark.parametrize("rows,cols", [(1, 1), (4, 15), (4, 17)])
def test_gm_to_ub_compact(dtype, rows, cols):
    """Compact 模式：GM 与 UB 均紧凑连续（每行不足 32B），合并为单段连续搬运。"""

    @T.prim_func
    def main(src: T.Tensor((rows, cols), dtype), out: T.Tensor((rows, cols), dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((rows, cols), dtype)
            T.copy(src, ub)
            T.copy(ub, out)

    source = _matrix(rows, cols, dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(source.npu())
    torch.testing.assert_close(actual.cpu(), source, rtol=0, atol=0)


@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="PyTorch float4_e2m1fn_x2 unavailable")
def test_gm_to_ub_fp4_packed():
    """打包 FP4 偶数列边界的 GM→UB→GM 往返搬运。"""

    @T.prim_func
    def main(src: T.Tensor((64,), "float4_e2m1fn"), out: T.Tensor((64,), "float4_e2m1fn")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float4_e2m1fn")
            T.copy(src, ub)
            T.copy(ub, out)

    src_bytes = torch.arange(32, dtype=torch.uint8)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(src_bytes.view(torch.float4_e2m1fn_x2).npu())
    torch.testing.assert_close(actual.view(torch.uint8).cpu(), src_bytes, rtol=0, atol=0)
