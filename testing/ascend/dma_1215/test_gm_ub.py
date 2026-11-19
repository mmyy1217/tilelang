"""1215 GM 与 UB 之间的搬运和 GM 输出保护区。"""

import re

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


DTYPES = ["int8", "int16", "int32", "float16", "bfloat16", "float32"]
SHAPES = [(1, 1, 0), (4, 15, 7), (4, 16, 0), (4, 17, 7)]


def _matrix(rows, cols, dtype):
    return (torch.arange(rows * cols).reshape(rows, cols) % 61 - 30).to(getattr(torch, dtype))


@tilelang.testing.requires_ascend
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


@pytest.mark.parametrize("policy,mode", [("normal_fv", 0), ("normal_lv", 1), ("notalloc_clean", 5)])
def test_gm_to_ub_l2_policy(policy, mode):
    """检查 GM→UB 指令的 L2 读策略，回程保持默认。"""

    @T.prim_func
    def main(src: T.Tensor((4, 16), "float32"), out: T.Tensor((4, 16), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 16), "float32")
            T.copy(src, ub, l2_cache_ctrl=policy)
            T.copy(ub, out)

    source = tilelang.lower(main, target="ascend").kernel_source
    assert re.search(rf"\basc_copy_gm2ub_align\([^;]*, static_cast<asc_load_l2_cache_mode>\({mode}\),", source), source


@tilelang.testing.requires_ascend
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


@pytest.mark.parametrize("policy,mode", [("normal_fv", 0), ("normal_lv", 1), ("notalloc_ci", 4), ("notalloc_pw", 5)])
def test_ub_to_gm_l2_policy(policy, mode):
    """检查 UB→GM 指令的 L2 写策略，读入保持默认。"""

    @T.prim_func
    def main(src: T.Tensor((4, 16), "float32"), out: T.Tensor((4, 16), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 16), "float32")
            T.copy(src, ub)
            T.copy(ub, out, l2_cache_ctrl=policy)

    source = tilelang.lower(main, target="ascend").kernel_source
    assert re.search(rf"\basc_copy_ub2gm_align\([^;]*, static_cast<asc_store_l2_cache_mode>\({mode}\),", source), source
