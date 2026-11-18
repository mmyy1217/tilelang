import pytest
from tilelang.layout import (
    make_ascend_l0c_layout,
)
from tvm import tirx
from testing.ascend._ir import calls
from testing.ascend.transform.test_ascend_lower_copy import _region, _lower, _equal


@pytest.mark.parametrize("dual,src_shape", [(0, (32, 64)), (1, (64, 64)), (2, (32, 128))], ids=["single", "split-m", "split-n"])
def test_fix_nz_uses_physical_panel_pitch_and_slice_origin(dual, src_shape):
    # 检查单目标和 M/N 双目标的 NZ 模式、分形列距及 UB 写入偏移。
    src = tirx.decl_buffer(src_shape, "float32", name="src", scope="shared.l0c")
    dst = tirx.decl_buffer((2, 64, 96), "float32", name="dst", scope="shared")
    after = _lower(
        src,
        dst,
        dst_region=_region(dst, [1, 32, 64], [1, 16, 16]),
        layouts={src: make_ascend_l0c_layout(src), dst: make_ascend_l0c_layout(dst)},
        annotations={"dual_dst_ctl": dual},
        unit_flag_ctrl=3,
    )
    (copy,) = calls(after, "tl.ascend_copy_matrix_cc_to_ub")
    _equal(copy.args[3:8], [src_shape[1], src_shape[0], 64 * 16, src_shape[0], dual])
    _equal([copy.args[0].args[2], copy.args[10], copy.args[14]], [64 * 96 + 64 * 16 + 16 * 16, 3, 0])
