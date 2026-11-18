"""1215 的指令参数检查，使用仓库已有的 lowering 辅助函数。"""

import pytest
from tvm import tirx
from tilelang.layout import make_ascend_l0c_layout, make_ascend_major_k_layout, make_ascend_nz_layout
from testing.ascend._ir import calls
from testing.ascend.transform.test_ascend_lower_copy import _lower


L2_POLICIES = [("normal_fv", 0), ("normal_lv", 1)]
CAST_MODES = [("float32", 0), ("float16", 1), ("bfloat16", 16)]


@pytest.mark.parametrize("policy,value", L2_POLICIES)
def test_gm_to_ub_l2_policy(policy, value):
    """GM→UB 读策略进入对齐搬入指令。"""
    src = tirx.decl_buffer((16, 32), "float32", scope="global")
    dst = tirx.decl_buffer((16, 32), "float32", scope="shared.dyn")
    (op,) = calls(_lower(src, dst, l2_cache_ctrl=policy), "tl.ascend_copy_gm_to_ubuf")
    assert int(op.args[8]) == value


@pytest.mark.parametrize("policy,value", L2_POLICIES)
def test_ub_to_gm_l2_policy(policy, value):
    """UB→GM 写策略进入对齐写回指令。"""
    src = tirx.decl_buffer((16, 32), "float32", scope="shared.dyn")
    dst = tirx.decl_buffer((16, 32), "float32", scope="global")
    (op,) = calls(_lower(src, dst, l2_cache_ctrl=policy), "tl.ascend_copy_ubuf_to_gm")
    assert int(op.args[5]) == value


@pytest.mark.parametrize("policy,value", L2_POLICIES)
def test_l0c_to_gm_l2_policy(policy, value):
    """L0C→GM 写策略进入 FIX 指令。"""
    src = tirx.decl_buffer((16, 32), "float32", scope="shared.l0c")
    dst = tirx.decl_buffer((16, 32), "float32", scope="global")
    (op,) = calls(_lower(src, dst, layouts={src: make_ascend_l0c_layout(src)}, l2_cache_ctrl=policy), "tl.ascend_copy_matrix_cc_to_gm")
    assert int(op.args[7]) == value


@pytest.mark.parametrize("dtype,mode", CAST_MODES)
def test_l0c_to_gm_cast_descriptor(dtype, mode):
    """检查 FIX 写回 GM 的 Cast 枚举和 NZ2ND 标志。"""
    src = tirx.decl_buffer((32, 64), "float32", scope="shared.l0c")
    dst = tirx.decl_buffer((32, 64), dtype, scope="global")
    after = _lower(src, dst, layouts={src: make_ascend_l0c_layout(src)})
    (op,) = calls(after, "tl.ascend_copy_matrix_cc_to_gm")
    assert int(op.args[10]) == mode
    assert int(op.args[13]) == 1


@pytest.mark.parametrize("target", [0, 1], ids=["aiv0", "aiv1"])
@pytest.mark.parametrize("dtype,mode", CAST_MODES)
def test_l0c_to_ub_cast_descriptor(dtype, mode, target):
    """检查 FIX 写入 UB 的 Cast、NZ2ND 和 AIV 编号。"""
    src = tirx.decl_buffer((32, 64), "float32", scope="shared.l0c")
    dst = tirx.decl_buffer((32, 64), dtype, scope="shared.dyn")
    after = _lower(src, dst, layouts={src: make_ascend_l0c_layout(src)}, sub_blockid=target)
    (op,) = calls(after, "tl.ascend_copy_matrix_cc_to_ub")
    assert int(op.args[8]) == target
    assert int(op.args[11]) == mode
    assert int(op.args[14]) == 1


def test_contiguous_ub_l1():
    """512字节原样搬运须生成 UB→L1 指令并覆盖16个32B单元。"""
    src = tirx.decl_buffer((256,), "int16", scope="shared.dyn")
    dst = tirx.decl_buffer((256,), "int16", scope="shared.l1")
    (op,) = calls(_lower(src, dst), "tl.ascend_copy_ubuf_to_cbuf")
    assert int(op.args[3]) * int(op.args[4]) == 16
    assert int(op.args[5]) == int(op.args[6]) == 0


@pytest.mark.parametrize("scope,intrin", [("shared.l0a", "ca"), ("shared.l0b", "cb")])
def test_l0_transpose_operand(scope, intrin):
    """分别检查 A、B 操作数使用自己的转置加载指令。"""
    src = tirx.decl_buffer((32, 16), "bfloat16", scope="shared.l1")
    dst = tirx.decl_buffer((16, 32), "bfloat16", scope=scope)
    layouts = {src: make_ascend_nz_layout(src), dst: make_ascend_major_k_layout(dst)}
    (op,) = calls(_lower(src, dst, layouts=layouts, transpose=True), "tl.ascend_load_cbuf_to_" + intrin)
    assert int(op.args[8]) == 1
