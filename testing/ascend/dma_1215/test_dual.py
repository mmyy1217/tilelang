import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx
from testing.ascend._ir import calls
from testing.ascend.transform.test_ascend_rewrite_dual_copy import _module, _cthread


def _check_split(copy, *, full_side, axis, manual):
    # 在两个 AIV 的 cthread 作用域检查偏移和切片长度。
    body = _cthread(tirx.Evaluate(copy))
    sid = body.node.var
    if manual:
        body = tirx.SBlock([], [], [], "VECTOR", body)
    before = _module(body)
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": not manual}):
        after = transform.RewriteDualCopy()(before)
    (copy,) = calls(after, "tl.tileop.ascend_copy")
    full, half = copy.args[full_side], copy.args[1 - full_side]
    extent = half.args[axis + 2]
    expected_index = sid * extent
    tvm.ir.assert_structural_equal(full.args[0].indices[axis], expected_index)
    tvm.ir.assert_structural_equal(full.args[axis + 2], extent)
    assert all(int(index) == 0 for dimension, index in enumerate(full.args[0].indices) if dimension != axis)
    assert not any(key in copy.annotations for key in ("double", "dual_dst_ctl", "unit_flag_ctrl"))


@pytest.mark.parametrize("manual", [False, True], ids=["automatic", "manual-vector"])
@pytest.mark.parametrize("src_shape,dst_shape,axis", [((128,), (64,), 0), ((64, 64), (32, 64), 0)], ids=["1d", "2d"])
def test_gm_to_ub_dual_copy(manual, src_shape, dst_shape, axis):
    """GM 源按 AIV 分半，UB 目标从本地起点写入。"""
    src = tirx.decl_buffer(src_shape, "float32", scope="global")
    dst = tirx.decl_buffer(dst_shape, "float32", scope="shared.dyn")
    copy = T.dual_copy(src, dst, unit_flag_ctrl=3)
    _check_split(copy, full_side=0, axis=axis, manual=manual)


@pytest.mark.parametrize("manual", [False, True], ids=["automatic", "manual-vector"])
@pytest.mark.parametrize("src_shape,dst_shape,axis", [((64,), (128,), 0), ((64, 32), (64, 64), 1)], ids=["1d", "2d"])
def test_ub_to_gm_dual_copy(manual, src_shape, dst_shape, axis):
    """各 AIV 的 UB 写回 GM 的不同半区。"""
    src = tirx.decl_buffer(src_shape, "float32", scope="shared.dyn")
    dst = tirx.decl_buffer(dst_shape, "float32", scope="global")
    copy = T.dual_copy(src, dst, unit_flag_ctrl=3)
    _check_split(copy, full_side=1, axis=axis, manual=manual)


@pytest.mark.parametrize("manual", [False, True], ids=["automatic", "manual-vector"])
def test_ub_to_l1_dual_copy(manual):
    """各 AIV 的 UB 写入同一 L1 分配的不同半区。"""
    src_shape, dst_shape, axis = ((64,), (128,), 0)
    src = tirx.decl_buffer(src_shape, "float32", scope="shared.dyn")
    dst = tirx.decl_buffer(dst_shape, "float32", scope="shared.l1")
    copy = T.dual_copy(src, dst, unit_flag_ctrl=3)
    _check_split(copy, full_side=1, axis=axis, manual=manual)
