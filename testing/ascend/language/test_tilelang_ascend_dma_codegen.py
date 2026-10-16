"""DMA address expressions must emit valid C++ statements."""

import re

import pytest
import tilelang
from tilelang.ascend import language as T
from tilelang import tvm


def runtime_divisor_copy():
    @T.prim_func
    def kernel(
        src: T.Tensor((16, 8), "float32"),
        dst: T.Tensor((16, 8), "float32"),
        num_cols: T.int32,
    ):
        with T.Kernel(1):
            buf = T.alloc_shared((4, 8), "float32")
            for i in T.serial(4):
                row = i // num_cols
                T.copy(src[row * 4, 0], buf)
                T.copy(buf, dst[row * 4, 0])

    return kernel


def test_mte_copy_hoists_let_bindings_before_calls():
    source = tilelang.lower(runtime_divisor_copy(), target="ascend").kernel_source

    for call_name in ("asc_copy_gm2ub_align", "asc_copy_ub2gm_align"):
        call_start = source.index(call_name)
        call_end = source.index(");", call_start)
        call = source[call_start:call_end]
        assert not re.search(r"\bint(?:32|64)_t\s+\w+\s*=", call), source


@pytest.mark.parametrize("scatter", [False, True], ids=["post-copy", "scatter"])
@pytest.mark.parametrize("c_type", ["unknown", "int8_t"])
def test_nd2nz_codegen_rejects_unknown_dtype(scatter, c_type):
    tirx = tvm.tirx
    if scatter:
        name = "tl.ascend_nd2nz_scatter"
        arguments = [T.uint64(0), T.uint64(0), T.int32(32), T.int32(256), tirx.StringImm(c_type), tirx.StringImm(c_type)]
    else:
        name = "tl.ascend_nd2nz_post_copy"
        arguments = [T.uint64(0), T.uint64(0), T.int32(32), T.int32(256), T.int32(32), tirx.StringImm(c_type)]
    body = tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get(name), arguments))
    function = tirx.PrimFunc([], body).with_attr("global_symbol", "main").with_attr("calling_conv", 2)
    build = tvm.get_global_func("target.build.tilelang_ascend_without_compile")
    with pytest.raises(tvm.error.InternalError, match="Unsupported ND->NZ dtype name"):
        build(tvm.IRModule({"main": function}), tvm.target.Target("ascend"))
