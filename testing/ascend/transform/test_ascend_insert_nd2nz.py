"""InsertNd2Nz recognizes physical layouts before scheduling."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_nz_layout
from tvm import tirx
from testing.ascend._ir import calls


def _module(
    *,
    rows=16,
    cols=128,
    padding=1,
    strides=None,
    staged=False,
    reshape=False,
    vf=None,
    src_dtype="bfloat16",
    dst_dtype="float32",
    dst_scope="shared.dyn",
):
    src = tirx.decl_buffer((2, rows, cols) if staged else (rows, cols), src_dtype, name="src", scope="shared.dyn", strides=strides)
    owner = tirx.decl_buffer((rows // 2, cols * 2), src_dtype, name="storage", scope="shared.dyn", data=src.data) if reshape else src
    dst = tirx.decl_buffer((2, rows + padding, cols) if staged else (rows + padding, cols), dst_dtype, name="dst", scope=dst_scope)
    i = tirx.Var("stage", "int32")

    def region(buffer):
        ranges = [tvm.ir.Range.from_min_extent(i, 1)] if staged else []
        return tirx.BufferRegion(buffer, ranges + [tvm.ir.Range(0, rows), tvm.ir.Range(0, cols)])

    body = tirx.Evaluate(T.copy(region(src), region(dst)))
    if vf:
        body = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], vf, body))
    if staged:
        body = tirx.For(i, 0, 2, tirx.ForKind.SERIAL, body)
    layout = make_ascend_nz_layout(dst) if dst_scope == "shared.l1" else make_ascend_compact_nz_layout(dst)
    root = tirx.SBlock([], [], [], "root", body, alloc_buffers=[owner, dst], annotations={"layout_map": {dst: layout}})
    return tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, root))})


@pytest.mark.parametrize("case", ["plain", "staged", "reshape", "simd", "simt"])
def test_scatter_regions_and_vf_boundary(case):
    before = _module(staged=case == "staged", reshape=case == "reshape", vf={"simd": "SIMD_VF", "simt": "SIMT_VF"}.get(case))
    after = transform.InsertNd2Nz()(before)
    if case == "simt":
        tvm.ir.assert_structural_equal(after, before)
        return
    (scatter,) = calls(after, "tl.ascend_nd2nz_scatter")
    assert [int(x) for x in scatter.args[2:4]] == [16, 128]
    assert [str(x.value) for x in scatter.args[4:]] == ["float", "bfloat16_t"]
    assert not calls(after, "tl.tileop.ascend_copy")
    # The destination footprint includes the bank-conflict padding row.
    analyzer = tvm.arith.Analyzer()
    assert int(analyzer.simplify(scatter.args[0].args[1])) == 16 * 128
    assert int(analyzer.simplify(scatter.args[1].args[1])) == 17 * 128
    if case == "staged":
        src_index = scatter.args[0].args[0].indices[0]
        dst_index = scatter.args[1].args[0].indices[0]
        assert src_index.same_as(dst_index)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"padding": 0}, "reserve exactly one padding row"),
        ({"strides": (256, 1)}, "compact trailing source matrix"),
        ({"staged": True, "strides": (2047, 128, 1)}, "32-byte-aligned source address"),
        ({"staged": True, "strides": (2048, 128, 1)}, None),
    ],
)
def test_scatter_storage_requirements(kwargs, error):
    before = _module(**kwargs)
    if error:
        with pytest.raises(ValueError, match=error):
            transform.InsertNd2Nz()(before)
    else:
        assert len(calls(transform.InsertNd2Nz()(before), "tl.ascend_nd2nz_scatter")) == 1


@pytest.mark.parametrize("dst_scope", ["shared.dyn", "shared.l1"])
def test_fp8_scatter_lowering(dst_scope):
    dtype, c_type = "float8_e4m3fn", "fp8_e4_t"
    mod = _module(src_dtype=dtype, dst_dtype=dtype, dst_scope=dst_scope, padding=16 if dst_scope == "shared.l1" else 1, cols=288)
    after = transform.InsertNd2Nz()(mod)
    (scatter,) = calls(after, "tl.ascend_nd2nz_scatter")
    assert [str(x.value) for x in scatter.args[4:]] == [c_type, c_type]
    assert len(calls(after, "tl.ascend_nd2nz_post_copy")) == int(dst_scope == "shared.l1")


@pytest.mark.parametrize("dst_scope", ["shared.dyn", "shared.l1"])
@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"rows": 17}, "rows divisible by 16"),
        ({"cols": 48}, "32-byte-aligned row|columns divisible by C0"),
        ({"dst_dtype": "float16"}, "does not support dtype conversion"),
        ({"src_dtype": "float32"}, "does not support dtype conversion"),
        ({"src_dtype": "float4_e2m1fn", "dst_dtype": "float4_e2m1fn"}, "Unsupported nd2nz dtype"),
    ],
)
def test_fp8_scatter_rejects_unsupported_shapes_and_casts(dst_scope, kwargs, error):
    options = dict(src_dtype="float8_e4m3fn", dst_dtype="float8_e4m3fn", dst_scope=dst_scope, padding=16 if dst_scope == "shared.l1" else 1)
    options.update(kwargs)
    with pytest.raises(ValueError, match=error):
        transform.InsertNd2Nz()(_module(**options))


def test_fp8_l1_scatter_rejects_noncompact_source():
    before = _module(src_dtype="float8_e4m3fn", dst_dtype="float8_e4m3fn", dst_scope="shared.l1", padding=16, strides=(256, 1))
    with pytest.raises(ValueError, match="compact ND source"):
        transform.InsertNd2Nz()(before)


def test_fp8_l1_scatter_rejects_unaligned_destination_rows():
    before = _module(src_dtype="float8_e4m3fn", dst_dtype="float8_e4m3fn", dst_scope="shared.l1", padding=1)
    with pytest.raises(ValueError, match="L1 row extent divisible by 16"):
        transform.InsertNd2Nz()(before)
