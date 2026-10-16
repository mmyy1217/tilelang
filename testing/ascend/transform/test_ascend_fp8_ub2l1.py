"""FP8 UB-to-L1 copies preserve physical gaps and prepacked NZ data."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_major_k_layout
from tvm import tirx
from testing.ascend._ir import calls
from testing.ascend.transform.test_ascend_lower_copy import _equal, _lower, _region


def test_fp8_prepacked_nz_half_tile_to_l1():
    dtype = "float8_e4m3fn"
    src = tirx.decl_buffer((105, 256), dtype, name="src", scope="shared.dyn")
    dst = tirx.decl_buffer((208, 256), dtype, name="dst", scope="shared.l1")
    body = tirx.Evaluate(T.copy(_region(src, [104, 256]), _region(dst, [104, 256], [104, 0])))
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=[src, dst],
        annotations={"layout_map": {src: make_ascend_compact_nz_layout(src), dst: make_ascend_major_k_layout(dst)}},
    )
    target = tvm.target.Target("ascend")
    module = tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, root)).with_attr("target", target)})
    with target:
        after = transform.AscendLowerTileOp()(transform.InsertNd2Nz()(module))
    (copy,) = calls(after, "tl.ascend_copy_ubuf_to_cbuf")
    _equal(copy.args[3:], [8, 104, 1, 104])
    _equal([copy.args[0].args[2]], [104 * 32])


@pytest.mark.parametrize(
    "src_width,dst_width,expected",
    [
        (32, 32, [1, 4, 0, 0]),
        (64, 32, [4, 1, 1, 0]),
        (32, 96, [4, 1, 0, 2]),
        (64, 96, [4, 1, 1, 2]),
    ],
)
def test_fp8_raw_pitches(src_width, dst_width, expected):
    dtype = "float8_e4m3fn"
    src = tirx.decl_buffer((4, src_width), dtype, scope="shared.dyn")
    dst = tirx.decl_buffer((4, dst_width), dtype, scope="shared.l1")
    after = _lower(src, dst, _region(src, [4, 32]), _region(dst, [4, 32]))
    (copy,) = calls(after, "tl.ascend_copy_ubuf_to_cbuf")
    _equal(copy.args[3:], expected)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("rows,row,col", [(104, 0, 0), (72, 104, 32), (7, 3, 64)])
def test_fp8_nz_offsets_and_tails(compact, rows, row, col):
    dtype = "float8_e4m3fn"
    src = tirx.decl_buffer((2, 105 if compact else 112, 256), dtype, scope="shared.dyn")
    dst = tirx.decl_buffer((3, 208, 256), dtype, scope="shared.l1")
    layout = make_ascend_compact_nz_layout if compact else make_ascend_major_k_layout
    after = _lower(
        src,
        dst,
        _region(src, [1, rows, 128], [1, 1, 32]),
        _region(dst, [1, rows, 128], [2, row, col]),
        layouts={src: layout(src), dst: make_ascend_major_k_layout(dst)},
    )
    (copy,) = calls(after, "tl.ascend_copy_ubuf_to_cbuf")
    pitch = 105 if compact else 112
    _equal(copy.args[3:], [4, rows, pitch - rows, 208 - rows])
    _equal([copy.args[1].args[2], copy.args[0].args[2]], [pitch * 256 + pitch * 32 + 32, 2 * 208 * 256 + col // 32 * 208 * 32 + row * 32])


@pytest.mark.parametrize(
    "width,src_pitch,dst_pitch,match",
    [
        (31, 64, 64, "32-byte aligned burst length"),
        (32, 65, 64, "32-byte aligned source gap"),
        (32, 64, 65, "32-byte aligned destination gap"),
        (64, 32, 96, "source gap"),
    ],
)
def test_fp8_raw_rejects_invalid_dma_geometry(width, src_pitch, dst_pitch, match):
    src = tirx.decl_buffer((4, width), "float8_e4m3fn", scope="shared.dyn", strides=[src_pitch, 1])
    dst = tirx.decl_buffer((4, width), "float8_e4m3fn", scope="shared.l1", strides=[dst_pitch, 1])
    with pytest.raises(tvm.error.InternalError, match=match):
        _lower(src, dst)


@pytest.mark.parametrize("count,match", [(4095, None), (4096, "burst count")])
def test_fp8_raw_burst_count_limit(count, match):
    src = tirx.decl_buffer((count, 64), "float8_e4m3fn", scope="shared.dyn")
    dst = tirx.decl_buffer((count, 64), "float8_e4m3fn", scope="shared.l1")
    if match:
        with pytest.raises(tvm.error.InternalError, match=match):
            _lower(src, dst, _region(src, [count, 32]), _region(dst, [count, 32]))
    else:
        after = _lower(src, dst, _region(src, [count, 32]), _region(dst, [count, 32]))
        (copy,) = calls(after, "tl.ascend_copy_ubuf_to_cbuf")
        _equal(copy.args[3:], [count, 1, 1, 1])


def test_fp8_raw_rejects_unaligned_offset():
    src = tirx.decl_buffer((4, 64), "float8_e4m3fn", scope="shared.dyn")
    dst = tirx.decl_buffer((4, 64), "float8_e4m3fn", scope="shared.l1")
    with pytest.raises(tvm.error.InternalError, match="source offset"):
        _lower(src, dst, _region(src, [4, 32], [0, 1]), _region(dst, [4, 32]))


def test_fp8_nz_rejects_partial_c0():
    src = tirx.decl_buffer((32, 128), "float8_e4m3fn", scope="shared.dyn")
    dst = tirx.decl_buffer((32, 128), "float8_e4m3fn", scope="shared.l1")
    with pytest.raises(tvm.error.InternalError, match="whole 32-element column groups"):
        _lower(
            src,
            dst,
            _region(src, [32, 16]),
            _region(dst, [32, 16]),
            layouts={src: make_ascend_compact_nz_layout(src), dst: make_ascend_major_k_layout(dst)},
        )


def test_fp8_copy_does_not_convert_dtype_or_layout():
    src = tirx.decl_buffer((32, 128), "float8_e4m3fn", scope="shared.dyn")
    dst = tirx.decl_buffer((32, 128), "int8", scope="shared.l1")
    with pytest.raises(tvm.error.InternalError, match="identical scalar dtypes"):
        _lower(src, dst)
    dst = tirx.decl_buffer((32, 128), "float8_e4m3fn", scope="shared.l1")
    with pytest.raises(tvm.error.InternalError, match="cannot convert ND/NZ"):
        _lower(src, dst, layouts={dst: make_ascend_major_k_layout(dst)})


def test_fp8_pointer_backed_nz_view_preserves_binding_and_maps_offset():
    arena = tirx.decl_buffer((8192,), "uint8", scope="shared.dyn")
    src = tirx.decl_buffer((2, 33, 64), "float8_e4m3fn", name="nz_view", scope="shared.dyn")
    dst = tirx.decl_buffer((32, 64), "float8_e4m3fn", scope="shared.l1")
    body = tirx.SeqStmt(
        [
            tirx.Bind(src.data, arena.access_ptr("rw", offset=256)),
            tirx.Evaluate(T.copy(_region(src, [1, 16, 32], [1, 3, 32]), _region(dst, [16, 32]))),
        ]
    )
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=[arena, dst],
        annotations={"layout_map": {src: make_ascend_compact_nz_layout(src), dst: make_ascend_major_k_layout(dst)}},
    )
    target = tvm.target.Target("ascend")
    module = tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, root)).with_attr("target", target)})
    with target:
        after = transform.AscendLowerTileOp()(module)
    (copy,) = calls(after, "tl.ascend_copy_ubuf_to_cbuf")
    _equal([copy.args[1].args[2]], [33 * 64 + 33 * 32 + 3 * 32])
    assert not tirx.analysis.undefined_vars(after["main"].body)


def test_fp8_nz_dynamic_rows_keep_physical_pitch():
    src = tirx.decl_buffer((105, 256), "float8_e4m3fn", scope="shared.dyn")
    dst = tirx.decl_buffer((208, 256), "float8_e4m3fn", scope="shared.l1")
    rows = tirx.Var("rows", "int32")
    after = _lower(
        src,
        dst,
        _region(src, [rows, 256]),
        _region(dst, [rows, 256]),
        layouts={src: make_ascend_compact_nz_layout(src), dst: make_ascend_major_k_layout(dst)},
        wrap=lambda body: tirx.For(rows, 1, 104, tirx.ForKind.SERIAL, body),
    )
    (copy,) = calls(after, "tl.ascend_copy_ubuf_to_cbuf")
    _equal(copy.args[3:], [8, rows, 105 - rows, 208 - rows])


def test_fp8_empty_region_emits_no_dma():
    src = tirx.decl_buffer((4, 64), "float8_e4m3fn", scope="shared.dyn")
    dst = tirx.decl_buffer((4, 64), "float8_e4m3fn", scope="shared.l1")
    after = _lower(src, dst, _region(src, [0, 32]), _region(dst, [0, 32]))
    assert not calls(after, "tl.ascend_copy_ubuf_to_cbuf")
