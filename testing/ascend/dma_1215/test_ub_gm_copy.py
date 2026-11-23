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


EXTRA_DTYPES = [
    ("uint8", 1),
    ("uint16", 2),
    ("uint32", 4),
    ("float8_e4m3fn", 1),
    ("float8_e8m0fnu", 1),
    ("int64", 8),
    ("uint64", 8),
    ("float64", 8),
]


@pytest.mark.parametrize("dtype,elem_bytes", EXTRA_DTYPES)
def test_ub_to_gm_all_remaining_dtypes(dtype, elem_bytes):
    """补齐 uint8/16/32、fp8(e4m3/e8m0) 及 64-bit 的 Normal 跨步写回与四周保护区校验，按字节精确比对。"""
    if not hasattr(torch, dtype):
        pytest.skip(f"PyTorch {dtype} unavailable")
    torch_dtype = getattr(torch, dtype)
    rows, cols = 4, 15

    @T.prim_func
    def main(src: T.Tensor((rows, cols), dtype), dst: T.Tensor((rows + 2, cols + 16), dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((rows, 32), dtype)
            T.copy(src, ub[:, :cols])
            T.copy(ub[:, :cols], dst[1 : rows + 1, 8 : 8 + cols])

    src_raw = (torch.arange(rows * cols * elem_bytes, dtype=torch.uint8) * 17 + 3).reshape(rows, cols, elem_bytes)
    source = src_raw.view(torch_dtype).reshape(rows, cols)
    guard_raw = torch.full((rows + 2, cols + 16, elem_bytes), 0x5A, dtype=torch.uint8)
    expected_raw = guard_raw.clone()
    expected_raw[1 : rows + 1, 8 : 8 + cols, :] = src_raw
    output = guard_raw.view(torch_dtype).reshape(rows + 2, cols + 16).npu()
    tilelang.compile(main, target="ascend")(source.npu(), output)
    torch.testing.assert_close(output.cpu().view(torch.uint8), expected_raw.reshape(rows + 2, (cols + 16) * elem_bytes), rtol=0, atol=0)


@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="PyTorch float4_e2m1fn_x2 unavailable")
def test_ub_to_gm_fp4_packed():
    """打包 FP4 偶数列边界的 UB→GM 写回与邻近保护字节校验。"""

    @T.prim_func
    def main(src: T.Tensor((64,), "float4_e2m1fn"), dst: T.Tensor((96,), "float4_e2m1fn")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float4_e2m1fn")
            T.copy(src, ub)
            T.copy(ub, dst[16:80])

    src_bytes = torch.arange(32, dtype=torch.uint8)
    expected = torch.full((48,), 0x5A, dtype=torch.uint8)
    expected[8:40] = src_bytes
    output = torch.full((48,), 0x5A, dtype=torch.uint8).view(torch.float4_e2m1fn_x2).npu()
    tilelang.compile(main, target="ascend")(src_bytes.view(torch.float4_e2m1fn_x2).npu(), output)
    torch.testing.assert_close(output.view(torch.uint8).cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["int8", "bfloat16", "float32"])
def test_ub_to_gm_5d_and_dynamic_sources(dtype):
    """3D 流水 UB 结合 T.dynamic 维度、分核 bx 与标量偏移跨步写回 5D GM 单平面子区域，核对四周保护区。"""
    M = T.dynamic("M")
    N = T.dynamic("N")

    @T.prim_func
    def main(
        src: T.Tensor((2, 4, 15), dtype),
        row_off: T.int32,
        col_off: T.int32,
        dst: T.Tensor((2, 2, 2, M, N), dtype),
    ):
        with T.Kernel(2) as bx:
            T.assume(M >= 20)
            T.assume(N >= 30)
            T.assume(row_off >= 0)
            T.assume(col_off >= 0)
            ub = T.alloc_shared((2, 4, 64), dtype)
            for ko in T.serial(1):
                stage = (ko + 1) % 2
                r0 = bx * 4 + row_off
                c0 = ko * 15 + col_off
                T.copy(src[bx, :, :], ub[stage, :, :15])
                T.copy(ub[stage, :, :15], dst[1, 0, 1, r0 : r0 + 4, c0 : c0 + 15])

    source = (torch.arange(2 * 4 * 15).reshape(2, 4, 15) % 61 - 30).to(getattr(torch, dtype))
    kernel = tilelang.compile(main, target="ascend")
    for shape, row_off, col_off in [((23, 37), 2, 3), ((31, 45), 5, 7)]:
        expected = torch.full((2, 2, 2, *shape), -77, dtype=source.dtype)
        expected[1, 0, 1, row_off : row_off + 4, col_off : col_off + 15] = source[0]
        expected[1, 0, 1, 4 + row_off : 8 + row_off, col_off : col_off + 15] = source[1]
        output = torch.full_like(expected, -77).npu()
        kernel(source.npu(), row_off, col_off, output)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

