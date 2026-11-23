"""1215 GM→L1 普通矩阵搬入并转成 NZ 分形（ND→NZ）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


# DMA 使用原 dtype，L1 读回使用同位宽且 Cube 支持的 dtype。
READBACK = {
    "int8": "int8",
    "uint8": "int8",
    "float8_e4m3fn": "int8",
    "int16": "float16",
    "uint16": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "int32": "float32",
    "uint32": "float32",
    "float32": "float32",
}

DEVICE = tilelang.testing.requires_ascend


def _matrix_data(shape, dtype):
    rng = torch.Generator().manual_seed(1215)
    return torch.randint(1, 63, shape, generator=rng).to(getattr(torch, READBACK[dtype]))


def _readback_dtypes(dtype):
    read_dtype = READBACK[dtype]
    return read_dtype, "int32" if read_dtype == "int8" else "float32"


@T.macro
def _read_l1(matrix, identity, out, rows, cols, read_dtype, acc_dtype):
    # 只读回 L1。用同位宽视图和单位矩阵保留 DMA 搬入的数据。
    weight = T.alloc_l1((cols, cols), read_dtype)
    acc = T.alloc_l0c((rows, cols), acc_dtype)
    T.copy(identity, weight, transpose=False)
    payload = T.view(matrix, dtype=read_dtype)
    T.gemm(payload, weight, acc, transpose_B=True, clear_accum=True)
    T.copy(acc, out)


@T.macro
def _read_l1_l0(matrix, identity, out, rows, cols, out_cols, dtype, read_dtype, acc_dtype):
    # 非整分形 L1 经 L0A/L0B 读回，验证编译器对非整分形 alloc_l1 的自动补齐。
    weight = T.alloc_l1((out_cols, cols), read_dtype)
    a_l0 = T.alloc_l0a((rows, cols), dtype)
    b_l0 = T.alloc_l0b((out_cols, cols), read_dtype)
    acc = T.alloc_l0c((rows, out_cols), acc_dtype)
    T.copy(identity, weight, transpose=False)
    T.copy(matrix, a_l0, transpose=False)
    T.copy(weight, b_l0, transpose=False)
    T.gemm(T.view(a_l0, dtype=read_dtype), b_l0, acc, transpose_B=True, clear_accum=True)
    T.copy(acc, out)


def _assert_readback(main, dtype, expected, *inputs):
    identity = torch.eye(expected.shape[-1]).to(getattr(torch, READBACK[dtype])).npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    actual = kernel(*inputs, identity).cpu()
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
def test_nd_to_nz(dtype):
    """完整 ND 矩阵 16×32 逐元素搬入 L1。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(src: T.Tensor((16, 32), dtype), eye: T.Tensor((32, 32), read_dtype), out: T.Tensor((16, 32), acc_dtype)):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 32), dtype)
            T.copy(src, l1, transpose=False)
            _read_l1(l1, eye, out, 16, 32, read_dtype, acc_dtype)

    source = _matrix_data((16, 32), dtype)
    _assert_readback(main, dtype, source, source.view(getattr(torch, dtype)).npu())


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
def test_nd_to_nz_gm_slice(dtype):
    """从非 32B 对齐行步长的 (20, 37) GM 矩阵中切取界内子块 [2:18, 3:35] 搬入 L1。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(src: T.Tensor((20, 37), dtype), eye: T.Tensor((32, 32), read_dtype), out: T.Tensor((16, 32), acc_dtype)):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 32), dtype)
            T.copy(src[2:18, 3:35], l1, transpose=False)
            _read_l1(l1, eye, out, 16, 32, read_dtype, acc_dtype)

    source = _matrix_data((20, 37), dtype)
    _assert_readback(main, dtype, source[2:18, 3:35], source.view(getattr(torch, dtype)).npu())


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
def test_nd_to_nz_non_fractal_alloc(dtype):
    """直接申请非整分形逻辑 shape (13, 29) 的 L1 并从 GM 界内切片搬入。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)
    rows, cols, out_cols = 13, 29, 32

    @T.prim_func
    def main(
        src: T.Tensor((20, 37), dtype),
        eye: T.Tensor((out_cols, cols), read_dtype),
        out: T.Tensor((rows, out_cols), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((rows, cols), dtype)
            T.copy(src[2 : 2 + rows, 3 : 3 + cols], l1, transpose=False)
            _read_l1_l0(l1, eye, out, rows, cols, out_cols, dtype, read_dtype, acc_dtype)

    source = _matrix_data((20, 37), dtype)
    identity = torch.eye(out_cols, dtype=getattr(torch, read_dtype))[:, :cols].contiguous().npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    actual = kernel(source.view(getattr(torch, dtype)).npu(), identity).cpu()
    expected = torch.zeros((rows, out_cols), dtype=actual.dtype)
    expected[:, :cols] = source[2 : 2 + rows, 3 : 3 + cols].to(actual.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
def test_nd_to_nz_non_fractal_range(dtype):
    """向整分形 (16, 32) L1 的非整分形子范围 [:13, :29] 写入界内切片，检查有效区与未写入行。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)
    rows, cols = 13, 29

    @T.prim_func
    def main(
        src: T.Tensor((20, 37), dtype),
        poison: T.Tensor((16, 32), dtype),
        eye: T.Tensor((32, 32), read_dtype),
        out: T.Tensor((16, 32), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 32), dtype)
            T.copy(poison, l1, transpose=False)
            T.copy(src[2 : 2 + rows, 3 : 3 + cols], l1[:rows, :cols], transpose=False)
            _read_l1(l1, eye, out, 16, 32, read_dtype, acc_dtype)

    source = _matrix_data((20, 37), dtype)
    poison = torch.full((16, 32), 7, dtype=source.dtype)
    identity = torch.eye(32, dtype=getattr(torch, read_dtype)).npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    actual = kernel(
        source.view(getattr(torch, dtype)).npu(),
        poison.view(getattr(torch, dtype)).npu(),
        identity,
    ).cpu()
    torch.testing.assert_close(actual[:rows, :cols], source[2 : 2 + rows, 3 : 3 + cols].to(actual.dtype), rtol=0, atol=0)
    torch.testing.assert_close(actual[rows:, :], poison[rows:, :].to(actual.dtype), rtol=0, atol=0)


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
def test_nd_to_nz_l1_subregion(dtype):
    """在 (32, 64) L1 中向分形边界对齐的子区域 [16:32, 32:64] 写入，核对写入区与周围保护区。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(
        src: T.Tensor((20, 37), dtype),
        poison: T.Tensor((32, 64), dtype),
        eye: T.Tensor((64, 64), read_dtype),
        out: T.Tensor((32, 64), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((32, 64), dtype)
            T.copy(poison, l1, transpose=False)
            T.copy(src[2:18, 3:35], l1[16:32, 32:64], transpose=False)
            _read_l1(l1, eye, out, 32, 64, read_dtype, acc_dtype)

    source = _matrix_data((20, 37), dtype)
    poison = torch.full((32, 64), 7, dtype=source.dtype)
    expected = poison.clone()
    expected[16:32, 32:64] = source[2:18, 3:35]
    _assert_readback(
        main,
        dtype,
        expected,
        source.view(getattr(torch, dtype)).npu(),
        poison.view(getattr(torch, dtype)).npu(),
    )


@DEVICE
@pytest.mark.parametrize("dtype", ["int8", "bfloat16", "float32"])
@pytest.mark.parametrize("keep_unit_slice", [False, True], ids=["scalar-idx", "unit-slice"])
def test_nd_to_nz_5d_gm_and_3d_l1_stage(dtype, keep_unit_slice):
    """从 5D GM 单平面切片搬入带流水 stage 维的 3D L1，检查写入 stage 与未写入 stage 保护区。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(
        src: T.Tensor((2, 2, 2, 20, 37), dtype),
        poison: T.Tensor((16, 32), dtype),
        eye: T.Tensor((32, 32), read_dtype),
        out: T.Tensor((2, 16, 32), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((2, 16, 32), dtype)
            weight = T.alloc_l1((32, 32), read_dtype)
            acc = T.alloc_l0c((16, 32), acc_dtype)
            T.copy(poison, l1[0, :, :], transpose=False)
            T.copy(poison, l1[1, :, :], transpose=False)
            for ko in T.serial(1):
                stage = (ko + 1) % 2
                if keep_unit_slice:
                    T.copy(src[1:2, 0:1, 1:2, 2:18, 3:35], l1[stage : stage + 1, :, :], transpose=False)
                else:
                    T.copy(src[1, 0, 1, 2:18, 3:35], l1[stage, :, :], transpose=False)
            T.copy(eye, weight, transpose=False)
            payload = T.view(l1, dtype=read_dtype)
            for s in T.serial(2):
                T.gemm(payload[s, :, :], weight, acc, transpose_B=True, clear_accum=True)
                T.copy(acc, out[s, :, :])

    source = _matrix_data((2, 2, 2, 20, 37), dtype)
    poison = torch.full((16, 32), 7, dtype=source.dtype)
    expected = torch.stack((poison, source[1, 0, 1, 2:18, 3:35]))
    _assert_readback(
        main,
        dtype,
        expected,
        source.view(getattr(torch, dtype)).npu(),
        poison.view(getattr(torch, dtype)).npu(),
    )


@DEVICE
@pytest.mark.parametrize("dtype", ["int8", "bfloat16", "float32"])
def test_nd_to_nz_dynamic_sources(dtype):
    """同一已编译 kernel 结合 T.dynamic 维度、分核 bx、循环 ko 与标量入参动态切片搬入 L1。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)
    M = T.dynamic("M")
    K = T.dynamic("K")

    @T.prim_func
    def main(
        src: T.Tensor((M, K), dtype),
        eye: T.Tensor((32, 32), read_dtype),
        row_off: T.int32,
        col_off: T.int32,
        out: T.Tensor((2, 16, 32), acc_dtype),
    ):
        with T.Kernel(2) as bx:
            T.assume(M >= 40)
            T.assume(K >= 40)
            T.assume(row_off >= 0)
            T.assume(col_off >= 0)
            l1 = T.alloc_l1((16, 32), dtype)
            for ko in T.serial(1):
                r0 = bx * 16 + row_off
                c0 = ko * 32 + col_off
                T.copy(src[r0 : r0 + 16, c0 : c0 + 32], l1, transpose=False)
            _read_l1(l1, eye, out[bx, :, :], 16, 32, read_dtype, acc_dtype)

    identity = torch.eye(32, dtype=getattr(torch, read_dtype)).npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    for shape, row_off, col_off in [((45, 53), 2, 3), ((64, 71), 5, 7)]:
        source = _matrix_data(shape, dtype)
        actual = kernel(source.view(getattr(torch, dtype)).npu(), identity, row_off, col_off).cpu()
        expected = torch.stack(
            (
                source[row_off : row_off + 16, col_off : col_off + 32],
                source[16 + row_off : 32 + row_off, col_off : col_off + 32],
            )
        )
        torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("src_rows,src_cols", [(17, 32), (16, 33)], ids=["exceed-rows", "exceed-cols"])
def test_nd_to_nz_rejects_source_exceeding_destination(src_rows, src_cols):
    """源 GM 切片行数或列数超出目的 L1 切片区域时，编译期直接报错拒绝。"""

    @T.prim_func
    def main(src: T.Tensor((32, 64), "float16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 32), "float16")
            T.copy(src[:src_rows, :src_cols], l1[:16, :32], transpose=False)

    with pytest.raises(Exception):
        tilelang.lower(main, target="ascend")

