"""1215 GM→L1 矩阵越界尾块补零（OOB Padding）。"""

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
# (GM 行数, GM 列数, L1 行数, L1 列数)，请求均从 GM 第 96 列开始。
OOB_SHAPES = [
    pytest.param(15, 128, 16, 32, id="row-tail"),
    pytest.param(16, 100, 16, 32, id="column-tail"),
    pytest.param(15, 100, 16, 32, id="both-tails"),
    # 31 列有效，余下 33 列覆盖最后一组 C0 内的缺列及后续完整 C0 组。
    pytest.param(15, 127, 16, 64, id="both-tails-partial-and-full-c0"),
    # 32 行跨两个 NZ 行分形，4 列有效，验证两组行都补满 64 列。
    pytest.param(32, 100, 32, 64, id="two-row-fractals"),
    pytest.param(16, 96, 16, 32, id="fully-oob"),
]


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


def _assert_readback(main, dtype, expected, *inputs):
    identity = torch.eye(expected.shape[-1]).to(getattr(torch, READBACK[dtype])).npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    actual = kernel(*inputs, identity).cpu()
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)


def _padded(data, rows, cols):
    # CPU 参考独立补齐到整块。空切片对应完全越界。
    expected = torch.zeros((rows, cols), dtype=data.dtype)
    expected[: data.shape[0], : data.shape[1]] = data
    return expected


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("rows,cols,tile_rows,tile_cols", OOB_SHAPES)
def test_oob_tail_padding_nd(dtype, rows, cols, tile_rows, tile_cols):
    """ND 从第 96 列请求完整 NZ tile，检查目标行尾和列尾。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(
        src: T.Tensor((rows, cols), dtype),
        poison: T.Tensor((tile_rows, tile_cols), dtype),
        eye: T.Tensor((tile_cols, tile_cols), read_dtype),
        out: T.Tensor((tile_rows, tile_cols), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((tile_rows, tile_cols), dtype)
            T.copy(poison, l1, transpose=False)
            T.copy(src[:tile_rows, 96 : 96 + tile_cols], l1, transpose=False)
            _read_l1(l1, eye, out, tile_rows, tile_cols, read_dtype, acc_dtype)

    data = _matrix_data((rows, cols), dtype)
    source = data.view(getattr(torch, dtype)).npu()
    poison = torch.full((tile_rows, tile_cols), 7, dtype=data.dtype).view(getattr(torch, dtype)).npu()
    _assert_readback(main, dtype, _padded(data[:, 96 : 96 + tile_cols], tile_rows, tile_cols), source, poison)


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("rows,cols,tile_rows,tile_cols", OOB_SHAPES)
def test_oob_tail_padding_dn(dtype, rows, cols, tile_rows, tile_cols):
    """DN 从转置后的第 96 列请求完整 NZ tile，检查目标行尾和列尾。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(
        src: T.Tensor((cols, rows), dtype),
        poison: T.Tensor((tile_rows, tile_cols), dtype),
        eye: T.Tensor((tile_cols, tile_cols), read_dtype),
        out: T.Tensor((tile_rows, tile_cols), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((tile_rows, tile_cols), dtype)
            T.copy(poison, l1, transpose=False)
            T.copy(src[96 : 96 + tile_cols, :tile_rows], l1, transpose=True)
            _read_l1(l1, eye, out, tile_rows, tile_cols, read_dtype, acc_dtype)

    data = _matrix_data((rows, cols), dtype)
    source = data.T.contiguous().view(getattr(torch, dtype)).npu()
    poison = torch.full((tile_rows, tile_cols), 7, dtype=data.dtype).view(getattr(torch, dtype)).npu()
    _assert_readback(main, dtype, _padded(data[:, 96 : 96 + tile_cols], tile_rows, tile_cols), source, poison)


@DEVICE
@pytest.mark.parametrize("dtype", ["int8", "bfloat16", "float32"])
@pytest.mark.parametrize("transpose", [False, True], ids=["nd", "dn"])
def test_oob_tail_padding_dynamic(dtype, transpose):
    """运行时动态传入列偏移，分别验证界内搬运与越界尾块自动补零。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)
    rows, cols, tile_rows, tile_cols = 32, 100, 32, 64
    src_shape = (cols, rows) if transpose else (rows, cols)

    @T.prim_func
    def main(
        src: T.Tensor(src_shape, dtype),
        poison: T.Tensor((tile_rows, tile_cols), dtype),
        eye: T.Tensor((tile_cols, tile_cols), read_dtype),
        col_offset: T.int32,
        out: T.Tensor((tile_rows, tile_cols), acc_dtype),
    ):
        with T.Kernel(1):
            T.assume(col_offset >= 0)
            l1 = T.alloc_l1((tile_rows, tile_cols), dtype)
            T.copy(poison, l1, transpose=False)
            if transpose:
                T.copy(src[col_offset : col_offset + tile_cols, :tile_rows], l1, transpose=True)
            else:
                T.copy(src[:tile_rows, col_offset : col_offset + tile_cols], l1, transpose=False)
            _read_l1(l1, eye, out, tile_rows, tile_cols, read_dtype, acc_dtype)

    data = _matrix_data((rows, cols), dtype)
    source = (data.T.contiguous() if transpose else data).view(getattr(torch, dtype)).npu()
    poison = torch.full((tile_rows, tile_cols), 7, dtype=data.dtype).view(getattr(torch, dtype)).npu()
    identity = torch.eye(tile_cols, dtype=getattr(torch, read_dtype)).npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    for col_offset in (32, 96):
        actual = kernel(source, poison, identity, col_offset).cpu()
        expected = _padded(data[:, col_offset : col_offset + tile_cols], tile_rows, tile_cols)
        torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)
