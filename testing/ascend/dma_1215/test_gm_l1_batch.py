"""1215 GM→L1 多矩阵批量连续/跨步搬入（Batch ND→NZ / DN→NZ）。"""

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
BATCH_LAYOUTS = ["compact", "gapped", "interleaved"]


def _matrix_data(shape, dtype):
    rng = torch.Generator().manual_seed(1215)
    return torch.randint(1, 63, shape, generator=rng).to(getattr(torch, READBACK[dtype]))


def _readback_dtypes(dtype):
    read_dtype = READBACK[dtype]
    return read_dtype, "int32" if read_dtype == "int8" else "float32"


@T.macro
def _read_l1_batch(matrices, identity, out, read_dtype, acc_dtype):
    weight = T.alloc_l1((32, 32), read_dtype)
    acc = T.alloc_l0c((16, 32), acc_dtype)
    T.copy(identity, weight, transpose=False)
    payload = T.view(matrices, dtype=read_dtype)
    for batch in T.serial(3):
        T.gemm(payload[batch, :, :], weight, acc, transpose_B=True, clear_accum=True)
        T.copy(acc, out[batch, :, :])


def _assert_readback(main, dtype, expected, *inputs):
    identity = torch.eye(expected.shape[-1]).to(getattr(torch, READBACK[dtype])).npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    actual = kernel(*inputs, identity).cpu()
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)


def _batch_source(rows, cols, dtype, layout):
    torch_dtype = getattr(torch, dtype)
    if layout == "interleaved":
        data = _matrix_data((rows, 3, cols), dtype)
        return data.transpose(0, 1), data.view(torch_dtype).npu().transpose(0, 1)

    pad_rows = 2 if layout == "gapped" else 0
    data = _matrix_data((3, rows + pad_rows, cols), dtype)
    return data[:, :rows, :], data.view(torch_dtype).npu()[:, :rows, :]


@T.macro
def _copy_batch_nd(source, target):
    T.copy(source, target, transpose=False)


@T.macro
def _copy_batch_dn(source, target):
    T.copy(source, target, transpose=True)


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("batch_layout", BATCH_LAYOUTS)
def test_batch_nd_to_nz(dtype, batch_layout):
    """一次调用搬入三个 ND 矩阵，覆盖紧凑、带间隔与交错排布，逐批核对读回数据。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)
    data, source = _batch_source(16, 32, dtype, batch_layout)
    strides = source.stride()
    assert strides == data.stride()

    @T.prim_func
    def main(
        src: T.StridedTensor((3, 16, 32), strides, dtype),
        eye: T.Tensor((32, 32), read_dtype),
        out: T.Tensor((3, 16, 32), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((3, 16, 32), dtype)
            T.copy(src, l1, transpose=False)
            _read_l1_batch(l1, eye, out, read_dtype, acc_dtype)

    _assert_readback(main, dtype, data, source)


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("batch_layout", BATCH_LAYOUTS)
def test_batch_dn_to_nz(dtype, batch_layout):
    """一次调用搬入三个 DN 矩阵，覆盖紧凑、带间隔与交错排布，与逐批转置的参考比较。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)
    data, source = _batch_source(32, 16, dtype, batch_layout)
    strides = source.stride()
    assert strides == data.stride()

    @T.prim_func
    def main(
        src: T.StridedTensor((3, 32, 16), strides, dtype),
        eye: T.Tensor((32, 32), read_dtype),
        out: T.Tensor((3, 16, 32), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((3, 16, 32), dtype)
            T.copy(src, l1, transpose=True)
            _read_l1_batch(l1, eye, out, read_dtype, acc_dtype)

    _assert_readback(main, dtype, data.transpose(1, 2), source)


@DEVICE
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize(
    "rows,cols,copy_rows,copy_cols,interleaved,transpose,batch_begin,compact",
    [
        (32, 32, 32, 32, False, False, 0, True),
        (32, 64, 32, 64, False, True, 0, True),
        (32, 32, 32, 32, True, False, 0, False),
        (32, 32, 32, 32, False, False, 0, False),
        (20, 10, 32, 32, False, False, 1, False),
        (32, 64, 32, 64, False, True, 0, False),
        (32, 64, 32, 64, True, True, 0, False),
        (20, 10, 32, 64, False, True, 1, False),
    ],
    ids=[
        "compact-nd",
        "compact-dn",
        "interleaved-source",
        "larger-matrix-pitch",
        "explicit-column-fill",
        "transpose-rectangular",
        "transpose-interleaved-source",
        "transpose-batch-m-n-tails",
    ],
)
def test_batched_nd2nz_strides_and_transpose_device(dtype, rows, cols, copy_rows, copy_cols, interleaved, transpose, batch_begin, compact):
    """用单位矩阵读回每个批次，检查转置、跨步、尾块、L1 更大 pitch 与相邻矩阵保护区。"""
    source_batches = 3
    # 跨步场景在源行之间留空，有的场景还交错存放多个 batch。
    row_pitch = cols if compact else cols * 2
    strides = (row_pitch, source_batches * row_pitch, 1) if interleaved else (rows * row_pitch, row_pitch, 1)
    dst_rows, dst_cols = (copy_cols, copy_rows) if transpose else (copy_rows, copy_cols)
    copy_batch = _copy_batch_dn if transpose else _copy_batch_nd
    # 行列尾块使用目标的完整列区间。
    target_rows, target_cols = (dst_rows, dst_cols) if rows < copy_rows or cols < copy_cols else (64, 64)

    @T.prim_func
    def main(
        source: T.StridedTensor((source_batches, rows, cols), strides, dtype),
        identity: T.Tensor((target_cols, target_cols), dtype),
        poison: T.Tensor((5, target_rows, target_cols), dtype),
        output: T.Tensor((5, target_rows, target_cols), "float32"),
    ):
        with T.Kernel(1):
            target = T.alloc_l1((5, target_rows, target_cols), dtype)
            weight = T.alloc_l1((target_cols, target_cols), dtype)
            a = T.alloc_l0a((target_rows, target_cols), dtype)
            b = T.alloc_l0b((target_cols, target_cols), dtype)
            acc = T.alloc_l0c((target_rows, target_cols), "float32")
            T.copy(poison, target, transpose=False)
            copy_batch(
                source[batch_begin : batch_begin + 4, :copy_rows, :copy_cols],
                target[1:5, :dst_rows, :dst_cols],
            )
            T.copy(identity, weight, transpose=False)
            T.copy(weight, b)
            for batch in T.serial(5):
                T.copy(target[batch : batch + 1, :, :], a)
                T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
                T.copy(acc, output[batch, :, :])

    kernel = tilelang.compile(main, target="ascend", out_idx=-1, pass_configs={"tl.enable_auto_schedule": True})
    torch_dtype = getattr(torch, dtype)
    storage_shape = (rows, source_batches, row_pitch) if interleaved else (source_batches, rows, row_pitch)
    storage = ((torch.arange(source_batches * rows * row_pitch).reshape(storage_shape) % 31) + 1).to(torch_dtype).npu()
    source = (storage.permute(1, 0, 2) if interleaved else storage)[:, :, :cols]
    assert source.stride() == strides
    valid_batches = min(4, max(source_batches - batch_begin, 0))
    expected = torch.full((5, target_rows, target_cols), -7.0)
    expected[1 : 1 + valid_batches, :dst_rows, :dst_cols] = 0
    values = source[batch_begin : batch_begin + valid_batches].cpu().float()
    if transpose:
        values = values.transpose(1, 2)
    expected[1 : 1 + valid_batches, : values.shape[1], : values.shape[2]] = values
    identity = torch.eye(target_cols, dtype=torch_dtype, device="npu")
    poison = torch.full((5, target_rows, target_cols), -7.0, dtype=torch_dtype, device="npu")
    actual = kernel(source, identity, poison)
    torch.npu.synchronize()
    # 连同未搬运的行列和 batch 一起检查，确认保护值保留。
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
