import pytest
import torch
import tilelang
import tilelang.testing
from tilelang.ascend import language as T


@T.macro
def _copy_batch_nd(source, target):
    T.copy(source, target, transpose=False)


@T.macro
def _copy_batch_dn(source, target):
    T.copy(source, target, transpose=True)


@tilelang.testing.requires_ascend
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
    """用单位矩阵读回每个批次，检查转置、跨步、尾块和相邻矩阵。"""
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
