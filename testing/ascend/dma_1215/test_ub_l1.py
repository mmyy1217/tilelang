import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()


ND2NZ_CASES = [
    (32, 128, "float32", "bfloat16"),
    (32, 64, "float32", "float32"),
    (32, 128, "bfloat16", "float32"),
]


@T.macro
def _read_l1(payload, eye, out, rows, cols, dst_dtype):
    weights = T.alloc_l1((cols, cols), dst_dtype)
    acc = T.alloc_l0c((rows, cols), "float32")
    T.copy(eye, weights, transpose=False)
    T.gemm(payload, weights, acc, transpose_B=True, clear_accum=True)
    T.copy(acc, out)


def _check_matrix(main, rows, cols, src_dtype, dst_dtype):
    target_dtype = getattr(torch, dst_dtype)
    data = torch.arange(rows * cols).reshape(rows, cols).to(getattr(torch, src_dtype))
    eye = torch.eye(cols, dtype=target_dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(data.npu(), eye.npu())
    torch.testing.assert_close(actual.cpu(), data.to(target_dtype).float(), rtol=0, atol=0)


@pytest.mark.parametrize("rows,cols,src_dtype,dst_dtype", ND2NZ_CASES)
def test_ub_to_l1_nd_to_nz(rows, cols, src_dtype, dst_dtype):
    """AIV0 将整张 ND 矩阵从 UB 重排写入 L1。"""

    @T.prim_func
    def main(src: T.Tensor((rows, cols), src_dtype), eye: T.Tensor((cols, cols), dst_dtype), out: T.Tensor((rows, cols), "float32")):
        with T.MixedKernel(1) as (_, sid):
            ub = T.alloc_shared((rows, cols), src_dtype)
            l1 = T.alloc_l1((rows, cols), dst_dtype)
            if sid == 0:
                T.copy(src, ub)
                T.copy(ub, l1)
            _read_l1(l1, eye, out, rows, cols, dst_dtype)

    _check_matrix(main, rows, cols, src_dtype, dst_dtype)


@pytest.mark.parametrize("rows,cols,src_dtype,dst_dtype", ND2NZ_CASES)
def test_ub_to_l1_dual_nd_to_nz(rows, cols, src_dtype, dst_dtype):
    """每个 AIV 写入 L1 的一半行，读回完整矩阵检查错位或覆盖。"""

    @T.prim_func
    def main(src: T.Tensor((rows, cols), src_dtype), eye: T.Tensor((cols, cols), dst_dtype), out: T.Tensor((rows, cols), "float32")):
        with T.MixedKernel(1):
            ub = T.alloc_shared((rows // 2, cols), src_dtype)
            l1 = T.alloc_l1((rows, cols), dst_dtype)
            T.dual_copy(src, ub)
            T.dual_copy(ub, l1)
            _read_l1(l1, eye, out, rows, cols, dst_dtype)

    _check_matrix(main, rows, cols, src_dtype, dst_dtype)
