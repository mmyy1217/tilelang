import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_compact_nz_layout


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


def test_ub_to_l1_contiguous_nz():
    """把 CPU 预排好的 32×32 BF16 NZ 块经 UB 原样搬到 L1。"""

    @T.prim_func
    def main(src: T.Tensor((32, 32), "bfloat16"), eye: T.Tensor((32, 32), "bfloat16"), out: T.Tensor((32, 32), "float32")):
        with T.MixedKernel(1, sids=1):
            ub = T.alloc_shared((32, 32), "bfloat16")
            l1 = T.alloc_l1((32, 32), "bfloat16")
            T.annotate_layout({ub: make_ascend_compact_nz_layout(ub)})
            T.copy(src, ub)
            T.copy(ub, l1)
            _read_l1(l1, eye, out, 32, 32, "bfloat16")

    rng = torch.Generator().manual_seed(1215)
    data = torch.randint(-64, 64, (32, 32), generator=rng).to(torch.bfloat16)
    # NZ 的两个列分形在外，块内按 16×16 连续存储。
    packed = data.reshape(2, 16, 2, 16).permute(2, 0, 1, 3).reshape(32, 32)
    eye = torch.eye(32, dtype=torch.bfloat16)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(packed.npu(), eye.npu())
    torch.testing.assert_close(actual.cpu(), data.float(), rtol=0, atol=0)


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
