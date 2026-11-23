"""1215 UB→L1 连续原样搬入（不重排格式，源 UB 已排为紧凑 NZ）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_compact_nz_layout


pytestmark = tilelang.testing.requires_ascend.marks()


@T.macro
def _read_l1(payload, eye, out, rows, cols, dst_dtype, acc_dtype):
    weights = T.alloc_l1((cols, cols), dst_dtype)
    acc = T.alloc_l0c((rows, cols), acc_dtype)
    T.copy(eye, weights, transpose=False)
    T.gemm(payload, weights, acc, transpose_B=True, clear_accum=True)
    T.copy(acc, out)


def _contiguous_nz_kernel(dtype, acc_dtype):
    @T.prim_func
    def main(src: T.Tensor((32, 32), dtype), eye: T.Tensor((32, 32), dtype), out: T.Tensor((32, 32), acc_dtype)):
        with T.MixedKernel(1, sids=1):
            ub = T.alloc_shared((32, 32), dtype)
            l1 = T.alloc_l1((32, 32), dtype)
            T.annotate_layout({ub: make_ascend_compact_nz_layout(ub)})
            T.copy(src, ub)
            T.copy(ub, l1)
            _read_l1(l1, eye, out, 32, 32, dtype, acc_dtype)

    return main


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_ub_to_l1_contiguous_nz(dtype):
    """把 CPU 预排好的 32×32 NZ 块经 UB 原样连续搬入 L1，用单位矩阵读回逐元素校验。"""
    main = _contiguous_nz_kernel(dtype, "float32")
    torch_dtype = getattr(torch, dtype)
    c0 = 32 // torch.empty((), dtype=torch_dtype).element_size()
    rng = torch.Generator().manual_seed(1215)
    data = torch.randint(-64, 64, (32, 32), generator=rng).to(torch_dtype)
    # NZ 的列分形在外，块内按 16×C0 连续存储。
    packed = data.reshape(2, 16, 32 // c0, c0).permute(2, 0, 1, 3).reshape(32, 32)
    eye = torch.eye(32, dtype=torch_dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(packed.npu(), eye.npu())
    torch.testing.assert_close(actual.cpu(), data.to(actual.dtype), rtol=0, atol=0)


def test_ub_to_l1_contiguous_nz_rejects_int8():
    main = _contiguous_nz_kernel("int8", "int32")
    with pytest.raises(ValueError, match="Unsupported nd2nz dtype: int8"):
        tilelang.compile(main, target="ascend", out_idx=-1)
