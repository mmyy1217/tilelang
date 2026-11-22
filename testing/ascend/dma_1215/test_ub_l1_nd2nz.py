"""1215 UB→L1 普通行优先矩阵重排为 NZ 分形再搬入 L1（ND→NZ，覆盖 5 种支持 dtype 组合）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()

ND2NZ_CASES = [
    (32, 64, "float16", "float16"),
    (32, 64, "bfloat16", "bfloat16"),
    (32, 64, "float32", "float32"),
    (32, 128, "float32", "bfloat16"),
    (32, 128, "bfloat16", "float32"),
]


@T.macro
def _read_l1(payload, eye, out, rows, cols, dst_dtype):
    weights = T.alloc_l1((cols, cols), dst_dtype)
    acc = T.alloc_l0c((rows, cols), "float32")
    T.copy(eye, weights, transpose=False)
    T.gemm(payload, weights, acc, transpose_B=True, clear_accum=True)
    T.copy(acc, out)


@pytest.mark.parametrize("rows,cols,src_dtype,dst_dtype", ND2NZ_CASES)
def test_ub_to_l1_nd_to_nz(rows, cols, src_dtype, dst_dtype):
    """AIV0 将整张 ND 矩阵从 UB 自动重排成 NZ 并写入 L1，单位矩阵读回校验。"""

    @T.prim_func
    def main(src: T.Tensor((rows, cols), src_dtype), eye: T.Tensor((cols, cols), dst_dtype), out: T.Tensor((rows, cols), "float32")):
        with T.MixedKernel(1) as (_, sid):
            ub = T.alloc_shared((rows, cols), src_dtype)
            l1 = T.alloc_l1((rows, cols), dst_dtype)
            if sid == 0:
                T.copy(src, ub)
                T.copy(ub, l1)
            _read_l1(l1, eye, out, rows, cols, dst_dtype)

    target_dtype = getattr(torch, dst_dtype)
    data = (torch.arange(rows * cols).reshape(rows, cols) % 61 - 30).to(getattr(torch, src_dtype))
    eye = torch.eye(cols, dtype=target_dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(data.npu(), eye.npu())
    torch.testing.assert_close(actual.cpu(), data.to(target_dtype).float(), rtol=0, atol=0)
