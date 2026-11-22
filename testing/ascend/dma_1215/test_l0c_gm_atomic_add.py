"""1215 L0C→GM 搬出时原子累加到 GM（Atomic Add，覆盖 FP32/INT32/FP16/BF16）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()


@pytest.mark.parametrize(
    "in_dtype,acc_dtype,out_dtype",
    [
        ("float16", "float32", "float32"),
        ("float16", "float32", "float16"),
        ("float16", "float32", "bfloat16"),
        ("int8", "int32", "int32"),
    ],
)
def test_l0c_atomic_add(in_dtype, acc_dtype, out_dtype):
    """两个 Cube 各贡献一个 16×16 矩阵，GM 预置初值 7 参与累加，覆盖 4 种支持输出 dtype。"""
    k = 32 if in_dtype == "int8" else 16

    @T.prim_func
    def main(a: T.Tensor((32, k), in_dtype), b: T.Tensor((16, k), in_dtype), out: T.Tensor((16, 16), out_dtype)):
        with T.Kernel(2) as block:
            x = T.alloc_l1((16, k), in_dtype)
            y = T.alloc_l1((16, k), in_dtype)
            acc = T.alloc_l0c((16, 16), acc_dtype)
            T.copy(a[block * 16 : (block + 1) * 16, :], x, transpose=False)
            T.copy(b, y, transpose=False)
            T.gemm(x, y, acc, transpose_B=True, clear_accum=True)
            T.set_atomic("add", out_dtype)
            T.copy(acc, out)
            T.set_atomic_none()

    torch_in = getattr(torch, in_dtype)
    torch_out = getattr(torch, out_dtype)
    if in_dtype == "int8":
        a = (torch.arange(32 * k).reshape(32, k) % 7 - 3).to(torch_in)
        eye = torch.zeros((16, k), dtype=torch_in)
        eye[:, :16] = torch.eye(16, dtype=torch_in)
        contrib0 = a[:16, :16].to(torch_out)
        contrib1 = a[16:, :16].to(torch_out)
    else:
        a = (torch.arange(32 * k).reshape(32, k) % 13 - 6).to(torch_in)
        eye = torch.eye(16, dtype=torch_in)
        contrib0 = a[:16].to(torch_out)
        contrib1 = a[16:].to(torch_out)

    initial = torch.full((16, 16), 7, dtype=torch_out)
    out = initial.npu()
    tilelang.compile(main, target="ascend")(a.npu(), eye.npu(), out)
    torch.testing.assert_close(out.cpu(), initial + contrib0 + contrib1, rtol=0, atol=0)
