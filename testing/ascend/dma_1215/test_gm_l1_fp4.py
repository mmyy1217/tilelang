"""1215 GM→L1 FP4（E2M1）每字节双值打包搬入（ND→NZ）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


FP4 = "float4_e2m1fn"
DEVICE = tilelang.testing.requires_ascend


def _fp4_expected(a_codes, b_codes):
    e2m1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    return e2m1[a_codes.long()] @ e2m1[b_codes.long()].T


def _pack_fp4(codes, storage_dtype):
    low = codes[:, ::2]
    high = codes[:, 1::2] << 4
    return (low | high).view(getattr(torch, storage_dtype)).npu()


@DEVICE
@pytest.mark.parametrize("storage_dtype", ["float4_e2m1fn_x2", "int8"], ids=["packed-dtype", "byte-view"])
def test_fp4_packed_nd_to_nz(storage_dtype):
    """按 ND 搬入 FP4。CPU 独立解码两个半字节，再与 blockscaled GEMM 比较。"""
    gm_shape, gm_dtype = {
        "float4_e2m1fn_x2": ((16, 256), FP4),
        "int8": ((16, 128), "int8"),
    }[storage_dtype]

    @T.prim_func
    def main(
        src: T.Tensor(gm_shape, gm_dtype),
        weight: T.Tensor(gm_shape, gm_dtype),
        scale: T.Tensor((4, 16), "uint16"),
        out: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((16, 256), FP4)
            b = T.alloc_l1((16, 256), FP4)
            a_scale = T.alloc_l1((16, 4), "uint16")
            b_scale = T.alloc_l1((16, 4), "uint16")
            acc = T.alloc_l0c((16, 16), "float32")
            T.copy(T.view(src, (16, 256), dtype=FP4), a, transpose=False)
            T.copy(T.view(weight, (16, 256), dtype=FP4), b, transpose=False)
            T.copy(scale, a_scale, transpose=True)
            T.copy(scale, b_scale, transpose=True)
            T.gemm_blockscaled(a, b, acc, a_scale, b_scale, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    rng = torch.Generator().manual_seed(1215)
    a_codes = torch.randint(0, 16, (16, 256), dtype=torch.uint8, generator=rng)
    b_codes = torch.randint(0, 16, (16, 256), dtype=torch.uint8, generator=rng)
    # E8M0 指数 127 对应缩放系数 1，每两个字节装入一个 uint16。
    scale = torch.full((4, 16), 0x7F7F, dtype=torch.uint16).npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    actual = kernel(_pack_fp4(a_codes, storage_dtype), _pack_fp4(b_codes, storage_dtype), scale).cpu()
    torch.testing.assert_close(actual, _fp4_expected(a_codes, b_codes), rtol=0, atol=0)


@DEVICE
@pytest.mark.parametrize("storage_dtype", ["float4_e2m1fn_x2", "int8"], ids=["packed-dtype", "byte-view"])
def test_fp4_packed_nd_to_nz_gm_slice(storage_dtype):
    """从非 C0 对齐行步长的 (20, 262) FP4 GM 切取偶数列起点子块 [2:18, 6:262] 搬入 L1。"""
    gm_shape, gm_dtype = {
        "float4_e2m1fn_x2": ((20, 262), FP4),
        "int8": ((20, 131), "int8"),
    }[storage_dtype]
    weight_shape = (16, 256) if storage_dtype == "float4_e2m1fn_x2" else (16, 128)

    @T.prim_func
    def main(
        src: T.Tensor(gm_shape, gm_dtype),
        weight: T.Tensor(weight_shape, gm_dtype),
        scale: T.Tensor((4, 16), "uint16"),
        out: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((16, 256), FP4)
            b = T.alloc_l1((16, 256), FP4)
            a_scale = T.alloc_l1((16, 4), "uint16")
            b_scale = T.alloc_l1((16, 4), "uint16")
            acc = T.alloc_l0c((16, 16), "float32")
            src_fp4 = T.view(src, (20, 262), dtype=FP4)
            T.copy(src_fp4[2:18, 6:262], a, transpose=False)
            T.copy(T.view(weight, (16, 256), dtype=FP4), b, transpose=False)
            T.copy(scale, a_scale, transpose=True)
            T.copy(scale, b_scale, transpose=True)
            T.gemm_blockscaled(a, b, acc, a_scale, b_scale, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    rng = torch.Generator().manual_seed(1215)
    a_full = torch.randint(0, 16, (20, 262), dtype=torch.uint8, generator=rng)
    b_codes = torch.randint(0, 16, (16, 256), dtype=torch.uint8, generator=rng)
    scale = torch.full((4, 16), 0x7F7F, dtype=torch.uint16).npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    actual = kernel(_pack_fp4(a_full, storage_dtype), _pack_fp4(b_codes, storage_dtype), scale).cpu()
    torch.testing.assert_close(actual, _fp4_expected(a_full[2:18, 6:262], b_codes), rtol=0, atol=0)
