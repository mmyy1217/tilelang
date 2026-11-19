"""1215 L1→L0A/B MX 缩放因子搬运。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


def _mx_input(rows, k, *, dtype, transpose, rng):
    # 先生成 [M/N, K] 的逻辑值，再按 GEMM 的输入布局打包给 GM。
    if dtype == "float4_e2m1fn":
        codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, generator=rng)
        values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
        gm = codes.T.contiguous() if transpose else codes
        packed = gm[:, ::2] | (gm[:, 1::2] << 4)
        return packed.view(torch.float4_e2m1fn_x2), values[codes.long()]

    data = (torch.randint(-4, 5, (rows, k), generator=rng).float() / 8).to(torch.float8_e4m3fn)
    return (data.T.contiguous() if transpose else data), data.float()


def _apply_e8m0(data, scale_bytes):
    return data * torch.pow(2.0, scale_bytes.float() - 127).repeat_interleave(32, dim=1)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("data_dtype", ["float8_e4m3fn", "float4_e2m1fn"], ids=["mxfp8", "mxfp4"])
@pytest.mark.parametrize("sf_dtype", ["uint16", "uint8", "int8", "float8_e8m0fnu"])
@pytest.mark.parametrize(
    "m,n,trans_a,trans_b",
    [
        (1, 15, False, True),
        (15, 15, False, True),
        (32, 32, False, True),
        (100, 100, False, True),
        (64, 64, False, False),
        (64, 64, True, False),
        (64, 64, True, True),
    ],
)
def test_blockscaled_l0_gemm(m, n, trans_a, trans_b, data_dtype, sf_dtype):
    """将两路 E8M0 搬到专用 L0 槽，用 CPU 展开后的矩阵乘验证。"""
    k = 128
    sf_cols = k // (64 if sf_dtype == "uint16" else 32)
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    @T.prim_func
    def gemm(
        a: T.Tensor(a_shape, data_dtype),
        b: T.Tensor(b_shape, data_dtype),
        sa: T.Tensor((m, sf_cols), sf_dtype),
        sb: T.Tensor((n, sf_cols), sf_dtype),
        out: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1(a_shape, data_dtype)
            b_l1 = T.alloc_l1(b_shape, data_dtype)
            sa_l1 = T.alloc_l1((m, sf_cols), sf_dtype)
            sb_l1 = T.alloc_l1((n, sf_cols), sf_dtype)
            a_l0 = T.alloc_l0a(a_shape, data_dtype)
            b_l0 = T.alloc_l0b(b_shape, data_dtype)
            # Scale 行维始终对应逻辑 M/N，和主矩阵是否转置无关。
            a_sf = T.alloc_l0a_sf(a_l0, sf_dtype=sf_dtype, sf_shape=(m, sf_cols))
            b_sf = T.alloc_l0b_sf(b_l0, sf_dtype=sf_dtype, sf_shape=(n, sf_cols))
            acc = T.alloc_l0c((m, n), "float32")
            T.copy(a, a_l1, transpose=False)
            T.copy(b, b_l1, transpose=False)
            T.copy(sa, sa_l1, transpose=False)
            T.copy(sb, sb_l1, transpose=False)
            T.copy(a_l1, a_l0, transpose=False)
            T.copy(b_l1, b_l0, transpose=False)
            T.copy(sa_l1, a_sf, transpose=False)
            T.copy(sb_l1, b_sf, transpose=False)
            T.gemm_blockscaled(a_l0, b_l0, acc, a_sf, b_sf, transpose_A=trans_a, transpose_B=trans_b, clear_accum=True)
            T.copy(acc, out)

    rng = torch.Generator().manual_seed(1215)
    a_gm, a = _mx_input(m, k, dtype=data_dtype, transpose=trans_a, rng=rng)
    b_gm, b = _mx_input(n, k, dtype=data_dtype, transpose=not trans_b, rng=rng)
    sa = torch.randint(124, 131, (m, k // 32), dtype=torch.uint8, generator=rng)
    sb = torch.randint(124, 131, (n, k // 32), dtype=torch.uint8, generator=rng)
    expected = _apply_e8m0(a, sa) @ _apply_e8m0(b, sb).T
    # 按位解释指数，uint16 每元素两组，字节类型每元素一组。
    torch_sf_dtype = getattr(torch, sf_dtype)
    sa_gm = sa.view(torch_sf_dtype)
    sb_gm = sb.view(torch_sf_dtype)
    kernel = tilelang.compile(gemm, target="ascend", out_idx=-1)
    actual = kernel(a_gm.npu(), b_gm.npu(), sa_gm.npu(), sb_gm.npu())
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
