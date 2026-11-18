import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


@tilelang.testing.requires_ascend
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
def test_blockscaled_l0_gemm(m, n, trans_a, trans_b):
    """将两路 E8M0 搬到专用 L0 槽，用 CPU 展开后的矩阵乘验证。"""
    k, sf_cols = 128, 2
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    @T.prim_func
    def gemm(
        a: T.Tensor(a_shape, "float8_e4m3fn"),
        b: T.Tensor(b_shape, "float8_e4m3fn"),
        sa: T.Tensor((sf_cols, m), "uint16"),
        sb: T.Tensor((sf_cols, n), "uint16"),
        out: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1(a_shape, "float8_e4m3fn")
            b_l1 = T.alloc_l1(b_shape, "float8_e4m3fn")
            sa_l1 = T.alloc_l1((m, sf_cols), "uint16")
            sb_l1 = T.alloc_l1((n, sf_cols), "uint16")
            a_l0 = T.alloc_l0a(a_shape, "float8_e4m3fn")
            b_l0 = T.alloc_l0b(b_shape, "float8_e4m3fn")
            # Transposed data tiles need the explicit logical SF shape (rows
            # always follow M/N, not the tile's leading axis).
            a_sf = T.alloc_l0a_sf(a_l0, sf_shape=(m, sf_cols))
            b_sf = T.alloc_l0b_sf(b_l0, sf_shape=(n, sf_cols))
            acc = T.alloc_l0c((m, n), "float32")
            T.copy(a, a_l1, transpose=False)
            T.copy(b, b_l1, transpose=False)
            T.copy(sa, sa_l1, transpose=True)
            T.copy(sb, sb_l1, transpose=True)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.copy(sa_l1, a_sf)
            T.copy(sb_l1, b_sf)
            T.gemm_blockscaled(a_l0, b_l0, acc, a_sf, b_sf, transpose_A=trans_a, transpose_B=trans_b, clear_accum=True)
            T.copy(acc, out)

    rng = torch.Generator().manual_seed(1215)
    a = (torch.randint(-4, 5, (m, 128), generator=rng).float() / 8).to(torch.float8_e4m3fn)
    b = (torch.randint(-4, 5, (n, 128), generator=rng).float() / 8).to(torch.float8_e4m3fn)
    sa = torch.randint(124, 131, (m, 4), dtype=torch.uint8, generator=rng)
    sb = torch.randint(124, 131, (n, 4), dtype=torch.uint8, generator=rng)
    expected = (a.float() * torch.pow(2.0, sa.float() - 127).repeat_interleave(32, dim=1)) @ (
        b.float() * torch.pow(2.0, sb.float() - 127).repeat_interleave(32, dim=1)
    ).T
    kernel = tilelang.compile(gemm, target="ascend", out_idx=-1)
    actual = kernel(
        (a.T.contiguous() if trans_a else a).npu(),
        (b if trans_b else b.T.contiguous()).npu(),
        sa.view(torch.uint16).T.contiguous().npu(),
        sb.view(torch.uint16).T.contiguous().npu(),
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
