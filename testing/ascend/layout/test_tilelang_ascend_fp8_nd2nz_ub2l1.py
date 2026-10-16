"""FP8 ND-to-NZ scatter supplies either GEMM operand through UB-to-L1 DMA."""

import pytest
import torch
import tilelang
import tilelang.testing
from tilelang.ascend import language as T
from tilelang.layout import make_ascend_compact_nz_layout

pytest.importorskip("torch_npu")


def make_kernel(dtype, cols, route, operand):
    rows = 32
    sids = 2 if route == "dual_copy" else 1
    part_rows = rows // sids

    @T.prim_func
    def kernel(a: T.Tensor((2, rows, cols), dtype), b: T.Tensor((2, rows, cols), dtype), out: T.Tensor((2, rows, rows), "float32")):
        with T.MixedKernel(1, sids=sids):
            nd = T.alloc_shared((part_rows, cols), dtype)
            nz = T.alloc_shared((part_rows + 1, cols), dtype)
            a1 = T.alloc_l1((rows, cols), dtype)
            b1 = T.alloc_l1((rows, cols), dtype)
            a0 = T.alloc_l0a((rows, cols), dtype)
            b0 = T.alloc_l0b((rows, cols), dtype)
            c0 = T.alloc_l0c((rows, rows), "float32")
            T.annotate_layout({nz: make_ascend_compact_nz_layout(nz)})
            for stage in T.Pipelined(2):
                if operand == "A":
                    T.copy(b[stage, :, :], b1)
                    if route == "dual_copy":
                        T.dual_copy(a[stage, :, :], nd)
                        T.dual_copy(nd, a1)
                    else:
                        T.copy(a[stage, :, :], nd)
                        if route == "explicit":
                            T.copy(nd, nz[:part_rows, :])
                            T.copy(nz[:part_rows, :], a1)
                        else:
                            T.copy(nd, a1)
                else:
                    T.copy(a[stage, :, :], a1)
                    if route == "dual_copy":
                        T.dual_copy(b[stage, :, :], nd)
                        T.dual_copy(nd, b1)
                    else:
                        T.copy(b[stage, :, :], nd)
                        if route == "explicit":
                            T.copy(nd, nz[:part_rows, :])
                            T.copy(nz[:part_rows, :], b1)
                        else:
                            T.copy(nd, b1)
                T.copy(a1, a0)
                T.copy(b1, b0)
                T.gemm(a0, b0, c0, transpose_B=True, clear_accum=True)
                T.copy(c0, out[stage, :, :])

    return kernel


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("cols", [32, 96, 256, 288])
@pytest.mark.parametrize("route", ["copy", "dual_copy", "explicit"])
@pytest.mark.parametrize("operand", ["A", "B"])
def test_fp8_nd2nz_ub_to_l1_gemm(cols, route, operand):
    dtype = "float8_e4m3fn"
    kernel = tilelang.compile(make_kernel(dtype, cols, route, operand), target="ascend", execution_backend="tvm_ffi")
    torch.manual_seed(31)
    a = torch.randint(-4, 5, (2, 32, cols)).to(getattr(torch, dtype))
    b = torch.randint(-4, 5, (2, 32, cols)).to(getattr(torch, dtype))
    out = torch.full((2, 32, 32), float("nan"), device="npu")
    kernel(a.npu(), b.npu(), out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), a.float() @ b.float().transpose(-1, -2), rtol=0, atol=0)
