"""Smoke coverage for the FP8 ND-to-NZ GEMM example's copy modes."""

import pytest
import torch
import tilelang
import tilelang.testing
from example_fp8_nd2nz_gemm import fp8_nd2nz_gemm


@tilelang.testing.requires_ascend
@pytest.mark.skipif(tilelang.env.get_default_target() == "pto", reason="FP8 ND-to-NZ scatter requires the AscendC backend")
@pytest.mark.parametrize("dual_copy", [False, True], ids=["copy", "dual-copy"])
def test_fp8_nd2nz_gemm(dual_copy):
    generator = torch.Generator().manual_seed(0)
    a = torch.randint(-4, 5, (32, 288), generator=generator).to(torch.float8_e4m3fn)
    b = torch.randint(-4, 5, (32, 288), generator=generator).to(torch.float8_e4m3fn)
    kernel = tilelang.compile(fp8_nd2nz_gemm(dual_copy), target="ascend", out_idx=-1, execution_backend="tvm_ffi")
    torch.testing.assert_close(kernel(a.npu(), b.npu()).cpu(), a.float() @ b.float().T, rtol=0, atol=0)
