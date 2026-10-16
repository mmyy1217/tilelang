"""Prepacked FP8 UB regions reach Cube without an ND-to-NZ conversion."""

import pytest
import torch
import tilelang
import tilelang.testing
from tilelang.ascend import language as T
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_major_k_layout

pytest.importorskip("torch_npu")


def make_kernel(dtype, rows, src_row, dst_row):
    cols = 128
    src_rows = rows + src_row + 1
    dst_rows = ((rows + dst_row + 15) // 16) * 16

    @T.prim_func
    def kernel(a: T.Tensor((rows, cols), dtype), b: T.Tensor((32, cols), dtype), out: T.Tensor((rows, 32), "float32")):
        with T.MixedKernel(1, sids=1):
            ub = T.alloc_shared((src_rows, cols), dtype)
            a1 = T.alloc_l1((dst_rows, cols), dtype)
            b1 = T.alloc_l1((32, cols), dtype)
            a0 = T.alloc_l0a((rows, cols), dtype)
            b0 = T.alloc_l0b((32, cols), dtype)
            c0 = T.alloc_l0c((rows, 32), "float32")
            T.annotate_layout({ub: make_ascend_compact_nz_layout(ub), a1: make_ascend_major_k_layout(a1)})
            with T.Vector(vector=1):
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(rows, cols):
                        ub[src_row + i, j] = a[i, j]
                T.ascend_set_flag("V_MTE3", 0)
                T.ascend_wait_flag("V_MTE3", 0)
                T.copy(ub[src_row : src_row + rows, :], a1[dst_row : dst_row + rows, :])
                T.ascend_cross_core_set_flag(4, "PIPE_MTE3", 0)
                T.ascend_pipe_barrier("PIPE_ALL")
            with T.Cube():
                T.copy(b, b1)
                T.ascend_set_flag("MTE2_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 0)
                T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 0)
                T.copy(a1[dst_row : dst_row + rows, :], a0)
                T.copy(b1, b0)
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 0)
                T.gemm(a0, b0, c0, transpose_B=True, clear_accum=True, unit_flag_ctrl=0)
                T.ascend_set_flag("M_FIX", 0)
                T.ascend_wait_flag("M_FIX", 0)
                T.copy(c0, out, unit_flag_ctrl=0)
                T.ascend_pipe_barrier("PIPE_ALL")

    return kernel


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("rows,src_row,dst_row", [(32, 0, 0), (32, 1, 16), (72, 1, 0), (104, 1, 0)])
def test_prepacked_fp8_ub_to_l1_gemm(rows, src_row, dst_row):
    dtype = "float8_e4m3fn"
    program = make_kernel(dtype, rows, src_row, dst_row)
    kernel = tilelang.compile(program, target="ascend", execution_backend="tvm_ffi", pass_configs={"tl.enable_auto_schedule": False})
    torch.manual_seed(31)
    a = torch.randint(-4, 5, (rows, 128)).to(getattr(torch, dtype))
    b = torch.randint(-4, 5, (32, 128)).to(getattr(torch, dtype))
    expected = a.float() @ b.float().T
    out = torch.full((rows, 32), float("nan"), device="npu")
    kernel(a.npu(), b.npu(), out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


def make_raw_kernel(dtype, src_gap, dst_gap):
    src_pitch = 1024 + src_gap
    dst_pitch = 1024 + dst_gap
    l1_rows = dst_pitch // 32

    @T.prim_func
    def kernel(data: T.Tensor((4, src_pitch), dtype), b: T.Tensor((32, 128), dtype), out: T.Tensor((32, 32), "float32")):
        with T.MixedKernel(1, sids=1):
            ub = T.alloc_shared((4, src_pitch), dtype)
            a1 = T.alloc_l1((l1_rows, 128), dtype)
            raw_l1 = T.make_tensor_from_addr(
                T.access_ptr(a1[0, 0], "w", extent=l1_rows * 128), (4, dst_pitch), dtype, storage_scope="shared.l1"
            )
            b1 = T.alloc_l1((32, 128), dtype)
            a0 = T.alloc_l0a((32, 128), dtype)
            b0 = T.alloc_l0b((32, 128), dtype)
            c0 = T.alloc_l0c((32, 32), "float32")
            with T.Vector(vector=1):
                T.copy(data, ub)
                T.ascend_set_flag("MTE2_MTE3", 0)
                T.ascend_wait_flag("MTE2_MTE3", 0)
                T.copy(ub[:, :1024], raw_l1[:, :1024])
                T.ascend_cross_core_set_flag(4, "PIPE_MTE3", 0)
                T.ascend_pipe_barrier("PIPE_ALL")
            with T.Cube():
                T.copy(b, b1)
                T.ascend_set_flag("MTE2_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 0)
                T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 0)
                T.copy(a1[:32, :], a0)
                T.copy(b1, b0)
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 0)
                T.gemm(a0, b0, c0, transpose_B=True, clear_accum=True, unit_flag_ctrl=0)
                T.ascend_set_flag("M_FIX", 0)
                T.ascend_wait_flag("M_FIX", 0)
                T.copy(c0, out, unit_flag_ctrl=0)
                T.ascend_pipe_barrier("PIPE_ALL")

    return kernel


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("src_gap,dst_gap", [(0, 0), (32, 0), (0, 1024), (32, 1024)])
def test_raw_fp8_ub_to_l1_pitches(src_gap, dst_gap):
    dtype = "float8_e4m3fn"
    program = make_raw_kernel(dtype, src_gap, dst_gap)
    kernel = tilelang.compile(program, target="ascend", execution_backend="tvm_ffi", pass_configs={"tl.enable_auto_schedule": False})
    torch.manual_seed(31)
    a = torch.randint(-4, 5, (32, 128)).to(getattr(torch, dtype))
    b = torch.randint(-4, 5, (32, 128)).to(getattr(torch, dtype))
    # Pack the host input as four NZ column groups, with a sentinel in each gap.
    packed = torch.full((4, 1024 + src_gap), 113, dtype=torch.uint8)
    packed[:, :1024] = a.view(torch.uint8).reshape(32, 4, 32).permute(1, 0, 2).reshape(4, 1024)
    out = torch.full((32, 32), float("nan"), device="npu")
    kernel(packed.view(getattr(torch, dtype)).npu(), b.npu(), out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), a.float() @ b.float().T, atol=0, rtol=0)


@tilelang.testing.requires_ascend
def test_prepacked_fp8_copy_with_auto_schedule():
    dtype = "float8_e4m3fn"

    @T.prim_func
    def program(a: T.Tensor((32, 128), dtype), b: T.Tensor((32, 128), dtype), out: T.Tensor((32, 32), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((33, 128), dtype)
            a1 = T.alloc_l1((32, 128), dtype)
            b1 = T.alloc_l1((32, 128), dtype)
            a0 = T.alloc_l0a((32, 128), dtype)
            b0 = T.alloc_l0b((32, 128), dtype)
            c0 = T.alloc_l0c((32, 32), "float32")
            T.annotate_layout({ub: make_ascend_compact_nz_layout(ub)})
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(32, 128):
                    ub[i, j] = a[i, j]
            T.copy(ub[:32, :], a1)
            T.copy(b, b1)
            T.copy(a1, a0)
            T.copy(b1, b0)
            T.gemm(a0, b0, c0, transpose_B=True, clear_accum=True)
            T.copy(c0, out)

    kernel = tilelang.compile(program, target="ascend", execution_backend="tvm_ffi")
    torch.manual_seed(31)
    a = torch.randint(-4, 5, (32, 128)).to(getattr(torch, dtype))
    b = torch.randint(-4, 5, (32, 128)).to(getattr(torch, dtype))
    out = torch.full((32, 32), float("nan"), device="npu")
    kernel(a.npu(), b.npu(), out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), a.float() @ b.float().T, atol=0, rtol=0)
