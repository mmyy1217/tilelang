"""
Test the layout-driven UB(ND)->UB(NZ) T.copy frontend.

Does: UB ND (32, 128) -> UB NZ (33, 128) via scatter, with optional cast.
Then writes the physical NZ result back to GM for verification.
"""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_nz_layout


def nd2nz_scatter_test(sd: torch.dtype, dd: torch.dtype, rows: int, cols: int, inside_vf=False):
    """Scatter a tile from ND layout to NZ layout in UB.

    ND layout:  (rows, cols) dense
    NZ layout:  (rows + 1, cols) with padding row to avoid bank conflict
    """

    @T.prim_func
    def main(
        input_gm: T.Tensor((rows, cols), sd),
        output_gm: T.Tensor((rows + 1, cols), dd),
    ):
        with T.Kernel(1):
            src_ub = T.alloc_shared((rows, cols), sd)  # ND layout
            dst_ub = T.alloc_shared((rows + 1, cols), dd)  # NZ layout
            T.annotate_layout({dst_ub: make_ascend_compact_nz_layout(dst_ub)})
            T.copy(input_gm, src_ub)
            if inside_vf:
                with T.SimdVF():
                    T.copy(src_ub, dst_ub[0:rows, 0:cols])
            else:
                T.copy(src_ub, dst_ub[0:rows, 0:cols])
            T.copy(dst_ub, output_gm)

    return main


ROWS = 32
COLS = 128


def _manual_compact_nz_dual_copy():
    @T.prim_func
    def main():
        with T.Kernel(1):
            x_l1 = T.alloc_l1((64, COLS), "float32")
            x_nz = T.alloc_shared((33, COLS), "float32")
            T.annotate_layout(
                {
                    x_l1: make_ascend_nz_layout(x_l1),
                    x_nz: make_ascend_compact_nz_layout(x_nz),
                }
            )
            with T.Cube():
                pass
            with T.Vector():
                T.dual_copy(x_nz[:32, :], x_l1[:, :])

    return main


def _manual_mixed_dual_copy():
    @T.prim_func
    def main(
        cube_a_gm: T.Tensor((16, 16), "bfloat16"),
        input_gm: T.Tensor((128, COLS), "float32"),
        output_gm: T.Tensor((128, COLS), "float32"),
    ):
        with T.Kernel(1):
            cube_a_l1 = T.alloc_l1((16, 16), "bfloat16")
            x_ub = T.alloc_shared((64, COLS), "float32")
            with T.Cube():
                T.copy(cube_a_gm, cube_a_l1)
            with T.Vector():
                T.dual_copy(input_gm, x_ub)
                T.dual_copy(x_ub, output_gm)

    return main


def test_manual_pipeline_rewrites_compact_nz_dual_copy():
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False}):
        source = tilelang.lower(_manual_compact_nz_dual_copy(), target="ascend").kernel_source
    assert "__global__ __mix__(1, 2)" in source
    assert source.count("asc_get_sub_block_id()") == 1
    assert source.count("asc_copy_ub2l1") == 1


def test_manual_mixed_pipeline_shares_sub_block_id():
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False}):
        source = tilelang.lower(_manual_mixed_dual_copy(), target="ascend").kernel_source
    assert "__global__ __mix__(1, 2)" in source
    assert source.count("asc_get_sub_block_id()") == 1
    assert "asc_copy_gm2ub_align" in source
    assert "asc_copy_ub2gm_align" in source


@pytest.mark.parametrize(
    "sd,dd,inside_vf",
    [
        (torch.float16, torch.float16, False),
        (torch.bfloat16, torch.bfloat16, False),
        (torch.bfloat16, torch.float32, False),
        (torch.float32, torch.float32, False),
        (torch.float32, torch.bfloat16, False),
        pytest.param(torch.bfloat16, torch.float32, True, id="inside-simd-vf"),
    ],
)
def test_nd2nz_scatter(sd, dd, inside_vf):
    # Prepare input: a recognizable pattern
    input_gm = torch.arange(ROWS * COLS, dtype=sd, device="npu").reshape(ROWS, COLS) + 1.0

    program = nd2nz_scatter_test(sd, dd, ROWS, COLS, inside_vf)
    kernel = tilelang.compile(
        program,
        target="ascend",
        out_idx=-1,
    )

    output_gm = kernel(input_gm)
    torch.npu.synchronize()

    frac_len = 32 // output_gm.element_size()
    input_cast = input_gm.to(dtype=dd)
    output_back = output_gm.view(COLS // frac_len, ROWS + 1, frac_len)[:, :-1, :].permute(1, 0, 2).reshape(ROWS, COLS)
    assert torch.equal(input_cast, output_back), f"sd={sd} dd={dd}"


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("cols", [*range(32, 513, 32), 544, 608])
@pytest.mark.parametrize("inside_vf", [False, True])
def test_fp8_nd2nz_scatter_preserves_bits(cols, inside_vf):
    dtype = torch.float8_e4m3fn
    rows = 32
    # Include every encoding, including NaNs and signed zero: this is a permutation.
    bits = torch.randint(0, 256, (rows * cols,), dtype=torch.uint8, generator=torch.Generator().manual_seed(31))
    bits[:256] = torch.arange(256, dtype=torch.uint8)
    bits = bits.reshape(rows, cols)
    kernel = tilelang.compile(
        nd2nz_scatter_test(dtype, dtype, rows, cols, inside_vf), target="ascend", out_idx=-1, execution_backend="tvm_ffi"
    )
    output = kernel(bits.view(dtype).npu()).cpu().view(torch.uint8)
    unpacked = output.reshape(cols // 32, rows + 1, 32)[:, :rows, :].permute(1, 0, 2).reshape(rows, cols)
    torch.testing.assert_close(unpacked, bits, rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("cols", range(32, 513, 32))
def test_fp8_nd2nz_source_at_ub_boundary(cols):
    rows = 32
    ub_bytes = 248 * 1024
    dtype = "float8_e4m3fn"

    @T.prim_func
    def main(input_gm: T.Tensor((rows, cols), dtype), output_gm: T.Tensor((rows + 1, cols), dtype)):
        with T.Kernel(1):
            # Ascend950 exposes 248 KiB UB; place the final source byte at its end.
            arena = T.alloc_shared((ub_bytes,), dtype)
            src = T.make_tensor_from_addr(
                T.access_ptr(arena[ub_bytes - rows * cols], "rw", extent=rows * cols),
                (rows, cols),
                dtype,
                storage_scope="shared.dyn",
            )
            nz = T.make_tensor_from_addr(
                T.access_ptr(arena[0], "rw", extent=(rows + 1) * cols),
                (rows + 1, cols),
                dtype,
                storage_scope="shared.dyn",
            )
            T.annotate_layout({nz: make_ascend_compact_nz_layout(nz)})
            T.copy(input_gm, src)
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 0)
            with T.SimdVF():
                T.copy(src, nz[:rows, :])
            T.ascend_set_flag("V_MTE3", 0)
            T.ascend_wait_flag("V_MTE3", 0)
            T.copy(nz, output_gm)

    kernel = tilelang.compile(
        main, target="ascend", out_idx=-1, execution_backend="tvm_ffi", pass_configs={"tl.enable_auto_schedule": False}
    )
    bits = torch.arange(rows * cols, dtype=torch.int32).to(torch.uint8).reshape(rows, cols)
    output = kernel(bits.view(torch.float8_e4m3fn).npu()).cpu().view(torch.uint8)
    unpacked = output.reshape(cols // 32, rows + 1, 32)[:, :rows, :].permute(1, 0, 2).reshape(rows, cols)
    torch.testing.assert_close(unpacked, bits, rtol=0, atol=0)
