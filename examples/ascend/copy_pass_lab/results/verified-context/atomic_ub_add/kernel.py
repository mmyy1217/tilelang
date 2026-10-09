"""Additional copy routes, state controls, and explicit unsupported boundaries."""

from functools import partial

import tilelang.ascend.language as T
from tilelang.layout import (
    make_ascend_compact_nz_layout,
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
    make_ascend_nz_layout,
)


def raw_ub_l1(dual=False, strided=False, short=False):
    rows = 64 if dual else 32
    cols = 5 if short else 64

    @T.prim_func
    def main(src: T.Tensor((32, cols), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((32, cols), "float32")
            l1 = T.alloc_l1((rows, cols), "float32")
            T.copy(src, ub)
            if strided:
                T.copy(ub[:, :32], l1[:, :32])
            elif short:
                T.copy(ub[0, :], l1[0, :])
            elif dual:
                T.dual_copy(ub, l1)
            else:
                T.copy(ub, l1)

    return main


def nz_pack(src_dtype="bfloat16", dst_dtype="bfloat16", inside_vf=False, post=False, dual=False, manual=False):
    rows, cols = 32, 128
    full_rows = rows * 2 if dual else rows

    @T.prim_func
    def main(src: T.Tensor((rows, cols), src_dtype), dst: T.Tensor((rows + 1, cols), dst_dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((rows, cols), src_dtype)
            nz = T.alloc_shared((rows + 1, cols), dst_dtype)
            l1 = T.alloc_l1((full_rows, cols), dst_dtype)
            T.annotate_layout({nz: make_ascend_compact_nz_layout(nz), l1: make_ascend_nz_layout(l1)})
            T.copy(src, ub)
            if inside_vf:
                with T.SimdVF():
                    T.copy(ub, nz[:rows, :])
            else:
                T.copy(ub, nz[:rows, :])
            if manual:
                T.ascend_nd2nz_post_copy(l1[0, 0], nz[0, 0], rows, cols, full_rows, dst_dtype)
            elif dual:
                T.dual_copy(nz[:rows, :], l1)
            elif post:
                T.copy(nz[:rows, :], l1)
            T.copy(nz, dst)

    return main


def nz_l1(src_dtype="float32", dst_dtype="bfloat16", single_c0=False, dual=False):
    rows, cols = 32, 16 if single_c0 else 128

    @T.prim_func
    def main(src: T.Tensor((rows, cols), src_dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((rows, cols), src_dtype)
            l1 = T.alloc_l1((rows * 2 if dual else rows, cols), dst_dtype)
            T.annotate_layout({l1: make_ascend_nz_layout(l1)})
            T.copy(src, ub)
            if dual:
                T.dual_copy(ub, l1)
            else:
                T.copy(ub, l1)

    return main


def gm_l1_transpose():
    @T.prim_func
    def main(src: T.Tensor((64, 32), "bfloat16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((32, 64), "bfloat16")
            T.annotate_layout({l1: make_ascend_nz_layout(l1)})
            T.copy(src, l1, transpose=True)

    return main


def l1_l0(scope="a", transpose=False, implicit=False, subregion=False):
    @T.prim_func
    def main(src: T.Tensor((64, 64), "bfloat16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((64, 64), "bfloat16")
            l0 = T.alloc_l0a((32, 32), "bfloat16") if scope == "a" else T.alloc_l0b((32, 32), "bfloat16")
            T.annotate_layout(
                {l1: make_ascend_nz_layout(l1), l0: make_ascend_major_mn_layout(l0) if implicit else make_ascend_major_k_layout(l0)}
            )
            T.copy(src, l1)
            if subregion:
                T.copy(l1[16:48, 16:48], l0, transpose=transpose)
            else:
                T.copy(l1[:32, :32], l0, transpose=transpose)

    return main


def fixpipe_output(dst_dtype="float32", ub_output=False, dual=False, split_n=False, atomic=None, unit=False):
    @T.prim_func
    def main(a: T.Tensor((32, 64), "bfloat16"), b: T.Tensor((32, 64), "bfloat16"), dst: T.Tensor((32, 32), dst_dtype)):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((32, 64), "bfloat16")
            b_l1 = T.alloc_l1((32, 64), "bfloat16")
            acc = T.alloc_l0c((32, 32), "float32")
            ub = T.alloc_shared((32 if split_n or not dual else 16, 16 if split_n else 32), dst_dtype)
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True, unit_flag_ctrl=3 if unit else None)
            if atomic is not None:
                T.set_atomic(atomic, dst_dtype)
            if ub_output:
                if dual:
                    T.dual_copy(acc, ub)
                    T.dual_copy(ub, dst)
                else:
                    T.copy(acc, ub)
                    T.copy(ub, dst)
            else:
                T.copy(acc, dst, unit_flag_ctrl=3 if unit else None)
            if atomic is not None:
                T.set_atomic_none()

    return main


def vf_cast(mode="simd", src_dtype="float32", dst_dtype="bfloat16", scalar=False):
    size = 1 if scalar else 256

    @T.prim_func
    def main(src: T.Tensor((size,), src_dtype), dst: T.Tensor((size,), dst_dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((size,), src_dtype)
            b = T.alloc_shared((size,), dst_dtype)
            T.copy(src, a)
            if mode == "simd":
                with T.SimdVF():
                    T.copy(a, b)
            else:
                with T.SimtVF(threads=128):
                    T.copy(a, b)
            T.copy(b, dst)

    return main


def dynamic_shape(dynamic_ub=False):
    n = T.dynamic("n")

    @T.prim_func
    def main(src: T.Tensor((n,), "float32"), dst: T.Tensor((n,), "float32")):
        with T.Kernel(1 if dynamic_ub else T.ceildiv(n, 64)) as bx:
            ub = T.alloc_shared((T.ceildiv(n, 64) * 64 if dynamic_ub else 64,), "float32")
            if dynamic_ub:
                T.copy(src, ub[:n])
                T.copy(ub[:n], dst)
            else:
                T.copy(src[bx * 64 : (bx + 1) * 64], ub)
                T.copy(ub, dst[bx * 64 : (bx + 1) * 64])

    return main


def empty_tail():
    @T.prim_func
    def main(src: T.Tensor((80,), "float32"), dst: T.Tensor((80,), "float32")):
        with T.Kernel(3) as bx:
            ub = T.alloc_shared((64,), "float32")
            T.copy(src[bx * 64 : (bx + 1) * 64], ub)
            T.copy(ub, dst[bx * 64 : (bx + 1) * 64])

    return main


def pipeline(stages=1):
    @T.prim_func
    def main(src: T.Tensor((8, 256), "float32"), dst: T.Tensor((8, 256), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((256,), "float32")
            T.annotate_buffer_versions({ub: stages})
            for i in T.Pipelined(8, num_stages=stages):
                T.copy(src[i, :], ub)
                T.copy(ub, dst[i, :])

    return main


def typed_dma(dtype="float8_e4m3fn", odd_fp4=False, atomic=None, cache=False):
    size = 64

    @T.prim_func
    def main(src: T.Tensor((size,), dtype), dst: T.Tensor((size,), dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((size,), dtype)
            if odd_fp4:
                T.copy(src[:3], ub[:3])
                T.copy(ub[:3], dst[:3])
            else:
                T.copy(src, ub, l2_cache_ctrl=1 if cache else None)
                if atomic is not None:
                    T.set_atomic(atomic, dtype)
                T.copy(ub, dst, l2_cache_ctrl=4 if cache else None)
                if atomic is not None:
                    T.set_atomic_none()

    return main


def mx_scales(placement="tile", sf_dtype="uint16", data_dtype="float8_e4m3fn"):
    width = 2 if sf_dtype == "uint16" else 4

    @T.prim_func
    def main(
        a: T.Tensor((16, 256), data_dtype),
        b: T.Tensor((16, 128), data_dtype),
        sa: T.Tensor((width, 16), sf_dtype),
        sb: T.Tensor((width, 16), sf_dtype),
        dst: T.Tensor((2, 16, 16), "float32"),
    ):
        with T.Kernel(1):
            a1 = T.alloc_l1((16, 256), data_dtype)
            b1 = T.alloc_l1((16, 128), data_dtype)
            sa1 = T.alloc_l1((16, width), sf_dtype)
            sb1 = T.alloc_l1((16, width), sf_dtype)
            a0 = T.alloc_l0a((16, 128), data_dtype)
            b0 = T.alloc_l0b((16, 128), data_dtype)
            sa0 = T.alloc_l0a_sf(a0, sf_dtype=sf_dtype)
            sb0 = T.alloc_l0b_sf(b0, sf_dtype=sf_dtype)
            acc = T.alloc_l0c((16, 16), "float32")
            T.copy(a, a1)
            T.copy(b, b1)
            T.copy(sa, sa1, transpose=True)
            T.copy(sb, sb1, transpose=True)
            T.copy(b1, b0)
            T.copy(sb1, sb0)
            if placement == "kernel":
                T.copy(sa1, sa0)
            for i in T.Pipelined(2, num_stages=2):
                T.copy(a1[:, i * 128 : (i + 1) * 128], a0)
                if placement == "tile":
                    T.copy(sa1, sa0)
                T.gemm_blockscaled(a0, b0, acc, sa0, sb0, transpose_B=True, clear_accum=True)
                T.copy(acc, dst[i, :, :])

    return main


def scalar_copy():
    @T.prim_func
    def main(src: T.Tensor((1,), "float32"), dst: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            T.copy(src[0], dst[0])

    return main


def illegal_dma_cast():
    @T.prim_func
    def main(src: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "bfloat16")
            T.copy(src, ub)

    return main


def offset_strided():
    @T.prim_func
    def main(src: T.StridedTensor((8, 64), (80, 1), "float32"), dst: T.Tensor((8, 64), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), "float32")
            T.copy(src[2:6, 16:48], ub)
            T.copy(ub, dst[3:7, 8:40])

    return main


def explicit_pad():
    @T.prim_func
    def main(src: T.Tensor((4, 30), "float32"), dst: T.Tensor((4, 32), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), "float32")
            T.ascend_set_copy_pad_value(2.0, dtype="float32")
            T.copy(src, ub[:, :30], data_select=1)
            T.copy(ub, dst)

    return main


def simt_global():
    @T.prim_func
    def main(src: T.Tensor((256,), "float32"), dst: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((256,), "float32")
            with T.SimtVF(threads=128):
                T.copy(src, ub)
            T.copy(ub, dst)

    return main


def unsupported_3d_dma():
    @T.prim_func
    def main(src: T.StridedTensor((2, 3, 4), (20, 5, 1), "bfloat16")):
        with T.Kernel(1):
            ub = T.alloc_shared((2, 3, 4), "bfloat16")
            T.copy(src, ub)

    return main


CASES = {
    "offset_strided": offset_strided,
    "explicit_pad": explicit_pad,
    "simt_global": simt_global,
    "raw_ub_l1": raw_ub_l1,
    "dual_raw_ub_l1": partial(raw_ub_l1, dual=True),
    "compact_nz": nz_pack,
    "compact_nz_cast_down": partial(nz_pack, src_dtype="float32"),
    "compact_nz_cast_up": partial(nz_pack, dst_dtype="float32"),
    "compact_nz_simd": partial(nz_pack, inside_vf=True),
    "prepacked_nz_l1": partial(nz_pack, post=True),
    "dual_prepacked_nz_l1": partial(nz_pack, dual=True),
    "manual_post_copy": partial(nz_pack, manual=True),
    "nz_l1_cast_down": nz_l1,
    "nz_l1_cast_up": partial(nz_l1, src_dtype="bfloat16", dst_dtype="float32"),
    "nz_l1_single_c0": partial(nz_l1, src_dtype="bfloat16", single_c0=True),
    "dual_nz_l1": partial(nz_l1, src_dtype="bfloat16", dual=True),
    "gm_l1_transpose": gm_l1_transpose,
    "l1_l0a_transpose": partial(l1_l0, transpose=True),
    "l1_l0b_transpose": partial(l1_l0, scope="b", transpose=True),
    "l1_l0_layout_transpose": partial(l1_l0, implicit=True),
    "l1_l0_subtile": partial(l1_l0, subregion=True),
    "l0c_ub": partial(fixpipe_output, ub_output=True),
    "l0c_ub_bf16": partial(fixpipe_output, dst_dtype="bfloat16", ub_output=True),
    "l0c_gm_bf16": partial(fixpipe_output, dst_dtype="bfloat16"),
    "l0c_gm_fp16": partial(fixpipe_output, dst_dtype="float16"),
    "dual_l0c_n": partial(fixpipe_output, ub_output=True, dual=True, split_n=True),
    "unit_flag": partial(fixpipe_output, unit=True),
    "atomic_fixpipe": partial(fixpipe_output, atomic="add"),
    "atomic_ub_add": partial(typed_dma, dtype="float32", atomic="add"),
    "atomic_ub_max": partial(typed_dma, dtype="float32", atomic="max"),
    "atomic_ub_min": partial(typed_dma, dtype="float32", atomic="min"),
    "simd_cast_down": vf_cast,
    "simd_cast_fp16": partial(vf_cast, dst_dtype="float16"),
    "reject_simd_cast_up": partial(vf_cast, src_dtype="bfloat16", dst_dtype="float32"),
    "simt_cast": partial(vf_cast, mode="simt"),
    "simt_scalar": partial(vf_cast, mode="simt", dst_dtype="float32", scalar=True),
    "scalar_global": scalar_copy,
    "dynamic_shape": dynamic_shape,
    "dynamic_ub": partial(dynamic_shape, dynamic_ub=True),
    "empty_tail": empty_tail,
    "pipeline_1": pipeline,
    "pipeline_3": partial(pipeline, stages=3),
    "pipeline_4": partial(pipeline, stages=4),
    "fp8_e4m3": typed_dma,
    "fp8_e5m2": partial(typed_dma, dtype="float8_e5m2"),
    "cache_policy": partial(typed_dma, dtype="float32", cache=True),
    "mx_sf_tile": mx_scales,
    "mx_sf_lifetime": partial(mx_scales, placement="kernel"),
    "mx_sf_byte": partial(mx_scales, sf_dtype="uint8"),
    "mx_fp4": partial(mx_scales, data_dtype="float4_e2m1fn"),
    "reject_dma_3d": unsupported_3d_dma,
    "reject_simd_short": partial(vf_cast, dst_dtype="float32", scalar=True),
    "reject_raw_stride": partial(raw_ub_l1, strided=True),
    "reject_raw_payload": partial(raw_ub_l1, short=True),
    "reject_fp4_odd": partial(typed_dma, dtype="float4_e2m1fn", odd_fp4=True),
    "reject_dma_cast": illegal_dma_cast,
    "reject_dual_cast": partial(fixpipe_output, dst_dtype="bfloat16", ub_output=True, dual=True),
}

EXPECTED_ERRORS = {
    "reject_dma_3d": ">2D strided copy not yet supported",
    "reject_simd_short": "total elements divisible by VReg",
    "reject_raw_stride": "contiguous source region",
    "reject_raw_payload": "multiple of 32 bytes",
    "reject_fp4_odd": "packed sub-byte row must be byte-aligned",
    "reject_dma_cast": "DMA copies cannot perform type casting",
    "reject_dual_cast": "Type conversion during cc->ub dual-destination copy is NOT supported",
    "reject_simd_cast_up": "cast copy only supports f32->bf16 or f32->half",
}


def make_kernel():
    return CASES['atomic_ub_add']()
