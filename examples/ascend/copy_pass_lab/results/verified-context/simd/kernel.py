"""Small Ascend kernels for offline copy lowering inspection."""

import tilelang.ascend.language as T
from tilelang.layout import make_ascend_nz_layout


def dma_roundtrip():
    @T.prim_func
    def main(src: T.Tensor((4, 64), "float32"), dst: T.Tensor((4, 64), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 64), "float32")
            T.copy(src, ub)
            T.copy(ub, dst)

    return main


def strided_dma():
    @T.prim_func
    def main(src: T.StridedTensor((4, 30), (32, 1), "float32"), dst: T.Tensor((4, 30), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), "float32")
            T.copy(src, ub[:, :30])
            T.copy(ub[:, :30], dst)

    return main


def padded_dma():
    @T.prim_func
    def main(src: T.Tensor((4, 30), "float32"), dst: T.Tensor((4, 32), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), "float32")
            T.copy(src, ub[:, :30], pad_value=-1.0)
            T.copy(ub, dst)

    return main


def tail_dma():
    @T.prim_func
    def main(src: T.Tensor((80,), "float32"), dst: T.Tensor((80,), "float32")):
        with T.Kernel(2) as block:
            ub = T.alloc_shared((64,), "float32")
            T.copy(src[block * 64 : (block + 1) * 64], ub)
            T.copy(ub, dst[block * 64 : (block + 1) * 64])

    return main


def dynamic_dma():
    @T.prim_func
    def main(src: T.Tensor((4, 2, 32), "float32"), dst: T.Tensor((4, 32), "float32"), n: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), "float32")
            if n > 0:
                T.copy(src[:n, 0, :], ub[:n, :])
                T.copy(ub[:n, :], dst[:n, :])

    return main


def gemm_copy():
    @T.prim_func
    def main(a: T.Tensor((31, 47), "bfloat16"), b: T.Tensor((19, 47), "bfloat16"), c: T.Tensor((31, 19), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((32, 64), "bfloat16")
            b_l1 = T.alloc_l1((32, 64), "bfloat16")
            a_l0 = T.alloc_l0a((32, 64), "bfloat16")
            b_l0 = T.alloc_l0b((32, 64), "bfloat16")
            acc = T.alloc_l0c((32, 32), "float32")
            T.copy(a[:32, :64], a_l1)
            T.copy(b[:32, :64], b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, c[:32, :32])

    return main


def dual_dma():
    @T.prim_func
    def main(src: T.Tensor((128, 64), "float32"), dst: T.Tensor((128, 64), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((64, 64), "float32")
            T.dual_copy(src, ub)
            T.dual_copy(ub, dst)

    return main


def dual_gemm():
    @T.prim_func
    def main(a: T.Tensor((32, 64), "bfloat16"), b: T.Tensor((32, 64), "bfloat16"), c: T.Tensor((32, 32), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((32, 64), "bfloat16")
            b_l1 = T.alloc_l1((32, 64), "bfloat16")
            acc = T.alloc_l0c((32, 32), "float32")
            ub = T.alloc_shared((16, 32), "float32")
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc, ub)
            T.dual_copy(ub, c)

    return main


def nz_scatter():
    @T.prim_func
    def main(src: T.Tensor((32, 64), "bfloat16"), dst: T.Tensor((32, 64), "bfloat16")):
        with T.Kernel(1):
            ub = T.alloc_shared((32, 64), "bfloat16")
            l1 = T.alloc_l1((32, 64), "bfloat16")
            T.annotate_layout({l1: make_ascend_nz_layout(l1)})
            T.copy(src, ub)
            T.copy(ub, l1)
            T.copy(ub, dst)

    return main


def vf_copy(mode="simd"):
    @T.prim_func
    def main(src: T.Tensor((256,), "float32"), dst: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((256,), "float32")
            b = T.alloc_shared((256,), "float32")
            T.copy(src, a)
            if mode == "simd":
                with T.SimdVF():
                    T.copy(a, b)
            else:
                with T.SimtVF(threads=128):
                    T.copy(a, b)
            T.copy(b, dst)

    return main


def pipelined_dma():
    @T.prim_func
    def main(src: T.Tensor((4, 256), "float32"), dst: T.Tensor((4, 256), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((256,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for step in T.Pipelined(4, num_stages=2):
                T.copy(src[step, :], ub)
                T.copy(ub, dst[step, :])

    return main


def fp4_dma():
    @T.prim_func
    def main(src: T.Tensor((64,), "float4_e2m1fn"), dst: T.Tensor((64,), "float4_e2m1fn")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float4_e2m1fn")
            T.copy(src, ub)
            T.copy(ub, dst)

    return main


CASES = {
    "dma": dma_roundtrip,
    "strided": strided_dma,
    "padding": padded_dma,
    "tail": tail_dma,
    "dynamic": dynamic_dma,
    "gemm": gemm_copy,
    "dual_dma": dual_dma,
    "dual_gemm": dual_gemm,
    "nz_scatter": nz_scatter,
    "simd": lambda: vf_copy("simd"),
    "simt": lambda: vf_copy("simt"),
    "pipeline": pipelined_dma,
    "fp4": fp4_dma,
}


def make_kernel():
    return CASES['simd']()
