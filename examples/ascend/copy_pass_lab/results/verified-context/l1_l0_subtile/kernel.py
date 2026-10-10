import tilelang.ascend.language as T
from tilelang.layout import make_ascend_major_k_layout, make_ascend_nz_layout


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((64, 64), "bfloat16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((64, 64), "bfloat16")
            l0 = T.alloc_l0a((32, 32), "bfloat16")
            T.annotate_layout({l1: make_ascend_nz_layout(l1), l0: make_ascend_major_k_layout(l0)})
            T.copy(src, l1)
            T.copy(l1[16:48, 16:48], l0, transpose=False)

    return main
