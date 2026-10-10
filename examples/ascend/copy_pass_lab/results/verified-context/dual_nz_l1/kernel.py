import tilelang.ascend.language as T
from tilelang.layout import make_ascend_nz_layout


def make_kernel():
    rows, cols = (32, 128)

    @T.prim_func
    def main(src: T.Tensor((rows, cols), "bfloat16")):
        with T.Kernel(1):
            ub = T.alloc_shared((rows, cols), "bfloat16")
            l1 = T.alloc_l1((rows * 2, cols), "bfloat16")
            T.annotate_layout({l1: make_ascend_nz_layout(l1)})
            T.copy(src, ub)
            T.dual_copy(ub, l1)

    return main
