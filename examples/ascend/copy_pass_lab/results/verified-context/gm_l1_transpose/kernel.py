import tilelang.ascend.language as T
from tilelang.layout import make_ascend_nz_layout


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((64, 32), "bfloat16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((32, 64), "bfloat16")
            T.annotate_layout({l1: make_ascend_nz_layout(l1)})
            T.copy(src, l1, transpose=True)

    return main
