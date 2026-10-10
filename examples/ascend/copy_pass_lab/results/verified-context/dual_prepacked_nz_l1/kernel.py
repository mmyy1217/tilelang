import tilelang.ascend.language as T
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_nz_layout


def make_kernel():
    rows, cols = (32, 128)
    full_rows = rows * 2

    @T.prim_func
    def main(src: T.Tensor((rows, cols), "bfloat16"), dst: T.Tensor((rows + 1, cols), "bfloat16")):
        with T.Kernel(1):
            ub = T.alloc_shared((rows, cols), "bfloat16")
            nz = T.alloc_shared((rows + 1, cols), "bfloat16")
            l1 = T.alloc_l1((full_rows, cols), "bfloat16")
            T.annotate_layout({nz: make_ascend_compact_nz_layout(nz), l1: make_ascend_nz_layout(l1)})
            T.copy(src, ub)
            T.copy(ub, nz[:rows, :])
            T.dual_copy(nz[:rows, :], l1)
            T.copy(nz, dst)

    return main
