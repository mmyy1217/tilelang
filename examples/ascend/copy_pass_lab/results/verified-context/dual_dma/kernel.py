import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((128, 64), "float32"), dst: T.Tensor((128, 64), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((64, 64), "float32")
            T.dual_copy(src, ub)
            T.dual_copy(ub, dst)

    return main
