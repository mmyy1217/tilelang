import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.StridedTensor((2, 3, 4), (20, 5, 1), "bfloat16")):
        with T.Kernel(1):
            ub = T.alloc_shared((2, 3, 4), "bfloat16")
            T.copy(src, ub)

    return main
