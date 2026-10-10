import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.StridedTensor((8, 64), (80, 1), "float32"), dst: T.Tensor((8, 64), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), "float32")
            T.copy(src[2:6, 16:48], ub)
            T.copy(ub, dst[3:7, 8:40])

    return main
