import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((80,), "float32"), dst: T.Tensor((80,), "float32")):
        with T.Kernel(3) as bx:
            ub = T.alloc_shared((64,), "float32")
            T.copy(src[bx * 64 : (bx + 1) * 64], ub)
            T.copy(ub, dst[bx * 64 : (bx + 1) * 64])

    return main
