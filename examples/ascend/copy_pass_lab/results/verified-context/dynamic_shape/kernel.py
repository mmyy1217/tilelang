import tilelang.ascend.language as T


def make_kernel():
    n = T.dynamic("n")

    @T.prim_func
    def main(src: T.Tensor((n,), "float32"), dst: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 64)) as bx:
            ub = T.alloc_shared((64,), "float32")
            T.copy(src[bx * 64 : (bx + 1) * 64], ub)
            T.copy(ub, dst[bx * 64 : (bx + 1) * 64])

    return main
