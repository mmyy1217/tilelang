import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((80,), "float32"), dst: T.Tensor((80,), "float32")):
        with T.Kernel(2) as block:
            ub = T.alloc_shared((64,), "float32")
            T.copy(src[block * 64 : (block + 1) * 64], ub)
            T.copy(ub, dst[block * 64 : (block + 1) * 64])

    return main
