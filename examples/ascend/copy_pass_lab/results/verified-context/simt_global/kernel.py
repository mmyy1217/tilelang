import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((256,), "float32"), dst: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((256,), "float32")
            with T.SimtVF(threads=128):
                T.copy(src, ub)
            T.copy(ub, dst)

    return main
