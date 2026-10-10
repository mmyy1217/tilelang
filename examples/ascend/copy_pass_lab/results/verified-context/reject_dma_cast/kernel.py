import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "bfloat16")
            T.copy(src, ub)

    return main
