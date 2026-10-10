import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((1,), "float32"), dst: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            T.copy(src[0], dst[0])

    return main
