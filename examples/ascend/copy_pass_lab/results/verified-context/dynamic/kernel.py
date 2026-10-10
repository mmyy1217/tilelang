import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((4, 2, 32), "float32"), dst: T.Tensor((4, 32), "float32"), n: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), "float32")
            if n > 0:
                T.copy(src[:n, 0, :], ub[:n, :])
                T.copy(ub[:n, :], dst[:n, :])

    return main
