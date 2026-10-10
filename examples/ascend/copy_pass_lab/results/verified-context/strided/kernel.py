import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.StridedTensor((4, 30), (32, 1), "float32"), dst: T.Tensor((4, 30), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), "float32")
            T.copy(src, ub[:, :30])
            T.copy(ub[:, :30], dst)

    return main
