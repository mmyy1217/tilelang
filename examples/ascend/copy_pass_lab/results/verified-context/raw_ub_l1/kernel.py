import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((32, 64), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((32, 64), "float32")
            l1 = T.alloc_l1((32, 64), "float32")
            T.copy(src, ub)
            T.copy(ub, l1)

    return main
