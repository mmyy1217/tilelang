import tilelang.ascend.language as T


def make_kernel():
    n = T.dynamic("n")

    @T.prim_func
    def main(src: T.Tensor((n,), "float32"), dst: T.Tensor((n,), "float32")):
        with T.Kernel(1) as bx:
            ub = T.alloc_shared((T.ceildiv(n, 64) * 64,), "float32")
            T.copy(src, ub[:n])
            T.copy(ub[:n], dst)

    return main
