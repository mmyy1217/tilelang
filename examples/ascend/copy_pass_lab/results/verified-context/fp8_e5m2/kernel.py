import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((64,), "float8_e5m2"), dst: T.Tensor((64,), "float8_e5m2")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float8_e5m2")
            T.copy(src, ub, l2_cache_ctrl=None)
            T.copy(ub, dst, l2_cache_ctrl=None)

    return main
