import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((64,), "float32"), dst: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            T.copy(src, ub, l2_cache_ctrl=None)
            T.set_atomic("add", "float32")
            T.copy(ub, dst, l2_cache_ctrl=None)
            T.set_atomic_none()

    return main
