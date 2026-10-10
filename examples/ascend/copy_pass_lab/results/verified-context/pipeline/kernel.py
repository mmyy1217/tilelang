import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((4, 256), "float32"), dst: T.Tensor((4, 256), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((256,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for step in T.Pipelined(4, num_stages=2):
                T.copy(src[step, :], ub)
                T.copy(ub, dst[step, :])

    return main
