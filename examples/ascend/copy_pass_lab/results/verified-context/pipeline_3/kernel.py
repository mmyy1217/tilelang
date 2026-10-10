import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((8, 256), "float32"), dst: T.Tensor((8, 256), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((256,), "float32")
            T.annotate_buffer_versions({ub: 3})
            for i in T.Pipelined(8, num_stages=3):
                T.copy(src[i, :], ub)
                T.copy(ub, dst[i, :])

    return main
