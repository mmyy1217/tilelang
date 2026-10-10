import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((256,), "float32"), dst: T.Tensor((256,), "bfloat16")):
        with T.Kernel(1):
            a = T.alloc_shared((256,), "float32")
            b = T.alloc_shared((256,), "bfloat16")
            T.copy(src, a)
            with T.SimtVF(threads=128):
                T.copy(a, b)
            T.copy(b, dst)

    return main
