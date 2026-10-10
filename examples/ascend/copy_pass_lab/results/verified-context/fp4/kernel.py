import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((64,), "float4_e2m1fn"), dst: T.Tensor((64,), "float4_e2m1fn")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float4_e2m1fn")
            T.copy(src, ub)
            T.copy(ub, dst)

    return main
