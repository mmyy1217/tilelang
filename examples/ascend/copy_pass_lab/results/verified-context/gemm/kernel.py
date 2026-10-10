import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(a: T.Tensor((31, 47), "bfloat16"), b: T.Tensor((19, 47), "bfloat16"), c: T.Tensor((31, 19), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((32, 64), "bfloat16")
            b_l1 = T.alloc_l1((32, 64), "bfloat16")
            a_l0 = T.alloc_l0a((32, 64), "bfloat16")
            b_l0 = T.alloc_l0b((32, 64), "bfloat16")
            acc = T.alloc_l0c((32, 32), "float32")
            T.copy(a[:32, :64], a_l1)
            T.copy(b[:32, :64], b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, c[:32, :32])

    return main
