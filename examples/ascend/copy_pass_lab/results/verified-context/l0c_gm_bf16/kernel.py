import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(a: T.Tensor((32, 64), "bfloat16"), b: T.Tensor((32, 64), "bfloat16"), dst: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((32, 64), "bfloat16")
            b_l1 = T.alloc_l1((32, 64), "bfloat16")
            acc = T.alloc_l0c((32, 32), "float32")
            ub = T.alloc_shared((32, 32), "bfloat16")
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True, unit_flag_ctrl=None)
            T.copy(acc, dst, unit_flag_ctrl=None)

    return main
