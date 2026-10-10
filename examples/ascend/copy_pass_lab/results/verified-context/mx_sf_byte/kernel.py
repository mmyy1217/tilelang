import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(
        a: T.Tensor((16, 256), "float8_e4m3fn"),
        b: T.Tensor((16, 128), "float8_e4m3fn"),
        sa: T.Tensor((4, 16), "uint8"),
        sb: T.Tensor((4, 16), "uint8"),
        dst: T.Tensor((2, 16, 16), "float32"),
    ):
        with T.Kernel(1):
            a1 = T.alloc_l1((16, 256), "float8_e4m3fn")
            b1 = T.alloc_l1((16, 128), "float8_e4m3fn")
            sa1 = T.alloc_l1((16, 4), "uint8")
            sb1 = T.alloc_l1((16, 4), "uint8")
            a0 = T.alloc_l0a((16, 128), "float8_e4m3fn")
            b0 = T.alloc_l0b((16, 128), "float8_e4m3fn")
            sa0 = T.alloc_l0a_sf(a0, sf_dtype="uint8")
            sb0 = T.alloc_l0b_sf(b0, sf_dtype="uint8")
            acc = T.alloc_l0c((16, 16), "float32")
            T.copy(a, a1)
            T.copy(b, b1)
            T.copy(sa, sa1, transpose=True)
            T.copy(sb, sb1, transpose=True)
            T.copy(b1, b0)
            T.copy(sb1, sb0)
            for i in T.Pipelined(2, num_stages=2):
                T.copy(a1[:, i * 128 : (i + 1) * 128], a0)
                T.copy(sa1, sa0)
                T.gemm_blockscaled(a0, b0, acc, sa0, sb0, transpose_B=True, clear_accum=True)
                T.copy(acc, dst[i, :, :])

    return main
