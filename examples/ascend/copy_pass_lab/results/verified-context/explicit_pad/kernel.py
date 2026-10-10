import tilelang.ascend.language as T


def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((4, 30), "float32"), dst: T.Tensor((4, 32), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), "float32")
            T.ascend_set_copy_pad_value(2.0, dtype="float32")
            T.copy(src, ub[:, :30], data_select=1)
            T.copy(ub, dst)

    return main
