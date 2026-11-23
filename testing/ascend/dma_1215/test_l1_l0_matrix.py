"""1215 L1→L0A/B 二维矩阵块加载（非转置，含子块切片、l0_pitch 拼装与非整分形）。"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


pytestmark = tilelang.testing.requires_ascend.marks()

# DMA 使用原 dtype。GEMM 用同位宽的 Cube 类型读回原始位模式。
READBACK = {
    "int8": "int8",
    "uint8": "int8",
    "float8_e4m3fn": "int8",
    "int16": "float16",
    "uint16": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "int32": "float32",
    "uint32": "float32",
    "float32": "float32",
}


def _matrix_data(shape, dtype):
    rng = torch.Generator().manual_seed(1215)
    return torch.randint(1, 63, shape, generator=rng).to(getattr(torch, READBACK[dtype]))


def _readback_dtypes(dtype):
    read_dtype = READBACK[dtype]
    return read_dtype, "int32" if read_dtype == "int8" else "float32"


def _check_gemm(main, a, b, expected):
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected.to(actual.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("operand", ["l0a", "l0b"])
def test_l1_to_l0_matrix(dtype, operand):
    """把 16×32 完整块按 10 种普通 dtype 分别加载到 L0A 和 L0B，用单位矩阵读回。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    if operand == "l0a":
        @T.prim_func
        def main(src: T.Tensor((16, 32), dtype), eye: T.Tensor((32, 32), read_dtype), out: T.Tensor((16, 32), acc_dtype)):
            with T.Kernel(1):
                a_l1 = T.alloc_l1((16, 32), dtype)
                b_l1 = T.alloc_l1((32, 32), read_dtype)
                a_l0 = T.alloc_l0a((16, 32), dtype)
                b_l0 = T.alloc_l0b((32, 32), read_dtype)
                acc = T.alloc_l0c((16, 32), acc_dtype)
                T.copy(src, a_l1, transpose=False)
                T.copy(eye, b_l1, transpose=False)
                T.copy(a_l1, a_l0, transpose=False)
                T.copy(b_l1, b_l0, transpose=False)
                T.gemm(T.view(a_l0, dtype=read_dtype), b_l0, acc, transpose_B=True, clear_accum=True)
                T.copy(acc, out)

        data = _matrix_data((16, 32), dtype)
        identity = torch.eye(32, dtype=getattr(torch, read_dtype))
        _check_gemm(main, data.view(getattr(torch, dtype)), identity, data)
    else:
        @T.prim_func
        def main(eye: T.Tensor((32, 32), read_dtype), src: T.Tensor((16, 32), dtype), out: T.Tensor((32, 16), acc_dtype)):
            with T.Kernel(1):
                a_l1 = T.alloc_l1((32, 32), read_dtype)
                b_l1 = T.alloc_l1((16, 32), dtype)
                a_l0 = T.alloc_l0a((32, 32), read_dtype)
                b_l0 = T.alloc_l0b((16, 32), dtype)
                acc = T.alloc_l0c((32, 16), acc_dtype)
                T.copy(eye, a_l1, transpose=False)
                T.copy(src, b_l1, transpose=False)
                T.copy(a_l1, a_l0, transpose=False)
                T.copy(b_l1, b_l0, transpose=False)
                T.gemm(a_l0, T.view(b_l0, dtype=read_dtype), acc, transpose_B=True, clear_accum=True)
                T.copy(acc, out)

        data = _matrix_data((16, 32), dtype)
        identity = torch.eye(32, dtype=getattr(torch, read_dtype))
        _check_gemm(main, identity, data.view(getattr(torch, dtype)), data.T)


@pytest.mark.parametrize("operand", ["l0a", "l0b"])
def test_l1_to_l0_matrix_fp4(operand):
    """用单位缩放的 MX GEMM 读回非转置加载到 L0A / L0B 的打包 FP4。"""
    dtype = "float4_e2m1fn"

    @T.prim_func
    def main(
        a: T.Tensor((64, 64), dtype), b: T.Tensor((64, 64), dtype), scale: T.Tensor((1, 64), "uint16"), out: T.Tensor((64, 64), "float32")
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((64, 64), dtype)
            b_l1 = T.alloc_l1((64, 64), dtype)
            scale_l1 = T.alloc_l1((64, 1), "uint16")
            a_l0 = T.alloc_l0a((64, 64), dtype)
            b_l0 = T.alloc_l0b((64, 64), dtype)
            a_sf = T.alloc_l0a_sf(a_l0)
            b_sf = T.alloc_l0b_sf(b_l0)
            acc = T.alloc_l0c((64, 64), "float32")
            T.copy(a, a_l1, transpose=False)
            T.copy(b, b_l1, transpose=False)
            T.copy(scale, scale_l1, transpose=True)
            T.copy(a_l1, a_l0, transpose=False)
            T.copy(b_l1, b_l0, transpose=False)
            T.copy(scale_l1, a_sf)
            T.copy(scale_l1, b_sf)
            T.gemm_blockscaled(a_l0, b_l0, acc, a_sf, b_sf, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    rng = torch.Generator().manual_seed(1215)
    codes = torch.randint(0, 16, (64, 64), generator=rng, dtype=torch.uint8)
    identity_codes = torch.eye(64, dtype=torch.uint8) * 2  # E2M1 的编码 2 对应 1。
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    data = values[codes.long()]
    a, b, expected = (codes, identity_codes, data) if operand == "l0a" else (identity_codes, codes, data.T)

    def pack(matrix):
        return (matrix[:, ::2] | (matrix[:, 1::2] << 4)).view(torch.float4_e2m1fn_x2).npu()

    scale = torch.full((1, 64), 0x7F7F, dtype=torch.uint16).npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    torch.testing.assert_close(kernel(pack(a), pack(b), scale).cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["int8", "bfloat16", "float32"])
@pytest.mark.parametrize("operand", ["l0a", "l0b"])
def test_l1_to_l0_src_slice(dtype, operand):
    """从 (32, 64) L1 按分形对齐起点切片 [16:32, 32:64] 加载到 L0A / L0B。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    if operand == "l0a":
        @T.prim_func
        def main(src: T.Tensor((32, 64), dtype), eye: T.Tensor((32, 32), read_dtype), out: T.Tensor((16, 32), acc_dtype)):
            with T.Kernel(1):
                a_l1 = T.alloc_l1((32, 64), dtype)
                b_l1 = T.alloc_l1((32, 32), read_dtype)
                a_l0 = T.alloc_l0a((16, 32), dtype)
                b_l0 = T.alloc_l0b((32, 32), read_dtype)
                acc = T.alloc_l0c((16, 32), acc_dtype)
                T.copy(src, a_l1, transpose=False)
                T.copy(eye, b_l1, transpose=False)
                T.copy(a_l1[16:32, 32:64], a_l0, transpose=False)
                T.copy(b_l1, b_l0, transpose=False)
                T.gemm(T.view(a_l0, dtype=read_dtype), b_l0, acc, transpose_B=True, clear_accum=True)
                T.copy(acc, out)

        data = _matrix_data((32, 64), dtype)
        identity = torch.eye(32, dtype=getattr(torch, read_dtype))
        _check_gemm(main, data.view(getattr(torch, dtype)), identity, data[16:32, 32:64])
    else:
        @T.prim_func
        def main(eye: T.Tensor((32, 32), read_dtype), src: T.Tensor((32, 64), dtype), out: T.Tensor((32, 16), acc_dtype)):
            with T.Kernel(1):
                a_l1 = T.alloc_l1((32, 32), read_dtype)
                b_l1 = T.alloc_l1((32, 64), dtype)
                a_l0 = T.alloc_l0a((32, 32), read_dtype)
                b_l0 = T.alloc_l0b((16, 32), dtype)
                acc = T.alloc_l0c((32, 16), acc_dtype)
                T.copy(eye, a_l1, transpose=False)
                T.copy(src, b_l1, transpose=False)
                T.copy(a_l1, a_l0, transpose=False)
                T.copy(b_l1[16:32, 32:64], b_l0, transpose=False)
                T.gemm(a_l0, T.view(b_l0, dtype=read_dtype), acc, transpose_B=True, clear_accum=True)
                T.copy(acc, out)

        data = _matrix_data((32, 64), dtype)
        identity = torch.eye(32, dtype=getattr(torch, read_dtype))
        _check_gemm(main, identity, data.view(getattr(torch, dtype)), data[16:32, 32:64].T)


@pytest.mark.parametrize("operand", ["l0a", "l0b"])
def test_l1_to_l0_dst_offset_with_pitch(operand):
    """分两次以 l0_pitch=2 向 L0A / L0B 的 [0:16, :] 和 [16:32, :] 拼装 (32, 32) 矩阵。"""
    dtype = "bfloat16"

    if operand == "l0a":
        @T.prim_func
        def main(src: T.Tensor((32, 32), dtype), eye: T.Tensor((32, 32), dtype), out: T.Tensor((32, 32), "float32")):
            with T.Kernel(1):
                a_l1 = T.alloc_l1((32, 32), dtype)
                b_l1 = T.alloc_l1((32, 32), dtype)
                a_l0 = T.alloc_l0a((32, 32), dtype)
                b_l0 = T.alloc_l0b((32, 32), dtype)
                acc = T.alloc_l0c((32, 32), "float32")
                T.copy(src, a_l1, transpose=False)
                T.copy(eye, b_l1, transpose=False)
                T.copy(a_l1[0:16, :], a_l0[0:16, :], l0_pitch=2)
                T.copy(a_l1[16:32, :], a_l0[16:32, :], l0_pitch=2)
                T.copy(b_l1, b_l0, transpose=False)
                T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
                T.copy(acc, out)

        data = _matrix_data((32, 32), dtype)
        identity = torch.eye(32, dtype=torch.bfloat16)
        _check_gemm(main, data, identity, data)
    else:
        @T.prim_func
        def main(eye: T.Tensor((32, 32), dtype), src: T.Tensor((32, 32), dtype), out: T.Tensor((32, 32), "float32")):
            with T.Kernel(1):
                a_l1 = T.alloc_l1((32, 32), dtype)
                b_l1 = T.alloc_l1((32, 32), dtype)
                a_l0 = T.alloc_l0a((32, 32), dtype)
                b_l0 = T.alloc_l0b((32, 32), dtype)
                acc = T.alloc_l0c((32, 32), "float32")
                T.copy(eye, a_l1, transpose=False)
                T.copy(src, b_l1, transpose=False)
                T.copy(a_l1, a_l0, transpose=False)
                T.copy(b_l1[0:16, :], b_l0[0:16, :], l0_pitch=2)
                T.copy(b_l1[16:32, :], b_l0[16:32, :], l0_pitch=2)
                T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
                T.copy(acc, out)

        data = _matrix_data((32, 32), dtype)
        identity = torch.eye(32, dtype=torch.bfloat16)
        _check_gemm(main, identity, data, data.T)


@pytest.mark.parametrize("dtype", ["int8", "bfloat16", "float32"])
def test_l1_to_l0_non_fractal(dtype):
    """非整分形逻辑 shape (13, 29) 从 L1 加载到 L0A，通过 L0 GEMM 读回校验。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)
    rows, cols, out_cols = 13, 29, 32

    @T.prim_func
    def main(
        src: T.Tensor((rows, cols), dtype),
        eye: T.Tensor((out_cols, cols), read_dtype),
        out: T.Tensor((rows, out_cols), acc_dtype),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((rows, cols), dtype)
            b_l1 = T.alloc_l1((out_cols, cols), read_dtype)
            a_l0 = T.alloc_l0a((rows, cols), dtype)
            b_l0 = T.alloc_l0b((out_cols, cols), read_dtype)
            acc = T.alloc_l0c((rows, out_cols), acc_dtype)
            T.copy(src, a_l1, transpose=False)
            T.copy(eye, b_l1, transpose=False)
            T.copy(a_l1, a_l0, transpose=False)
            T.copy(b_l1, b_l0, transpose=False)
            T.gemm(T.view(a_l0, dtype=read_dtype), b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    data = _matrix_data((rows, cols), dtype)
    identity = torch.eye(out_cols, dtype=getattr(torch, read_dtype))[:, :cols].contiguous()
    expected = torch.zeros((rows, out_cols), dtype=getattr(torch, acc_dtype))
    expected[:, :cols] = data.to(expected.dtype)
    _check_gemm(main, data.view(getattr(torch, dtype)), identity, expected)


def test_l1_to_l0_3d_stage_and_loop_slice():
    """3D L1 与 3D L0 通过循环变量选择流水 stage 并在 K 维按 C0 倍数步进切片加载。"""
    dtype = "float16"

    @T.prim_func
    def main(src: T.Tensor((16, 64), dtype), eye: T.Tensor((32, 32), dtype), out: T.Tensor((2, 16, 32), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((2, 16, 64), dtype)
            b_l1 = T.alloc_l1((32, 32), dtype)
            a_l0 = T.alloc_l0a((2, 16, 32), dtype)
            b_l0 = T.alloc_l0b((32, 32), dtype)
            acc = T.alloc_l0c((16, 32), "float32")
            T.copy(src, a_l1[1, :, :], transpose=False)
            T.copy(eye, b_l1, transpose=False)
            T.copy(b_l1, b_l0, transpose=False)
            for ko in T.serial(2):
                stage = ko % 2
                T.copy(a_l1[1, :, ko * 32 : (ko + 1) * 32], a_l0[stage, :, :], transpose=False)
                T.gemm(a_l0[stage, :, :], b_l0, acc, transpose_B=True, clear_accum=True)
                T.copy(acc, out[ko, :, :])

    data = _matrix_data((16, 64), dtype)
    identity = torch.eye(32, dtype=torch.float16)
    expected = torch.stack((data[:, :32], data[:, 32:64]))
    _check_gemm(main, data, identity, expected)


@pytest.mark.parametrize("row_start,col_start", [(1, 0), (0, 1)], ids=["unaligned-row", "unaligned-col"])
def test_l1_to_l0_rejects_unaligned_origin(row_start, col_start):
    """源 L1 切片起点未按 16×C0 分形边界对齐时，编译期直接报错拒绝。"""

    @T.prim_func
    def main(src: T.Tensor((32, 64), "float16")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((32, 64), "float16")
            a_l0 = T.alloc_l0a((16, 32), "float16")
            T.copy(src, a_l1, transpose=False)
            T.copy(a_l1[row_start : row_start + 16, col_start : col_start + 32], a_l0, transpose=False)

    with pytest.raises(Exception):
        tilelang.lower(main, target="ascend")

