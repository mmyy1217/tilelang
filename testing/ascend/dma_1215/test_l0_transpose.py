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


def _check_gemm(main, a, b, expected):
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(a.npu(), b.npu())
    torch.testing.assert_close(actual.cpu(), expected.to(actual.dtype), rtol=0, atol=0)


def _readback_inputs(dtype, k):
    torch_dtype = getattr(torch, READBACK[dtype])
    rng = torch.Generator().manual_seed(1215)
    data = torch.randint(1, 63, (k, 32), generator=rng).to(torch_dtype)
    return data, data.view(getattr(torch, dtype)), torch.eye(k, dtype=torch_dtype)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_l1_to_l0_matrix(dtype):
    """分别把两个 16×32 完整块加载到 L0A 和 L0B。"""

    @T.prim_func
    def main(a: T.Tensor((16, 32), dtype), b: T.Tensor((16, 32), dtype), out: T.Tensor((16, 16), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((16, 32), dtype)
            b_l1 = T.alloc_l1((16, 32), dtype)
            a_l0 = T.alloc_l0a((16, 32), dtype)
            b_l0 = T.alloc_l0b((16, 32), dtype)
            acc = T.alloc_l0c((16, 16), "float32")
            T.copy(a, a_l1, transpose=False)
            T.copy(b, b_l1, transpose=False)
            T.copy(a_l1, a_l0, transpose=False)
            T.copy(b_l1, b_l0, transpose=False)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    rng = torch.Generator().manual_seed(1215)
    a = torch.randint(-4, 5, (16, 32), generator=rng).to(getattr(torch, dtype))
    b = torch.randint(-4, 5, (16, 32), generator=rng).to(getattr(torch, dtype))
    _check_gemm(main, a, b, a.float() @ b.float().T)


@pytest.mark.parametrize("dtype,rows,k", [*((dtype, 32, 64) for dtype in READBACK), ("bfloat16", 24, 32), ("float32", 25, 32)])
def test_l1_to_l0a_transpose(dtype, rows, k):
    """转置加载到 L0A，用 L0B 的单位矩阵逐元素读回，覆盖两种尾块。"""
    read_dtype = READBACK[dtype]
    acc_dtype = "int32" if read_dtype == "int8" else "float32"

    @T.prim_func
    def main(src: T.Tensor((k, 32), dtype), eye: T.Tensor((k, k), read_dtype), out: T.Tensor((rows, k), acc_dtype)):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((k, 32), dtype)
            b_l1 = T.alloc_l1((k, k), read_dtype)
            a_l0 = T.alloc_l0a((rows, k), dtype)
            b_l0 = T.alloc_l0b((k, k), read_dtype)
            acc = T.alloc_l0c((rows, k), acc_dtype)
            T.copy(src, a_l1, transpose=False)
            T.copy(eye, b_l1, transpose=False)
            T.copy(a_l1[:, :rows], a_l0, transpose=True)
            T.copy(b_l1, b_l0, transpose=False)
            T.gemm(T.view(a_l0, dtype=read_dtype), b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    data, source, identity = _readback_inputs(dtype, k=k)
    _check_gemm(main, source, identity, data[:, :rows].T)


@pytest.mark.parametrize("dtype", READBACK)
def test_l1_to_l0b_transpose(dtype):
    """转置加载到 L0B，用 L0A 的单位矩阵逐元素读回。"""
    read_dtype = READBACK[dtype]
    acc_dtype = "int32" if read_dtype == "int8" else "float32"

    @T.prim_func
    def main(eye: T.Tensor((64, 64), read_dtype), src: T.Tensor((64, 32), dtype), out: T.Tensor((64, 32), acc_dtype)):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((64, 64), read_dtype)
            b_l1 = T.alloc_l1((64, 32), dtype)
            a_l0 = T.alloc_l0a((64, 64), read_dtype)
            b_l0 = T.alloc_l0b((32, 64), dtype)
            acc = T.alloc_l0c((64, 32), acc_dtype)
            T.copy(eye, a_l1, transpose=False)
            T.copy(src, b_l1, transpose=False)
            T.copy(a_l1, a_l0, transpose=False)
            T.copy(b_l1, b_l0, transpose=True)
            T.gemm(a_l0, T.view(b_l0, dtype=read_dtype), acc, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    data, source, identity = _readback_inputs(dtype, k=64)
    _check_gemm(main, identity, source, data)


@pytest.mark.parametrize("operand", ["l0a", "l0b"])
def test_l1_to_l0_transpose_fp4(operand):
    """用单位缩放的 MX GEMM 读回打包 FP4，两侧分别输入非对称矩阵。"""
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
            T.copy(a_l1, a_l0, transpose=True)
            T.copy(b_l1, b_l0, transpose=True)
            T.copy(scale_l1, a_sf)
            T.copy(scale_l1, b_sf)
            T.gemm_blockscaled(a_l0, b_l0, acc, a_sf, b_sf, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    rng = torch.Generator().manual_seed(1215)
    codes = torch.randint(0, 16, (64, 64), generator=rng, dtype=torch.uint8)
    identity_codes = torch.eye(64, dtype=torch.uint8) * 2  # E2M1 的编码 2 对应 1。
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    data = values[codes.long()]
    a, b, expected = (codes, identity_codes, data.T) if operand == "l0a" else (identity_codes, codes, data)

    def pack(matrix):
        return (matrix[:, ::2] | (matrix[:, 1::2] << 4)).view(torch.float4_e2m1fn_x2).npu()

    scale = torch.full((1, 64), 0x7F7F, dtype=torch.uint16).npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    torch.testing.assert_close(kernel(pack(a), pack(b), scale).cpu(), expected, rtol=0, atol=0)
