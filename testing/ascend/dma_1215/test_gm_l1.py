"""1215 GM→L1 的数据搬运和 L2 读策略。"""

import re

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


# DMA 使用原 dtype，L1 读回使用同位宽且 Cube 支持的 dtype。
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
FP4 = "float4_e2m1fn"
L2_LOAD_POLICIES = [("normal_fv", 0), ("normal_lv", 1), ("notalloc_clean", 5)]

DEVICE = tilelang.testing.requires_ascend
BATCH_LAYOUTS = ["compact", "gapped", "interleaved"]
# (GM 行数, GM 列数, L1 行数, L1 列数)，请求均从 GM 第 96 列开始。
OOB_SHAPES = [
    pytest.param(15, 128, 16, 32, id="row-tail"),
    pytest.param(16, 100, 16, 32, id="column-tail"),
    pytest.param(15, 100, 16, 32, id="both-tails"),
    # 31 列有效，余下 33 列覆盖最后一组 C0 内的缺列及后续完整 C0 组。
    pytest.param(15, 127, 16, 64, id="both-tails-partial-and-full-c0"),
    # 32 行跨两个 NZ 行分形，4 列有效，验证两组行都补满 64 列。
    pytest.param(32, 100, 32, 64, id="two-row-fractals"),
    pytest.param(16, 96, 16, 32, id="fully-oob"),
]


def _matrix_data(shape, dtype):
    rng = torch.Generator().manual_seed(1215)
    return torch.randint(1, 63, shape, generator=rng).to(getattr(torch, READBACK[dtype]))


def _readback_dtypes(dtype):
    read_dtype = READBACK[dtype]
    return read_dtype, "int32" if read_dtype == "int8" else "float32"


@T.macro
def _read_l1(matrix, identity, out, rows, cols, read_dtype, acc_dtype):
    # 只读回 L1。用同位宽视图和单位矩阵保留 DMA 搬入的数据。
    weight = T.alloc_l1((cols, cols), read_dtype)
    acc = T.alloc_l0c((rows, cols), acc_dtype)
    T.copy(identity, weight, transpose=False)
    payload = T.view(matrix, dtype=read_dtype)
    T.gemm(payload, weight, acc, transpose_B=True, clear_accum=True)
    T.copy(acc, out)


@T.macro
def _read_l1_batch(matrices, identity, out, read_dtype, acc_dtype):
    weight = T.alloc_l1((32, 32), read_dtype)
    acc = T.alloc_l0c((16, 32), acc_dtype)
    T.copy(identity, weight, transpose=False)
    payload = T.view(matrices, dtype=read_dtype)
    for batch in T.serial(3):
        T.gemm(payload[batch, :, :], weight, acc, transpose_B=True, clear_accum=True)
        T.copy(acc, out[batch, :, :])


def _assert_readback(main, dtype, expected, *inputs):
    identity = torch.eye(expected.shape[-1]).to(getattr(torch, READBACK[dtype])).npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    actual = kernel(*inputs, identity).cpu()
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)


def _batch_source(rows, cols, dtype, layout):
    torch_dtype = getattr(torch, dtype)
    if layout == "interleaved":
        data = _matrix_data((rows, 3, cols), dtype)
        return data.transpose(0, 1), data.view(torch_dtype).npu().transpose(0, 1)

    pad_rows = 2 if layout == "gapped" else 0
    data = _matrix_data((3, rows + pad_rows, cols), dtype)
    return data[:, :rows, :], data.view(torch_dtype).npu()[:, :rows, :]


def _padded(data, rows, cols):
    # CPU 参考独立补齐到整块。空切片对应完全越界。
    expected = torch.zeros((rows, cols), dtype=data.dtype)
    expected[: data.shape[0], : data.shape[1]] = data
    return expected


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
def test_nd_to_nz(dtype):
    """完整 ND 矩阵 16×32 逐元素搬入 L1。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(src: T.Tensor((16, 32), dtype), eye: T.Tensor((32, 32), read_dtype), out: T.Tensor((16, 32), acc_dtype)):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 32), dtype)
            T.copy(src, l1, transpose=False)
            _read_l1(l1, eye, out, 16, 32, read_dtype, acc_dtype)

    source = _matrix_data((16, 32), dtype)
    _assert_readback(main, dtype, source, source.view(getattr(torch, dtype)).npu())


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
def test_dn_to_nz(dtype):
    """从 GM 的 32×16 矩阵转置搬入 L1 的 16×32。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(src: T.Tensor((32, 16), dtype), eye: T.Tensor((32, 32), read_dtype), out: T.Tensor((16, 32), acc_dtype)):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 32), dtype)
            T.copy(src, l1, transpose=True)
            _read_l1(l1, eye, out, 16, 32, read_dtype, acc_dtype)

    source = _matrix_data((32, 16), dtype)
    _assert_readback(main, dtype, source.T, source.view(getattr(torch, dtype)).npu())


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("batch_layout", BATCH_LAYOUTS)
def test_batch_nd_to_nz(dtype, batch_layout):
    """一次调用搬入三个 ND 矩阵，逐批核对读回数据。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)
    data, source = _batch_source(16, 32, dtype, batch_layout)
    strides = source.stride()
    assert strides == data.stride()

    @T.prim_func
    def main(
        src: T.StridedTensor((3, 16, 32), strides, dtype),
        eye: T.Tensor((32, 32), read_dtype),
        out: T.Tensor((3, 16, 32), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((3, 16, 32), dtype)
            T.copy(src, l1, transpose=False)
            _read_l1_batch(l1, eye, out, read_dtype, acc_dtype)

    _assert_readback(main, dtype, data, source)


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("batch_layout", BATCH_LAYOUTS)
def test_batch_dn_to_nz(dtype, batch_layout):
    """一次调用搬入三个 DN 矩阵，与逐批转置的参考比较。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)
    data, source = _batch_source(32, 16, dtype, batch_layout)
    strides = source.stride()
    assert strides == data.stride()

    @T.prim_func
    def main(
        src: T.StridedTensor((3, 32, 16), strides, dtype),
        eye: T.Tensor((32, 32), read_dtype),
        out: T.Tensor((3, 16, 32), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((3, 16, 32), dtype)
            T.copy(src, l1, transpose=True)
            _read_l1_batch(l1, eye, out, read_dtype, acc_dtype)

    _assert_readback(main, dtype, data.transpose(1, 2), source)


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("rows,cols,tile_rows,tile_cols", OOB_SHAPES)
def test_oob_tail_padding_nd(dtype, rows, cols, tile_rows, tile_cols):
    """ND 从第 96 列请求完整 NZ tile，检查目标行尾和列尾。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(
        src: T.Tensor((rows, cols), dtype),
        poison: T.Tensor((tile_rows, tile_cols), dtype),
        eye: T.Tensor((tile_cols, tile_cols), read_dtype),
        out: T.Tensor((tile_rows, tile_cols), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((tile_rows, tile_cols), dtype)
            T.copy(poison, l1, transpose=False)
            T.copy(src[:tile_rows, 96 : 96 + tile_cols], l1, transpose=False)
            _read_l1(l1, eye, out, tile_rows, tile_cols, read_dtype, acc_dtype)

    data = _matrix_data((rows, cols), dtype)
    source = data.view(getattr(torch, dtype)).npu()
    poison = torch.full((tile_rows, tile_cols), 7, dtype=data.dtype).view(getattr(torch, dtype)).npu()
    _assert_readback(main, dtype, _padded(data[:, 96 : 96 + tile_cols], tile_rows, tile_cols), source, poison)


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("rows,cols,tile_rows,tile_cols", OOB_SHAPES)
def test_oob_tail_padding_dn(dtype, rows, cols, tile_rows, tile_cols):
    """DN 从转置后的第 96 列请求完整 NZ tile，检查目标行尾和列尾。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(
        src: T.Tensor((cols, rows), dtype),
        poison: T.Tensor((tile_rows, tile_cols), dtype),
        eye: T.Tensor((tile_cols, tile_cols), read_dtype),
        out: T.Tensor((tile_rows, tile_cols), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((tile_rows, tile_cols), dtype)
            T.copy(poison, l1, transpose=False)
            T.copy(src[96 : 96 + tile_cols, :tile_rows], l1, transpose=True)
            _read_l1(l1, eye, out, tile_rows, tile_cols, read_dtype, acc_dtype)

    data = _matrix_data((rows, cols), dtype)
    source = data.T.contiguous().view(getattr(torch, dtype)).npu()
    poison = torch.full((tile_rows, tile_cols), 7, dtype=data.dtype).view(getattr(torch, dtype)).npu()
    _assert_readback(main, dtype, _padded(data[:, 96 : 96 + tile_cols], tile_rows, tile_cols), source, poison)


@DEVICE
@pytest.mark.parametrize("storage_dtype", ["float4_e2m1fn_x2", "int8"], ids=["packed-dtype", "byte-view"])
def test_fp4_packed_nd_to_nz(storage_dtype):
    """按 ND 搬入 FP4。CPU 独立解码两个半字节，再与 blockscaled GEMM 比较。"""
    gm_shape, gm_dtype = {
        "float4_e2m1fn_x2": ((16, 256), FP4),
        "int8": ((16, 128), "int8"),
    }[storage_dtype]

    @T.prim_func
    def main(
        src: T.Tensor(gm_shape, gm_dtype),
        weight: T.Tensor(gm_shape, gm_dtype),
        scale: T.Tensor((4, 16), "uint16"),
        out: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((16, 256), FP4)
            b = T.alloc_l1((16, 256), FP4)
            a_scale = T.alloc_l1((16, 4), "uint16")
            b_scale = T.alloc_l1((16, 4), "uint16")
            acc = T.alloc_l0c((16, 16), "float32")
            T.copy(T.view(src, (16, 256), dtype=FP4), a, transpose=False)
            T.copy(T.view(weight, (16, 256), dtype=FP4), b, transpose=False)
            T.copy(scale, a_scale, transpose=True)
            T.copy(scale, b_scale, transpose=True)
            T.gemm_blockscaled(a, b, acc, a_scale, b_scale, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    rng = torch.Generator().manual_seed(1215)
    a_codes = torch.randint(0, 16, (16, 256), dtype=torch.uint8, generator=rng)
    b_codes = torch.randint(0, 16, (16, 256), dtype=torch.uint8, generator=rng)
    e2m1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    expected = e2m1[a_codes.long()] @ e2m1[b_codes.long()].T

    def pack(codes):
        low = codes[:, ::2]
        high = codes[:, 1::2] << 4
        return (low | high).view(getattr(torch, storage_dtype)).npu()

    # E8M0 指数 127 对应缩放系数 1，每两个字节装入一个 uint16。
    scale = torch.full((4, 16), 0x7F7F, dtype=torch.uint16).npu()
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    actual = kernel(pack(a_codes), pack(b_codes), scale).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,cols", [*((dtype, 32) for dtype in READBACK), (FP4, 64)])
@pytest.mark.parametrize("policy,mode", L2_LOAD_POLICIES)
def test_gm_to_l1_l2_policy_nd(dtype, cols, policy, mode):
    """检查 ND→NZ 搬入指令收到指定的 L2 读策略，含打包 FP4。"""

    @T.prim_func
    def main(src: T.Tensor((16, cols), dtype)):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, cols), dtype)
            T.copy(src, l1, transpose=False, l2_cache_ctrl=policy)

    source = tilelang.lower(main, target="ascend").kernel_source
    assert re.search(rf"\basc_copy_gm2l1_nd2nz\([^;]*, static_cast<asc_load_l2_cache_mode>\({mode}\),", source), source


@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("policy,mode", L2_LOAD_POLICIES)
def test_gm_to_l1_l2_policy_dn(dtype, policy, mode):
    """检查 DN→NZ 搬入指令收到指定的 L2 读策略。"""

    @T.prim_func
    def main(src: T.Tensor((32, 16), dtype)):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 32), dtype)
            T.copy(src, l1, transpose=True, l2_cache_ctrl=policy)

    source = tilelang.lower(main, target="ascend").kernel_source
    assert re.search(rf"\basc_copy_gm2l1_dn2nz\([^;]*, static_cast<asc_load_l2_cache_mode>\({mode}\),", source), source
