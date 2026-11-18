"""GM→L1 的七项 1215 验收测试。"""

import pytest
import torch
import tilelang
import tilelang.testing
from tilelang import tvm
from tvm import tirx
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_nz_layout
from testing.ascend._ir import calls
from testing.ascend.transform.test_ascend_lower_copy import _lower


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

DEVICE = tilelang.testing.requires_ascend
BATCH_LAYOUTS = ["compact", "gapped", "interleaved"]
OOB_SHAPES = [
    pytest.param(15, 128, id="row-tail"),
    pytest.param(16, 100, id="column-tail"),
    pytest.param(15, 100, id="both-tails"),
    pytest.param(16, 96, id="fully-oob"),
]
FORMAT_TAILS = [
    pytest.param(0, 0, id="full"),
    pytest.param(1, 0, id="row-tail"),
    pytest.param(0, 1, id="column-tail"),
    pytest.param(1, 1, id="both-tails"),
]
L2_POLICIES = [("normal_fv", 0), ("normal_lv", 1), ("notalloc_clean", 5)]


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


def _padded(data, cols):
    # CPU 参考独立补齐到整块。空切片对应完全越界。
    expected = torch.zeros((16, cols), dtype=data.dtype)
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
    """用单条 ND 搬运三个矩阵，逐批核对数据和 stride。"""
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
    """用单条 DN 搬运三个矩阵，与逐批转置的参考比较。"""
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


@pytest.mark.parametrize("batch_layout", BATCH_LAYOUTS)
def test_batch_fp4_nd_descriptor(batch_layout):
    """保持 FP4 的 4 位逻辑类型，检查单条批量 DMA 按字节折算。"""
    strides = {
        "compact": (16 * 64, 64, 1),
        "gapped": (18 * 64, 64, 1),
        "interleaved": (64, 3 * 64, 1),
    }[batch_layout]

    @T.prim_func
    def main(src: T.StridedTensor((3, 16, 64), strides, FP4)):
        with T.Kernel(1):
            dst = T.alloc_l1((3, 16, 64), FP4)
            with T.Cube():
                T.copy(src, dst, transpose=False)

    source = tilelang.lower(main, target="ascend").kernel_source
    assert source.count("asc_copy_gm2l1_nd2nz(") == 1
    assert "asc_set_gm2l1_nz_para(3, 1, static_cast<uint16_t>(16), 16)" in source
    assert f", {strides[1] // 2}, static_cast<asc_load_l2_cache_mode>(0), 16, 32, {strides[0] // 2}, 0);" in source


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("rows,cols", OOB_SHAPES)
def test_oob_tail_padding_nd(dtype, rows, cols):
    """ND 从第 96 列请求 16×32 tile，检查越界区清零。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(
        src: T.Tensor((rows, cols), dtype),
        poison: T.Tensor((16, 32), dtype),
        eye: T.Tensor((32, 32), read_dtype),
        out: T.Tensor((16, 32), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 32), dtype)
            T.copy(poison, l1, transpose=False)
            T.copy(src[:16, 96:128], l1, transpose=False)
            _read_l1(l1, eye, out, 16, 32, read_dtype, acc_dtype)

    data = _matrix_data((rows, cols), dtype)
    source = data.view(getattr(torch, dtype)).npu()
    poison = torch.full((16, 32), 7, dtype=data.dtype).view(getattr(torch, dtype)).npu()
    _assert_readback(main, dtype, _padded(data[:, 96:128], 32), source, poison)


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("rows,cols", OOB_SHAPES)
def test_oob_tail_padding_dn(dtype, rows, cols):
    """DN 从转置后的第 96 列请求 16×32 tile，检查越界区清零。"""
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(
        src: T.Tensor((cols, rows), dtype),
        poison: T.Tensor((16, 32), dtype),
        eye: T.Tensor((32, 32), read_dtype),
        out: T.Tensor((16, 32), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 32), dtype)
            T.copy(poison, l1, transpose=False)
            T.copy(src[96:128, :16], l1, transpose=True)
            _read_l1(l1, eye, out, 16, 32, read_dtype, acc_dtype)

    data = _matrix_data((rows, cols), dtype)
    source = data.T.contiguous().view(getattr(torch, dtype)).npu()
    poison = torch.full((16, 32), 7, dtype=data.dtype).view(getattr(torch, dtype)).npu()
    _assert_readback(main, dtype, _padded(data[:, 96:128], 32), source, poison)


@pytest.mark.parametrize("rows,cols", OOB_SHAPES)
def test_oob_tail_padding_fp4_nd_descriptor(rows, cols):
    """FP4 以字节计列数，检查越界裁剪后下发的长度。"""

    @T.prim_func
    def main(src: T.Tensor((rows, 2 * cols), FP4)):
        with T.Kernel(1):
            dst = T.alloc_l1((16, 64), FP4)
            T.copy(src[:16, 192:256], dst, transpose=False)

    source = tilelang.lower(main, target="ascend").kernel_source
    valid_cols = min(32, cols - 96)
    if rows < 16 or not valid_cols:
        assert "asc_fill_l1(" in source
    if valid_cols:
        assert source.count("asc_copy_gm2l1_nd2nz(") == 1
        assert f", {cols}, static_cast<asc_load_l2_cache_mode>(0), {rows}, {valid_cols}, 0, 0);" in source
    else:
        assert "asc_copy_gm2l1_nd2nz(" not in source


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("row_tail,col_tail", FORMAT_TAILS)
def test_nz_format_padding_nd(dtype, row_tail, col_tail):
    """ND 源区域完全有效，检查目的分形的行尾和列尾。"""
    c0 = 256 // tvm.DataType(dtype).bits  # 每行分形的内维宽度为 32B。
    rows, cols = 16 - row_tail, c0 - col_tail
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(
        src: T.Tensor((rows, cols), dtype),
        poison: T.Tensor((16, c0), dtype),
        eye: T.Tensor((c0, c0), read_dtype),
        out: T.Tensor((16, c0), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, c0), dtype)
            T.copy(poison, l1, transpose=False)
            T.copy(src, l1[:rows, :cols], transpose=False)
            _read_l1(l1, eye, out, 16, c0, read_dtype, acc_dtype)

    data = _matrix_data((rows, cols), dtype)
    source = data.view(getattr(torch, dtype)).npu()
    poison = torch.full((16, c0), 7, dtype=data.dtype).view(getattr(torch, dtype)).npu()
    _assert_readback(main, dtype, _padded(data, c0), source, poison)


@DEVICE
@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("row_tail,col_tail", FORMAT_TAILS)
def test_nz_format_padding_dn(dtype, row_tail, col_tail):
    """DN 源区域完全有效，检查转置后的分形行尾和列尾。"""
    c0 = 256 // tvm.DataType(dtype).bits  # 每行分形的内维宽度为 32B。
    rows, cols = 16 - row_tail, c0 - col_tail
    read_dtype, acc_dtype = _readback_dtypes(dtype)

    @T.prim_func
    def main(
        src: T.Tensor((cols, rows), dtype),
        poison: T.Tensor((16, c0), dtype),
        eye: T.Tensor((c0, c0), read_dtype),
        out: T.Tensor((16, c0), acc_dtype),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, c0), dtype)
            T.copy(poison, l1, transpose=False)
            T.copy(src, l1[:rows, :cols], transpose=True)
            _read_l1(l1, eye, out, 16, c0, read_dtype, acc_dtype)

    data = _matrix_data((rows, cols), dtype)
    source = data.T.contiguous().view(getattr(torch, dtype)).npu()
    poison = torch.full((16, c0), 7, dtype=data.dtype).view(getattr(torch, dtype)).npu()
    _assert_readback(main, dtype, _padded(data, c0), source, poison)


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


@pytest.mark.parametrize("dtype,cols", [*(pytest.param(dtype, 32, id=dtype) for dtype in READBACK), pytest.param(FP4, 64, id="fp4")])
@pytest.mark.parametrize("policy,value", L2_POLICIES)
def test_gm_to_l1_l2_policy_nd(dtype, cols, policy, value):
    """检查 ND 搬运请求的读缓存策略枚举。"""
    src = tirx.decl_buffer((16, cols), dtype)
    dst = tirx.decl_buffer((16, cols), dtype, scope="shared.l1")
    after = _lower(src, dst, layouts={dst: make_ascend_nz_layout(dst)}, transpose=False, l2_cache_ctrl=policy)
    (copy,) = calls(after, "tl.ascend_copy_gm_to_cbuf")
    assert int(copy.args[4]) == value
    assert int(copy.args[9]) == 0


@pytest.mark.parametrize("dtype", READBACK)
@pytest.mark.parametrize("policy,value", L2_POLICIES)
def test_gm_to_l1_l2_policy_dn(dtype, policy, value):
    """检查 DN 搬运请求的读缓存策略枚举。"""
    src = tirx.decl_buffer((32, 16), dtype)
    dst = tirx.decl_buffer((16, 32), dtype, scope="shared.l1")
    after = _lower(src, dst, layouts={dst: make_ascend_nz_layout(dst)}, transpose=True, l2_cache_ctrl=policy)
    (copy,) = calls(after, "tl.ascend_copy_gm_to_cbuf")
    assert int(copy.args[4]) == value
    assert int(copy.args[9]) == 1
