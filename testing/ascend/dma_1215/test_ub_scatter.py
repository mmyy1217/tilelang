import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_compact_nz_layout


pytestmark = tilelang.testing.requires_ascend.marks()


ROWS = 32
COLS = 128


def _check_scatter(main, src_dtype, dst_dtype):
    data = torch.arange(ROWS * COLS, dtype=torch.float32).reshape(ROWS, COLS).to(src_dtype)
    actual = tilelang.compile(main, target="ascend", out_idx=-1)(data.npu()).cpu()
    c0 = 32 // actual.element_size()
    # NZ 的物理顺序是列块、行、块内列，末行用于存储对齐。
    expected = data.to(dst_dtype).reshape(ROWS, COLS // c0, c0).transpose(0, 1)
    torch.testing.assert_close(actual.reshape(COLS // c0, ROWS + 1, c0)[:, :ROWS, :], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
def test_ub_to_ub_nd_to_nz(src_dtype, dst_dtype):
    """只拷贝有效行，按 NZ 的物理地址顺序检查转换结果。"""

    @T.prim_func
    def main(src: T.Tensor((ROWS, COLS), src_dtype), out: T.Tensor((ROWS + 1, COLS), dst_dtype)):
        with T.Kernel(1):
            nd = T.alloc_shared((ROWS, COLS), src_dtype)
            nz = T.alloc_shared((ROWS + 1, COLS), dst_dtype)
            T.annotate_layout({nz: make_ascend_compact_nz_layout(nz)})
            T.copy(src, nd)
            T.copy(nd, nz[:ROWS, :])
            T.copy(nz, out)

    _check_scatter(main, src_dtype, dst_dtype)


def test_ub_to_ub_nd_to_nz_inside_simd_vf():
    """在显式 SimdVF 中将 BF16 ND 转成 FP32 NZ。"""

    @T.prim_func
    def main(src: T.Tensor((ROWS, COLS), "bfloat16"), out: T.Tensor((ROWS + 1, COLS), "float32")):
        with T.Kernel(1):
            nd = T.alloc_shared((ROWS, COLS), "bfloat16")
            nz = T.alloc_shared((ROWS + 1, COLS), "float32")
            T.annotate_layout({nz: make_ascend_compact_nz_layout(nz)})
            T.copy(src, nd)
            with T.SimdVF():
                T.copy(nd, nz[:ROWS, :])
            T.copy(nz, out)

    _check_scatter(main, torch.bfloat16, torch.float32)
