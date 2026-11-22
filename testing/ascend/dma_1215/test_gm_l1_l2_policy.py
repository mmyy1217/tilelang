"""1215 GM→L1 搬入时的 L2 Cache 读缓存策略。"""

import re

import pytest
import tilelang
import tilelang.ascend.language as T


DTYPES = (
    "int8",
    "uint8",
    "float8_e4m3fn",
    "int16",
    "uint16",
    "float16",
    "bfloat16",
    "int32",
    "uint32",
    "float32",
)
FP4 = "float4_e2m1fn"
L2_LOAD_POLICIES = [("normal_fv", 0), ("normal_lv", 1), ("notalloc_clean", 5)]


@pytest.mark.parametrize("dtype,cols", [*((dtype, 32) for dtype in DTYPES), (FP4, 64)])
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


@pytest.mark.parametrize("dtype", DTYPES)
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
