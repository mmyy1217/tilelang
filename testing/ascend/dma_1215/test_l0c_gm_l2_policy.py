"""1215 L0C→GM 搬出时的 L2 Cache 写缓存策略。"""

import re

import pytest
import tilelang
import tilelang.ascend.language as T


@T.macro
def _gemm_32x64(a_gm, b_gm, acc):
    a = T.alloc_l1(a_gm.shape, a_gm.dtype)
    b = T.alloc_l1(b_gm.shape, b_gm.dtype)
    T.copy(a_gm, a, transpose=False)
    T.copy(b_gm, b, transpose=False)
    T.gemm(a, b, acc, transpose_B=True, clear_accum=True)


L2_STORE_POLICIES = [
    ("normal_fv", 0),
    ("normal_lv", 1),
    ("normal_pers", 2),
    ("normal_red", 3),
    ("notalloc_ci", 4),
    ("notalloc_pw", 5),
    ("notalloc_pi", 6),
    ("notalloc_red", 7),
    ("wbh_fv", 8),
    ("wbh_lv", 9),
    ("wbh_pers", 10),
    ("wbh_red", 11),
    ("wts_fv", 12),
    ("wts_lv", 13),
    ("wts_pers", 14),
    ("wts_red", 15),
]


@pytest.mark.parametrize(
    "input_dtype,acc_dtype,output_dtype",
    [
        ("float16", "float32", "float32"),
        ("float16", "float32", "float16"),
        ("float16", "float32", "bfloat16"),
        ("int8", "int32", "int32"),
    ],
)
@pytest.mark.parametrize("policy,mode", L2_STORE_POLICIES)
def test_l0c_to_gm_l2_policy(input_dtype, acc_dtype, output_dtype, policy, mode):
    """检查各输出类型的 Fixpipe 写回指令收到指定的 L2 写策略。"""

    # K=32 覆盖 INT8 的一个完整 Cube K 分形。
    @T.prim_func
    def main(a: T.Tensor((32, 32), input_dtype), b: T.Tensor((64, 32), input_dtype), out: T.Tensor((32, 64), output_dtype)):
        with T.Kernel(1):
            acc = T.alloc_l0c((32, 64), acc_dtype)
            _gemm_32x64(a, b, acc)
            T.copy(acc, out, l2_cache_ctrl=policy)

    source = tilelang.lower(main, target="ascend").kernel_source
    assert re.search(rf"\basc_copy_l0c2gm\([^;]*, static_cast<asc_store_l2_cache_mode>\({mode}\),", source), source
