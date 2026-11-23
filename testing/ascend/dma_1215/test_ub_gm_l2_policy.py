"""1215 UB→GM 写回时的 L2 Cache 写缓存策略。"""

import re

import pytest
import tilelang
import tilelang.ascend.language as T


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


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int8"])
@pytest.mark.parametrize("policy,mode", L2_STORE_POLICIES)
def test_ub_to_gm_l2_policy(dtype, policy, mode):
    """检查 UB→GM 对齐写回指令收到指定的 L2 写策略。"""

    @T.prim_func
    def main(src: T.Tensor((4, 32), dtype), out: T.Tensor((4, 32), dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), dtype)
            T.copy(src, ub)
            T.copy(ub, out, l2_cache_ctrl=policy)

    source = tilelang.lower(main, target="ascend").kernel_source
    assert re.search(rf"\basc_copy_ub2gm_align\([^;]*, static_cast<asc_store_l2_cache_mode>\({mode}\),", source), source
