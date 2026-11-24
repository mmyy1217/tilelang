"""1215 GM→UB 搬入时的 L2 Cache 读缓存策略。"""

import re

import pytest
import tilelang
import tilelang.ascend.language as T


L2_LOAD_POLICIES = [
    ("normal_fv", 0),
    ("notalloc_keep", 4),
]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int8"])
@pytest.mark.parametrize("policy,mode", L2_LOAD_POLICIES)
def test_gm_to_ub_l2_policy(dtype, policy, mode):
    """检查 GM→UB 对齐搬入指令收到指定的 L2 读策略。"""

    @T.prim_func
    def main(src: T.Tensor((4, 32), dtype), out: T.Tensor((4, 32), dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), dtype)
            T.copy(src, ub, l2_cache_ctrl=policy)
            T.copy(ub, out)

    source = tilelang.lower(main, target="ascend").kernel_source
    assert re.search(rf"\basc_copy_gm2ub_align\([^;]*, static_cast<asc_load_l2_cache_mode>\({mode}\),", source), source
