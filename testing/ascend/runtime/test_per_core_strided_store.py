import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T

pytest.importorskip("torch_npu")

NUM_TOKENS = T.dynamic("num_tokens")


@T.prim_func
def interleaved_store(
    source: T.Tensor((2, 256), T.float32),
    output: T.Tensor((NUM_TOKENS,), T.float32),
):
    with T.MixedKernel(1) as (bx, sid):
        ub = T.alloc_shared((256,), T.float32)
        T.copy(source[sid, :], ub)
        stages = T.ceildiv(NUM_TOKENS - sid * 64, 128)
        with T.PerCoreTask():
            for stage in T.Serial(stages):
                begin = stage * 128 + sid * 64
                count = T.min(NUM_TOKENS - begin, 64)
                T.copy(ub[stage * 64 : stage * 64 + count], output[begin : begin + count])


@pytest.fixture(scope="module")
def compiled_store():
    return tilelang.compile(interleaved_store, target="ascend")


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("num_tokens", [0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512])
def test_per_core_interleaved_store(compiled_store, num_tokens):
    guard = 32
    indices = torch.arange(num_tokens, dtype=torch.int64)
    subcore = indices.remainder(128) // 64
    local_index = indices // 128 * 64 + indices.remainder(64)
    for repeat in range(3):
        reference_source = torch.arange(512, dtype=torch.float32).reshape(2, 256) + repeat * 1024
        source = reference_source.npu()
        expected = torch.full((num_tokens + 2 * guard,), -12345.0, dtype=torch.float32)
        storage = expected.npu()
        output = storage[guard : guard + num_tokens]
        compiled_store(source, output)
        torch.npu.synchronize()
        expected[guard : guard + num_tokens] = reference_source[subcore, local_index]
        torch.testing.assert_close(storage.cpu(), expected, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
