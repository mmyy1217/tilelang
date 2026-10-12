"""Pack a token window from BSHD storage into head-major UB tiles."""

import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import tilelang
from tilelang.ascend import language as T


def window_pack(batch=1, sequence=128, heads=32, channels=64, tile_heads=8, tile_tokens=8, num_blocks=1, dtype="float8_e4m3fn"):
    if dtype not in ("uint8", "float8_e4m3fn", "float8_e5m2"):
        raise ValueError("window_pack supports byte storage; use uint8 or one of the two FP8 formats")
    if min(batch, sequence, heads, channels, tile_heads, tile_tokens, num_blocks) <= 0:
        raise ValueError("window_pack requires positive dimensions and block count")
    if sequence % tile_tokens or heads % tile_heads or channels % 32:
        raise ValueError("window_pack requires divisible token/head tiles and a 32-byte-aligned channel row")
    tile_bytes = tile_heads * tile_tokens * channels
    if tile_bytes > 65536:
        raise ValueError(f"window_pack limits its UB tile to 65536 bytes, got {tile_bytes}; reduce tile_heads or tile_tokens")
    windows, head_groups = sequence // tile_tokens, heads // tile_heads
    tasks = batch * windows * head_groups
    if tasks % num_blocks:
        raise ValueError("window_pack requires the number of tiles to be divisible by num_blocks")
    output_shape = (batch, windows, head_groups, tile_heads, tile_tokens, channels)

    @T.prim_func
    def kernel(src: T.Tensor((batch, sequence, heads, channels), dtype), dst: T.Tensor(output_shape, dtype)):
        with T.Kernel(num_blocks) as bx:
            tile = T.alloc_shared((tile_heads, tile_tokens, channels), dtype)
            for step in T.serial(tasks // num_blocks):
                task = step * num_blocks + bx
                b = task // (windows * head_groups)
                w = (task // head_groups) % windows
                group = task % head_groups
                for h in T.serial(tile_heads):
                    T.copy(src[b, w * tile_tokens : (w + 1) * tile_tokens, group * tile_heads + h, :], tile[h, :, :])
                T.copy(tile, dst[b, w, group, :, :, :])

    return kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["uint8", "float8_e4m3fn", "float8_e5m2"], default="float8_e4m3fn")
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--num-blocks", type=int, default=1)
    args = parser.parse_args()
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    shape = (1, 128, 32, 64)
    batch, sequence, heads, channels = shape
    tile_heads, tile_tokens = 8, 8
    host = torch.randint(0, 256, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(3308))
    expected = host.reshape(batch, sequence // tile_tokens, tile_tokens, heads // tile_heads, tile_heads, channels)
    expected = expected.permute(0, 1, 3, 4, 2, 5).contiguous()
    storage_dtype = getattr(torch, args.dtype)
    src = host.to("npu").view(storage_dtype)
    dst = torch.full(expected.shape, 165, dtype=torch.uint8, device="npu").view(storage_dtype)
    program = window_pack(num_blocks=args.num_blocks, dtype=args.dtype)
    (args.dump_dir / "input.tir").write_text(program.script())
    compiled = tilelang.compile(program, target="ascend")
    (args.dump_dir / "kernel.cce").write_text(compiled.get_kernel_source())
    compiled(src, dst)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.view(torch.uint8).cpu(), expected, rtol=0, atol=0)
    print(f"PASS: {args.dtype}; {tile_heads} logical inbound copy calls per tile; {host.numel()} payload bytes")


if __name__ == "__main__":
    main()
