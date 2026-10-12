"""Verify descriptor address equivalence on CPU without importing TileLang."""

import argparse
import json
from pathlib import Path


def verify_case(batch, sequence, heads, channels, tile_heads, tile_tokens):
    windows, groups = sequence // tile_tokens, heads // tile_heads
    size = batch * sequence * heads * channels
    expected = [-1] * size
    for b in range(batch):
        for token in range(sequence):
            for head in range(heads):
                for d in range(channels):
                    destination = (
                        (((b * windows + token // tile_tokens) * groups + head // tile_heads) * tile_heads + head % tile_heads)
                        * tile_tokens
                        + token % tile_tokens
                    ) * channels + d
                    expected[destination] = ((b * sequence + token) * heads + head) * channels + d

    ordinary, loop, nddma = ([-1] * size for _ in range(3))
    tasks = batch * windows * groups
    tile_size = tile_heads * tile_tokens * channels
    for task in range(tasks):
        b, window, group = task // (windows * groups), (task // groups) % windows, task % groups
        base = ((b * sequence + window * tile_tokens) * heads + group * tile_heads) * channels
        output = task * tile_size
        for head in range(tile_heads):
            for burst in range(tile_tokens):
                src = base + head * channels + burst * heads * channels
                dst = output + head * tile_tokens * channels + burst * channels
                ordinary[dst : dst + channels] = range(src, src + channels)

        loop_sizes = (tile_heads, 1)
        loop_src_strides = (channels, 0)
        loop_dst_strides = (tile_tokens * channels, 0)
        for outer in range(loop_sizes[1]):
            for inner in range(loop_sizes[0]):
                src_base = base + outer * loop_src_strides[1] + inner * loop_src_strides[0]
                dst_base = output + outer * loop_dst_strides[1] + inner * loop_dst_strides[0]
                for burst in range(tile_tokens):
                    src, dst = src_base + burst * heads * channels, dst_base + burst * channels
                    loop[dst : dst + channels] = range(src, src + channels)

        sizes = (channels, tile_tokens, tile_heads)
        src_strides = (1, heads * channels, channels)
        dst_strides = (1, channels, tile_tokens * channels)
        for i2 in range(sizes[2]):
            for i1 in range(sizes[1]):
                for i0 in range(sizes[0]):
                    idx = (i0, i1, i2)
                    src = base + sum(i * stride for i, stride in zip(idx, src_strides))
                    dst = output + sum(i * stride for i, stride in zip(idx, dst_strides))
                    assert nddma[dst] == -1, "NDDMA destination writes overlap"
                    nddma[dst] = src
    assert ordinary == expected and loop == expected and nddma == expected
    assert sorted(expected) == list(range(size)), "Payload bytes must appear exactly once"
    loop_supported = channels % 32 == 0 and (tile_tokens * channels) % 32 == 0
    return {
        "shape_bshd_storage": [batch, sequence, heads, channels],
        "tile_htd_storage": [tile_heads, tile_tokens, channels],
        "bytes": size,
        "tiles": tasks,
        "ordinary_requests": tasks * tile_heads,
        "loop_requests": tasks if loop_supported else None,
        "nddma_requests": tasks,
        "loop_aligned": loop_supported,
        "address_equivalent": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [verify_case(1, 128, 32, 64, 8, 8)]
    for channels in (32, 64, 128):
        for tile_heads in (1, 2, 8):
            cases.append(verify_case(2, 16, 8, channels, tile_heads, 4))
    cases.append(verify_case(1, 16, 8, 48, 8, 4))
    assert cases[0]["ordinary_requests"] == 512 and cases[0]["nddma_requests"] == 64
    assert cases[-1]["loop_requests"] is None
    text = json.dumps({"validation": "CPU address model only", "cases": cases}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(f"PASS: {len(cases)} address cases; no device compilation or execution performed")


if __name__ == "__main__":
    main()
