"""Query installed CANN host tiling on CPU without initializing an NPU."""
import argparse
import ctypes
import hashlib
import itertools
import json
from pathlib import Path


class TilingV35(ctypes.LittleEndianStructure):
    _fields_ = [
        ("seqLenAccumSize", ctypes.c_uint32),
        ("batchCount", ctypes.c_int64), ("batchPerCore", ctypes.c_int64),
        ("needCoreNum", ctypes.c_uint32),
        ("kvCacheBlockSize", ctypes.c_int64), ("blockTableWidth", ctypes.c_int64),
        ("numBlocks", ctypes.c_int64), ("hiddenSizeK", ctypes.c_uint64),
        ("hiddenSizeV", ctypes.c_uint64), ("numTokens", ctypes.c_int64),
        ("maxUbHiddenSizeK", ctypes.c_uint64), ("maxUbHiddenSizeV", ctypes.c_uint64),
        ("maxUbHiddenSize", ctypes.c_uint64), ("nonContiguousFlag", ctypes.c_uint32),
        ("kCacheStride0", ctypes.c_int64), ("kCacheStride1", ctypes.c_int64),
        ("kCacheStride2", ctypes.c_int64), ("vCacheStride0", ctypes.c_int64),
        ("vCacheStride1", ctypes.c_int64), ("vCacheStride2", ctypes.c_int64),
        ("keyOutStride0", ctypes.c_int64), ("keyOutStride1", ctypes.c_int64),
        ("valueOutStride0", ctypes.c_int64), ("valueOutStride1", ctypes.c_int64),
        ("numHeadsK", ctypes.c_uint64), ("numHeadsV", ctypes.c_uint64),
    ]


def tensor(name, shape, dtype, fmt="ND"):
    return dict(name=name, shape=shape, ori_shape=shape, dtype=dtype, format=fmt, ori_format=fmt)


def request(layout, fp8, groups, tail):
    key_bytes = 1 if fp8 else 2
    key_dtype = "float8_e4m3fn" if fp8 else "bfloat16"
    blocks, tokens = groups * 9 + 3, groups * (897 if tail else 1024)
    nz = layout == "PA_NZ"
    fmt = "FRACTAL_NZ" if nz else "ND"
    ks = [blocks, 512 * key_bytes // 32, 128, 32 // key_bytes] if nz else [blocks, 128, 1, 512]
    vs = [blocks, 4, 128, 16] if nz else [blocks, 128, 1, 64]
    ko = tensor("key", [tokens, 512] if nz else [tokens, 1, 512], key_dtype)
    vo = tensor("value", [tokens, 64] if nz else [tokens, 1, 64], "bfloat16")
    inputs = [tensor("key_cache", ks, key_dtype, fmt), tensor("value_cache", vs, "bfloat16", fmt),
              tensor("block_tables", [groups, 9], "int32"), tensor("seq_lens", [groups], "int32"),
              ko, vo, tensor("seq_offset", [groups], "int32")]
    return dict(optype="GatherPaKvCache", compile_info={"device_id": None}, inputs=inputs,
                outputs=[ko, vo], attrs=[dict(name="cache_mode", dtype="str", value=layout),
                                        dict(name="is_seq_lens_cumsum", dtype="bool", value=False)])


def decode(data, layout, groups, tail, cores=64):
    if len(data) != ctypes.sizeof(TilingV35):
        raise RuntimeError(f"Expected V35 tiling ABI of {ctypes.sizeof(TilingV35)} bytes, got {len(data)}; inspect the installed schema")
    raw = TilingV35.from_buffer_copy(data)
    fields = {name: getattr(raw, name) for name, _ in TilingV35._fields_}
    per_core = (groups + cores - 1) // cores
    active_cores = (groups + per_core - 1) // per_core
    expected = dict(batchCount=groups, batchPerCore=per_core, needCoreNum=active_cores,
                    kvCacheBlockSize=128, blockTableWidth=9, numBlocks=groups * 9 + 3,
                    hiddenSizeK=512, hiddenSizeV=64, numTokens=groups * (897 if tail else 1024),
                    nonContiguousFlag=0, keyOutStride0=512, valueOutStride0=64)
    for key, value in expected.items():
        if fields[key] != value:
            raise RuntimeError(f"Unexpected {key}: expected {value}, got {fields[key]}; inspect raw tiling before interpreting")
    if layout == "Norm" and (fields["numHeadsK"], fields["numHeadsV"]) != (1, 1):
        raise RuntimeError("Unexpected ND head counts; inspect tiling ABI")
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--soc", choices=("Ascend950PR_9579", "Ascend950PR_9589"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from asc_op_compile_base.common.platform.platform_info import set_current_compile_soc_info
    from asc_op_compile_base.asc_op_compiler.op_tiling import do_op_tiling

    soc = args.soc
    cores = 56 if soc == "Ascend950PR_9579" else 64
    set_current_compile_soc_info(soc)
    summary = dict(soc=soc, execution="host_only_no_npu", source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[])
    try:
        for layout, fp8, groups, tail in itertools.product(("Norm", "PA_NZ"), (False, True), (1, 3, 64), (False, True)):
            name = f'{layout}-{"fp8" if fp8 else "bf16"}-g{groups}-tail{int(tail)}'
            req = request(layout, fp8, groups, tail)
            (args.output / f"{name}-request.json").write_text(json.dumps(req, indent=2) + "\n")
            result = do_op_tiling(**req)
            data = bytes(result["tiling_data"])
            result["tiling_data"] = data.hex()
            (args.output / f"{name}-raw.json").write_text(json.dumps(result, indent=2) + "\n")
            fields = decode(data, layout, groups, tail, cores)
            key = 1001 if layout == "PA_NZ" else 1101
            if result["tiling_key"] != key or result["block_dim"] != fields["needCoreNum"]:
                raise RuntimeError(f"Unexpected dispatch in {name}; inspect raw results")
            kb = 1 if fp8 else 2
            whole = fields["maxUbHiddenSizeK"] >= 128 * 512 and fields["maxUbHiddenSizeV"] >= 128 * 64
            item = dict(case=name, tiling_key=key, block_dim=result["block_dim"], workspace=result["workspaces"],
                        fields=fields, key_capacity_bytes=fields["maxUbHiddenSizeK"] * kb,
                        value_capacity_bytes=fields["maxUbHiddenSizeV"] * 2,
                        queue_slot_bytes=fields["maxUbHiddenSize"] * kb,
                        nz_source_branch=("whole_block_nddma" if whole else "token_dma_fallback") if layout == "PA_NZ" else "not_nz")
            summary["cases"].append(item)
            print(name, key, item["queue_slot_bytes"], item["nz_source_branch"], flush=True)
    finally:
        libraries = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            path = line.split()[-1]
            if path.startswith("/") and ("libophost" in path or "libregister.so" in path or "libplatform.so" in path):
                libraries.add(path)
        summary["loaded_libraries"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(libraries)}
        summary["complete"] = len(summary["cases"]) == 24
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
