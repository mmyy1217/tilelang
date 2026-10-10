"""Check archived pass indices, diagnostics, source snapshots, and codegen routes."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "dma": ["asc_copy_gm2ub_align", "asc_copy_ub2gm_align"],
    "strided": ["4, 120", "128, 128", "120, 128"],
    "padding": ["asc_set_copy_pad_val", "4, 120, 0, 2, 1"],
    "tail": ["asc_copy_gm2ub_align", "min(64"],
    "dynamic": ["asc_copy_gm2ub_align", "n"],
    "gemm": ["asc_fill_l1", "asc_copy_l12l0a", "asc_copy_l12l0b", "asc_copy_l0c2gm"],
    "dual_dma": ["sid", "asc_copy_gm2ub_align"],
    "dual_gemm": ["asc_copy_l0c2ub", "asc_dual_dst_mode>(1)", "sid * 512"],
    "nz_scatter": ["ascend_nd2nz_scatter", "asc_copy_ub2l1"],
    "simd": ["simd_inst::vlds_norm", "simd_inst::vsts_norm"],
    "simt": ["__simt_vf__", "threadIdx.x", "float2"],
    "pipeline": ["step & 1", "asc_sync_wait"],
    "fp4": ["float4_e2m1x2_t", "1, 32"],
    "offset_strided": ["src[176]", "dst[200]", "320, 128", "256, 128"],
    "explicit_pad": ["asc_set_copy_pad_val", "4, 120, 0, 2, 1"],
    "simt_global": ["__simt_vf__", "__gm__ float2*", "asc_copy_ub2gm_align"],
    "raw_ub_l1": ["asc_copy_ub2l1", "256, 1, 0, 0"],
    "dual_raw_ub_l1": ["asc_copy_ub2l1", "sid"],
    "compact_nz": ["ascend_nd2nz_scatter<32, 128, bfloat16_t, bfloat16_t>"],
    "compact_nz_cast_down": ["ascend_nd2nz_scatter<32, 128, float, bfloat16_t>"],
    "compact_nz_cast_up": ["ascend_nd2nz_scatter<32, 128, bfloat16_t, float>"],
    "compact_nz_simd": ["ascend_nd2nz_scatter_callee"],
    "prepacked_nz_l1": ["ascend_nd2nz_scatter", "asc_copy_ub2l1"],
    "dual_prepacked_nz_l1": ["ascend_nd2nz_scatter", "asc_copy_ub2l1", "sid"],
    "manual_post_copy": ["ascend_nd2nz_scatter", "asc_copy_ub2l1"],
    "nz_l1_cast_down": ["ascend_nd2nz_scatter<32, 128, float, bfloat16_t>", "asc_copy_ub2l1"],
    "nz_l1_cast_up": ["ascend_nd2nz_scatter<32, 128, bfloat16_t, float>", "asc_copy_ub2l1"],
    "nz_l1_single_c0": ["asc_copy_ub2l1"],
    "dual_nz_l1": ["ascend_nd2nz_scatter", "asc_copy_ub2l1", "sid"],
    "gm_l1_transpose": ["asc_copy_gm2l1_dn2nz"],
    "l1_l0a_transpose": ["asc_copy_l12l0a_transpose"],
    "l1_l0b_transpose": ["asc_copy_l12l0b_transpose"],
    "l1_l0_layout_transpose": ["asc_copy_l12l0a_transpose"],
    "l1_l0_subtile": ["asc_copy_l12l0a", ", 1, 1, 2, 2, 4, 2"],
    "l0c_ub": ["asc_copy_l0c2ub", "asc_dual_dst_mode>(0)"],
    "l0c_ub_bf16": ["asc_copy_l0c2ub", "asc_quant_mode>(16)"],
    "l0c_gm_bf16": ["asc_copy_l0c2gm", "asc_quant_mode>(16)"],
    "l0c_gm_fp16": ["asc_copy_l0c2gm", "asc_quant_mode>(1)"],
    "dual_l0c_n": ["asc_dual_dst_mode>(2)", "sid * 16", "32, 64"],
    "unit_flag": ["asc_unit_flag_mode>(3)", "(bool)1, 0, 3"],
    "atomic_fixpipe": ["asc_set_atomic_add", "asc_copy_l0c2gm", "asc_set_atomic_none"],
    "atomic_ub_add": ["asc_set_atomic_add", "asc_copy_ub2gm_align", "asc_set_atomic_none"],
    "atomic_ub_max": ["asc_set_atomic_max", "asc_set_atomic_none"],
    "atomic_ub_min": ["asc_set_atomic_min", "asc_set_atomic_none"],
    "simd_cast_down": ["simd_inst::vcvt<bfloat16_t>"],
    "simd_cast_fp16": ["simd_inst::vcvt<half>"],
    "simt_cast": ["__simt_vf__", "__float22bfloat162_rn"],
    "simt_scalar": ["__simt_vf__", "threadIdx.x"],
    "scalar_global": ["write_gm_bypass_dcache"],
    "dynamic_shape": ["int32_t n", "block_idx", "max((n"],
    "dynamic_ub": ["int32_t n", "(n * 4)"],
    "empty_tail": ["if (", "block_idx", "80 -"],
    "pipeline_1": ["asc_copy_gm2ub_align", "asc_sync_wait"],
    "pipeline_3": ["i % 3", "asc_sync_wait"],
    "pipeline_4": ["i & 3", "asc_sync_wait"],
    "fp8_e4m3": ["fp8_e4_t", "1, 64"],
    "fp8_e5m2": ["fp8_e5_t", "1, 64"],
    "cache_policy": ["asc_load_l2_cache_mode>(1)", "asc_store_l2_cache_mode>(4)"],
    "mx_sf_tile": ["asc_copy_l12l0a_mx", "asc_copy_l12l0b_mx", "asc_mmad_mx"],
    "mx_sf_lifetime": ["asc_copy_l12l0a_mx", "asc_copy_l12l0b_mx", "asc_mmad_mx"],
    "mx_sf_byte": ["asc_copy_l12l0a_mx", "__gm__ uint8_t* sa", "fp8_e8m0_t"],
    "mx_fp4": ["float4_e2m1x2_t", "__cbuf__ int8_t*", "asc_copy_l12l0a_mx", "asc_mmad_mx"],
}
REJECTIONS = {
    "reject_dma_3d": ">2D strided copy not yet supported",
    "reject_simd_short": "total elements divisible by VReg",
    "reject_raw_stride": "contiguous source region",
    "reject_raw_payload": "multiple of 32 bytes",
    "reject_fp4_odd": "packed sub-byte row must be byte-aligned",
    "reject_dma_cast": "DMA copies cannot perform type casting",
    "reject_dual_cast": "Type conversion during cc->ub dual-destination copy is NOT supported",
    "reject_simd_cast_up": "cast copy only supports f32->bf16 or f32->half",
}


def verify_case(directory):
    name = directory.name
    metadata = json.loads((directory / "manifest.json").read_text())
    log = (directory / "pass.log").read_text()
    lines = log.splitlines()
    records = metadata["passes"]
    assert metadata["case"] == name
    assert metadata["enable_device_compile"] is False
    assert metadata["enable_host_codegen"] is False
    assert metadata["pass_count"] == len(records)
    assert [p["sequence"] for p in records] == list(range(len(records)))
    for record in records:
        number = record["sequence"]
        assert lines[record["before_line"] - 1].startswith(f"[IR BEFORE #{number:04d}] {record['name']}")
        if record["status"] == "completed":
            assert lines[record["after_line"] - 1].startswith(f"[IR AFTER #{number:04d}] {record['name']}")
        else:
            assert name in REJECTIONS and record["status"] == "incomplete"
    tsv = (directory / "passes.tsv").read_text().splitlines()
    assert len(tsv) == len(records) + 1
    for row, record in zip(tsv[1:], records):
        fields = row.split("\t")
        assert int(fields[0]) == record["sequence"]
        assert fields[1:3] == [record["name"], record["status"]]
        assert int(fields[4]) == record["before_line"]
        if record["status"] == "completed":
            assert int(fields[5]) == record["after_line"]
    snapshot = directory / "kernel.py"
    if "kernel_snapshot_sha256" in metadata:
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == metadata["kernel_snapshot_sha256"]
    if metadata.get("kernel_snapshot_format") == "standalone_specialized_factory":
        module = ast.parse(snapshot.read_text())
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        assert len(functions) == 1 and functions[0].name == "make_kernel"
        assert all(isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for node in module.body)
    if name in REJECTIONS:
        assert metadata["status"] == "failed"
        assert metadata["expected_error_matched"] is True
        assert REJECTIONS[name] in metadata["error"]
        assert "[COMPILATION ERROR]" in log
        assert not (directory / "kernel.cpp").exists()
        return {"outcome": "expected_rejection", "pass_count": len(records), "diagnostic": REJECTIONS[name]}
    assert metadata["status"] == "lowered_to_cce_source"
    assert all(p["status"] == "completed" for p in records)
    for filename in ("input.tir", "host.tir", "device.tir", "kernel.cpp"):
        assert (directory / filename).is_file()
    source = (directory / "kernel.cpp").read_text()
    for token in EXPECTED[name]:
        assert token in source, (name, "missing expected codegen", token)
    if name.startswith(("raw_", "dual_raw_")) or name == "nz_l1_single_c0":
        assert "ascend_nd2nz_scatter" not in source
    if name == "simt_global":
        assert "asc_copy_gm2ub_align" not in source
    if name == "mx_sf_lifetime":
        assert source.index("asc_copy_l12l0a_mx") < source.index("for (int32_t i")
    changed = [p for p in records if p.get("changed")]
    preferred = "InsertNd2Nz" if "nz" in name else "AscendLowerTileOp"
    if name.startswith("pipeline") and name != "pipeline_1":
        preferred = "MaterializeMultiBuffer"
    elif name.startswith("dual"):
        preferred = "RewriteDualCopy"
    elif name.startswith("simd"):
        preferred = "AscendSimdVFLowerParallel"
    elif name in ("tail", "empty_tail"):
        preferred = "AscendInsertOOBPadding"
    elif name.startswith("fp4"):
        preferred = "RewriteFp4ToFp4x2"
    record = next((p for p in changed if preferred in p["name"]), changed[0])
    return {
        "outcome": "lowered_to_cce_source",
        "pass_count": len(records),
        "changed_pass": record["name"],
        "before_line": record["before_line"],
        "after_line": record["after_line"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).with_name("results") / "verified-context")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    actual = {p.name for p in args.root.iterdir() if p.is_dir()}
    assert actual == EXPECTED.keys() | REJECTIONS.keys(), ("case set mismatch", actual ^ (EXPECTED.keys() | REJECTIONS.keys()))
    results = {name: verify_case(args.root / name) for name in sorted(actual)}
    summary = {
        "total_cases": len(results),
        "lowered_cases": len(EXPECTED),
        "expected_rejections": len(REJECTIONS),
        "total_pass_callbacks": sum(case["pass_count"] for case in results.values()),
        "cases": results,
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.summary:
        args.summary.write_text(text)
    print(f"Verified {len(results)} cases: {len(EXPECTED)} lowered, {len(REJECTIONS)} expected rejections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
