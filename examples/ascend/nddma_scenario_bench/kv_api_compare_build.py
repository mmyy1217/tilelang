"""Build a same-process installed-API versus loop comparison for 950PR_9579."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = Path(__file__).resolve().parent
    original = (base / "kv_cache.asc").read_text()
    prefix, body = original.split("#ifndef TRANSPOSE", 1)
    body = "#ifndef TRANSPOSE" + body.split("#define KERNEL(Name,Mode)", 1)[0]
    if body.count("BUF=98304") != 1:
        raise RuntimeError("Original UB declaration changed; inspect the source before generating the comparison")
    lines = [prefix]
    macros = ("SOURCE_ROWS", "SOURCE_COLS", "TAKE_ROWS", "TAKE_COLS", "TILE_GROUPS", "TRANSPOSE", "SOURCE_OFFSET")
    for width, budget in ((512, 65536), (1024, 122336)):
        values = (128, width, 7, 128, 96, 0, 0)
        lines += [f"#define {k} {v}" for k, v in zip(macros, values)]
        lines += [f"namespace candidate{width} {{", body.replace("BUF=98304", f"BUF={budget}")]
        lines += [f"__global__ __vector__ void KvApiLoop{width}(__gm__ uint8_t* yk, __gm__ uint8_t* yv,",
                  "__gm__ uint8_t* xk, __gm__ uint8_t* xv, __gm__ uint32_t* table,",
                  "__gm__ uint32_t* lens, __gm__ uint32_t* offsets, uint32_t batches) {",
                  "Run<4>(yk,yv,xk,xv,table,lens,offsets,batches);", "}", "}"]
        lines += ["#undef " + k for k in macros]
    lines += ["#define KV_COMPARE_LOOP 1", '#include "kv_aclnn_probe.cpp"',
              "void KvApiLoopLaunch(int keyBytes, int groups, aclrtStream stream, void* kcache, void* vcache,",
              "void* table, void* lengths, void* offsets, void* key, void* value) {",
              "const uint32_t per = (groups + 55) / 56;",
              "const uint32_t blocks = (groups + per - 1) / per;",
              "if (keyBytes == 1) candidate512::KvApiLoop512<<<blocks,nullptr,stream>>>(",
              "(uint8_t*)key,(uint8_t*)value,(uint8_t*)kcache,(uint8_t*)vcache,",
              "(uint32_t*)table,(uint32_t*)lengths,(uint32_t*)offsets,groups);",
              "else candidate1024::KvApiLoop1024<<<blocks,nullptr,stream>>>(",
              "(uint8_t*)key,(uint8_t*)value,(uint8_t*)kcache,(uint8_t*)vcache,",
              "(uint32_t*)table,(uint32_t*)lengths,(uint32_t*)offsets,groups);", "}"]
    for name in ("kv_cache.asc", "kv_aclnn_probe.cpp", "kv_reference_upstream.h"):
        (out / name).write_bytes((base / name).read_bytes())
    source = out / "compare.asc"
    source.write_text("\n".join(lines) + "\n")
    command = ["bisheng", str(source), "-o", str(out / "kernel"), "--npu-arch=dav-3510",
               "-O2", "-std=c++17", "-B/usr/lib/gcc/x86_64-linux-gnu/9",
               "-L/usr/lib/gcc/x86_64-linux-gnu/9", "-lm", "-lopapi", "-lnnopbase", "-I" + str(out)]
    record = dict(command=command, source_hashes={name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                      for name in ("compare.asc", "kv_cache.asc", "kv_aclnn_probe.cpp", "kv_reference_upstream.h")},
                  settings=dict(soc="Ascend950PR_9579", vector_cores=56, tile_tokens=96,
                                fp8_slot_bytes=65536, bf16_slot_bytes=122336, table_width=9),
                  status="started")
    try:
        with (out / "compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        record.update(status="built_not_executed", binary_sha256=hashlib.sha256((out / "kernel").read_bytes()).hexdigest())
    finally:
        (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record["status"])


if __name__ == "__main__":
    main()
