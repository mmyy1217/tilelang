"""Build and profile the three AscendC window-pack implementations on an NPU."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess


CASES = [
    ("fp8_h8", "fp8_e4m3fn_t", 128, 64, 8, 1),
    ("fp8_h1", "fp8_e4m3fn_t", 128, 64, 1, 1),
    ("fp8_wide", "fp8_e4m3fn_t", 128, 256, 8, 1),
    ("fp8_many_tiles", "fp8_e4m3fn_t", 2048, 64, 8, 1),
    ("fp8_blocks16", "fp8_e4m3fn_t", 2048, 64, 8, 16),
    ("fp4_h8", "fp4x2_e2m1_t", 128, 64, 8, 1),
    ("uint8_h8", "uint8_t", 128, 64, 8, 1),
]


def run(command, log, cwd):
    with log.open("w") as output:
        subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=180)


def summarize(values):
    return {"samples_us": values, "median_us": statistics.median(values), "min_us": min(values), "max_us": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-gcc-dir", type=Path)
    parser.add_argument("--cases", nargs="+", choices=[case[0] for case in CASES])
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--binaries-from", type=Path)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 1000:
        parser.error("rounds must be between 1 and 1000")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("reference_window_pack.asc").resolve()
    report = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "warmup": 10, "rounds": args.rounds, "cases": []}
    previous = None
    if args.binaries_from:
        previous = json.loads((args.binaries_from / "results.json").read_text())
        if previous["source_sha256"] != report["source_sha256"]:
            raise RuntimeError("Reused binaries must have the same source SHA256")
    (root / "npu-before.txt").write_text(subprocess.check_output(["npu-smi", "info"], text=True))
    for name, dtype, sequence, channel_bytes, tile_heads, blocks in CASES:
        if args.cases and name not in args.cases:
            continue
        folder = root / name
        folder.mkdir(exist_ok=True)
        binary = folder / "reference"
        command = [
            "bisheng",
            str(source),
            "-o",
            str(binary),
            "--npu-arch=dav-3510",
            "-O2",
            "-std=c++17",
            f"-DCOPY_STORAGE={dtype}",
            f"-DPACK_SEQUENCE={sequence}",
            f"-DPACK_CHANNEL_BYTES={channel_bytes}",
            f"-DPACK_TILE_HEADS={tile_heads}",
            f"-DPACK_BLOCKS={blocks}",
        ]
        if args.host_gcc_dir:
            command += [f"-B{args.host_gcc_dir}", f"-L{args.host_gcc_dir}"]
        if previous is None:
            (folder / "compile-command.json").write_text(json.dumps(command, indent=2) + "\n")
            run(command, folder / "compile.log", folder)
        else:
            matching = [case for case in previous["cases"] if case["name"] == name]
            keys = ("storage_type", "sequence", "channel_bytes", "tile_heads", "blocks")
            if len(matching) != 1 or [matching[0][k] for k in keys] != [dtype, sequence, channel_bytes, tile_heads, blocks]:
                raise RuntimeError(f"{name}: reused binary parameters do not match")
            binary = (args.binaries_from / name / "reference").resolve()
            (folder / "compile-command.json").write_text((binary.parent / "compile-command.json").read_text())
        run([str(binary), str(args.rounds)], folder / "events.log", folder)
        log = (folder / "events.log").read_text()
        if log.count(": PASS;") != 3:
            raise RuntimeError(f"{name}: expected correctness PASS for all three kernels")
        event_samples = {mode: [] for mode in range(3)}
        for line in log.splitlines():
            if line.startswith("SAMPLE,"):
                _, _, mode, duration = line.split(",")
                event_samples[int(mode)].append(float(duration))
        if any(len(values) != args.rounds for values in event_samples.values()):
            raise RuntimeError(f"{name}: incomplete ACL event samples")
        result = {
            "name": name,
            "storage_type": dtype,
            "sequence": sequence,
            "channel_bytes": channel_bytes,
            "tile_heads": tile_heads,
            "blocks": blocks,
            "correctness": "PASS",
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "acl_events": {str(mode): summarize(values) for mode, values in event_samples.items()},
        }
        if not args.no_profile:
            run(
                [
                    "msprof",
                    f"--output={folder / 'profile'}",
                    "--task-time=on",
                    "--ai-core=off",
                    "--sys-lp=off",
                    f"--application={shlex.join([str(binary), str(args.rounds)])}",
                ],
                folder / "profile.log",
                folder,
            )
            summaries = list((folder / "profile").rglob("op_summary*.csv"))
            result["profiler_csvs"] = [str(p.relative_to(root)) for p in summaries]
            result["profiler_rows"] = []
            for summary in summaries:
                with summary.open() as stream:
                    result["profiler_rows"].extend(list(csv.DictReader(stream)))
            result["device_tasks"] = {}
            for mode, symbol in enumerate(("PackOrdinary", "PackLoop", "PackNddma")):
                rows = [row for row in result["profiler_rows"] if symbol in row["Op Name"]]
                rows.sort(key=lambda row: float(row["Task Start Time(us)"]))
                if len(rows) != args.rounds + 11:
                    raise RuntimeError(f"{name}/{symbol}: expected {args.rounds + 11} tasks, got {len(rows)}")
                result["device_tasks"][str(mode)] = summarize([float(row["Task Duration(us)"]) for row in rows[11:]])
        report["cases"].append(result)
        (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, {mode: round(values["median_us"], 3) for mode, values in result["acl_events"].items()}, flush=True)
    (root / "npu-after.txt").write_text(subprocess.check_output(["npu-smi", "info"], text=True))


if __name__ == "__main__":
    main()
