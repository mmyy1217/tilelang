"""Isolate NDDMA storage width, entry invalidation, and destination layout."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from run_ascendc_comparison import run, summarize


CASES = {
    "paired_head": {"PACK_COMPARE_WIDTHS": 1},
    "paired_wide": {"PACK_COMPARE_WIDTHS": 1, "PACK_CHANNEL_BYTES": 256},
    "paired_token": {"PACK_COMPARE_WIDTHS": 1, "PACK_TOKEN_MAJOR": 1},
    "paired_linear": {"PACK_COMPARE_WIDTHS": 1, "PACK_TOKEN_MAJOR": 1, "PACK_TILE_HEADS": 32},
    "head_b8": {},
    "head_no_dci": {"NDDMA_SKIP_DCI": 1},
    "head_b16": {"NDDMA_STORAGE": "uint16_t"},
    "head_b32": {"NDDMA_STORAGE": "uint32_t"},
    "wide_b8": {"PACK_CHANNEL_BYTES": 256},
    "wide_b32": {"PACK_CHANNEL_BYTES": 256, "NDDMA_STORAGE": "uint32_t"},
    "token_b8": {"PACK_TOKEN_MAJOR": 1},
    "token_b32": {"PACK_TOKEN_MAJOR": 1, "NDDMA_STORAGE": "uint32_t"},
    "linear_b8": {"PACK_TOKEN_MAJOR": 1, "PACK_TILE_HEADS": 32},
    "linear_b32": {"PACK_TOKEN_MAJOR": 1, "PACK_TILE_HEADS": 32, "NDDMA_STORAGE": "uint32_t"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binaries-from", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--rounds", type=int, default=51)
    parser.add_argument("--host-gcc-dir", type=Path, default=Path("/usr/lib/gcc/x86_64-linux-gnu/9"))
    args = parser.parse_args()
    if not 1 <= args.rounds <= 1000:
        parser.error("rounds must be between 1 and 1000")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("reference_window_pack.asc").resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report = {"source_sha256": digest, "warmup": 10, "rounds": args.rounds, "cases": []}
    previous = None
    if args.binaries_from:
        previous = json.loads((args.binaries_from / "results.json").read_text())
        if previous["source_sha256"] != digest:
            raise RuntimeError("Reused binaries must have the same source SHA256")
    if not args.build_only:
        (root / "npu-before.txt").write_text(subprocess.check_output(["npu-smi", "info"], text=True))
    for name, options in CASES.items():
        if args.cases and name not in args.cases:
            continue
        folder = root / name
        symbols = ["PackOrdinary", "PackLoop", "PackNddma"]
        if options.get("PACK_COMPARE_WIDTHS"):
            symbols += ["PackB16", "PackB32", "PackNoDci"]
        folder.mkdir(exist_ok=True)
        binary = folder / "reference"
        if previous is None:
            command = [
                "bisheng",
                str(source),
                "-o",
                str(binary),
                "--npu-arch=dav-3510",
                "-O2",
                "-std=c++17",
                f"-B{args.host_gcc_dir}",
                f"-L{args.host_gcc_dir}",
                *[f"-D{key}={value}" for key, value in options.items()],
            ]
            (folder / "compile-command.json").write_text(json.dumps(command, indent=2) + "\n")
            run(command, folder / "compile.log", folder)
        else:
            prior = next(case for case in previous["cases"] if case["name"] == name)
            if prior["options"] != options:
                raise RuntimeError(f"{name}: reused binary options do not match")
            binary = (args.binaries_from / name / "reference").resolve()
            if hashlib.sha256(binary.read_bytes()).hexdigest() != prior["binary_sha256"]:
                raise RuntimeError(f"{name}: reused binary SHA256 does not match")
            (folder / "compile-command.json").write_text((binary.parent / "compile-command.json").read_text())
        result = {"name": name, "options": options, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
        if not args.build_only:
            run(["stdbuf", "-oL", "-eL", str(binary), str(args.rounds)], folder / "events.log", folder)
            log = (folder / "events.log").read_text()
            if log.count(": PASS;") != len(symbols):
                raise RuntimeError(f"{name}: expected correctness PASS for all {len(symbols)} kernels")
            samples = {str(mode): [] for mode in range(len(symbols))}
            for line in log.splitlines():
                if line.startswith("SAMPLE,"):
                    _, _, mode, duration = line.split(",")
                    samples[mode].append(float(duration))
            if any(len(values) != args.rounds for values in samples.values()):
                raise RuntimeError(f"{name}: incomplete ACL event samples")
            result["correctness"] = "PASS"
            result["acl_events"] = {mode: summarize(values) for mode, values in samples.items()}
            if args.profile:
                command = [
                    "msprof",
                    f"--output={folder / 'profile'}",
                    "--task-time=on",
                    "--ai-core=off",
                    "--sys-lp=off",
                    f"--application={shlex.join([str(binary), str(args.rounds)])}",
                ]
                run(command, folder / "profile.log", folder)
                rows = []
                for path in (folder / "profile").rglob("op_summary*.csv"):
                    with path.open() as stream:
                        rows.extend(csv.DictReader(stream))
                result["device_tasks"] = {}
                for mode, symbol in enumerate(symbols):
                    selected = sorted((row for row in rows if symbol in row["Op Name"]), key=lambda row: float(row["Task Start Time(us)"]))
                    if len(selected) != args.rounds + 11:
                        raise RuntimeError(f"{name}/{symbol}: unexpected task count {len(selected)}")
                    result["device_tasks"][str(mode)] = summarize([float(row["Task Duration(us)"]) for row in selected[11:]])
        report["cases"].append(result)
        (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        key = "device_tasks" if args.profile else "acl_events"
        print(name, "compiled" if args.build_only else {m: round(s["median_us"], 3) for m, s in result[key].items()}, flush=True)
    if not args.build_only:
        (root / "npu-after.txt").write_text(subprocess.check_output(["npu-smi", "info"], text=True))


if __name__ == "__main__":
    main()
