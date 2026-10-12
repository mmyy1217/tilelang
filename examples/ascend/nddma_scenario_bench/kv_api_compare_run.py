"""Check or profile the installed API and loop candidate on identical allocations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from scenario_bundle import require_idle_device
from run import stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("check", "profile"))
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", choices=("bf16", "fp8"), required=True)
    p.add_argument("--groups", type=int, choices=(1, 3, 64), required=True)
    p.add_argument("--tail", type=int, choices=(0, 1), required=True)
    p.add_argument("--device", type=int, choices=(0, 1), default=0)
    p.add_argument("--allow-occupied", action="store_true")
    args = p.parse_args()
    if args.allow_occupied and args.device != 0:
        p.error("The occupied-device authorization covers device 0 only")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    build = args.build.resolve()
    prior = json.loads((build / "manifest.json").read_text())
    binary = build / "kernel"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != prior["binary_sha256"]:
        raise RuntimeError("Comparison binary hash mismatch; rebuild or restore the recorded artifact")
    record = dict(build=prior, device=args.device, dtype=args.dtype, groups=args.groups, tail=args.tail,
                  occupied_device_authorized=args.allow_occupied, phase=args.phase, status="started")
    try:
        before = subprocess.check_output(["npu-smi", "info"], text=True)
        (root / "npu-before.txt").write_text(before)
        if not args.allow_occupied:
            require_idle_device(before, args.device)
        repeats = 121 if args.phase == "profile" else 0
        app = [str(binary), str(args.device), "PA_NZ", args.dtype, str(args.groups), str(args.tail), str(repeats)]
        cmd = ["msprof", f"--output={root / 'profile'}", "--task-time=on", "--sys-lp=off",
               "--ai-core=off", f"--application={shlex.join(app)}"] if repeats else app
        record["command"] = cmd
        try:
            with (root / "run.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        finally:
            after = subprocess.check_output(["npu-smi", "info"], text=True)
            (root / "npu-after.txt").write_text(after)
        log = (root / "run.log").read_text(errors="replace")
        if (log.count("\nCHECK,") != 3 or log.count("CANDIDATE_CHECK,") != 3 or
                log.count("POST_CHECK,") != 2 or "RESULT,PASS," not in log or "ERROR," in log):
            raise RuntimeError("Incomplete paired correctness checks; inspect run.log")
        record["correctness"] = "PASS"
        record["status"] = "checked"
        if repeats:
            rows = []
            for path in (root / "profile").rglob("op_summary*.csv"):
                with path.open() as f:
                    rows.extend(csv.DictReader(f))
            inventory = {}
            for row in rows:
                name = row["Op Name"]
                inventory[name] = inventory.get(name, 0) + 1
            record["all_task_counts"] = inventory
            official = [r for r in rows if r.get("Op Type") == "GatherPaKvCache" or "GatherPaKvCache" in r["Op Name"]]
            candidate = [r for r in rows if "KvApiLoop" in r["Op Name"]]
            if len(official) != repeats + 4 or len(candidate) != repeats + 4:
                raise RuntimeError(f"Unexpected task count: official={len(official)}, candidate={len(candidate)}; expected {repeats + 4} each")
            if len(rows) != len(official) + len(candidate):
                raise RuntimeError("Additional device tasks present; account for the complete API before comparing")
            for name, items in (("official", official), ("loop", candidate)):
                items.sort(key=lambda r: float(r["Task Start Time(us)"]))
                record[name] = stats([float(r["Task Duration(us)"]) for r in items[23:-1]])
            record["latency_reduction_pct"] = 100 * (1 - record["loop"]["median_us"] / record["official"]["median_us"])
            record["warmup_per_mode"] = 20
            record["timed_samples_per_mode"] = 101
            record["performance_status"] = "provisional_nonexclusive" if args.allow_occupied else "endpoint_audit_only"
            if not args.allow_occupied:
                try:
                    require_idle_device(after, args.device)
                except RuntimeError:
                    record["performance_status"] = "provisional_contended"
            record["status"] = "profiled"
    except Exception as error:
        record.update(status="not_completed", error=str(error))
        raise
    finally:
        (root / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("status", "dtype", "groups", "tail", "correctness", "official", "loop", "performance_status") if k in record}))


if __name__ == "__main__":
    main()
