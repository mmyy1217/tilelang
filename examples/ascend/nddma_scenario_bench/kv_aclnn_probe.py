"""Build or launch the installed GatherPaKvCache API with an idle-device gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from scenario_bundle import require_idle_device


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "check"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--device", type=int, choices=(0, 1), default=1)
    parser.add_argument("--layout", choices=("Norm", "PA_NZ"), default="PA_NZ")
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--groups", type=int, choices=(1, 3, 64), default=1)
    parser.add_argument("--tail", type=int, choices=(0, 1), default=0)
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--debug-logs", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.repeats <= 200:
        parser.error("repeats must be between 0 and 200")
    if args.phase == "check" and args.build is None:
        parser.error("check requires --build")
    runtime = Path(os.environ["ASCEND_HOME_PATH"])
    source = Path(__file__).with_suffix(".cpp")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / source.name).write_bytes(source.read_bytes())
    record = {"phase": args.phase, "source_sha256": digest(source),
              "driver_sha256": digest(__file__), "runtime": str(runtime),
              "api_library_sha256": digest(runtime / "lib64/libopapi.so"),
              "api_header_sha256": digest(runtime / "include/aclnnop/aclnn_gather_pa_kv_cache.h"),
              "status": "started"}
    try:
        if args.phase == "build":
            command = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", str(source),
                       "-I" + str(runtime / "include"), "-L" + str(runtime / "lib64"),
                       "-lopapi", "-lascendcl", "-lnnopbase", "-o", str(output / "kv_aclnn_probe")]
            record["command"] = command
            with (output / "build.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
            record["binary_sha256"] = digest(output / "kv_aclnn_probe")
            record["status"] = "built_not_executed"
        else:
            build = args.build.resolve()
            prior = json.loads((build / "manifest.json").read_text())
            binary = build / "kv_aclnn_probe"
            for key in ("source_sha256", "api_library_sha256", "api_header_sha256", "runtime"):
                if record[key] != prior[key]:
                    raise RuntimeError(f"Build mismatch for {key}; rebuild against the current runtime")
            if digest(binary) != prior["binary_sha256"]:
                raise RuntimeError("Binary hash mismatch; rebuild before launching")
            record["binary_sha256"] = prior["binary_sha256"]
            before = subprocess.check_output(["npu-smi", "info"], text=True)
            (output / "npu-before.txt").write_text(before)
            require_idle_device(before, args.device)
            command = [str(binary), str(args.device), args.layout, args.dtype,
                       str(args.groups), str(args.tail), str(args.repeats)]
            env = os.environ.copy()
            if args.debug_logs:
                env.update(ASCEND_GLOBAL_LOG_LEVEL="0", ASCEND_SLOG_PRINT_TO_STDOUT="1")
            record["command"] = command
            record["debug_logs"] = args.debug_logs
            try:
                with (output / "run.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   check=True, timeout=120)
            finally:
                after = subprocess.check_output(["npu-smi", "info"], text=True)
                (output / "npu-after.txt").write_text(after)
            record["status"] = "correctness_pass"
            if "RESULT,PASS," not in (output / "run.log").read_text():
                raise RuntimeError("Missing complete correctness result")
            try:
                require_idle_device(after, args.device)
                record["occupancy_audit"] = "empty_at_both_endpoints_only"
            except RuntimeError as error:
                record["occupancy_audit"] = "contended_do_not_use_for_performance"
                record["occupancy_error"] = str(error)
            record["performance"] = "not_measured"
    except Exception as error:
        record["status"] = "not_completed"
        record["error"] = str(error)
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
