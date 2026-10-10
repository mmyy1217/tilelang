"""Dump real Ascend pass inputs/outputs without device compilation or launch."""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import inspect
from functools import partial
import re
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import tilelang
from tilelang import tvm
from tilelang.instrumentation import (
    PassEventObserver,
    PassInstrumentationTool,
    StackedPassInstrument,
    compile_pass_instrumentation,
)

from kernels import CASES as BASE_CASES
from kernels_extended import CASES as EXTENDED_CASES, EXPECTED_ERRORS
from kernel_snapshot import standalone_kernel

CASES = {**BASE_CASES, **EXTENDED_CASES}


ROOT = Path(__file__).resolve().parents[3]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


class DumpTool(PassInstrumentationTool, PassEventObserver):
    def __init__(self, stream):
        self.stream = stream
        self.line = 1
        self.sequence = 0
        self.records = []
        self.instruments = []

    def write(self, text):
        start = self.line
        self.stream.write(text)
        self.stream.flush()
        self.line += text.count("\n")
        return start

    def allocate_sequence(self):
        value = self.sequence
        self.sequence += 1
        return value

    def create_pass_instrument(self):
        instrument = StackedPassInstrument(self, capture_nested=True, sequence_allocator=self.allocate_sequence)
        self.instruments.append(instrument)
        return instrument

    def pass_started(self, mod, event):
        text = mod.script(show_meta=True)
        record = {
            **asdict(event),
            "status": "started",
            "before_sha256": digest(text),
            "before_line": self.write(f"\n[IR BEFORE #{event.sequence:04d}] {event.name} depth={event.depth}\n{text}\n") + 1,
        }
        self.records.append(record)
        return record, time.perf_counter()

    def pass_finished(self, mod, event, state):
        record, start = state
        text = mod.script(show_meta=True)
        output_hash = digest(text)
        record.update(
            status="completed",
            after_sha256=output_hash,
            changed=output_hash != record["before_sha256"],
            elapsed_ms=(time.perf_counter() - start) * 1000,
            after_line=self.write(f"\n[IR AFTER #{event.sequence:04d}] {event.name} depth={event.depth}\n{text}\n") + 1,
        )

    def passes_incomplete(self, passes, error):
        for frame in passes:
            if frame.state is not None:
                record, _ = frame.state
                record["status"] = "incomplete"
                self.write(f"\n[IR INCOMPLETE #{record['sequence']:04d}] {record['name']}\n")

    def callback_mismatch(self, actual, expected):
        raise RuntimeError(f"Pass callback mismatch: got {actual}, expected {expected}")

    def run_codegen(self, event, next_call):
        self.write(f"\n[CODEGEN INPUT] {event.name}\n{event.mod.script(show_meta=True)}\n")
        return next_call()


def external_factory(spec):
    filename, separator, name = spec.rpartition(":")
    if not separator:
        raise ValueError("--kernel must use /path/to/file.py:factory_name")
    path = Path(filename).resolve()
    module_spec = importlib.util.spec_from_file_location("copy_pass_user_kernel", path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"Cannot load kernel module: {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(path.parent))
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    factory = getattr(module, name)
    if not callable(factory):
        raise TypeError(f"Expected a no-argument kernel factory, got {factory!r}")
    return factory, path


def run_case(name, factory, directory, kernel_path, expected_error=None):
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "case": name,
        "tilelang_revision": git_revision(ROOT),
        "tvm_revision": git_revision(ROOT / "3rdparty/tvm"),
        "tilelang_root": str(ROOT),
        "python": sys.executable,
        "kernel_file": str(kernel_path),
        "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        "enable_device_compile": False,
        "enable_host_codegen": False,
        "target": "ascend",
        "status": "started",
        "expected_error": expected_error,
    }
    if name in CASES:
        snapshot = standalone_kernel(factory)
        (directory / "kernel.py").write_text(snapshot)
        metadata["kernel_snapshot_sha256"] = digest(snapshot)
        metadata["kernel_snapshot_format"] = "standalone_specialized_factory"
        if isinstance(factory, partial):
            metadata["factory_parameters"] = {"args": factory.args, "kwargs": factory.keywords}
    success = False
    with (directory / "pass.log").open("w") as stream:
        tool = DumpTool(stream)
        try:
            with (
                compile_pass_instrumentation(tools=[tool], include_default_tools=False, reuse_existing=False) as session,
                tvm.target.Target("ascend"),
                tvm.transform.PassContext(opt_level=3, instruments=session.create_pass_instruments()),
            ):
                function = factory()
                if not isinstance(function, (tvm.tirx.PrimFunc, tvm.IRModule)):
                    raise TypeError("Kernel factory must return a PrimFunc or IRModule")
                initial = function.script(show_meta=True)
                (directory / "input.tir").write_text(initial + "\n")
                tool.write(f"[INITIAL IR]\n{initial}\n")
                artifact = tilelang.lower(
                    function,
                    target="ascend",
                    enable_host_codegen=False,
                    enable_device_compile=False,
                )
                (directory / "host.tir").write_text(artifact.host_mod.script(show_meta=True) + "\n")
                (directory / "device.tir").write_text(artifact.device_mod.script(show_meta=True) + "\n")
                (directory / "kernel.cpp").write_text(artifact.kernel_source)
                tool.write(f"\n[GENERATED CCE SOURCE]\n{artifact.kernel_source}\n")
            if not tool.records or any(record["status"] != "completed" for record in tool.records):
                raise RuntimeError("Pass dump is empty or contains incomplete callbacks")
            metadata["status"] = "lowered_to_cce_source"
            success = expected_error is None
            if expected_error is not None:
                metadata["expectation_error"] = "Expected a diagnostic, but lowering succeeded"
        except Exception:
            metadata["status"] = "failed"
            metadata["error"] = traceback.format_exc()
            tool.write(f"\n[COMPILATION ERROR]\n{metadata['error']}\n")
            if expected_error is not None:
                success = re.search(expected_error, metadata["error"], re.IGNORECASE) is not None
                metadata["expected_error_matched"] = success
        metadata["passes"] = tool.records
        metadata["pass_count"] = len(tool.records)
        (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        index = ["sequence\tpass\tstatus\ttext_changed\tbefore_line\tafter_line\n"]
        for record in tool.records:
            index.append(
                f"{record['sequence']}\t{record['name']}\t{record['status']}\t"
                f"{record.get('changed', '')}\t{record['before_line']}\t{record.get('after_line', '')}\n"
            )
        (directory / "passes.tsv").write_text("".join(index))
    metadata_label = "expected_rejection" if metadata.get("expected_error_matched") else metadata["status"]
    print(f"{name}: {metadata_label}, {metadata['pass_count']} passes, {directory / 'pass.log'}", flush=True)
    if not success:
        print(metadata.get("error", "Compilation failed"), file=sys.stderr)
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case", choices=CASES, default=None)
    group.add_argument("--all", action="store_true")
    group.add_argument("--kernel", help="Python file:factory returning a PrimFunc/IRModule")
    parser.add_argument("--out", type=Path, help="Fresh output directory; existing case directories are never overwritten")
    parser.add_argument("--list", action="store_true", help="List built-in cases")
    args = parser.parse_args()
    if args.list:
        print("\n".join(CASES))
        return 0
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = args.out or ROOT.parent / "work/experiments/ascend-copy-pass-lab/runs" / stamp
    if args.kernel:
        factory, kernel_path = external_factory(args.kernel)
        selected = {"custom": factory}
    else:
        kernel_path = Path(__file__).with_name("kernels.py")
        selected = CASES if args.all else {args.case or "dma": CASES[args.case or "dma"]}
    results = []
    for name, factory in selected.items():
        source_factory = factory.func if isinstance(factory, partial) else factory
        source_path = kernel_path if args.kernel else Path(inspect.getsourcefile(source_factory))
        results.append(run_case(name, factory, output / name, source_path, EXPECTED_ERRORS.get(name) if not args.kernel else None))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
