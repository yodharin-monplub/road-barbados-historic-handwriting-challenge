#!/usr/bin/env python3
"""Wait for one exact all-data run, then build the guarded Model35 candidate.

This runner performs no upload and has no network operation.  Every substantive
step delegates to the existing fail-closed Model35/36 CLIs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-kraken/bin/python"
STATUS = ROOT / "outputs/model70/deployment_continuation_status.json"


class DeploymentContinuationError(RuntimeError):
    pass


def _process_record(pid: int) -> tuple[str, str, str] | None:
    """Return (start ticks, state, command) for a process, or None if absent."""
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text(encoding="utf-8")
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
    except (FileNotFoundError, ProcessLookupError):
        return None
    closing = stat.rfind(")")
    if closing < 0:
        raise DeploymentContinuationError(f"cannot parse /proc/{pid}/stat")
    fields = stat[closing + 2 :].split()
    if len(fields) <= 19:
        raise DeploymentContinuationError(f"short /proc/{pid}/stat")
    return fields[19], fields[0], command


def _write_status(state: str, **extra: object) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"status": state, **extra}, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(STATUS)


def wait_for_training(pid: int, timeout_seconds: int, poll_seconds: int) -> None:
    initial = _process_record(pid)
    if initial is None:
        raise DeploymentContinuationError(f"training PID {pid} is absent")
    start_ticks, state, command = initial
    if state == "Z" or "model36/train.py --all-data" not in command:
        raise DeploymentContinuationError(
            f"PID {pid} is not the expected live all-data wrapper: {command!r}"
        )
    _write_status(
        "waiting_for_exact_training_pid",
        training_pid=pid,
        process_start_ticks=start_ticks,
        command=command,
    )
    started = time.monotonic()
    while True:
        current = _process_record(pid)
        if current is None or current[1] == "Z":
            return
        if current[0] != start_ticks or current[2] != command:
            raise DeploymentContinuationError("training PID identity changed")
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError("timed out waiting for all-data training")
        time.sleep(poll_seconds)


def _run(arguments: list[str]) -> None:
    print("RUN", " ".join(arguments), flush=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(arguments, cwd=ROOT, env=environment, check=True)


def deployment_commands() -> list[list[str]]:
    deployment = ROOT / "outputs/model36/deployment_epoch07"
    return [
        [str(PYTHON), "model35/check_candidate.py"],
        [str(PYTHON), "model36/export_checkpoint.py"],
        [
            str(PYTHON),
            "model36/infer.py",
            "--checkpoint",
            str(deployment / "epoch07.safetensors"),
            "--input",
            "data/processed/train_folds.csv",
            "--output",
            str(deployment / "train_predictions.csv"),
            "--no-constrain-charset",
            "--batch-size",
            "1",
        ],
        [
            str(PYTHON),
            "model36/infer.py",
            "--checkpoint",
            str(deployment / "epoch07.safetensors"),
            "--input",
            "data/raw/Test.csv",
            "--output",
            str(deployment / "test_predictions.csv"),
            "--no-constrain-charset",
            "--batch-size",
            "1",
        ],
        [
            str(PYTHON),
            "model35/build_submission.py",
            "--oof-report",
            "outputs/model35/allfold_epoch07_verified/report.json",
            "--training-manifest",
            "outputs/model36/foldall/run_manifest.json",
            "--training-predictions",
            str(deployment / "train_predictions.csv"),
            "--test-predictions",
            str(deployment / "test_predictions.csv"),
            "--conversion-receipt",
            str(deployment / "epoch07.conversion.json"),
            "--output-dir",
            "outputs/model35/deployment_epoch07",
        ],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--wait-timeout", type=int, default=10800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.training_pid <= 1:
        parser.error("--training-pid must identify a non-system process")
    if args.wait_timeout <= 0 or args.poll_seconds <= 0:
        parser.error("timeouts must be positive")
    for path in (
        ROOT / "outputs/model36/deployment_epoch07",
        ROOT / "outputs/model35/deployment_epoch07",
    ):
        if path.exists():
            raise FileExistsError(path)
    wait_for_training(args.training_pid, args.wait_timeout, args.poll_seconds)
    _write_status("training_exited_checking_artifacts", training_pid=args.training_pid)
    for index, command in enumerate(deployment_commands(), start=1):
        _write_status(
            "running_deployment_step",
            step=index,
            total_steps=len(deployment_commands()),
            command=command,
        )
        _run(command)
    report = ROOT / "outputs/model35/deployment_epoch07/report.json"
    if not report.is_file():
        raise DeploymentContinuationError("guarded deployment report is missing")
    _write_status(
        "validated_candidate_not_uploaded",
        report=str(report),
        submission=str(report.parent / "submission.csv"),
    )
    print("DEPLOYMENT CONTINUATION COMPLETE; NOTHING UPLOADED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        _write_status("failed", error_type=type(error).__name__, error=str(error))
        print(
            f"DEPLOYMENT CONTINUATION FAILED: {type(error).__name__}: {error}",
            flush=True,
        )
        raise
