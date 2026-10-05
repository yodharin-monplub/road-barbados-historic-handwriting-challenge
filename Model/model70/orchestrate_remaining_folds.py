#!/usr/bin/env python3
"""Finish Model36 folds 2--4 and build the fixed-policy OOF reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-kraken/bin/python"
TRUTH = ROOT / "data/processed/train_folds.csv"
POLICY = "fixed_zero_based_epoch_7_selected_once_on_fold0"


class PipelineError(RuntimeError):
    pass


def environment() -> dict[str, str]:
    value = dict(os.environ)
    value["PYTHONPATH"] = str(ROOT / "src")
    value["PYTHONUNBUFFERED"] = "1"
    return value


def run(arguments: list[str]) -> None:
    print("RUN", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, env=environment(), check=True)


def checkpoints(fold: int) -> list[Path]:
    return sorted((ROOT / f"outputs/model36/fold{fold}").glob("checkpoint_*.ckpt"))


def gpu_compute_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,name",
            "--format=csv,noheader",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def wait_for_attached_training(fold: int, timeout_seconds: int) -> None:
    started = time.monotonic()
    last_count = -1
    while True:
        found = checkpoints(fold)
        if len(found) != last_count:
            print(f"WAIT fold={fold} checkpoints={len(found)}/8", flush=True)
            last_count = len(found)
        if len(found) > 8:
            raise PipelineError(f"fold {fold} has more than eight checkpoints")
        if len(found) == 8:
            active = gpu_compute_processes()
            if not active:
                print(f"READY fold={fold} training complete and CUDA idle", flush=True)
                return
            print(f"WAIT fold={fold} checkpoint matrix complete; CUDA still active", flush=True)
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"timed out waiting for fold {fold} training")
        time.sleep(30)


def leaderboard_record(fold: int, epoch: int) -> dict[str, object]:
    path = ROOT / f"outputs/model36/checkpoint_evaluation/fold{fold}/leaderboard.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("fold", -1)) != fold:
        raise PipelineError(f"fold {fold} leaderboard provenance mismatch")
    matches = [row for row in payload.get("results", []) if int(row["epoch"]) == epoch]
    if len(matches) != 1:
        raise PipelineError(f"fold {fold} leaderboard does not have exactly one epoch {epoch}")
    return matches[0]


def evaluate_and_infer(fold: int) -> dict[str, object]:
    evaluation = ROOT / f"outputs/model36/checkpoint_evaluation/fold{fold}"
    all_rows = ROOT / f"outputs/model36/fold{fold}/train_and_validation_epoch07.csv"
    if evaluation.exists() or all_rows.exists():
        raise FileExistsError(f"fold {fold} evaluation/inference output already exists")
    run(
        [
            str(PYTHON),
            "model22/evaluate_checkpoints.py",
            "--checkpoint-dir",
            f"outputs/model36/fold{fold}",
            "--input",
            str(TRUTH),
            "--output-dir",
            f"outputs/model36/checkpoint_evaluation/fold{fold}",
            "--fold",
            str(fold),
            "--config",
            "model36/config.yaml",
            "--batch-size",
            "1",
        ]
    )
    record = leaderboard_record(fold, 7)
    run(
        [
            str(PYTHON),
            "model36/infer.py",
            "--checkpoint",
            str(record["weights"]),
            "--input",
            str(TRUTH),
            "--output",
            str(all_rows),
            "--no-constrain-charset",
            "--batch-size",
            "1",
        ]
    )
    return record


def start_postprocess(fold: int, record: dict[str, object]) -> tuple[subprocess.Popen[str], object]:
    output = ROOT / f"outputs/model35/fold{fold}_epoch07_verified"
    if output.exists():
        raise FileExistsError(output)
    logs = ROOT / "outputs/model70/logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = (logs / f"model35_fold{fold}.log").open("w", encoding="utf-8")
    predictions = Path(str(record["predictions"]))
    arguments = [
        str(PYTHON),
        "model35/run_fold.py",
        "--truth",
        str(TRUTH),
        "--predictions",
        f"outputs/model36/fold{fold}/train_and_validation_epoch07.csv",
        "--inference-manifest",
        f"outputs/model36/fold{fold}/train_and_validation_epoch07.manifest.json",
        "--training-manifest",
        f"outputs/model36/fold{fold}/run_manifest.json",
        "--heldout-oof-predictions",
        str(predictions),
        "--oof-evaluation-receipt",
        str(predictions.with_suffix(".evaluation.json")),
        "--heldout-fold",
        str(fold),
        "--checkpoint-epoch",
        "7",
        "--checkpoint-policy",
        POLICY,
        "--policy-selection-fold",
        "0",
        "--model-family",
        "model36",
        "--output-dir",
        str(output),
    ]
    print(f"START CPU postprocess fold={fold} log={log.name}", flush=True)
    process = subprocess.Popen(
        arguments,
        cwd=ROOT,
        env=environment(),
        text=True,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log


def train(fold: int) -> None:
    output = ROOT / f"outputs/model36/fold{fold}"
    if output.exists():
        raise FileExistsError(output)
    run([str(PYTHON), "model36/train.py", "--fold", str(fold)])
    if len(checkpoints(fold)) != 8:
        raise PipelineError(f"fold {fold} training did not produce eight checkpoints")


def wait_postprocess(jobs: list[tuple[int, subprocess.Popen[str], object]]) -> None:
    for fold, process, log in jobs:
        code = process.wait()
        log.close()
        if code:
            raise subprocess.CalledProcessError(code, process.args)
        report = ROOT / f"outputs/model35/fold{fold}_epoch07_verified/report.json"
        if not report.is_file():
            raise PipelineError(f"fold {fold} postprocess did not write its report")
        print(f"DONE CPU postprocess fold={fold}", flush=True)


def combine_and_select() -> None:
    fixed_output = ROOT / "outputs/model35/allfold_epoch07_verified"
    if fixed_output.exists():
        raise FileExistsError(fixed_output)
    combine = [
        str(PYTHON),
        "model35/combine_oof.py",
        "--truth",
        str(TRUTH),
    ]
    for fold in range(5):
        combine.extend(
            ["--fold-report", f"outputs/model35/fold{fold}_epoch07_verified/report.json"]
        )
    combine.extend(["--output-dir", str(fixed_output)])
    run(combine)

    selection_output = ROOT / "outputs/model69/oof"
    if selection_output.exists():
        raise FileExistsError(selection_output)
    select = [
        str(PYTHON),
        "model69/select_crossfold_epochs.py",
        "--truth",
        str(TRUTH),
    ]
    select.extend(
        [
            "--fold-leaderboard",
            "0=outputs/model22/checkpoint_evaluation/fold0/leaderboard.json",
        ]
    )
    for fold in range(1, 5):
        select.extend(
            [
                "--fold-leaderboard",
                f"{fold}=outputs/model36/checkpoint_evaluation/fold{fold}/leaderboard.json",
            ]
        )
    select.extend(["--output-dir", str(selection_output)])
    run(select)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--attach-fold", type=int, choices=(2,))
    mode.add_argument(
        "--restart-fold",
        type=int,
        choices=(3,),
        help="start a clean deterministic fold-3 run, then continue through fold 4",
    )
    parser.add_argument("--wait-timeout", type=int, default=7200)
    args = parser.parse_args()
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    jobs: list[tuple[int, subprocess.Popen[str], object]] = []
    if args.restart_fold is None:
        attached = args.attach_fold if args.attach_fold is not None else 2
        wait_for_attached_training(attached, args.wait_timeout)
        record = evaluate_and_infer(attached)
        process, log = start_postprocess(attached, record)
        jobs.append((attached, process, log))
        first_training_fold = attached + 1
    else:
        first_training_fold = args.restart_fold
    for fold in range(first_training_fold, 5):
        train(fold)
        record = evaluate_and_infer(fold)
        process, log = start_postprocess(fold, record)
        jobs.append((fold, process, log))
    wait_postprocess(jobs)
    combine_and_select()
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"PIPELINE FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
