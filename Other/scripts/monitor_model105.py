#!/usr/bin/env python3
"""Supervise the authorized Vast instance; preserve artifacts before stopping it."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import math
import struct
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HOST = "root@145.236.167.14"
PORT = "17109"
INSTANCE = "50076619"
REMOTE = "/workspace/road-htr"
CLI = "/home/msi/.local/share/vastai-cli/bin/vastai"
OUT = ROOT / "outputs/model105"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-p", PORT, HOST]
SCP = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-P", PORT]


def run(args, timeout=180):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed ({result.returncode}): {result.stderr[-1500:]}")
    return result.stdout


def record(stage, **fields):
    state = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), stage=stage, **fields)
    tmp = OUT / "supervisor_status.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(OUT / "supervisor_status.json")
    print(json.dumps(state), flush=True)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def beep():
    """A brief plain tone, using the desktop audio server when available."""
    path = OUT / "completion_beep.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"".join(struct.pack("<h", int(3500 * math.sin(2 * math.pi * 880 * i / 24000)))
                                  for i in range(6000)))
    try:
        run(["pw-play", str(path)], timeout=10)
    except Exception:
        print("Audible beep unavailable; see supervisor_status.json", flush=True)


def snapshot():
    code = '''
import json, pathlib, subprocess
root = pathlib.Path('/workspace/road-htr')
jobs = []
for p in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        args = (p / 'cmdline').read_bytes().decode().split('\\0')
        if any(a in ('model105/train.py', 'model105/infer.py', 'model92/infer.py') for a in args):
            jobs.append({'pid': int(p.name), 'args': args[:8]})
    except (OSError, UnicodeError):
        pass
logs = {}
for name in ('train', 'infer'):
    p = root / ('outputs/model105/' + name + '.log')
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0, p.stat().st_size - 1400))
            logs[name] = f.read().decode(errors='replace')[-1000:]
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], text=True).strip()
gpu_jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
print(json.dumps({'jobs': jobs, 'gpu': gpu, 'gpu_jobs': gpu_jobs, 'logs': logs,
 'complete': (root / 'outputs/model105/test.manifest.json').is_file()}))
'''
    return json.loads(run(SSH + ["python3 -c " + shlex.quote(code)]))


def collect(relative_files, target):
    target.mkdir(parents=True, exist_ok=True)
    run(SCP + [f"{HOST}:{REMOTE}/{f}" for f in relative_files] + [str(target)], timeout=1800)


def verify_prediction(path):
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    assert digest(path) == manifest["output_sha256"], f"CSV checksum: {path}"
    assert digest(path.with_suffix(".jsonl")) == manifest["detail_sha256"], f"Details checksum: {path}"
    assert digest(ROOT / "data/raw/Test.csv") == manifest["input_sha256"], "Wrong test input"
    with (ROOT / "data/raw/Test.csv").open(encoding="utf-8-sig") as handle:
        table = csv.DictReader(handle)
        id_column = table.fieldnames[0]
        expected = [row[id_column] for row in table]
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(expected) == manifest["rows"] == 1374
    assert [row[id_column] for row in rows] == expected, "Wrong ID order"
    empty_ids = sorted(row[id_column] for row in rows if not row["prediction"].strip())
    assert empty_ids == sorted(manifest["empty_prediction_ids"]), "Blank-output manifest mismatch"
    return manifest


def collect_folds():
    try:
        for fold in range(5):
            verify_prediction(ROOT / f"outputs/model92/test_fold{fold}_step400.csv")
        return
    except (OSError, ValueError, AssertionError, KeyError):
        pass
    names = [f"outputs/model92/test_fold{fold}_step400{suffix}"
             for fold in range(5) for suffix in (".csv", ".jsonl", ".manifest.json")]
    collect(names, ROOT / "outputs/model92")
    for fold in range(5):
        verify_prediction(ROOT / f"outputs/model92/test_fold{fold}_step400.csv")


def collect_final():
    collect([f"outputs/model105/{name}" for name in (
        "test.csv", "test.jsonl", "test.manifest.json", "train.log", "infer.log")], OUT)
    manifest = verify_prediction(OUT / "test.csv")
    (OUT / "all").mkdir(exist_ok=True)
    collect(["outputs/model105/all/run_manifest.json"], OUT / "all")
    run(SCP + ["-r", f"{HOST}:{REMOTE}/outputs/model105/all/best_adapter", str(OUT / "all")], timeout=1800)
    for name, expected in manifest["adapter_files_sha256"].items():
        assert digest(OUT / "all/best_adapter" / name) == expected, f"Adapter checksum: {name}"
    assert manifest["adapter_files_sha256"], "Missing adapter checksum ledger"
    assert digest(ROOT / "model105/config.yaml") == manifest["config_sha256"], "Wrong model config"
    training = json.loads((OUT / "all/run_manifest.json").read_text())
    assert training["train_rows"] == 4095 and training["validation_rows"] == 0
    return manifest


def stop_instance():
    current = snapshot()
    if current["jobs"] or current["gpu_jobs"]:
        raise RuntimeError("Instance still has active model/GPU processes; refusing to stop")
    record("stopping_instance", instance=INSTANCE)
    run([CLI, "stop", "instance", INSTANCE, "--raw"])
    for attempt in range(12):
        state = json.loads(run([CLI, "show", "instance", INSTANCE, "--raw"]))
        if isinstance(state, dict) and "instances" in state:
            state = state["instances"]
        if isinstance(state, list):
            state = state[0]
        if str(state.get("id")) != INSTANCE:
            raise RuntimeError("Unexpected instance ID from Vast API")
        if state.get("actual_status") == "stopped" or (
            state.get("actual_status") == "exited"
            and state.get("cur_state") == "stopped"
            and state.get("intended_status") == "stopped"
        ):
            return
        time.sleep(10)
    raise RuntimeError("Stop requested but stopped status is not yet confirmed")


def build_submission():
    from road_htr.crossfit_candidate_selector import apply_deployment
    from road_htr.submission import validate_submission
    submission = ROOT / "submissions/model104_model105_selector.csv"
    # Empty raw OCR is retained in the original artifact for audit. The existing
    # Model102 prediction is the deterministic deployment fallback for those IDs.
    with (ROOT / "outputs/model102/test.csv").open(encoding="utf-8-sig") as handle:
        primary = {row["ID"]: row["prediction"] for row in csv.DictReader(handle)}
    with (OUT / "test.csv").open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    fallback_ids = []
    for row in rows:
        if not row["prediction"].strip():
            row["prediction"] = primary[row["ID"]]
            fallback_ids.append(row["ID"])
    challenger = OUT / "test_with_blank_fallback.csv"
    with challenger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = apply_deployment(
        ROOT / "data/raw/Test.csv", ROOT / "outputs/model102/test.csv",
        challenger, ROOT / "outputs/model104/deployment.json",
        ROOT / "outputs/model104/test.csv",
        sample_path=ROOT / "data/raw/SampleSubmission.csv", submission_path=submission)
    result["validation"] = validate_submission(submission, ROOT / "data/raw/SampleSubmission.csv")
    result["submission"] = str(submission)
    result["sha256"] = digest(submission)
    result["blank_fallback_ids"] = fallback_ids
    (ROOT / "outputs/model104/deployment_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Read-only remote status check")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.once:
        print(json.dumps(snapshot(), indent=2))
        return
    with (OUT / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record("collecting_model92")
        collect_folds()
        record("model92_verified", folds=5, rows_per_fold=1374)
        failures = 0
        while True:
            try:
                state = snapshot()
                record("monitoring", **state)
                if state["complete"] and not state["jobs"]:
                    break
                if not state["jobs"]:
                    raise RuntimeError("Model105 exited without completed inference; check logs")
                failures = 0
            except Exception as exc:
                failures += 1
                record("check_failed", error=str(exc), consecutive_failures=failures)
                if failures >= 3:
                    raise
            time.sleep(600)
        record("collecting_model105")
        collect_final()
        record("model105_verified")
        stop_instance()
        record("instance_stopped", instance=INSTANCE)
        result = build_submission()
        record("complete", instance=INSTANCE, instance_status="stopped", **result)
        beep()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--once" not in sys.argv:
            record("needs_attention", error=str(exc))
            beep()
        raise
