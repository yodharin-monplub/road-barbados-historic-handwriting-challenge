"""Sequential local seed pilots, with quiet ten-minute background monitoring."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import save_json

OUT = ROOT / "outputs/model114"
PYTHON = str(ROOT / ".venv-lighton/bin/python")


def record(stage, **fields):
    value = {"stage": stage, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "cloud_used": False, "submission_ready": False, **fields}
    save_json(OUT / "seed_pair_status.json", value)
    print(json.dumps(value), flush=True)


def free_gpu():
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip()
    if apps or int(memory.splitlines()[0]) < 5000:
        raise RuntimeError("GPU is occupied or has <5000MiB free; no other job was interrupted")


def run(action, number=114):
    if action in ("smoke", "train", "infer"):
        free_gpu()
    log_path = ROOT / f"outputs/model{number}/{action}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x") as log:
        process = subprocess.Popen([PYTHON, "-u", "model114/pilot.py", action, "--model", str(number)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        record(action, model=number, pid=process.pid, log=str(log_path.relative_to(ROOT)))
        while True:
            try:
                code = process.wait(timeout=600)
                break
            except subprocess.TimeoutExpired:
                with log_path.open("rb") as progress:
                    progress.seek(max(0, log_path.stat().st_size - 3000))
                    tail = progress.read().decode("utf-8", errors="replace")
                record(action, model=number, pid=process.pid, elapsed_minutes=round((time.monotonic()-started)/60, 1), log_tail=tail)
        if code:
            raise RuntimeError(f"Model{number} {action} failed with exit {code}; see {log_path.relative_to(ROOT)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / "seed_pair_status.json").exists():
            raise RuntimeError("Supervisor already ran; inspect state before any explicit recovery")
        run("smoke")
        record("smoke_passed")
        for number in (114, 115):
            run("train", number)
            run("infer", number)
            run("evaluate", number)
        reports = {str(n): json.loads((ROOT / f"outputs/model{n}/report.json").read_text()) for n in (114, 115)}
        record("pilots_complete", results={n: {k: r[k] for k in ("ensemble_gain", "advance_to_confirmation_screen")} for n, r in reports.items()})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        record("needs_attention", error=str(error))
        raise
