"""Sequential Model121 local GPU pipeline with quiet600-second supervision."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import save_json

OUT = ROOT / "outputs/model121"


def record(stage, **fields):
    value = {"stage": stage, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "cloud_used": False, "submission_ready": False, **fields}
    save_json(OUT / "status.json", value)
    print(json.dumps(value), flush=True)


def wait_gpu():
    attempt = 0
    while True:
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
        if not apps and free >= 5000:
            return
        delay = 10 if attempt < 6 else 600
        record("waiting_for_gpu", free_mib=free, next_check_seconds=delay)
        time.sleep(delay)
        attempt += 1


def run(action):
    if action != "evaluate":
        wait_gpu()
    log_path = OUT / f"{action}.log"
    with log_path.open("x") as log:
        process = subprocess.Popen([str(ROOT / ".venv-lighton/bin/python"), "-u", "model121/experiment.py", action],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        record(action, pid=process.pid, log=str(log_path.relative_to(ROOT)))
        start = time.monotonic()
        while True:
            try:
                code = process.wait(timeout=600)
                break
            except subprocess.TimeoutExpired:
                with log_path.open("rb") as handle:
                    handle.seek(max(0, log_path.stat().st_size - 2500))
                    tail = handle.read().decode(errors="replace")
                record(action, pid=process.pid, elapsed_minutes=round((time.monotonic()-start)/60, 1), log_tail=tail)
        if code:
            raise RuntimeError(f"{action} exited{code}; see {log_path.relative_to(ROOT)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / "status.json").exists():
            raise RuntimeError("Supervisor already ran; inspect state before recovery")
        try:
            jobs = [f"fold{f}_{n}_{m}" for f in range(1, 5) for n in ("model73", "model107") for m in ("greedy", "beam")]
            for action in (*jobs, "evaluate"):
                run(action)
            report = json.loads((OUT / "report.json").read_text())
            record("pilot_complete", scores=report["scores"], gains=report["gains"], passes_screen=report["confirmation_passed"])
        except Exception as error:
            record("needs_attention", error=str(error))
            raise


if __name__ == "__main__":
    main()
