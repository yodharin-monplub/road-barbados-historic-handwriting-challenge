"""Sequential Model118 local GPU pipeline with quiet600-second supervision."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import save_json

OUT = ROOT / "outputs/model118"


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
    if action in ("train", "infer"):
        wait_gpu()
    log_path = OUT / f"{action}.log"
    with log_path.open("x") as log:
        process = subprocess.Popen([str(ROOT / ".venv-lighton/bin/python"), "-u", "model118/experiment.py", action],
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


def wait_for_model117():
    while True:
        state = subprocess.check_output(["systemctl", "--user", "show", "road-model117.service", "--property=ActiveState", "--value"], text=True).strip()
        if state in ("active", "activating", "deactivating", "reloading"):
            record("waiting_for_model117", service_state=state, next_check_seconds=600)
            time.sleep(600)
            continue
        result = ROOT / "outputs/model117/status.json"
        if not result.exists() or json.loads(result.read_text()).get("stage") != "pilot_complete":
            raise RuntimeError("Model117 is not live but did not complete successfully; inspect before using GPU")
        return


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / "status.json").exists():
            raise RuntimeError("Supervisor already ran; inspect state before recovery")
        try:
            wait_for_model117()
            for action in ("infer", "evaluate"):
                run(action)
            report = json.loads((OUT / "report.json").read_text())
            record("pilot_complete", standalone_gain=report["standalone_gain_vs_matched_control"],
                   ensemble_gain=report["ensemble_gain"], passes_screen=report["advance_to_confirmation_screen"])
        except Exception as error:
            record("needs_attention", error=str(error))
            raise


if __name__ == "__main__":
    main()
