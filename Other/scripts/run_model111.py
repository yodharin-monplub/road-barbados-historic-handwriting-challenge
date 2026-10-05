"""Local CUDA likelihood pilot; sequential judges and ten-minute progress checks."""
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/model111"


def record(stage, **fields):
    value = {"stage": stage, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
    temporary = OUT / "status.tmp"
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(OUT / "status.json")
    print(json.dumps(value), flush=True)


def main():
    for judge in ("model73", "model107"):
        record("scoring", judge=judge)
        with (OUT / f"{judge}.log").open("a") as log:
            process = subprocess.Popen([str(ROOT / ".venv-lighton/bin/python"), "model111/score.py", "--judge", judge],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = process.wait(timeout=600)
                    break
                except subprocess.TimeoutExpired:
                    output = OUT / f"fold0/{judge}_scores.jsonl"
                    with output.open() as handle: rows = sum(1 for _ in handle)
                    record("scoring", judge=judge, completed_rows=rows, total_rows=819)
            if code: raise RuntimeError(f"{judge} scoring exited {code}; see its log")
    record("evaluating")
    with (OUT / "evaluation.log").open("a") as log:
        subprocess.run([str(ROOT / ".venv-lighton/bin/python"), "model111/evaluate.py"], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((OUT / "fold0/report.json").read_text())
    record("pilot_complete", clean_gain=report["clean_primary_gain"], passes_screen=report["passes_development_screen"],
           submission_ready=False, cloud_used=False)


if __name__ == "__main__":
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "supervisor.lock").open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            main()
    except Exception as error:
        record("needs_attention", error=str(error))
        raise
