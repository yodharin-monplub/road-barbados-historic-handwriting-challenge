"""Background local preparation; wait between checks rather than occupying chat."""
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/model110"


def record(stage, **fields):
    value = {"stage": stage, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
    temporary = OUT / "preparation_status.tmp"
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(OUT / "preparation_status.json")
    print(json.dumps(value), flush=True)


def main():
    out = OUT / "fold0/words"
    while True:
        path = out / "generation_manifest.json"
        generated = json.loads(path.read_text()) if path.exists() else {}
        record("waiting_for_generation", generated_words=len(generated.get("rows", [])))
        if generated.get("complete") and (out / "labels.csv").exists(): break
        # The existing generator is deliberately not relaunched or interrupted.
        active = False
        for proc in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                if "model110/generate_words.py" in proc.read_bytes().decode().split("\0"):
                    active = True; break
            except (OSError, UnicodeError): pass
        if not active:
            raise RuntimeError("Word generation ended before completion; inspect before resuming")
        time.sleep(600)
    for checker in ("kraken", "lighton"):
        record("checking_generated_words", checker=checker)
        # Completed receipts are checked again by compose's provenance validator.
        if not (out / f"{checker}_audit_receipt.json").exists():
            with (OUT / f"{checker}_audit.log").open("a") as log:
                subprocess.run([str(ROOT / ".venv-lighton/bin/python"), "model108/audit.py", "--directory", str(out),
                                "--checker", checker], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    record("composing_passages")
    if not (OUT / "fold0/preparation_report.json").exists():
        subprocess.run([str(ROOT / ".venv-lighton/bin/python"), "model110/compose.py"], cwd=ROOT, check=True)
    subprocess.run([str(ROOT / ".venv-lighton/bin/python"), "model110/train.py", "--prepare-only"], cwd=ROOT, check=True)
    report = json.loads((OUT / "fold0/preparation_report.json").read_text())
    record("ready_for_ocr_ablation", report=report)


if __name__ == "__main__":
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "preparation.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main()
    except Exception as error:
        record("needs_attention", error=str(error))
        raise
