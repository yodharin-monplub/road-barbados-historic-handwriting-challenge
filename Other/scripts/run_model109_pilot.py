"""Sequential local generator fine-tuning and paired 200-word quality audits."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import save_json


def stage(name, args, environment=".venv-lighton"):
    save_json(ROOT / "outputs/model109/status.json", {"stage": name, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print(f"Stage: {name}", flush=True)
    subprocess.run([str(ROOT / environment / "bin/python"), *args], cwd=ROOT, check=True)


def main():
    stage("fine_tuning", ["model109/train.py"])
    stage("generating_pretrained_baseline", ["model108/generate.py"])
    stage("generating_fine_tuned_pilot", ["model108/generate.py", "--adapter", "outputs/model109/fold0/adapter.pt",
                                        "--output", "outputs/model109/fold0/pilot"])
    for model in [108, 109]:
        directory = f"outputs/model{model}/fold0/pilot"
        for checker in ["kraken", "lighton", "score"]:
            stage(f"model{model}_{checker}_audit", ["model108/audit.py", "--directory", directory, "--checker", checker],
                  ".venv-kraken" if checker == "kraken" else ".venv-lighton")
    reports = {str(model): json.loads((ROOT / f"outputs/model{model}/fold0/pilot/quality_report.json").read_text()) for model in [108, 109]}
    save_json(ROOT / "outputs/model109/comparison.json", {"reports": reports,
        "acceptance_delta": reports["109"]["acceptance_rate"] - reports["108"]["acceptance_rate"],
        "next_step": "Review text-fidelity screen before scaling synthetic OCR training.", "submission_ready": False})
    save_json(ROOT / "outputs/model109/status.json", {"stage": "pilot_complete", "submission_ready": False})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save_json(ROOT / "outputs/model109/status.json", {"stage": "failed", "error": str(error)})
        raise
