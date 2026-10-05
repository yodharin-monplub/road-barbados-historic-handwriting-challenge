"""Prepare the four confirmation folds locally before renting GPU time."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import build_bank, synthesize, save_json


def main():
    report = json.loads((ROOT / "outputs/model107/fold0_report.json").read_text())
    if not report["development_gate_passed"]:
        raise ValueError("Fold-0 gate failed; confirmation is not authorized by the recipe")
    base = json.loads((ROOT / "model107/config.json").read_text())
    for fold in range(1, 5):
        directory = ROOT / f"outputs/model107/fold{fold}"
        checkpoints = list((ROOT / f"outputs/model36/checkpoint_evaluation/fold{fold}/converted").glob("checkpoint_07-*.safetensors"))
        if len(checkpoints) != 1:
            raise ValueError(f"Ambiguous epoch-7 aligner for fold {fold}")
        checkpoint = checkpoints[0]
        config = dict(base, fold=fold,
                      checkpoint=str(checkpoint.relative_to(ROOT)),
                      conversion_receipt=str(checkpoint.with_suffix(".conversion.json").relative_to(ROOT)),
                      training_manifest=f"outputs/model36/fold{fold}/run_manifest.json")
        save_json(directory / "prepare_config.json", config)
        save_json(ROOT / "outputs/model107/preparation_status.json", {"stage": "aligning", "fold": fold})
        print(f"Preparing fold {fold}: fixed epoch-7 fold-excluded aligner", flush=True)
        build_bank(config, directory / "bank", device="cuda")
        training = directory / "training"
        if not (training / "manifest.json").exists():
            synthesize(config, directory / "bank/bank.json", training, count=819)
        subprocess.run([sys.executable, "model107/train.py", "--fold", str(fold), "--prepare-only"], cwd=ROOT, check=True)
    save_json(ROOT / "outputs/model107/preparation_status.json", {"stage": "complete", "folds": [1, 2, 3, 4]})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save_json(ROOT / "outputs/model107/preparation_status.json", {"stage": "failed", "error": str(error)})
        raise
