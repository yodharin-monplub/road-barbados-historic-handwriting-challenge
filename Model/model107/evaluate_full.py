"""Frozen full-OOF selector screen and conditional promotion decision."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.schema import read_table, predictions_by_id
from road_htr.stackmix import save_json, sha256


def main():
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    out = ROOT / "outputs/model107"
    truth = read_table(ROOT / "data/processed/train_folds.csv", require_target=True)
    combined = {}
    for fold in range(5):
        path = out / f"fold{fold}_oof.csv"
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        if manifest["fold"] != fold or manifest["output_sha256"] != sha256(path):
            raise ValueError("Wrong fold/hash in OOF artifact")
        predictions = predictions_by_id(read_table(path), "prediction")
        expected = {r[truth.id_column] for r in truth.rows if int(r["fold"]) == fold}
        if set(predictions) != expected or set(combined) & set(predictions):
            raise ValueError("OOF partition mismatch")
        combined.update(predictions)
    combined_path = out / "full_oof.csv"
    with combined_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "prediction"])
        writer.writerows((r[truth.id_column], combined[r[truth.id_column]]) for r in truth.rows)
    baseline = "outputs/model104/full_oof/crossfit_predictions.csv"
    selector = "outputs/model107/selector_oof"
    subprocess.run([sys.executable, "model104/evaluate.py", "--truth", "data/processed/train_folds.csv",
                    "--candidate-a", baseline, "--candidate-b", str(combined_path),
                    "--config", "model104/config.json", "--output-dir", selector], cwd=ROOT, check=True)
    bootstrap = out / "confirmation_bootstrap.json"
    subprocess.run([sys.executable, "scripts/bootstrap_stage_gains.py", "--truth", "data/processed/train_folds.csv",
                    "--prediction", f"model104={baseline}",
                    "--prediction", f"model107={selector}/crossfit_predictions.csv",
                    "--samples", "5000", "--output", str(bootstrap)], cwd=ROOT, check=True)
    comparison = json.loads(bootstrap.read_text())["comparisons"][0]
    passed = comparison["observed_delta"] >= 0.0005 and comparison["percentile_95"][0] > 0
    save_json(out / "promotion_decision.json", {
        "passes_submission_gate": passed, "comparison": comparison,
        "next_step": "all-data refit and test inference" if passed else "retain Model104 submission",
        "submission_ready": False,
        "caveat": "Development OOF estimate; reused upstream selectors and model selection can bias it. Bootstrap is conditional on predictions, not a private-leaderboard guarantee."})


if __name__ == "__main__":
    main()
