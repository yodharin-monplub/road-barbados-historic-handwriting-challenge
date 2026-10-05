"""Small explicit helpers for the image-likelihood pilot."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import sha256, save_json
from road_htr.schema import read_table, predictions_by_id

OUT = ROOT / "outputs/model111"
TRUTH = ROOT / "data/processed/train_folds.csv"
JUDGES = {
    "model73": {"adapter": "outputs/model73/fold0/checkpoint-400", "config": "model73/config.yaml",
                "receipt": "outputs/model73/fold0_oof_step400.manifest.json",
                "predictions": "outputs/model73/fold0_oof_step400.csv"},
    "model107": {"adapter": "outputs/model107/train/fold0/best_adapter", "config": "outputs/model107/config_fold0.yaml",
                 "receipt": "outputs/model107/fold0_oof.manifest.json", "predictions": "outputs/model107/fold0_oof.csv"}}


def predictions(path):
    table = read_table(ROOT / path)
    return dict(predictions_by_id(table, "prediction" if "prediction" in table.fieldnames else table.target_column))


def check_judge(name):
    info = JUDGES[name]
    receipt = json.loads((ROOT / info["receipt"]).read_text())
    if receipt["fold"] != 0 or receipt["rows"] != 819 or receipt["input_sha256"] != sha256(TRUTH):
        raise ValueError("Judge receipt has wrong fold/labels")
    if receipt["output_sha256"] != sha256(ROOT / info["predictions"]):
        raise ValueError("Candidate prediction checksum mismatch")
    if receipt["config_sha256"] != sha256(ROOT / info["config"]):
        raise ValueError("Original judge config checksum mismatch")
    verified = {}
    for filename, expected in receipt["adapter_files_sha256"].items():
        if filename in ("optimizer.pt", "rng_state.pth", "scheduler.pt", "trainer_state.json", "training_args.bin", "README.md"):
            continue
        actual = sha256(ROOT / info["adapter"] / filename)
        if actual != expected: raise ValueError(f"Judge inference asset checksum mismatch: {filename}")
        verified[filename] = actual
    if "adapter_model.safetensors" not in verified: raise ValueError("Missing judge weights")
    return {**info, "adapter_files_sha256": verified, "receipt_sha256": sha256(ROOT / info["receipt"])}


def choose(primary, challenger, scores, conservative=False):
    """Scores are (primary NLL, challenger NLL) pairs for each judge."""
    if not scores: raise ValueError("No judges")
    import math
    if any(not math.isfinite(value) for pair in scores for value in pair):
        raise ValueError("Non-finite likelihood")
    switch = all(b < a for a, b in scores) if conservative else sum(b-a for a, b in scores) < 0
    return challenger if switch else primary
