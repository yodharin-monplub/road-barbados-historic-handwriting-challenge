"""Matched rank16 pilot; reuse verified training and ensemble evaluation."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model114"))
import pilot

OUT = ROOT / "outputs/model117"
CONFIG = ROOT / "model117/config.json"
PROTOCOL = OUT / "protocol.json"


def prepare():
    original = pilot.check()
    if PROTOCOL.exists():
        raise RuntimeError("Refusing to overwrite protocol")
    baseline = json.loads((ROOT / "model114/config.json").read_text())
    candidate = json.loads(CONFIG.read_text())
    if candidate["lora"]["rank"] != 16 or candidate["lora"]["alpha"] != 32:
        raise ValueError("Expected rank16 alpha32")
    candidate["lora"]["rank"] = baseline["lora"]["rank"]
    candidate["lora"]["alpha"] = baseline["lora"]["alpha"]
    candidate["output_dir"] = baseline["output_dir"]
    if candidate != baseline:
        raise ValueError("Unexpected settings change versus matched Model114")
    protocol = copy.deepcopy(original)
    paths = ["model117/config.json", "model117/experiment.py", "scripts/run_model117.py",
             "model114/pilot.py", "outputs/model114/protocol.json",
             "outputs/model114/fold0_oof_step400.csv", "outputs/model114/fold0_oof_step400.manifest.json",
             "outputs/model114/report.json"]
    protocol["hashes"].update({p: pilot.sha256(ROOT / p) for p in paths})
    protocol.update({"models": {"117": 20260914}, "matched_control": "model114",
                     "standalone_gain_gate": 0.001,
                     "intervention": "LoRA rank64->16, alpha128->32; all other settings unchanged"})
    pilot.save_json(PROTOCOL, protocol)
    print("Rank16 protocol frozen; same seed, data, runtime and400-step budget as Model114", flush=True)


def evaluate():
    pilot.evaluate(117)
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    protocol = pilot.check()
    table, truth = pilot.canonical()
    ids = protocol["validation_ids"]
    matched = pilot.read_predictions(ROOT / "outputs/model114/fold0_oof_step400.csv")
    if set(matched) != set(ids):
        raise ValueError("Matched control does not cover heldout IDs")
    score = pilot.competition_error([truth[i][table.target_column] for i in ids], [matched[i] for i in ids])
    report["scores"]["matched_model114"] = score
    gain = report["scores"]["candidate"]["multi_score"] - score["multi_score"]
    report["standalone_gain_vs_matched_control"] = gain
    report["advance_to_confirmation_screen"] = bool(
        report["ensemble_gain"] >= protocol["confirmation_screen_gain"] and gain >= protocol["standalone_gain_gate"])
    pilot.save_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "check", "train", "infer", "evaluate"))
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    pilot.PROTOCOL = PROTOCOL
    pilot.check()
    if action == "train":
        if (OUT / "fold0").exists():
            raise RuntimeError("Training output already exists")
        pilot.train(CONFIG)
    elif action == "infer":
        path = OUT / "fold0_oof_step400.csv"
        if path.exists():
            raise RuntimeError("Predictions already exist")
        pilot.infer(CONFIG, pilot.TRUTH, path, adapter_path=OUT / "fold0/best_adapter", fold=0)
    elif action == "evaluate":
        evaluate()


if __name__ == "__main__":
    main()
