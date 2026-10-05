"""Prepare, train and score the matched shuffled-short-word pilot."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "model114"))
import pilot as previous
import data as shuffled
from road_htr import stackmix, vlm_cli, vlm_data
from road_htr.stackmix import sha256, save_json

OUT = ROOT / "outputs/model116"
CONFIG = ROOT / "model116/config.json"
PROTOCOL = OUT / "protocol.json"


def check():
    previous.check()
    p = json.loads(PROTOCOL.read_text())
    for path, expected in p["hashes"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Frozen Model116 input/code changed: {path}")
    for name, expected in p["versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Runtime changed: {name}")
    return p


def prepare():
    old = previous.check()
    if PROTOCOL.exists():
        raise RuntimeError("Model116 protocol already exists")
    baseline = json.loads((ROOT / "model114/config.json").read_text())
    candidate = json.loads(CONFIG.read_text())
    candidate.pop("synthetic_manifest")
    candidate["output_dir"] = baseline["output_dir"]
    if candidate != baseline:
        raise ValueError("Settings differ from matched Model114 control")
    shuffled.generate()
    table, eligible = shuffled.eligible_rows()
    rows = shuffled.validate_synthetic(shuffled.OUT / "manifest.json", shuffled.TRUTH,
                                      0, eligible, table.target_column)
    # Hook is process-local; old shared code and old manifests are not edited.
    stackmix.validate_synthetic = shuffled.validate_synthetic
    cfg = json.loads(CONFIG.read_text())
    dataset = vlm_data.build_vlm_dataset(
        shuffled.TRUTH, ROOT / cfg["image_dir"], prompt=cfg["prompt"], fold=0,
        fold_mode="train", exclude_ids=vlm_data.read_id_file(ROOT / cfg["exclude_ids"]),
        image_variant="full", augmentation=cfg["augmentation"],
        synthetic_manifest=shuffled.OUT / "manifest.json")
    if len(dataset) != 4095 or dataset[3276]["id"] != rows[0]["id"]:
        raise ValueError("Synthetic dataset integration failed")
    paths = ["model116/config.json", "model116/data.py", "model116/experiment.py",
             "model116/test_model116.py", "scripts/run_model116.py",
             "outputs/model116/synthetic/manifest.json", "outputs/model116/synthetic/labels.csv",
             "outputs/model107/fold0/bank/bank.json", "outputs/model114/protocol.json",
             "outputs/model114/report.json", "outputs/model114/fold0_oof_step400.csv",
             "outputs/model114/fold0_oof_step400.manifest.json",
             "outputs/model114/replacement_ensemble_fold0.csv", "model114/pilot.py"]
    save_json(PROTOCOL, {
        "hashes": {p: sha256(ROOT / p) for p in paths}, "versions": old["versions"],
        "train_ids": old["train_ids"], "validation_ids": old["validation_ids"],
        "sampling_seed": shuffled.SEED, "training_seed": cfg["seed"],
        "real_rows": 3276, "synthetic_rows": 819, "validation_rows": 819,
        "standalone_gain_gate": 0.001, "ensemble_gain_gate": 0.0005,
        "matched_control": "model114", "ensemble_baseline": "model112",
        "ensemble_rule": old["ensemble_rule"], "limitations": old["limitations"]})
    check()
    print(json.dumps(json.loads((shuffled.OUT / "summary.json").read_text()), indent=2), flush=True)
    print("Fold-isolation, donor validation and real/synthetic dataset loading passed", flush=True)


def evaluate():
    p = check()
    path = OUT / "fold0_oof_step400.csv"
    receipt = json.loads(path.with_suffix(".manifest.json").read_text())
    adapter = OUT / "fold0/best_adapter"
    if (receipt["fold"] != 0 or receipt["rows"] != 819 or receipt["input_sha256"] != sha256(shuffled.TRUTH)
            or receipt["output_sha256"] != sha256(path) or receipt["config_sha256"] != sha256(CONFIG)
            or Path(receipt["adapter"]).resolve() != adapter.resolve()):
        raise ValueError("Inference receipt mismatch")
    for name, expected in receipt["adapter_files_sha256"].items():
        if sha256(adapter / name) != expected:
            raise ValueError("Inference adapter checksum mismatch")
    checkpoint = OUT / "fold0/checkpoint-400"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state["global_step"] != 400 or sha256(adapter / "adapter_model.safetensors") != sha256(checkpoint / "adapter_model.safetensors"):
        raise ValueError("Expected final step400 adapter")
    run = json.loads((OUT / "fold0/run_manifest.json").read_text())
    if (run["real_train_rows"], run["synthetic_rows"], run["validation_rows"]) != (3276, 819, 819):
        raise ValueError("Wrong training/validation mix")
    if run["synthetic_manifest_sha256"] != sha256(shuffled.OUT / "manifest.json"):
        raise ValueError("Training used a different synthetic dataset")
    table, truth = previous.canonical()
    ids = p["validation_ids"]
    candidate = previous.read_predictions(path)
    matched = previous.read_predictions(ROOT / "outputs/model114/fold0_oof_step400.csv")
    if set(candidate) != set(ids) or set(matched) != set(ids):
        raise ValueError("Prediction IDs differ from complete heldout fold")
    source, _ = previous.sources(truth)
    anchor = previous.read_predictions(ROOT / "outputs/model104/full_oof/crossfit_predictions.csv")
    old = previous.read_predictions(ROOT / "outputs/model112/diagnostic_family_word_oof.csv")
    matched_ensemble = previous.read_predictions(ROOT / "outputs/model114/replacement_ensemble_fold0.csv")
    replacement = {}
    for i in ids:
        donors = {n: source[n][i] for n in previous.NAMES}
        if previous.fuse(anchor[i], donors, True, False) != old[i]:
            raise ValueError("Model112 reproduction mismatch")
        donors["model92"] = candidate[i]
        replacement[i] = previous.fuse(anchor[i], donors, True, False) if candidate[i].strip() else anchor[i]
    refs = [truth[i][table.target_column] for i in ids]
    scores = {name: previous.competition_error(refs, [values[i] for i in ids]) for name, values in
              (("model116", candidate), ("matched_model114", matched), ("model112", old),
               ("replacement_ensemble", replacement), ("model114_replacement_ensemble", matched_ensemble))}
    standalone_gain = scores["model116"]["multi_score"] - scores["matched_model114"]["multi_score"]
    ensemble_gain = scores["replacement_ensemble"]["multi_score"] - scores["model112"]["multi_score"]
    if (OUT / "report.json").exists():
        raise RuntimeError("Refusing to overwrite completed result")
    with (OUT / "replacement_ensemble_fold0.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "prediction"])
        writer.writerows((i, replacement[i]) for i in ids)
    report = {"model": 116, "fold": 0, "rows": 819, "scores": scores,
              "standalone_gain_vs_matched_control": standalone_gain, "ensemble_gain": ensemble_gain,
              "advance_to_confirmation_screen": standalone_gain >= p["standalone_gain_gate"] and ensemble_gain >= p["ensemble_gain_gate"],
              "empty_predictions": sum(not candidate[i].strip() for i in ids),
              "changed_ensemble_lines": sum(replacement[i] != old[i] for i in ids),
              "protocol_sha256": sha256(PROTOCOL), "prediction_sha256": sha256(path),
              "submission_ready": False, "limitations": p["limitations"]}
    save_json(OUT / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "check", "train", "infer", "evaluate"))
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
        return
    check()
    if action == "train":
        if (OUT / "fold0").exists():
            raise RuntimeError("Refusing to overwrite training outputs")
        stackmix.validate_synthetic = shuffled.validate_synthetic
        vlm_cli.train(CONFIG)
    elif action == "infer":
        path = OUT / "fold0_oof_step400.csv"
        if path.exists():
            raise RuntimeError("Refusing to overwrite predictions")
        vlm_cli.infer(CONFIG, shuffled.TRUTH, path, OUT / "fold0/best_adapter", fold=0)
    elif action == "evaluate":
        evaluate()


if __name__ == "__main__":
    main()
