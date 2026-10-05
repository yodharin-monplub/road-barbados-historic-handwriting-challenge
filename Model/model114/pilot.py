"""Prepare and validate the two fixed, local seed-diversity pilots."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "model112"))
from road_htr.schema import read_table, predictions_by_id
from road_htr.stackmix import sha256, save_json
from road_htr.metrics import competition_error
from road_htr.vlm_data import read_id_file
from road_htr.vlm_cli import train, infer
from run import sources, read as read_predictions
from fusion import fuse, NAMES

OUT = ROOT / "outputs/model114"
TRUTH = ROOT / "data/processed/train_folds.csv"
PROTOCOL = OUT / "protocol.json"


def canonical():
    table = read_table(TRUTH, require_target=True)
    excluded = read_id_file(ROOT / "config/known_bad_train_ids.txt")
    truth = {r[table.id_column]: r for r in table.rows if r[table.id_column] not in excluded}
    assert len(truth) == 4095 and sum(int(r["fold"]) == 0 for r in truth.values()) == 819
    return table, truth


def prepare():
    if PROTOCOL.exists():
        raise RuntimeError("Protocol already frozen; use check rather than overwrite")
    table, truth = canonical()
    _, ledger = sources(truth)
    paths = ["model114/pilot.py", "scripts/run_seed_pilots.py",
             "model114/config.json", "model115/config.json", "model114/smoke_config.json",
             "src/road_htr/vlm_cli.py", "src/road_htr/vlm_data.py",
             "src/road_htr/schema.py", "src/road_htr/metrics.py", "src/road_htr/stackmix.py",
             "src/road_htr/preprocess.py", "src/road_htr/word_rover.py",
             "model112/fusion.py", "model112/run.py", "model99/evaluate.py",
             "data/processed/train_folds.csv", "config/known_bad_train_ids.txt",
             "outputs/model104/full_oof/crossfit_predictions.csv",
             "outputs/model112/diagnostic_family_word_oof.csv"]
    paths += [entry["path"] for entry in ledger.values()]
    # These training-only examples include the earlier largest-image/longest-text
    # memory cases; the smoke's three validation images stay held out.
    required = ["T85PtmjHWp5dZ2ZZ", "gP7kGWfDFYMbZ8nO", "t0UrASljcgzvBAnO"]
    train_ids = sorted(i for i, r in truth.items() if int(r["fold"]) != 0)
    val_ids = sorted(i for i, r in truth.items() if int(r["fold"]) == 0)
    if not set(required) <= set(train_ids):
        raise ValueError("Smoke stress rows must all be training rows")
    selected = required + [i for i in train_ids if i not in required][:29] + val_ids[:3]
    smoke = OUT / "trainer_smoke/rows.csv"
    smoke.parent.mkdir(parents=True, exist_ok=True)
    if smoke.exists():
        raise RuntimeError("Refusing to overwrite smoke inputs")
    with smoke.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table.fieldnames)
        writer.writeheader()
        writer.writerows(truth[i] for i in selected)
    paths.append(str(smoke.relative_to(ROOT)))
    versions = {p: importlib.metadata.version(p) for p in
                ("torch", "transformers", "trl", "datasets", "peft", "accelerate", "safetensors")}
    save_json(PROTOCOL, {
        "schema_version": 1, "fold": 0, "models": {"114": 20260914, "115": 20260915},
        "hashes": {p: sha256(ROOT / p) for p in paths}, "versions": versions,
        "train_ids": train_ids, "validation_ids": val_ids,
        "ensemble_rule": "Replace only Model92 with the new seed in Model112 family_word; fixed original Model104 anchor",
        "confirmation_screen_gain": 0.0005,
        "limitations": "Development folds have prior selection history; no unbiased score or automatic submission."})
    check()
    print("Frozen both seed pilots: 3276 training / 819 validation rows", flush=True)


def check():
    protocol = json.loads(PROTOCOL.read_text())
    for path, expected in protocol["hashes"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"Frozen input/code changed: {path}")
    for package, expected in protocol["versions"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Runtime changed: {package}")
    return protocol


def evaluate(number):
    protocol = check()
    table, truth = canonical()
    ids = protocol["validation_ids"]
    path = ROOT / f"outputs/model{number}/fold0_oof_step400.csv"
    receipt = json.loads(path.with_suffix(".manifest.json").read_text())
    adapter = ROOT / f"outputs/model{number}/fold0/best_adapter"
    if (receipt["fold"] != 0 or receipt["rows"] != len(ids)
            or receipt["input_sha256"] != sha256(TRUTH)
            or receipt["output_sha256"] != sha256(path)
            or receipt["config_sha256"] != sha256(ROOT / f"model{number}/config.json")
            or Path(receipt["adapter"]).resolve() != adapter.resolve()):
        raise ValueError("Prediction provenance mismatch")
    for name, expected in receipt["adapter_files_sha256"].items():
        if sha256(adapter / name) != expected:
            raise ValueError("Adapter checksum mismatch")
    state = json.loads((adapter.parent / "checkpoint-400/trainer_state.json").read_text())
    if state["global_step"] != 400:
        raise ValueError("Expected final step400 adapter")
    if sha256(adapter / "adapter_model.safetensors") != sha256(adapter.parent / "checkpoint-400/adapter_model.safetensors"):
        raise ValueError("Saved adapter is not checkpoint400")
    candidate = predictions_by_id(read_table(path), "prediction")
    if set(candidate) != set(ids):
        raise ValueError("Candidate must cover the complete held-out fold")
    source, _ = sources(truth)
    anchor = read_predictions(ROOT / "outputs/model104/full_oof/crossfit_predictions.csv")
    old = read_predictions(ROOT / "outputs/model112/diagnostic_family_word_oof.csv")
    replacement = {}
    for i in ids:
        original = {n: source[n][i] for n in NAMES}
        if fuse(anchor[i], original, True, False) != old[i]:
            raise ValueError("Cannot reproduce Model112 baseline")
        original["model92"] = candidate[i]
        replacement[i] = fuse(anchor[i], original, True, False) if candidate[i].strip() else anchor[i]
    refs = [truth[i][table.target_column] for i in ids]
    score = lambda values: competition_error(refs, [values[i] for i in ids])
    scores = {"candidate": score(candidate), "model92": score(source["model92"]),
              "model112": score(old), "replacement_ensemble": score(replacement)}
    gain = scores["replacement_ensemble"]["multi_score"] - scores["model112"]["multi_score"]
    report_path = path.parent / "report.json"
    if report_path.exists():
        raise RuntimeError("Refusing to overwrite completed evaluation")
    with (path.parent / "replacement_ensemble_fold0.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "prediction"])
        writer.writerows((i, replacement[i]) for i in ids)
    save_json(report_path, {
        "model": number, "fold": 0, "rows": len(ids), "scores": scores,
        "ensemble_gain": gain, "advance_to_confirmation_screen": gain >= protocol["confirmation_screen_gain"],
        "changed_ensemble_lines": sum(replacement[i] != old[i] for i in ids),
        "different_from_model92_lines": sum(candidate[i] != source["model92"][i] for i in ids),
        "empty_predictions": sum(not candidate[i].strip() for i in ids),
        "protocol_sha256": sha256(PROTOCOL), "prediction_sha256": sha256(path),
        "submission_ready": False, "limitations": protocol["limitations"]})
    print(json.dumps(json.loads(report_path.read_text()), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "check", "smoke", "train", "infer", "evaluate"])
    parser.add_argument("--model", type=int, choices=[114, 115], default=114)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
        return
    check()
    config = ROOT / f"model{args.model}/config.json"
    output = ROOT / f"outputs/model{args.model}"
    if args.action in ("train", "smoke"):
        target = OUT / "trainer_smoke/run" if args.action == "smoke" else output / "fold0"
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite training directory: {target}")
        train(ROOT / "model114/smoke_config.json" if args.action == "smoke" else config)
    elif args.action == "infer":
        target = output / "fold0_oof_step400.csv"
        if target.exists():
            raise RuntimeError("Refusing to overwrite predictions")
        infer(config, TRUTH, target, adapter_path=output / "fold0/best_adapter", fold=0)
    elif args.action == "evaluate":
        evaluate(args.model)


if __name__ == "__main__":
    main()
