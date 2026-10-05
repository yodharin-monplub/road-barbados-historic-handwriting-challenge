#!/usr/bin/env python3
"""Model69: select each fold's checkpoint using only other-fold OOF scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from road_htr.metrics import competition_error
from road_htr.schema import predictions_by_id, read_table

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (ROOT / "outputs/model69").resolve()


class Model69Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "model_number": 69,
        "source_models": ["model22_fold0", "model36_folds1_to_4"],
        "eligible_epochs_zero_based": list(range(8)),
        "selection_policy": "leave_one_outer_fold_out_mean_exact_multi_score",
        "tie_break": "earliest_epoch",
        "deployment_policy": "maximum_all_fold_mean_exact_multi_score_then_train_all_data",
        "output_root": "outputs/model69",
    }
    if config != expected:
        raise Model69Error("Model69 config is immutable")
    return config


def select_epoch(scores_by_fold: Mapping[int, Mapping[int, float]], fit_folds: Sequence[int], epochs: Sequence[int]) -> tuple[int, dict[int, float]]:
    means = {
        epoch: sum(float(scores_by_fold[fold][epoch]) for fold in fit_folds) / len(fit_folds)
        for epoch in epochs
    }
    selected = min(epochs, key=lambda epoch: (-means[epoch], epoch))
    return selected, means


def parse_fold_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use FOLD=PATH")
    raw_fold, raw_path = value.split("=", 1)
    try:
        fold = int(raw_fold)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fold must be an integer") from exc
    return fold, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--truth", type=Path, default=ROOT / "data/processed/train_folds.csv")
    parser.add_argument("--fold-leaderboard", action="append", type=parse_fold_path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    truth_path = args.truth.resolve()
    truth = read_table(truth_path, require_target=True)
    if truth.target_column is None or "fold" not in truth.fieldnames:
        raise Model69Error("truth must contain target and fold")
    supplied = dict(args.fold_leaderboard)
    if set(supplied) != set(range(5)) or len(args.fold_leaderboard) != 5:
        raise Model69Error("exactly one leaderboard for each fold 0..4 is required")
    epochs = [int(value) for value in config["eligible_epochs_zero_based"]]
    truth_hash = sha256(truth_path)
    scores: dict[int, dict[int, float]] = {}
    prediction_paths: dict[int, dict[int, Path]] = {}
    leaderboard_receipts: dict[str, Any] = {}
    for fold, unresolved in sorted(supplied.items()):
        path = unresolved.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("fold", -1)) != fold or payload.get("input_sha256") != truth_hash:
            raise Model69Error(f"leaderboard provenance mismatch for fold {fold}")
        by_epoch = {int(row["epoch"]): row for row in payload.get("results", [])}
        if any(epoch not in by_epoch for epoch in epochs):
            raise Model69Error(f"fold {fold} lacks an eligible checkpoint")
        fold_rows = [row for row in truth.rows if int(row["fold"]) == fold]
        fold_ids = [row[truth.id_column].strip() for row in fold_rows]
        references = [row[truth.target_column] for row in fold_rows]
        scores[fold] = {}
        prediction_paths[fold] = {}
        for epoch in epochs:
            record = by_epoch[epoch]
            prediction_path = Path(record["predictions"]).resolve()
            if sha256(prediction_path) != record["predictions_sha256"]:
                raise Model69Error(f"prediction hash mismatch for fold {fold}, epoch {epoch}")
            table = read_table(prediction_path)
            column = table.target_column or next(name for name in table.fieldnames if name != table.id_column)
            values = predictions_by_id(table, column)
            if set(values) != set(fold_ids):
                raise Model69Error(f"prediction IDs mismatch for fold {fold}, epoch {epoch}")
            metric = competition_error(references, [values[identifier] for identifier in fold_ids])
            if abs(metric["multi_score"] - float(record["multi_score"])) > 1e-12:
                raise Model69Error(f"score mismatch for fold {fold}, epoch {epoch}")
            scores[fold][epoch] = metric["multi_score"]
            prediction_paths[fold][epoch] = prediction_path
        leaderboard_receipts[str(fold)] = {"path": str(path), "sha256": sha256(path)}

    selected_by_fold: dict[int, int] = {}
    selection_details: dict[str, Any] = {}
    for heldout in range(5):
        fit_folds = [fold for fold in range(5) if fold != heldout]
        epoch, means = select_epoch(scores, fit_folds, epochs)
        selected_by_fold[heldout] = epoch
        selection_details[str(heldout)] = {
            "selected_epoch": epoch,
            "fit_folds": fit_folds,
            "heldout_fold_used_for_selection": False,
            "mean_scores_by_epoch": {str(key): value for key, value in means.items()},
        }
    deployment_epoch, deployment_means = select_epoch(scores, list(range(5)), epochs)

    destination = args.output_dir.resolve()
    if destination == OUTPUT_ROOT or OUTPUT_ROOT not in destination.parents or destination.exists():
        raise Model69Error("output directory must be new and below outputs/model69")
    destination.mkdir(parents=True)
    output_path = destination / "allfold_oof.csv"
    combined: dict[str, str] = {}
    selected_sources: dict[str, Any] = {}
    for fold in range(5):
        epoch = selected_by_fold[fold]
        path = prediction_paths[fold][epoch]
        table = read_table(path)
        column = table.target_column or next(name for name in table.fieldnames if name != table.id_column)
        combined.update(predictions_by_id(table, column))
        selected_sources[str(fold)] = {"epoch": epoch, "path": str(path), "sha256": sha256(path)}
    ids = [row[truth.id_column].strip() for row in truth.rows]
    references = [row[truth.target_column] for row in truth.rows]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[truth.id_column, "prediction"])
        writer.writeheader()
        writer.writerows({truth.id_column: identifier, "prediction": combined[identifier]} for identifier in ids)
    report = {
        "status": "complete_exploratory_crossfold_epoch_selection",
        "claim_scope": (
            "Each row is predicted by a model excluding its outer fold, and its "
            "fold score is excluded from epoch selection. However, models for "
            "the other folds were trained on this fold's labels. Selection is "
            "therefore not fully nested: this is exploratory OOF evidence, "
            "not an independent generalization estimate or promotion gate."
        ),
        "fully_nested_validation": False,
        "model_number": 69,
        "rows": len(ids),
        "score": competition_error(references, [combined[identifier] for identifier in ids]),
        "selected_epoch_by_fold": {str(key): value for key, value in selected_by_fold.items()},
        "selection_details": selection_details,
        "deployment_epoch": deployment_epoch,
        "deployment_mean_scores_by_epoch": {str(key): value for key, value in deployment_means.items()},
        "leaderboards": leaderboard_receipts,
        "selected_sources": selected_sources,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "truth": {"path": str(truth_path), "sha256": truth_hash},
        "artifact": {"path": str(output_path), "sha256": sha256(output_path)},
    }
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
