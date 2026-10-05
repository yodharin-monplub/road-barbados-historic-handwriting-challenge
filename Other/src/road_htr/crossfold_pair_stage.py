"""Shared receipted runner for deterministic cross-fold prediction-pair stages."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from model54.restore_final_tokens import (
    _atomic_json,
    _prediction_map,
    _write_changes,
    _write_predictions,
    resolve,
    sha256,
)
from road_htr.metrics import competition_error
from road_htr.schema import read_table
from road_htr.submission import write_submission


class CrossfoldPairStageError(RuntimeError):
    pass


def _destination(path: str | Path, output_root: Path) -> Path:
    value = resolve(path)
    if value == output_root or output_root not in value.parents:
        raise CrossfoldPairStageError(f"output directory must be below {output_root}")
    if value.exists():
        raise CrossfoldPairStageError("output directory must be new")
    return value


def evaluate_pair_stage(
    *,
    model_number: int,
    config: Mapping[str, Any],
    config_path: Path,
    implementation_path: Path,
    output_root: Path,
    truth_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    build_maps: Callable[[Sequence[str], Sequence[str], Sequence[int], Mapping[str, Any]], Mapping[int, Mapping[Any, Any]]],
    repair: Callable[[str, Mapping[Any, Any]], tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    truth = read_table(resolve(truth_path), require_target=True)
    if truth.target_column is None or "fold" not in truth.fieldnames:
        raise CrossfoldPairStageError("OOF truth must contain target and fold")
    folds = [int(row["fold"]) for row in truth.rows]
    if Counter(folds) != Counter({fold: 819 for fold in range(5)}):
        raise CrossfoldPairStageError("truth is not the canonical balanced five-fold table")
    ids = [row[truth.id_column].strip() for row in truth.rows]
    references = [row[truth.target_column] for row in truth.rows]
    source_map, source_receipt = _prediction_map(predictions_path, ids)
    baseline = [source_map[identifier] for identifier in ids]
    maps = build_maps(baseline, references, folds, config)
    values: list[str] = []
    row_operations: list[list[dict[str, Any]]] = []
    for value, fold in zip(baseline, folds, strict=True):
        result, operations = repair(value, maps[fold])
        values.append(result)
        row_operations.append(operations)
    changes = [
        {"ID": identifier, "fold": fold, "before": before, "after": after, "operations": operations}
        for identifier, fold, before, after, operations in zip(
            ids, folds, baseline, values, row_operations, strict=True
        )
        if operations
    ]
    destination = _destination(output_dir, output_root)
    destination.mkdir(parents=True, exist_ok=False)
    predictions_output, changes_output = destination / "allfold_oof.csv", destination / "changes.jsonl"
    _write_predictions(predictions_output, truth.id_column, ids, values)
    _write_changes(changes_output, changes)
    fold_scores: dict[str, Any] = {}
    for heldout in range(5):
        positions = [index for index, fold in enumerate(folds) if fold == heldout]
        before = competition_error([references[i] for i in positions], [baseline[i] for i in positions])
        after = competition_error([references[i] for i in positions], [values[i] for i in positions])
        fold_scores[str(heldout)] = {
            "baseline": before,
            f"model{model_number}": after,
            "multi_score_gain": after["multi_score"] - before["multi_score"],
            "fit_rows": len(ids) - len(positions),
            "fit_folds": [fold for fold in range(5) if fold != heldout],
            "mapping_size": len(maps[heldout]),
        }
    baseline_score, score = competition_error(references, baseline), competition_error(references, values)
    report = {
        "status": "complete_five_fold_oof",
        "model_number": model_number,
        "rows": len(ids),
        "changed_rows": len(changes),
        "changed_operations": sum(len(row["operations"]) for row in changes),
        "fit_uses_heldout_fold": False,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "implementation_sha256": sha256(implementation_path),
        "shared_runner_sha256": sha256(Path(__file__).resolve()),
        "source": source_receipt,
        "baseline_score": baseline_score,
        "score": score,
        "multi_score_gain": score["multi_score"] - baseline_score["multi_score"],
        "fold_scores": fold_scores,
        "all_folds_improve": all(fold_scores[str(fold)]["multi_score_gain"] > 0 for fold in range(5)),
        "artifacts": {
            "predictions": str(predictions_output), "predictions_sha256": sha256(predictions_output),
            "changes": str(changes_output), "changes_sha256": sha256(changes_output),
        },
    }
    _atomic_json(destination / "report.json", report)
    return report


def apply_pair_stage(
    *,
    model_number: int,
    config: Mapping[str, Any],
    config_path: Path,
    implementation_path: Path,
    output_root: Path,
    truth_path: str | Path,
    training_predictions_path: str | Path,
    test_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    fit_mapping: Callable[[Sequence[str], Sequence[str], Mapping[str, Any]], Mapping[Any, Any]],
    repair: Callable[[str, Mapping[Any, Any]], tuple[str, list[dict[str, Any]]]],
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> dict[str, Any]:
    truth, test = read_table(resolve(truth_path), require_target=True), read_table(resolve(test_path))
    assert truth.target_column is not None
    train_ids = [row[truth.id_column].strip() for row in truth.rows]
    references = [row[truth.target_column] for row in truth.rows]
    training_predictions, training_receipt = _prediction_map(training_predictions_path, train_ids)
    mapping = fit_mapping([training_predictions[identifier] for identifier in train_ids], references, config)
    test_ids = [row[test.id_column].strip() for row in test.rows]
    source, source_receipt = _prediction_map(predictions_path, test_ids)
    values: list[str] = []
    changes: list[dict[str, Any]] = []
    for identifier in test_ids:
        before = source[identifier]
        result, operations = repair(before, mapping)
        values.append(result)
        if operations:
            changes.append({"ID": identifier, "before": before, "after": result, "operations": operations})
    if any(not value.strip() for value in values):
        raise CrossfoldPairStageError("repair produced a blank test prediction")
    destination = _destination(output_dir, output_root)
    destination.mkdir(parents=True, exist_ok=False)
    predictions_output, changes_output = destination / "test.csv", destination / "changes.jsonl"
    _write_predictions(predictions_output, test.id_column, test_ids, values)
    _write_changes(changes_output, changes)
    if submission_path is not None:
        if sample_path is None:
            raise CrossfoldPairStageError("sample is required with submission")
        write_submission(resolve(sample_path), resolve(submission_path), dict(zip(test_ids, values, strict=True)))
    report = {
        "status": "complete_test_candidate_no_leaderboard_score_claimed",
        "model_number": model_number,
        "rows": len(test_ids),
        "changed_rows": len(changes),
        "changed_operations": sum(len(row["operations"]) for row in changes),
        "mapping_fit_rows": len(train_ids),
        "mapping_fit_uses_only_oof_pairs": True,
        "mapping_size": len(mapping),
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "implementation_sha256": sha256(implementation_path),
        "shared_runner_sha256": sha256(Path(__file__).resolve()),
        "training_predictions": training_receipt,
        "source": source_receipt,
        "artifacts": {
            "predictions": str(predictions_output), "predictions_sha256": sha256(predictions_output),
            "changes": str(changes_output), "changes_sha256": sha256(changes_output),
            "submission": str(resolve(submission_path)) if submission_path else None,
            "submission_sha256": sha256(resolve(submission_path)) if submission_path else None,
        },
    }
    _atomic_json(destination / "report.json", report)
    return report
