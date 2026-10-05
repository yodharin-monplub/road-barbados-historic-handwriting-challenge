#!/usr/bin/env python3
"""Model63: frequency-gated outer-fold context repair after Model59."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from model54.restore_final_tokens import (  # noqa: E402
    _atomic_json, _prediction_map, _write_changes, _write_predictions, resolve, sha256,
)
from model60.crossfold_context_repair import build_outer_fold_models, transform  # noqa: E402
from road_htr.contextual_corrector import TrainingLanguageModel  # noqa: E402
from road_htr.metrics import competition_error  # noqa: E402
from road_htr.schema import read_table  # noqa: E402
from road_htr.submission import write_submission  # noqa: E402


OUTPUT_ROOT = (ROOT / "outputs" / "model63").resolve()


class Model63GateError(RuntimeError):
    pass


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve(path)
    config = json.loads(resolved.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "model_number": 63,
        "source_model": "model59",
        "fit_policy": "outer_fold_excluded_truth_context_model_with_row_gate",
        "minimum_frequency": 1,
        "context_weight": 16.0,
        "distance_penalty": 32.0,
        "fallback_frequency": 8,
        "two_edit_min_length": 6,
        "seed_minimum_frequency": 3,
        "seed_ambiguity_ratio": 1.0,
        "minimum_candidate_frequency": 2,
        "row_gate": "accept_only_if_every_proposed_change_meets_minimum_candidate_frequency",
        "output_root": "outputs/model63",
    }
    if config != expected:
        raise Model63GateError("Model63 config is immutable")
    return config, resolved


def _require_new_output(path: str | Path) -> Path:
    output = resolve(path)
    if output == OUTPUT_ROOT or OUTPUT_ROOT not in output.parents:
        raise Model63GateError("output directory must be below outputs/model63")
    if output.exists():
        raise Model63GateError("output directory must be new")
    return output


def apply_gate(
    baseline: Sequence[str],
    proposals: Sequence[str],
    operations: Sequence[Sequence[Mapping[str, Any]]],
    minimum_candidate_frequency: int,
) -> tuple[list[str], list[bool]]:
    if not (len(baseline) == len(proposals) == len(operations)):
        raise Model63GateError("gate input length mismatch")
    accepted = [
        bool(row_operations)
        and all(int(operation["frequency"]) >= minimum_candidate_frequency for operation in row_operations)
        for row_operations in operations
    ]
    return [
        proposal if keep else source
        for source, proposal, keep in zip(baseline, proposals, accepted, strict=True)
    ], accepted


def _run(
    baseline: Sequence[str],
    models: Mapping[int, TrainingLanguageModel],
    folds: Sequence[int],
    config: Mapping[str, Any],
) -> tuple[list[str], list[list[dict[str, Any]]], list[bool]]:
    proposals, operations = transform(baseline, models, folds, config)
    values, accepted = apply_gate(
        baseline, proposals, operations, int(config["minimum_candidate_frequency"])
    )
    return values, operations, accepted


def evaluate_oof(config_path: str | Path, truth_path: str | Path, predictions_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config, config_resolved = load_config(config_path)
    truth = read_table(resolve(truth_path), require_target=True)
    if truth.target_column is None or "fold" not in truth.fieldnames:
        raise Model63GateError("OOF truth must contain target and fold")
    folds = [int(row["fold"]) for row in truth.rows]
    if Counter(folds) != Counter({fold: 819 for fold in range(5)}):
        raise Model63GateError("truth is not the canonical balanced five-fold table")
    ids = [row[truth.id_column].strip() for row in truth.rows]
    references = [row[truth.target_column] for row in truth.rows]
    source, source_receipt = _prediction_map(predictions_path, ids)
    baseline = [source[identifier] for identifier in ids]
    models = build_outer_fold_models(references, folds)
    values, operations, accepted = _run(baseline, models, folds, config)
    changes = [
        {"ID": identifier, "fold": fold, "before": before, "after": after, "operations": row_operations}
        for identifier, fold, before, after, row_operations, keep in zip(
            ids, folds, baseline, values, operations, accepted, strict=True
        ) if keep
    ]
    destination = _require_new_output(output_dir)
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
            "baseline": before, "model63": after,
            "multi_score_gain": after["multi_score"] - before["multi_score"],
            "fit_rows": len(ids) - len(positions),
            "fit_folds": [fold for fold in range(5) if fold != heldout],
        }
    baseline_score, score = competition_error(references, baseline), competition_error(references, values)
    report = {
        "status": "complete_five_fold_oof", "model_number": 63, "rows": len(ids),
        "proposed_rows": sum(bool(item) for item in operations), "accepted_rows": len(changes),
        "rejected_rows": sum(bool(item) and not keep for item, keep in zip(operations, accepted, strict=True)),
        "accepted_tokens": sum(len(row["operations"]) for row in changes),
        "fit_uses_heldout_fold": False,
        "config": {"path": str(config_resolved), "sha256": sha256(config_resolved)},
        "implementation_sha256": sha256(Path(__file__).resolve()), "source": source_receipt,
        "baseline_score": baseline_score, "score": score,
        "multi_score_gain": score["multi_score"] - baseline_score["multi_score"],
        "fold_scores": fold_scores,
        "all_folds_improve": all(fold_scores[str(f)]["multi_score_gain"] > 0 for f in range(5)),
        "artifacts": {"predictions": str(predictions_output), "predictions_sha256": sha256(predictions_output), "changes": str(changes_output), "changes_sha256": sha256(changes_output)},
    }
    _atomic_json(destination / "report.json", report)
    return report


def apply_test(config_path: str | Path, truth_path: str | Path, test_path: str | Path, predictions_path: str | Path, output_dir: str | Path, sample_path: str | Path | None = None, submission_path: str | Path | None = None) -> dict[str, Any]:
    config, config_resolved = load_config(config_path)
    truth, test = read_table(resolve(truth_path), require_target=True), read_table(resolve(test_path))
    assert truth.target_column is not None
    references = [row[truth.target_column] for row in truth.rows]
    ids = [row[test.id_column].strip() for row in test.rows]
    source, source_receipt = _prediction_map(predictions_path, ids)
    baseline = [source[identifier] for identifier in ids]
    model = TrainingLanguageModel(references)
    values, operations, accepted = _run(baseline, {0: model}, [0] * len(ids), config)
    changes = [
        {"ID": identifier, "before": before, "after": after, "operations": row_operations}
        for identifier, before, after, row_operations, keep in zip(ids, baseline, values, operations, accepted, strict=True) if keep
    ]
    if any(not value.strip() for value in values):
        raise Model63GateError("repair produced a blank test prediction")
    destination = _require_new_output(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    predictions_output, changes_output = destination / "test.csv", destination / "changes.jsonl"
    _write_predictions(predictions_output, test.id_column, ids, values)
    _write_changes(changes_output, changes)
    if submission_path is not None:
        if sample_path is None: raise Model63GateError("sample is required with submission")
        write_submission(resolve(sample_path), resolve(submission_path), dict(zip(ids, values, strict=True)))
    report = {
        "status": "complete_test_candidate_no_leaderboard_score_claimed", "model_number": 63,
        "rows": len(ids), "proposed_rows": sum(bool(item) for item in operations), "accepted_rows": len(changes),
        "rejected_rows": sum(bool(item) and not keep for item, keep in zip(operations, accepted, strict=True)),
        "accepted_tokens": sum(len(row["operations"]) for row in changes), "language_model_fit_rows": len(references),
        "language_model_fit_uses_test_labels": False,
        "config": {"path": str(config_resolved), "sha256": sha256(config_resolved)},
        "implementation_sha256": sha256(Path(__file__).resolve()), "source": source_receipt,
        "artifacts": {"predictions": str(predictions_output), "predictions_sha256": sha256(predictions_output), "changes": str(changes_output), "changes_sha256": sha256(changes_output), "submission": str(resolve(submission_path)) if submission_path else None, "submission_sha256": sha256(resolve(submission_path)) if submission_path else None},
    }
    _atomic_json(destination / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Model63 gated context repair")
    subparsers = parser.add_subparsers(dest="command", required=True)
    oof = subparsers.add_parser("oof"); oof.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json")); oof.add_argument("--truth", type=Path, default=ROOT / "data/processed/train_folds.csv"); oof.add_argument("--predictions", type=Path, required=True); oof.add_argument("--output-dir", type=Path, required=True)
    apply_parser = subparsers.add_parser("apply"); apply_parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json")); apply_parser.add_argument("--truth", type=Path, default=ROOT / "data/processed/train_folds.csv"); apply_parser.add_argument("--test", type=Path, default=ROOT / "data/raw/Test.csv"); apply_parser.add_argument("--predictions", type=Path, required=True); apply_parser.add_argument("--output-dir", type=Path, required=True); apply_parser.add_argument("--sample", type=Path); apply_parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    report = evaluate_oof(args.config, args.truth, args.predictions, args.output_dir) if args.command == "oof" else apply_test(args.config, args.truth, args.test, args.predictions, args.output_dir, args.sample, args.submission)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
