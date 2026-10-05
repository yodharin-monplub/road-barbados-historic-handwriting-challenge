#!/usr/bin/env python3
"""Model62: exact-context recurring insertion repair after Model61."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from model54.restore_final_tokens import (  # noqa: E402
    _atomic_json,
    _prediction_map,
    _write_changes,
    _write_predictions,
    resolve,
    sha256,
)
from road_htr.metrics import competition_error  # noqa: E402
from road_htr.schema import read_table  # noqa: E402
from road_htr.submission import write_submission  # noqa: E402


OUTPUT_ROOT = (ROOT / "outputs" / "model62").resolve()
Boundary = tuple[str, str]
Span = tuple[str, ...]


class Model62GateError(RuntimeError):
    pass


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve(path)
    config = json.loads(resolved.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "model_number": 62,
        "source_model": "model61",
        "fit_policy": "outer_fold_excluded_exact_boundary_insertions",
        "maximum_inserted_tokens": 2,
        "minimum_support": 3,
        "minimum_precision": 0.5,
        "operation": "difflib_insert_blocks_with_exact_left_right_context",
        "output_root": "outputs/model62",
    }
    if config != expected:
        raise Model62GateError("Model62 config is immutable")
    return config, resolved


def _require_new_output(path: str | Path) -> Path:
    output = resolve(path)
    if output == OUTPUT_ROOT or OUTPUT_ROOT not in output.parents:
        raise Model62GateError("output directory must be below outputs/model62")
    if output.exists():
        raise Model62GateError("output directory must be new")
    return output


def fit_mapping(
    predictions: Sequence[str], references: Sequence[str], config: Mapping[str, Any]
) -> dict[Boundary, dict[str, Any]]:
    if len(predictions) != len(references):
        raise Model62GateError("prediction/reference length mismatch")
    maximum_inserted = int(config["maximum_inserted_tokens"])
    occurrences: Counter[Boundary] = Counter()
    outcomes: dict[Boundary, Counter[Span]] = defaultdict(Counter)
    for prediction, reference in zip(predictions, references, strict=True):
        source = prediction.split()
        target = reference.split()
        bounded = ["<s>", *source, "</s>"]
        occurrences.update(zip(bounded, bounded[1:], strict=False))
        matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
        for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
            target_length = target_end - target_start
            if tag != "insert" or not 1 <= target_length <= maximum_inserted:
                continue
            if source_start != source_end:
                raise Model62GateError("difflib insertion unexpectedly consumes source")
            boundary = (
                source[source_start - 1] if source_start else "<s>",
                source[source_start] if source_start < len(source) else "</s>",
            )
            outcomes[boundary][tuple(target[target_start:target_end])] += 1
    mapping: dict[Boundary, dict[str, Any]] = {}
    for boundary, counts in outcomes.items():
        target, support = max(counts.items(), key=lambda item: (item[1], item[0]))
        observations = occurrences[boundary]
        precision = support / observations
        if (
            support >= int(config["minimum_support"])
            and precision >= float(config["minimum_precision"])
        ):
            mapping[boundary] = {
                "target": target,
                "support": support,
                "observations": observations,
                "precision": precision,
            }
    return mapping


def build_outer_fold_maps(
    predictions: Sequence[str],
    references: Sequence[str],
    folds: Sequence[int],
    config: Mapping[str, Any],
) -> dict[int, dict[Boundary, dict[str, Any]]]:
    return {
        heldout: fit_mapping(
            [value for value, fold in zip(predictions, folds, strict=True) if fold != heldout],
            [value for value, fold in zip(references, folds, strict=True) if fold != heldout],
            config,
        )
        for heldout in sorted(set(folds))
    }


def repair_context_insertions(
    value: str, mapping: Mapping[Boundary, Mapping[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    tokens = value.split()
    bounded = ["<s>", *tokens, "</s>"]
    corrected: list[str] = []
    changes: list[dict[str, Any]] = []
    for index in range(len(bounded) - 1):
        if index:
            corrected.append(bounded[index])
        boundary = (bounded[index], bounded[index + 1])
        rule = mapping.get(boundary)
        if rule is None:
            continue
        target = tuple(str(token) for token in rule["target"])
        corrected.extend(target)
        changes.append(
            {
                "boundary_index": index,
                "left": boundary[0],
                "right": boundary[1],
                "inserted": list(target),
                "support": int(rule["support"]),
                "observations": int(rule["observations"]),
                "precision": float(rule["precision"]),
            }
        )
    return " ".join(corrected), changes


def evaluate_oof(
    config_path: str | Path,
    truth_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config, config_resolved = load_config(config_path)
    truth = read_table(resolve(truth_path), require_target=True)
    if truth.target_column is None or "fold" not in truth.fieldnames:
        raise Model62GateError("OOF truth must contain target and fold")
    folds = [int(row["fold"]) for row in truth.rows]
    if Counter(folds) != Counter({fold: 819 for fold in range(5)}):
        raise Model62GateError("truth is not the canonical balanced five-fold table")
    ids = [row[truth.id_column].strip() for row in truth.rows]
    references = [row[truth.target_column] for row in truth.rows]
    source_map, source_receipt = _prediction_map(predictions_path, ids)
    baseline = [source_map[identifier] for identifier in ids]
    maps = build_outer_fold_maps(baseline, references, folds, config)
    values: list[str] = []
    operations: list[list[dict[str, Any]]] = []
    for value, fold in zip(baseline, folds, strict=True):
        result, changes = repair_context_insertions(value, maps[fold])
        values.append(result)
        operations.append(changes)
    changes = [
        {
            "ID": identifier,
            "fold": fold,
            "before": before,
            "after": after,
            "operations": row_operations,
        }
        for identifier, fold, before, after, row_operations in zip(
            ids, folds, baseline, values, operations, strict=True
        )
        if row_operations
    ]
    destination = _require_new_output(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    predictions_output = destination / "allfold_oof.csv"
    changes_output = destination / "changes.jsonl"
    _write_predictions(predictions_output, truth.id_column, ids, values)
    _write_changes(changes_output, changes)
    fold_scores: dict[str, Any] = {}
    for heldout in range(5):
        positions = [index for index, fold in enumerate(folds) if fold == heldout]
        before = competition_error(
            [references[index] for index in positions],
            [baseline[index] for index in positions],
        )
        after = competition_error(
            [references[index] for index in positions],
            [values[index] for index in positions],
        )
        fold_scores[str(heldout)] = {
            "baseline": before,
            "model62": after,
            "multi_score_gain": after["multi_score"] - before["multi_score"],
            "fit_rows": len(ids) - len(positions),
            "fit_folds": [fold for fold in range(5) if fold != heldout],
            "mapping_size": len(maps[heldout]),
        }
    baseline_score = competition_error(references, baseline)
    score = competition_error(references, values)
    report = {
        "status": "complete_five_fold_oof",
        "model_number": 62,
        "rows": len(ids),
        "changed_rows": len(changes),
        "changed_operations": sum(len(row["operations"]) for row in changes),
        "fit_uses_heldout_fold": False,
        "config": {"path": str(config_resolved), "sha256": sha256(config_resolved)},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "source": source_receipt,
        "baseline_score": baseline_score,
        "score": score,
        "multi_score_gain": score["multi_score"] - baseline_score["multi_score"],
        "fold_scores": fold_scores,
        "all_folds_improve": all(
            fold_scores[str(fold)]["multi_score_gain"] > 0 for fold in range(5)
        ),
        "artifacts": {
            "predictions": str(predictions_output),
            "predictions_sha256": sha256(predictions_output),
            "changes": str(changes_output),
            "changes_sha256": sha256(changes_output),
        },
    }
    _atomic_json(destination / "report.json", report)
    return report


def apply_test(
    config_path: str | Path,
    truth_path: str | Path,
    training_predictions_path: str | Path,
    test_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> dict[str, Any]:
    config, config_resolved = load_config(config_path)
    truth = read_table(resolve(truth_path), require_target=True)
    test = read_table(resolve(test_path))
    assert truth.target_column is not None
    train_ids = [row[truth.id_column].strip() for row in truth.rows]
    references = [row[truth.target_column] for row in truth.rows]
    training_predictions, training_receipt = _prediction_map(
        training_predictions_path, train_ids
    )
    mapping = fit_mapping(
        [training_predictions[identifier] for identifier in train_ids], references, config
    )
    test_ids = [row[test.id_column].strip() for row in test.rows]
    source, source_receipt = _prediction_map(predictions_path, test_ids)
    values: list[str] = []
    changes: list[dict[str, Any]] = []
    for identifier in test_ids:
        before = source[identifier]
        result, operations = repair_context_insertions(before, mapping)
        values.append(result)
        if operations:
            changes.append(
                {"ID": identifier, "before": before, "after": result, "operations": operations}
            )
    if any(not value.strip() for value in values):
        raise Model62GateError("repair produced a blank test prediction")
    destination = _require_new_output(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    predictions_output = destination / "test.csv"
    changes_output = destination / "changes.jsonl"
    _write_predictions(predictions_output, test.id_column, test_ids, values)
    _write_changes(changes_output, changes)
    if submission_path is not None:
        if sample_path is None:
            raise Model62GateError("sample is required with submission")
        write_submission(
            resolve(sample_path),
            resolve(submission_path),
            dict(zip(test_ids, values, strict=True)),
        )
    report = {
        "status": "complete_test_candidate_no_leaderboard_score_claimed",
        "model_number": 62,
        "rows": len(test_ids),
        "changed_rows": len(changes),
        "changed_operations": sum(len(row["operations"]) for row in changes),
        "mapping_fit_rows": len(train_ids),
        "mapping_fit_uses_only_oof_pairs": True,
        "mapping_size": len(mapping),
        "config": {"path": str(config_resolved), "sha256": sha256(config_resolved)},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "training_predictions": training_receipt,
        "source": source_receipt,
        "artifacts": {
            "predictions": str(predictions_output),
            "predictions_sha256": sha256(predictions_output),
            "changes": str(changes_output),
            "changes_sha256": sha256(changes_output),
            "submission": str(resolve(submission_path)) if submission_path else None,
            "submission_sha256": sha256(resolve(submission_path)) if submission_path else None,
        },
    }
    _atomic_json(destination / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Model62 context insertion repair")
    subparsers = parser.add_subparsers(dest="command", required=True)
    oof = subparsers.add_parser("oof")
    oof.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    oof.add_argument("--truth", type=Path, default=ROOT / "data/processed/train_folds.csv")
    oof.add_argument("--predictions", type=Path, required=True)
    oof.add_argument("--output-dir", type=Path, required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    apply_parser.add_argument("--truth", type=Path, default=ROOT / "data/processed/train_folds.csv")
    apply_parser.add_argument("--training-predictions", type=Path, required=True)
    apply_parser.add_argument("--test", type=Path, default=ROOT / "data/raw/Test.csv")
    apply_parser.add_argument("--predictions", type=Path, required=True)
    apply_parser.add_argument("--output-dir", type=Path, required=True)
    apply_parser.add_argument("--sample", type=Path)
    apply_parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if args.command == "oof":
        report = evaluate_oof(args.config, args.truth, args.predictions, args.output_dir)
    else:
        report = apply_test(
            args.config,
            args.truth,
            args.training_predictions,
            args.test,
            args.predictions,
            args.output_dir,
            args.sample,
            args.submission,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
