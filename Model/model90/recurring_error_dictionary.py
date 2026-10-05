#!/usr/bin/env python3
"""Fold-safe recurring OCR-error dictionary on top of Model88."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from model89.train_only_dictionary import FastMetric, score_folds, write_predictions
from road_htr.schema import predictions_by_id, read_table
from road_htr.submission import write_submission
from road_htr.supervised_error_corrector import (
    SupervisedErrorModel,
    repair_supervised_errors,
)


@dataclass(frozen=True)
class Configuration:
    maximum_distance: int
    minimum_support: int
    minimum_precision: float
    minimum_length: int
    source_must_be_unknown: bool


def configurations() -> list[Configuration]:
    return [
        Configuration(distance, support, precision, length, unknown)
        for distance in (1, 2)
        for support in (2, 3, 4, 5, 8)
        for precision in (0.8, 0.9, 1.0)
        for length in (3, 4, 5, 6)
        for unknown in (True, False)
    ]


def mappings(model: SupervisedErrorModel, config: Configuration) -> dict:
    result = model.mappings(
        config.maximum_distance,
        config.minimum_support,
        config.minimum_precision,
        config.minimum_length,
    )
    if not config.source_must_be_unknown:
        return result
    truth_words = {
        target
        for outcomes in model.outcomes.values()
        for target in outcomes
    }
    return {source: value for source, value in result.items() if source not in truth_words}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()

    truth = read_table(args.truth, require_target=True)
    oof = read_table(args.oof)
    assert truth.target_column is not None
    oof_column = oof.target_column or next(c for c in oof.fieldnames if c != oof.id_column)
    oof_map = predictions_by_id(oof, oof_column)
    ids = [row[truth.id_column].strip() for row in truth.rows]
    references = [row[truth.target_column] for row in truth.rows]
    folds = [int(row["fold"]) for row in truth.rows]
    source = [oof_map[identifier] for identifier in ids]
    fold_models = {
        fold: SupervisedErrorModel(
            (source[index], references[index])
            for index in range(len(ids))
            if folds[index] != fold
        )
        for fold in sorted(set(folds))
    }
    fast_metric = FastMetric(references, source)
    baseline = fast_metric.score(source)
    trials = []
    best_key = None
    best = None
    for order, config in enumerate(configurations()):
        fold_mappings = {fold: mappings(model, config) for fold, model in fold_models.items()}
        corrected = []
        operations = []
        for value, fold in zip(source, folds, strict=True):
            result, changes = repair_supervised_errors(value, fold_mappings[fold])
            corrected.append(result)
            operations.append(changes)
        score = fast_metric.score(corrected)
        changed_tokens = sum(map(len, operations))
        trial = {
            "config": config.__dict__,
            "changed_rows": sum(bool(row) for row in operations),
            "changed_tokens": changed_tokens,
            "score": score,
        }
        trials.append(trial)
        key = (float(score["mean_error"]), changed_tokens, order)
        if best_key is None or key < best_key:
            best_key = key
            best = config, corrected, operations
    assert best is not None
    config, corrected, operations = best

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_predictions(output / "crossfit_predictions.csv", truth.id_column, ids, corrected)
    with (output / "crossfit_changes.jsonl").open("w", encoding="utf-8") as handle:
        for identifier, before, after, changes in zip(ids, source, corrected, operations, strict=True):
            if changes:
                handle.write(json.dumps({"id": identifier, "before": before, "after": after, "changes": changes}, ensure_ascii=False) + "\n")

    train = read_table(args.train, require_target=True)
    test = read_table(args.test)
    test_predictions = read_table(args.test_predictions)
    assert train.target_column is not None
    test_column = test_predictions.target_column or next(
        c for c in test_predictions.fieldnames if c != test_predictions.id_column
    )
    test_map = predictions_by_id(test_predictions, test_column)
    final_model = SupervisedErrorModel(
        (source[index], references[index]) for index in range(len(ids))
    )
    final_mappings = mappings(final_model, config)
    test_ids = [row[test.id_column].strip() for row in test.rows]
    test_source = [test_map[identifier] for identifier in test_ids]
    test_corrected = []
    test_changes = []
    for identifier, value in zip(test_ids, test_source, strict=True):
        result, changes = repair_supervised_errors(value, final_mappings)
        test_corrected.append(result)
        if changes:
            test_changes.append({"id": identifier, "before": value, "after": result, "changes": changes})
    write_predictions(output / "test.csv", test.id_column, test_ids, test_corrected)
    with (output / "test_changes.jsonl").open("w", encoding="utf-8") as handle:
        for row in test_changes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_submission(args.sample, args.submission, dict(zip(test_ids, test_corrected, strict=True)))

    selected_score = fast_metric.score(corrected)
    report = {
        "method": "recurring OCR-to-truth token substitutions learned only from other folds",
        "rows": len(ids),
        "trial_count": len(trials),
        "selected_config": config.__dict__,
        "baseline_score": baseline,
        "selected_score": selected_score,
        "multi_score_delta": selected_score["multi_score"] - baseline["multi_score"],
        "baseline_fold_scores": score_folds(references, source, folds),
        "selected_fold_scores": score_folds(references, corrected, folds),
        "changed_rows": sum(bool(row) for row in operations),
        "changed_tokens": sum(map(len, operations)),
        "learned_full_mappings": len(final_mappings),
        "test_changed_rows": len(test_changes),
        "top_trials": sorted(trials, key=lambda row: (float(row["score"]["mean_error"]), int(row["changed_tokens"])))[:20],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
