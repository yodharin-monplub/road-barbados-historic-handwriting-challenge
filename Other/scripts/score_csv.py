#!/usr/bin/env python3
"""Score ID-aligned validation predictions with the documented Zindi metric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.metrics import competition_error  # noqa: E402
from road_htr.schema import SchemaError, predictions_by_id, read_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-column")
    parser.add_argument("--fold", type=int)
    parser.add_argument(
        "--per-fold",
        action="store_true",
        help="also report every fold when the truth table contains a fold column",
    )
    args = parser.parse_args()
    if args.fold is not None and args.per_fold:
        parser.error("--fold and --per-fold are mutually exclusive")

    truth = read_table(args.truth, require_target=True)
    predictions = read_table(args.predictions)
    if args.prediction_column is None and predictions.target_column is None:
        non_id = [c for c in predictions.fieldnames if c != predictions.id_column]
        if len(non_id) != 1:
            raise SchemaError(
                "pass --prediction-column because the prediction CSV is ambiguous"
            )
        prediction_column = non_id[0]
    else:
        prediction_column = args.prediction_column
    by_id = predictions_by_id(predictions, prediction_column)

    assert truth.target_column is not None
    truth_rows = [
        row
        for row in truth.rows
        if args.fold is None or ("fold" in truth.fieldnames and int(row["fold"]) == args.fold)
    ]
    if args.fold is not None and "fold" not in truth.fieldnames:
        raise SchemaError("--fold was supplied but the truth CSV has no fold column")
    truth_ids = [row[truth.id_column].strip() for row in truth_rows]
    missing = sorted(set(truth_ids) - set(by_id))
    extra = sorted(set(by_id) - set(truth_ids))
    if missing or extra:
        raise SchemaError(
            f"ID mismatch: {len(missing)} missing, {len(extra)} extra; "
            f"examples missing={missing[:5]}, extra={extra[:5]}"
        )
    references = [row[truth.target_column] for row in truth_rows]
    hypotheses = [by_id[identifier] for identifier in truth_ids]
    aggregate = competition_error(references, hypotheses)
    if not args.per_fold:
        print(json.dumps(aggregate, indent=2))
        return
    if "fold" not in truth.fieldnames:
        raise SchemaError("--per-fold requires a fold column in the truth CSV")
    fold_scores = {}
    for fold_value in sorted({int(row["fold"]) for row in truth.rows}):
        rows = [row for row in truth.rows if int(row["fold"]) == fold_value]
        fold_ids = [row[truth.id_column].strip() for row in rows]
        fold_scores[str(fold_value)] = competition_error(
            [row[truth.target_column] for row in rows],
            [by_id[identifier] for identifier in fold_ids],
        )
    print(json.dumps({"aggregate": aggregate, "folds": fold_scores}, indent=2))


if __name__ == "__main__":
    main()
