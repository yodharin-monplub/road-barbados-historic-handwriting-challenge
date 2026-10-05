#!/usr/bin/env python3
"""Strictly assemble the five Model92 held-fold prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from road_htr.metrics import competition_error
from road_htr.schema import SchemaError, predictions_by_id, read_table
from road_htr.vlm_cli import _sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    truth_path = args.truth.resolve()
    truth = read_table(truth_path, require_target=True)
    if "fold" not in truth.fieldnames or truth.target_column is None:
        raise SchemaError("truth must contain fold and target columns")

    combined: dict[str, str] = {}
    fold_reports = []
    for fold in range(5):
        prediction_path = (
            args.input_dir / f"fold{fold}_oof_step400.csv"
        ).resolve()
        table = read_table(prediction_path)
        value_columns = [name for name in table.fieldnames if name != table.id_column]
        value_column = table.target_column or (
            value_columns[0] if len(value_columns) == 1 else None
        )
        if value_column is None:
            raise SchemaError(f"fold {fold} predictions need one value column")
        predictions = dict(predictions_by_id(table, value_column))
        rows = [row for row in truth.rows if int(row["fold"]) == fold]
        expected = {row[truth.id_column].strip() for row in rows}
        if set(predictions) != expected or len(predictions) != len(expected):
            raise SchemaError(f"fold {fold} IDs do not exactly match heldout IDs")
        overlap = set(combined).intersection(predictions)
        if overlap:
            raise SchemaError(f"duplicate IDs across folds: {sorted(overlap)[:3]}")
        combined.update(predictions)
        score = competition_error(
            [row[truth.target_column] for row in rows],
            [predictions[row[truth.id_column].strip()] for row in rows],
        )
        fold_reports.append(
            {
                "fold": fold,
                "rows": len(rows),
                "path": str(prediction_path),
                "sha256": _sha256(prediction_path),
                "score": score,
            }
        )

    ordered_ids = [row[truth.id_column].strip() for row in truth.rows]
    if set(combined) != set(ordered_ids) or len(combined) != len(ordered_ids):
        raise SchemaError("combined predictions do not exactly cover training IDs")
    if any(not combined[identifier] for identifier in ordered_ids):
        raise SchemaError("combined predictions contain blanks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[truth.id_column, "prediction"])
        writer.writeheader()
        writer.writerows(
            {truth.id_column: identifier, "prediction": combined[identifier]}
            for identifier in ordered_ids
        )
    overall = competition_error(
        [row[truth.target_column] for row in truth.rows],
        [combined[identifier] for identifier in ordered_ids],
    )
    report = {
        "schema_version": 1,
        "rows": len(ordered_ids),
        "truth": str(truth_path),
        "truth_sha256": _sha256(truth_path),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "folds": fold_reports,
        "score": overall,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
