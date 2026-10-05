#!/usr/bin/env python3
"""Normalize PP-OCR's systematic plus-for-ampersand transcription error."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.metrics import competition_error  # noqa: E402
from road_htr.schema import SchemaError, predictions_by_id, read_table  # noqa: E402
from road_htr.submission import write_submission  # noqa: E402


def normalize_ampersands(text: str) -> str:
    """Return text with the model's systematic ``+`` glyph error repaired."""

    return text.replace("+", "&")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if (args.sample is None) != (args.submission is None):
        parser.error("--sample and --submission must be supplied together")

    table = read_table(args.predictions)
    prediction_column = table.target_column or next(
        (column for column in table.fieldnames if column != table.id_column), None
    )
    if prediction_column is None:
        raise SchemaError("prediction file has no prediction column")
    original = predictions_by_id(table, prediction_column)
    ids = [row[table.id_column].strip() for row in table.rows]
    normalized = {
        identifier: normalize_ampersands(original[identifier]) for identifier in ids
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[table.id_column, "prediction"])
        writer.writeheader()
        for identifier in ids:
            writer.writerow(
                {table.id_column: identifier, "prediction": normalized[identifier]}
            )

    result: dict[str, object] = {
        "rows": len(ids),
        "changed_rows": sum(original[i] != normalized[i] for i in ids),
        "replaced_characters": sum(original[i].count("+") for i in ids),
        "output": str(args.output),
    }
    if args.truth is not None:
        truth = read_table(args.truth, require_target=True)
        assert truth.target_column is not None
        truth_by_id = {
            row[truth.id_column].strip(): row[truth.target_column].strip()
            for row in truth.rows
        }
        if set(truth_by_id) != set(normalized):
            raise SchemaError("truth and prediction IDs differ")
        ordered_ids = [row[truth.id_column].strip() for row in truth.rows]
        result["score"] = competition_error(
            [truth_by_id[i] for i in ordered_ids],
            [normalized[i] for i in ordered_ids],
        )
        if "fold" in truth.fieldnames:
            result["fold_scores"] = {
                str(fold): competition_error(
                    [
                        row[truth.target_column].strip()
                        for row in truth.rows
                        if int(row["fold"]) == fold
                    ],
                    [
                        normalized[row[truth.id_column].strip()]
                        for row in truth.rows
                        if int(row["fold"]) == fold
                    ],
                )
                for fold in sorted({int(row["fold"]) for row in truth.rows})
            }

    if args.submission is not None:
        assert args.sample is not None
        write_submission(args.sample, args.submission, normalized)
        result["submission"] = str(args.submission)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
