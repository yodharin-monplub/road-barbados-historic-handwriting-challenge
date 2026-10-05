#!/usr/bin/env python3
"""Restore common historical-abbreviation carets from a fold-safe lexicon."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.metrics import competition_error  # noqa: E402
from road_htr.schema import SchemaError, predictions_by_id, read_table  # noqa: E402
from road_htr.submission import write_submission  # noqa: E402


def build_caret_map(
    labels: Iterable[str], minimum_count: int, minimum_ratio: float
) -> dict[str, str]:
    """Map caretless tokens to a dominant caret-bearing training spelling."""

    variants: dict[str, Counter[str]] = defaultdict(Counter)
    for label in labels:
        for token in label.split():
            variants[token.replace("^", "")][token] += 1
    mapping: dict[str, str] = {}
    for caretless, counts in variants.items():
        candidate, support = counts.most_common(1)[0]
        if (
            "^" in candidate
            and support >= minimum_count
            and support / sum(counts.values()) >= minimum_ratio
        ):
            mapping[caretless] = candidate
    return mapping


def restore(text: str, mapping: dict[str, str]) -> str:
    """Apply caret insertion only; never remove or relocate an existing caret."""

    return " ".join(
        mapping.get(token, token) if "^" not in token else token
        for token in text.split()
    )


def build_fold_maps(
    rows: Iterable[dict[str, str]],
    target_column: str,
    fold_column: str,
    minimum_count: int,
    minimum_ratio: float,
) -> dict[int, dict[str, str]]:
    """Build one caret map per held-out fold without using that fold's labels."""

    materialized = tuple(rows)
    folds = sorted({int(row[fold_column]) for row in materialized})
    return {
        fold: build_caret_map(
            (
                row[target_column]
                for row in materialized
                if int(row[fold_column]) != fold
            ),
            minimum_count,
            minimum_ratio,
        )
        for fold in folds
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--oof",
        action="store_true",
        help="build a separate label lexicon excluding each prediction row's fold",
    )
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if (args.sample is None) != (args.submission is None):
        parser.error("--sample and --submission must be supplied together")

    train = read_table(args.train, require_target=True)
    predictions = read_table(args.predictions)
    assert train.target_column is not None
    prediction_column = predictions.target_column or next(
        (
            column
            for column in predictions.fieldnames
            if column != predictions.id_column
        ),
        None,
    )
    if prediction_column is None:
        raise SchemaError("prediction file has no prediction column")
    source = predictions_by_id(predictions, prediction_column)
    ids = [row[predictions.id_column].strip() for row in predictions.rows]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    minimum_count = int(config["minimum_count"])
    minimum_ratio = float(config["minimum_ratio"])

    if args.oof:
        if "fold" not in train.fieldnames:
            raise SchemaError("--oof requires a fold column in --train")
        train_by_id = {row[train.id_column].strip(): row for row in train.rows}
        if set(ids) != set(train_by_id):
            raise SchemaError("OOF prediction and training IDs differ")
        fold_maps = build_fold_maps(
            train.rows,
            train.target_column,
            "fold",
            minimum_count,
            minimum_ratio,
        )
        normalized = {
            identifier: restore(
                source[identifier],
                fold_maps[int(train_by_id[identifier]["fold"])],
            )
            for identifier in ids
        }
        map_sizes: object = {str(fold): len(value) for fold, value in fold_maps.items()}
    else:
        mapping = build_caret_map(
            (row[train.target_column] for row in train.rows),
            minimum_count,
            minimum_ratio,
        )
        normalized = {i: restore(source[i], mapping) for i in ids}
        map_sizes = len(mapping)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[predictions.id_column, "prediction"]
        )
        writer.writeheader()
        for identifier in ids:
            writer.writerow(
                {
                    predictions.id_column: identifier,
                    "prediction": normalized[identifier],
                }
            )

    result: dict[str, object] = {
        "rows": len(ids),
        "changed_rows": sum(source[i] != normalized[i] for i in ids),
        "inserted_carets": sum(
            normalized[i].count("^") - source[i].count("^") for i in ids
        ),
        "map_sizes": map_sizes,
        "output": str(args.output),
    }
    if args.oof:
        train_by_id = {row[train.id_column].strip(): row for row in train.rows}
        references = [train_by_id[i][train.target_column].strip() for i in ids]
        result["score"] = competition_error(
            references, [normalized[i] for i in ids]
        )
        result["fold_scores"] = {
            str(fold): competition_error(
                [
                    train_by_id[i][train.target_column].strip()
                    for i in ids
                    if int(train_by_id[i]["fold"]) == fold
                ],
                [
                    normalized[i]
                    for i in ids
                    if int(train_by_id[i]["fold"]) == fold
                ],
            )
            for fold in sorted({int(row["fold"]) for row in train.rows})
        }
    if args.submission is not None:
        assert args.sample is not None
        write_submission(args.sample, args.submission, normalized)
        result["submission"] = str(args.submission)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
