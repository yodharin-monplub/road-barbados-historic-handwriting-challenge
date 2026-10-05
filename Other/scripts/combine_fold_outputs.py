#!/usr/bin/env python3
"""Combine non-overlapping fold CSV or JSONL outputs in truth-file order."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.schema import SchemaError, read_table  # noqa: E402


def parse_fold_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use FOLD=PATH")
    raw_fold, raw_path = value.split("=", 1)
    try:
        fold = int(raw_fold)
    except ValueError as error:
        raise argparse.ArgumentTypeError("fold must be an integer") from error
    if not raw_path:
        raise argparse.ArgumentTypeError("path cannot be blank")
    return fold, Path(raw_path)


def _expected_ids(truth_path: Path) -> tuple[str, list[str], dict[int, set[str]]]:
    truth = read_table(truth_path)
    if "fold" not in truth.fieldnames:
        raise SchemaError("truth table must contain a fold column")
    order = [row[truth.id_column].strip() for row in truth.rows]
    by_fold: dict[int, set[str]] = {}
    for row in truth.rows:
        by_fold.setdefault(int(row["fold"]), set()).add(row[truth.id_column].strip())
    return truth.id_column, order, by_fold


def _read_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = str(record.get("id", "")).strip()
            if not identifier or identifier in records:
                raise SchemaError(f"blank/duplicate id at {path}:{line_number}")
            records[identifier] = record
    return records


def combine_jsonl(
    order: list[str], expected: dict[int, set[str]], inputs: dict[int, Path], output: Path
) -> None:
    combined: dict[str, dict] = {}
    for fold, path in inputs.items():
        records = _read_jsonl(path)
        if set(records) != expected[fold]:
            raise SchemaError(
                f"fold {fold} ID mismatch: "
                f"missing={sorted(expected[fold]-set(records))[:5]}, "
                f"extra={sorted(set(records)-expected[fold])[:5]}"
            )
        combined.update(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for identifier in order:
            handle.write(json.dumps(combined[identifier], ensure_ascii=False) + "\n")


def combine_csv(
    id_column: str,
    order: list[str],
    expected: dict[int, set[str]],
    inputs: dict[int, Path],
    output: Path,
) -> None:
    combined: dict[str, str] = {}
    output_column: str | None = None
    for fold, path in inputs.items():
        table = read_table(path)
        value_columns = [column for column in table.fieldnames if column != table.id_column]
        if len(value_columns) != 1:
            raise SchemaError(f"{path} must contain one prediction column")
        if output_column is None:
            output_column = value_columns[0]
        records = {
            row[table.id_column].strip(): row[value_columns[0]] for row in table.rows
        }
        if set(records) != expected[fold]:
            raise SchemaError(
                f"fold {fold} ID mismatch: "
                f"missing={sorted(expected[fold]-set(records))[:5]}, "
                f"extra={sorted(set(records)-expected[fold])[:5]}"
            )
        combined.update(records)
    assert output_column is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_column, output_column])
        writer.writeheader()
        for identifier in order:
            writer.writerow({id_column: identifier, output_column: combined[identifier]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument(
        "--input", type=parse_fold_path, action="append", required=True, dest="inputs"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = dict(args.inputs)
    if len(inputs) != len(args.inputs):
        parser.error("each fold may be supplied only once")
    id_column, order, expected = _expected_ids(args.truth)
    if set(inputs) != set(expected):
        parser.error(
            f"input folds {sorted(inputs)} do not match truth folds {sorted(expected)}"
        )
    suffixes = {path.suffix.casefold() for path in inputs.values()}
    if suffixes == {".jsonl"} and args.output.suffix.casefold() == ".jsonl":
        combine_jsonl(order, expected, inputs, args.output)
    elif suffixes == {".csv"} and args.output.suffix.casefold() == ".csv":
        combine_csv(id_column, order, expected, inputs, args.output)
    else:
        parser.error("inputs and output must all be CSV or all be JSONL")
    print(f"combined {len(order)} rows from {len(inputs)} folds into {args.output}")


if __name__ == "__main__":
    main()
