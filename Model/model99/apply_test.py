#!/usr/bin/env python3
"""Apply the fold-0-selected character ROVER rule to the five test witnesses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate import fuse_line, prediction_map
from road_htr.schema import read_table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = (
    ROOT / "outputs/model83/model77_test/full.csv",
    ROOT / "outputs/model83/model77_test/enhanced.csv",
    ROOT / "outputs/model83/model77_test/center.csv",
    ROOT / "outputs/model83/model77_test/center_enhanced.csv",
    ROOT / "outputs/model35/deployment_epoch07/confusions_test.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/Test.csv")
    parser.add_argument("--base", type=Path, default=ROOT / "outputs/model87/test.csv")
    parser.add_argument("--threshold", type=int, default=3, choices=(3, 4, 5))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/model99/test.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/model99/test_report.json")
    args = parser.parse_args()

    table = read_table(args.input)
    ids = [row[table.id_column].strip() for row in table.rows]
    base = prediction_map(args.base)
    sources = [prediction_map(path) for path in DEFAULT_SOURCES]
    expected = set(ids)
    if set(base) != expected or any(set(source) != expected for source in sources):
        raise ValueError("test candidate IDs do not match Test.csv")
    predictions = {
        identifier: fuse_line(
            base[identifier], [source[identifier] for source in sources], args.threshold
        )
        for identifier in ids
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[table.id_column, "prediction"])
        writer.writeheader()
        writer.writerows(
            {table.id_column: identifier, "prediction": predictions[identifier]}
            for identifier in ids
        )
    result = {
        "schema_version": 1,
        "threshold": args.threshold,
        "rows": len(ids),
        "changed_rows_vs_model87": sum(
            predictions[identifier] != base[identifier] for identifier in ids
        ),
        "output": str(args.output.resolve()),
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
