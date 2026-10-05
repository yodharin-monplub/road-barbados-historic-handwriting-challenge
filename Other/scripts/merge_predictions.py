#!/usr/bin/env python3
"""Merge targeted OCR reruns into a complete prediction CSV by ID."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.schema import SchemaError  # noqa: E402
from road_htr.view_selector import read_prediction_map  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    id_column, merged = read_prediction_map(args.base)
    base_order = list(merged)
    replaced: set[str] = set()
    for path in args.override:
        _, values = read_prediction_map(path)
        unknown = set(values) - set(merged)
        if unknown:
            raise SchemaError(f"override IDs absent from base: {sorted(unknown)[:5]}")
        overlap = replaced.intersection(values)
        if overlap:
            raise SchemaError(f"IDs repeated across overrides: {sorted(overlap)[:5]}")
        merged.update(values)
        replaced.update(values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_column, "prediction"])
        writer.writeheader()
        for identifier in base_order:
            writer.writerow({id_column: identifier, "prediction": merged[identifier]})
    print(f"wrote {len(base_order)} rows; replaced {len(replaced)}")


if __name__ == "__main__":
    main()

