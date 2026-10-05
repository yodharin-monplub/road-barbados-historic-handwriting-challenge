#!/usr/bin/env python3
"""Filter consensus rows and append them to training data entirely in code."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.schema import SchemaError, read_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--agreement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-agreement", type=float, default=0.88)
    parser.add_argument("--min-exact-votes", type=int, default=2)
    args = parser.parse_args()

    train = read_table(args.train, require_target=True)
    assert train.target_column is not None
    if "fold" not in train.fieldnames:
        raise SchemaError("use the prepared train_folds.csv as --train")
    pseudo: list[dict[str, object]] = []
    with args.agreement.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                float(row["agreement"]) >= args.min_agreement
                and int(row["exact_votes"]) >= args.min_exact_votes
                and str(row["prediction"]).strip()
            ):
                pseudo.append(row)
    train_ids = {row[train.id_column].strip() for row in train.rows}
    overlap = train_ids.intersection(str(row["id"]) for row in pseudo)
    if overlap:
        raise SchemaError(f"pseudo-label IDs overlap train: {sorted(overlap)[:5]}")

    fields = [*train.fieldnames, "is_pseudo", "pseudo_agreement", "pseudo_exact_votes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in train.rows:
            writer.writerow(
                {
                    **row,
                    "is_pseudo": 0,
                    "pseudo_agreement": 1.0,
                    "pseudo_exact_votes": "",
                }
            )
        for row in pseudo:
            generated = {field: "" for field in train.fieldnames}
            generated[train.id_column] = str(row["id"])
            generated[train.target_column] = str(row["prediction"])
            generated["fold"] = -1
            writer.writerow(
                {
                    **generated,
                    "is_pseudo": 1,
                    "pseudo_agreement": row["agreement"],
                    "pseudo_exact_votes": row["exact_votes"],
                }
            )
    manifest = {
        "input_train": str(args.train),
        "input_agreement": str(args.agreement),
        "original_rows": len(train.rows),
        "pseudo_rows": len(pseudo),
        "min_agreement": args.min_agreement,
        "min_exact_votes": args.min_exact_votes,
        "fully_automated": True,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

