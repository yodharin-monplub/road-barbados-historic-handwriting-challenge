#!/usr/bin/env python3
"""Create deterministic length-stratified folds without altering labels."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.schema import read_table  # noqa: E402
from road_htr.vlm_data import read_id_file  # noqa: E402


def make_folds(targets: list[str], n_folds: int, seed: int) -> list[int]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(targets) < n_folds:
        raise ValueError("fewer rows than folds")

    # Joint word/character buckets keep both WER and CER validation weights
    # distributed. Assignment is deterministic for a given seed.
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        char_bin = min(len(target) // 16, 12)
        word_bin = min(len(target.split()) // 3, 8)
        buckets[(char_bin, word_bin)].append(index)

    rng = random.Random(seed)
    folds = [-1] * len(targets)
    offset = 0
    for key in sorted(buckets):
        members = buckets[key]
        rng.shuffle(members)
        for position, index in enumerate(members):
            folds[index] = (offset + position) % n_folds
        offset = (offset + len(members)) % n_folds
    assert all(fold >= 0 for fold in folds)
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--exclude-ids", type=Path)
    args = parser.parse_args()

    table = read_table(args.train_csv, require_target=True)
    assert table.target_column is not None
    excluded = read_id_file(args.exclude_ids)
    rows = tuple(
        row for row in table.rows if row[table.id_column].strip() not in excluded
    )
    unknown = excluded - {row[table.id_column].strip() for row in table.rows}
    if unknown:
        raise ValueError(f"excluded IDs absent from training data: {sorted(unknown)}")
    targets = [row[table.target_column] for row in rows]
    folds = make_folds(targets, args.folds, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*table.fieldnames, "fold"]
    if "fold" in table.fieldnames:
        raise ValueError("input already has a fold column")
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, fold in zip(rows, folds, strict=True):
            writer.writerow({**row, "fold": fold})
    counts = {fold: folds.count(fold) for fold in range(args.folds)}
    print(
        f"wrote {len(folds)} rows to {args.output}; fold sizes={counts}; "
        f"excluded={len(excluded)}"
    )


if __name__ == "__main__":
    main()
