#!/usr/bin/env python3
"""Impute Model81's one empty fold output from the other four-fold medoid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from road_htr.deployment_medoid import _mixed_distance
from road_htr.ensemble import write_prediction_csv


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_predictions(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "prediction"]:
            raise ValueError(f"unexpected schema in {path}: {reader.fieldnames}")
        rows = list(reader)
    ids = [row["ID"].strip() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ID in {path}")
    return ids, {row["ID"].strip(): row["prediction"].strip() for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--donor", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if len(args.donor) != 4 or len(set(args.donor)) != 4:
        raise ValueError("exactly four unique donor CSVs are required")

    ids, target = read_predictions(args.target)
    blanks = [identifier for identifier in ids if not target[identifier]]
    if len(blanks) != 1:
        raise ValueError(f"expected exactly one target blank, found {blanks}")

    donor_maps: list[dict[str, str]] = []
    for path in args.donor:
        donor_ids, predictions = read_predictions(path)
        if donor_ids != ids:
            raise ValueError(f"ID order mismatch in {path}")
        donor_maps.append(predictions)

    identifier = blanks[0]
    donor_predictions = [predictions[identifier] for predictions in donor_maps]
    if any(not value for value in donor_predictions):
        raise ValueError(f"a donor is blank for {identifier}")

    unique_candidates = sorted(set(donor_predictions))
    costs = {
        candidate: sum(
            _mixed_distance(source, candidate) for source in donor_predictions
        )
        / len(donor_predictions)
        for candidate in unique_candidates
    }
    chosen = min(unique_candidates, key=lambda candidate: (costs[candidate], candidate))
    target[identifier] = chosen

    write_prediction_csv(args.output, "ID", ids, [target[item] for item in ids])
    receipt = {
        "schema_version": 1,
        "policy": "leave-one-out four-fold mixed-distance medoid; lexical tie-break",
        "fixed_char_mix": 0.55,
        "target": str(args.target),
        "target_sha256": sha256(args.target),
        "donors": [
            {"path": str(path), "sha256": sha256(path)} for path in args.donor
        ],
        "imputed_id": identifier,
        "donor_predictions": donor_predictions,
        "candidate_costs": costs,
        "imputed_prediction": chosen,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "changed_rows": 1,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
