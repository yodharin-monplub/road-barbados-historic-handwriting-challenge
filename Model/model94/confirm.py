#!/usr/bin/env python3
"""Evaluate frozen Model94 on untouched confirmation folds 1 through 4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.crossfit_candidate_selector import (  # noqa: E402
    load_labeled_rows,
    read_predictions,
)
from road_htr.metrics import competition_error  # noqa: E402
from road_htr.model34_token_fusion import _row_cost, paired_bootstrap  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, rows = load_labeled_rows(args.truth)
    primary = read_predictions(args.primary)
    challenger = read_predictions(args.challenger)
    confirmation_folds = {int(value) for value in config["confirmation_folds"]}
    selected = [row for row in rows if row.fold in confirmation_folds]
    if {row.fold for row in selected} != confirmation_folds:
        raise ValueError("one or more configured confirmation folds is absent")
    missing_primary = {row.identifier for row in selected} - set(primary)
    missing_challenger = {row.identifier for row in selected} - set(challenger)
    if missing_primary or missing_challenger:
        raise ValueError(
            f"missing predictions: primary={sorted(missing_primary)[:5]}, "
            f"challenger={sorted(missing_challenger)[:5]}"
        )

    def score(fold_rows, values):
        return competition_error(
            [row.target for row in fold_rows],
            [values[row.identifier] for row in fold_rows],
        )

    baseline = score(selected, primary)
    corrected = score(selected, challenger)
    fold_scores = {}
    nonnegative = 0
    for fold in sorted(confirmation_folds):
        fold_rows = [row for row in selected if row.fold == fold]
        fold_baseline = score(fold_rows, primary)
        fold_corrected = score(fold_rows, challenger)
        delta = fold_corrected["multi_score"] - fold_baseline["multi_score"]
        nonnegative += int(delta >= 0)
        fold_scores[str(fold)] = {
            "baseline": fold_baseline,
            "corrected": fold_corrected,
            "delta": delta,
        }
    bootstrap = paired_bootstrap(
        [_row_cost(row.target, primary[row.identifier]) for row in selected],
        [_row_cost(row.target, challenger[row.identifier]) for row in selected],
        samples=int(config["bootstrap_samples"]),
        seed=20260906,
    )
    delta = corrected["multi_score"] - baseline["multi_score"]
    promoted = (
        delta >= float(config["confirmation_minimum_gain"])
        and nonnegative >= int(config["confirmation_minimum_nonnegative_folds"])
        and float(bootstrap["positive_fraction"])
        >= float(config["bootstrap_minimum_positive_fraction"])
    )
    report = {
        "schema_version": 1,
        "kind": "model94_confirmation",
        "development_fold": int(config["development_fold"]),
        "confirmation_folds": sorted(confirmation_folds),
        "rows": len(selected),
        "baseline_score": baseline,
        "corrected_score": corrected,
        "delta": delta,
        "nonnegative_confirmation_folds": nonnegative,
        "fold_scores": fold_scores,
        "bootstrap": bootstrap,
        "thresholds": {
            "minimum_gain": float(config["confirmation_minimum_gain"]),
            "minimum_nonnegative_folds": int(
                config["confirmation_minimum_nonnegative_folds"]
            ),
            "minimum_positive_fraction": float(
                config["bootstrap_minimum_positive_fraction"]
            ),
        },
        "promoted": promoted,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
