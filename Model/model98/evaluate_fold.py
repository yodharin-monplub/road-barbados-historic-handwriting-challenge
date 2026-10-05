#!/usr/bin/env python3
"""Cross-fitted Model88/Model95 selector on one development fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.crossfit_candidate_selector import evaluate_crossfit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--minimum-gain", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model98/fold0"))
    args = parser.parse_args()
    report = evaluate_crossfit(
        ROOT / "data/processed/train_folds.csv",
        ROOT / "outputs/model88/word_rover_vs_kraken_selector/crossfit_predictions.csv",
        args.direct,
        args.output_dir,
        ROOT / "model88/config.json",
        single_fold=args.fold,
    )
    summary = {
        "fold": args.fold,
        "rows": report["rows"],
        "model88_score": report["candidate_a_score"],
        "model95_direct_score": report["candidate_b_score"],
        "selected_score": report["score"],
        "selected_model95_rows": report["selected_b_rows"],
        "delta": report["score_delta_vs_a"],
        "minimum_gain": args.minimum_gain,
        "promoted": report["score_delta_vs_a"] >= args.minimum_gain,
        "note": "Uses the already-frozen Model88 selector configuration without tuning.",
    }
    (args.output_dir / "promotion.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
