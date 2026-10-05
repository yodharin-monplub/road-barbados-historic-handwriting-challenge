#!/usr/bin/env python3
"""Evaluate raw and inner-cross-fitted gated Model93 predictions on one fold."""

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
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--selector-config", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--minimum-gain", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_crossfit(
        args.truth,
        args.primary,
        args.llm,
        args.output_dir,
        args.selector_config,
        single_fold=args.fold,
    )
    summary = {
        "fold": args.fold,
        "baseline_score": report["candidate_a_score"],
        "aggressive_score": report["candidate_b_score"],
        "gated_score": report["score"],
        "aggressive_delta": report["candidate_b_score"]["multi_score"]
        - report["candidate_a_score"]["multi_score"],
        "gated_delta": report["score_delta_vs_a"],
        "minimum_gain": args.minimum_gain,
        "promoted": report["score_delta_vs_a"] >= args.minimum_gain,
        "decision_note": "Fold 0 is a development gate; confirmation folds remain untouched.",
    }
    path = args.output_dir / "promotion.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
