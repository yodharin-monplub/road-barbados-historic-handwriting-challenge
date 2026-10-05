#!/usr/bin/env python3
"""Read-only numeric and provenance gate; run before final all-data training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.model35_submission import check_oof_scores, verify_oof_sources
from road_htr.fold_model_postprocess import _absolute, _read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=ROOT / "data/processed/train_folds.csv")
    parser.add_argument("--oof-report", type=Path, default=ROOT / "outputs/model35/allfold_epoch07_verified/report.json")
    parser.add_argument("--fallback", type=Path, default=ROOT / "outputs/model66/oof/allfold_oof.csv")
    args = parser.parse_args()
    truth = _absolute(args.truth)
    report = _read_json(_absolute(args.oof_report))
    result = check_oof_scores(truth, report, _absolute(args.fallback))
    verify_oof_sources(truth, report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
