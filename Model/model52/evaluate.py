#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.vlm_ocr_guarded_selector import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen Model52 on one held fold")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--truth", type=Path, default=ROOT / "data/processed/train_folds.csv")
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.config, args.truth, args.fallback, args.vlm, args.fold, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
