#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.vlm_ocr_token_fusion import apply  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply frozen Model53 to test candidates")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/train_folds.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data/raw/Test.csv")
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    print(json.dumps(apply(args.config, args.train, args.test, args.fallback, args.vlm, args.output_dir, args.sample, args.submission), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
