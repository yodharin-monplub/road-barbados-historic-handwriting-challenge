#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.vlm_ocr_token_fusion import audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen Model53")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--truth", type=Path, default=ROOT / "data/processed/train_folds.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/model53/dry_run.json")
    args = parser.parse_args()
    print(json.dumps(audit(args.config, args.truth, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
