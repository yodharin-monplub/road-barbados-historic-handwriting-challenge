#!/usr/bin/env python3
"""Model154 inference entry point.

python Model/model154/infer.py --model <fold dir>/best_model --input <csv> \
    --output <csv> [--fold N]
Pass ABSOLUTE paths: road_htr.trocr_cli resolves relative ones against Other/.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))

from road_htr.trocr_cli import infer_main  # noqa: E402

infer_main(HERE / "config.yaml")
