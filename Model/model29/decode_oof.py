#!/usr/bin/env python3
"""Run exact-score nested cross-fitted Model29 decoding."""

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.model29_crossfit import main  # noqa: E402


if __name__ == "__main__":
    main(HERE / "config.yaml")
