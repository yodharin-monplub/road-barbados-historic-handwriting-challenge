#!/usr/bin/env python3
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.fold_model_postprocess import fold_main  # noqa: E402


if __name__ == "__main__":
    fold_main()

