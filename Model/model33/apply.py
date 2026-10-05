#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.crossfit_candidate_selector import apply_main  # noqa: E402


if __name__ == "__main__":
    apply_main()
