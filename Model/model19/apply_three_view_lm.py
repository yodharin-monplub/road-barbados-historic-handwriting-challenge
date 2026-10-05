#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.three_view_lm_selector import apply_main  # noqa: E402

apply_main()
