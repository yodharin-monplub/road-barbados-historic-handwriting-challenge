#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.confusion_corrector import tune_main  # noqa: E402

tune_main()
