#!/usr/bin/env python3
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.ctc_cli import infer_main  # noqa: E402

infer_main(HERE / "config.yaml")

