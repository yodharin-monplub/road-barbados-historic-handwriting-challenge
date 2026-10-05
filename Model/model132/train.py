#!/usr/bin/env python3
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))

from road_htr.kraken_cli import train_main  # noqa: E402

train_main(HERE / "config.yaml")
