#!/usr/bin/env python3
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.vlm_cli import train_main  # noqa: E402

if __name__ == "__main__":
    train_main(HERE / "config.yaml")
