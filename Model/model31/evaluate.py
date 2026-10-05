#!/usr/bin/env python3
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.byt5_corrector import evaluate_main  # noqa: E402


if __name__ == "__main__":
    evaluate_main(Path(__file__).with_name("config.yaml"))

