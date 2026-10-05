#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.llm_corrector import prepare_main  # noqa: E402

if __name__ == "__main__":
    prepare_main()
