#!/usr/bin/env python3
"""Apply the frozen Model34 rule to an exactly authenticated prediction pair."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.model34_token_fusion import main  # noqa: E402


if __name__ == "__main__":
    sys.argv.insert(1, "apply")
    main()
