#!/usr/bin/env python3
"""Run the fixed Model38 Model33 robustness panel."""

from __future__ import annotations

import os
import sys
from pathlib import Path


# Small 655x54 CPU matrix products are faster and substantially less disruptive
# to concurrent training when BLAS is kept to one worker.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.model38_selector_robustness import main  # noqa: E402


if __name__ == "__main__":
    main()
