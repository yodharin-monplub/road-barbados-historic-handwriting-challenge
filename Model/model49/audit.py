#!/usr/bin/env python3
"""Run Model49's static CPU contract and label-constraint audit."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model49.decode import main  # noqa: E402


if __name__ == "__main__":
    if "--contract-audit" not in sys.argv:
        sys.argv.append("--contract-audit")
    main()
