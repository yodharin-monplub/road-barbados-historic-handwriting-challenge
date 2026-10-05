#!/usr/bin/env python3
"""CPU-only Model44 lineage audit entry point; no model or GPU is loaded."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model44.decode import main  # noqa: E402


if __name__ == "__main__":
    if "--dry-audit" not in sys.argv:
        sys.argv.append("--dry-audit")
    main()
