#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model46.transfer import main  # noqa: E402
if __name__ == "__main__":
    if "--dry-audit" not in sys.argv:
        sys.argv.append("--dry-audit")
    main()
