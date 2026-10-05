#!/usr/bin/env python3
"""Reject malformed or empty submissions before using a Zindi submission slot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.submission import validate_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_submission(args.submission, args.sample), indent=2))


if __name__ == "__main__":
    main()

