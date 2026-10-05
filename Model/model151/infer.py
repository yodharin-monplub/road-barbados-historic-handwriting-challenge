#!/usr/bin/env python3
"""Model149 inference entry point.

Model22/infer.py adds <its parent>/src to sys.path, which was the pre-reorganisation
layout; under Model/ that path no longer exists. This resolves the project root by
AGENTS.md, exactly as Model122/train.py does.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))

from road_htr.kraken_cli import infer_main  # noqa: E402

infer_main()
