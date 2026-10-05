#!/usr/bin/env python3
"""Model154 training entry point: python Model/model154/train.py --fold N

Model3/train.py adds <its parent>/src to sys.path, the pre-reorganisation layout;
this resolves the project root by AGENTS.md instead (as Model149 does for Kraken).
road_htr.trocr_cli resolves the config's relative paths against Other/, so the host
needs Other/data -> ../Data (see go154.sh).
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))

from road_htr.trocr_cli import train_main  # noqa: E402

train_main(HERE / "config.yaml")
