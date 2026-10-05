#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.qwen2_vl_starter_highres import audit_main  # noqa: E402

audit_main(Path(__file__).with_name("config.yaml"))
