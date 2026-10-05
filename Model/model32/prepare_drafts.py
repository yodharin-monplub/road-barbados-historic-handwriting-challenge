#!/usr/bin/env python3
"""Generate canonical receipt-v2 drafts, including audited blank OCR rows."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.qwen2_vl_ocr_guided import prepare_drafts_main  # noqa: E402


prepare_drafts_main(Path(__file__).with_name("config.yaml"))
