#!/usr/bin/env python3
"""Scan batches and report the worst-case vision sequence length / mask size."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import Collator, LineDataset, absolute, load_rows  # noqa: E402

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else int(cfg["batch_size"])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 40
rows = load_rows(cfg, 0, "train")
processor = AutoProcessor.from_pretrained(str(absolute(cfg["base"])), fix_mistral_regex=True)
loader = DataLoader(LineDataset(rows, absolute(cfg["image_dir"]), cfg, train=True),
                    batch_size=batch_size, shuffle=True, num_workers=4,
                    collate_fn=Collator(processor, cfg, with_labels=True))
worst = 0
for step, batch in enumerate(loader):
    pixels = batch["pixel_values"]
    patches = (pixels.shape[-2] // 14) * (pixels.shape[-1] // 14) * pixels.shape[0]
    if patches > worst:
        worst = patches
        print(f"step {step} pixels {tuple(pixels.shape)} patches {patches} "
              f"mask {patches*patches*4/2**30:.2f} GiB text {batch['input_ids'].shape[1]}",
              flush=True)
    if step >= limit:
        break
print("worst patches", worst, "mask GiB", worst * worst * 4 / 2**30)
