#!/usr/bin/env python3
"""Print the tensor shapes Model122 feeds the vision tower, for memory sizing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import Collator, LineDataset, absolute, build_model, load_rows  # noqa: E402

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else int(cfg["batch_size"])
rows = load_rows(cfg, 0, "train")[:batch_size]
from transformers import AutoProcessor  # noqa: E402

processor = AutoProcessor.from_pretrained(str(absolute(cfg["base"])), fix_mistral_regex=True)
ds = LineDataset(rows, absolute(cfg["image_dir"]), cfg, train=True)
items = [ds[i] for i in range(len(rows))]
print("resized image sizes:", [item["image"].size for item in items])
batch = Collator(processor, cfg, with_labels=True)(items)
for key, value in batch.items():
    if hasattr(value, "shape"):
        print(key, tuple(value.shape), value.dtype)
print("image_sizes:", batch.get("image_sizes"))
patches = batch["pixel_values"].shape[-2] // 14 * (batch["pixel_values"].shape[-1] // 14)
total = patches * batch["pixel_values"].shape[0] if batch["pixel_values"].dim() == 4 else patches
print("patches per image:", patches, "total:", total,
      "dense mask GiB:", total * total * 4 / 2**30)
