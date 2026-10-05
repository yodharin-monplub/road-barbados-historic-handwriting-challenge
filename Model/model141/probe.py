#!/usr/bin/env python3
"""Zero-shot probe of GOT-OCR2 on this challenge's line crops.

GOT-OCR2 is an OCR-specialist (Apache-2.0, ~580M) whose architecture shares
nothing with the LightOnOCR/Qwen members or the PP-OCRv6 CTC model, so it is
the cheapest remaining source of the error diversity that has paid off twice.
Before spending anything, check that it reads these lines at all and how its
processor wants to be called - its interface is not the chat template the other
members use.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))

from road_htr.metrics import competition_error  # noqa: E402
from road_htr.submission import clean_prediction  # noqa: E402

BASE = ROOT / "Other" / "checkpoints" / "model141" / "got-ocr2-hf"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 24


def main() -> None:
    from transformers import AutoProcessor, GotOcr2ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(BASE))
    model = GotOcr2ForConditionalGeneration.from_pretrained(str(BASE), dtype=torch.bfloat16)
    model.to("cuda").eval()
    print("loaded:", type(model).__name__, type(processor).__name__)

    with (ROOT / "Data/processed/train_folds.csv").open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["fold"]) == 0][:LIMIT]

    references, predictions = [], []
    started = time.time()
    for row in rows:
        with Image.open(ROOT / "Data/raw/images" / f"{row['ID']}.jpg") as handle:
            image = handle.convert("RGB")
        inputs = processor(image, return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                       tokenizer=processor.tokenizer,
                                       stop_strings="<|im_end|>")
        text = processor.decode(generated[0, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        references.append(row["Target"])
        predictions.append(clean_prediction(text))

    for reference, prediction in list(zip(references, predictions))[:6]:
        print(f"\n  truth: {reference[:95]}\n  got  : {prediction[:95]}")
    scores = competition_error(references, predictions)
    print(f"\nrows {len(rows)}  multi_score {scores['multi_score']:.6f}  "
          f"wWER {scores['weighted_wer']:.4f}  wCER {scores['weighted_cer']:.4f}  "
          f"{(time.time()-started)/len(rows):.2f}s/row")


if __name__ == "__main__":
    main()

