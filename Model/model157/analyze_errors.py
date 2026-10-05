#!/usr/bin/env python3
"""Where does the deployed ensemble lose its score? (Model157, CPU only)

    python Model/model157/analyze_errors.py [--show-text]

--show-text also prints the reference and predicted text of the worst rows; keep
that output local, since the transcriptions are competition data.
Uses the complete out-of-fold predictions of the deployed twelve-member ensemble
(Model152), so every row is predicted by models that never saw it.

The competition score is 1 - (WER/12 + CER/55)/2 with length-weighted WER/CER, so
each row's share of the total loss is additive:
    loss_i = (w_word_i * wer_edits_i / sum w_word) / 24
           + (w_char_i * cer_edits_i / sum w_char) / 110,   w = sqrt(length)
The report ranks error categories by their share of that loss, and for each one
states the score gain if its rows were transcribed PERFECTLY - an upper bound on
what fixing that category could ever be worth.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
SHOW_TEXT = "--show-text" in sys.argv[1:]

from road_htr.metrics import (  # noqa: E402
    CER_NORMALIZER, WER_NORMALIZER, levenshtein, strip_conversation_tokens)

MEMBERS = [
    "model134_pseudo3", "model140_pseudo4", "model122_prod_lora", "model130_kraken",
    "model136_qwen25vl7b", "model143_qwen3vl4b_vision", "model144_qwen25vl7b_vision",
    "model145_qwen3vl4b_vision_pseudo4", "model146_qwen3vl8b_vision",
    "model147_qwen3vl8b_pseudo5", "model149_kraken_pseudo5", "model150_kraken_pseudo4",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.split()[0].strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def image_size(identifier: str) -> tuple[int, int] | None:
    path = ROOT / "Data" / "raw" / "images" / f"{identifier}.jpg"
    try:
        from PIL import Image
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "Model/model124/ensemble.py"), "oof",
                    "--members", *MEMBERS, "--min-votes", "2"],
                   cwd=ROOT, check=True, capture_output=True)
    pred = {r["ID"]: r["prediction"] for r in read_csv(ROOT / "Other/outputs/model124/oof_ensemble.csv")}
    rows = [r for r in read_csv(ROOT / "Data/processed/train_folds.csv") if r["ID"] in pred]
    bad = read_ids(ROOT / "Other/config/known_bad_train_ids.txt")
    suspect = read_ids(ROOT / "Other/config/community_reported_suspect_ids.txt")

    stats = []
    for r in rows:
        ref, hyp = strip_conversation_tokens(r["Target"]), strip_conversation_tokens(pred[r["ID"]])
        rw, hw = ref.split(), hyp.split()
        stats.append({
            "id": r["ID"], "ref": ref, "hyp": hyp,
            "ww": math.sqrt(max(len(rw), 1)), "wc": math.sqrt(max(len(ref), 1)),
            "ew": levenshtein(rw, hw), "ec": levenshtein(tuple(ref), tuple(hyp)),
            "nref": len(ref), "nhyp": len(hyp), "size": image_size(r["ID"]),
        })
    sww = sum(s["ww"] for s in stats)
    swc = sum(s["wc"] for s in stats)
    for s in stats:
        s["loss"] = (s["ww"] * s["ew"] / sww) / (2 * WER_NORMALIZER) + \
                    (s["wc"] * s["ec"] / swc) / (2 * CER_NORMALIZER)
    total = sum(s["loss"] for s in stats)
    print(f"rows {len(stats)}  ensemble OOF score {1 - total:.6f}  total loss {total:.6f}")
    print(f"rows with any error: {sum(1 for s in stats if s['ec'] > 0)}")

    heights = sorted(s["size"][1] for s in stats if s["size"])
    median_h = heights[len(heights) // 2] if heights else 0
    print(f"image sizes read: {len(heights)}  median height {median_h}px")

    def category(s) -> list[str]:
        cats = []
        ratio = (s["nhyp"] + 1) / (s["nref"] + 1)
        if s["nhyp"] == 0:
            cats.append("empty prediction")
        if ratio >= 1.5:
            cats.append("prediction >= 1.5x target length (over-transcribed)")
        elif ratio <= 0.67:
            cats.append("prediction <= 0.67x target length (truncated)")
        if s["size"]:
            w, h = s["size"]
            if h >= 1.8 * median_h:
                cats.append("tall crop (>= 1.8x median height: likely multi-line)")
            if w / max(h, 1) < 6:
                cats.append("low aspect ratio (< 6: likely multi-line)")
        if s["id"] in bad:
            cats.append("in known_bad_train_ids")
        if s["id"] in suspect:
            cats.append("in community_reported_suspect_ids")
        if s["nref"] >= 90:
            cats.append("long line (>= 90 chars)")
        if s["ec"] > 0 and s["ec"] <= 2:
            cats.append("near miss (1-2 char edits)")
        if any(ch.isdigit() for ch in s["ref"]):
            cats.append("target contains digits")
        if "^" in s["ref"]:
            cats.append("target contains superscript marker ^")
        return cats

    share = defaultdict(float)
    count = defaultdict(int)
    errrows = defaultdict(int)
    for s in stats:
        for c in category(s):
            share[c] += s["loss"]
            count[c] += 1
            errrows[c] += s["ec"] > 0
    print("\nCATEGORY (rows can be in several)            rows  w/err  loss share  max gain if perfect")
    for c in sorted(share, key=lambda k: -share[k]):
        print(f"  {c:<52} {count[c]:>5} {errrows[c]:>6}   {100 * share[c] / total:6.2f}%    +{share[c]:.5f}")

    worst = sorted(stats, key=lambda s: -s["loss"])[:40]
    print(f"\nTOP 40 rows hold {100 * sum(s['loss'] for s in worst) / total:.2f}% of all loss")
    for s in worst:
        size = f"{s['size'][0]}x{s['size'][1]}" if s["size"] else "?"
        print(f"  {s['id']} {size:>9} loss {s['loss']:.6f} ec {s['ec']:>3}")
        # Transcriptions are competition data (the licence bars publishing them), so
        # they are printed only on request, for local inspection.
        if SHOW_TEXT:
            print(f"      REF: {s['ref'][:150]}")
            print(f"      HYP: {s['hyp'][:150]}")
    decile = sorted(stats, key=lambda s: -s["loss"])
    top10 = sum(s["loss"] for s in decile[: len(decile) // 10])
    print(f"\nconcentration: worst 10% of rows = {100 * top10 / total:.1f}% of loss")
    json.dump([{k: s[k] for k in ("id", "loss", "ec", "ew", "nref", "nhyp")} | {"size": s["size"]}
               for s in stats], (ROOT / "Model/model157/row_errors.json").open("w"), indent=0)


if __name__ == "__main__":
    main()
