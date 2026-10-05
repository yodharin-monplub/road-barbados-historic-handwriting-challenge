#!/usr/bin/env python3
"""Which member should be the ensemble anchor?

vote_all starts from the anchor's prediction and only replaces a line when enough
members agree on the same alternative, so the anchor is a real hyper-parameter -
Model141 found that swapping it changed 12 test lines. It has never been chosen
deliberately: model134_pseudo3 has been the anchor since Model134 simply because
it was listed first.

Part 1 ranks every member as anchor on the complete OOF. Part 2 repeats the ranking
on folds 0-3 and scores the winner on fold 4, which took no part in the ranking.
Anchors are scored in parallel processes (one per anchor).
"""
from __future__ import annotations

import csv
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from road_htr.metrics import competition_error  # noqa: E402
from vote import read_predictions, vote_all  # noqa: E402

# The Model152 pruned twelve (deployed, public 0.919377).
MEMBERS = [
    "model134_pseudo3", "model140_pseudo4", "model122_prod_lora", "model130_kraken",
    "model136_qwen25vl7b", "model143_qwen3vl4b_vision", "model144_qwen25vl7b_vision",
    "model145_qwen3vl4b_vision_pseudo4", "model146_qwen3vl8b_vision",
    "model147_qwen3vl8b_pseudo5", "model149_kraken_pseudo5", "model150_kraken_pseudo4",
]


def find(member: str, fold: int) -> Path | None:
    base = ROOT / "Other" / "outputs" / member
    for pattern in (f"fold{fold}*/oof_final_beam*.csv", f"fold{fold}*/oof_ep*_greedy.csv"):
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[-1]
    return None


VECTORS = {m: {f: read_predictions(p) for f in range(5) if (p := find(m, f))} for m in MEMBERS}
with (ROOT / "Data/processed/train_folds.csv").open(encoding="utf-8") as handle:
    TRUTH = {row["ID"]: row["Target"] for row in csv.DictReader(handle)}


def score(anchor: str, folds) -> float:
    others = [m for m in MEMBERS if m != anchor]
    merged: dict[str, str] = {}
    for fold in folds:
        merged.update(vote_all(VECTORS[anchor][fold],
                               [VECTORS[m][fold] for m in others], 2))
    ids = [i for i in merged if i in TRUTH]
    return competition_error([TRUTH[i] for i in ids], [merged[i] for i in ids])["multi_score"]


def job(args):
    anchor, folds = args
    return score(anchor, folds), anchor


if __name__ == "__main__":
    with Pool(min(len(MEMBERS), 16)) as pool:
        full = sorted(pool.map(job, [(m, range(5)) for m in MEMBERS]), reverse=True)
        part = sorted(pool.map(job, [(m, [0, 1, 2, 3]) for m in MEMBERS]), reverse=True)
        best34 = part[0][1]
        held = dict((a, s) for s, a in pool.map(job, [("model134_pseudo3", [4]), (best34, [4])]))
    print("=== PART 1: every anchor, complete OOF", flush=True)
    for value, member in full:
        mark = "  <- current" if member == "model134_pseudo3" else ""
        print(f"  {member:<36} {value:.6f}{mark}", flush=True)
    print("\n=== PART 2: ranked on folds 0-3, checked on held-out fold 4", flush=True)
    print(f"  best anchor on folds 0-3: {best34} ({part[0][0]:.6f})")
    print(f"  held-out fold 4, current anchor: {held['model134_pseudo3']:.6f}")
    print(f"  held-out fold 4, that anchor:    {held[best34]:.6f}")
