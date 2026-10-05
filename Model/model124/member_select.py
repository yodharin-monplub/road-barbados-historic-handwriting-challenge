#!/usr/bin/env python3
"""Backward elimination over ensemble members, with an honest check of the procedure.

Part 1 selects on the complete OOF - the same data every earlier decision used, so
the winner is optimistic by construction.
Part 2 repeats the whole selection on folds 0-3 only and scores the chosen subset on
fold 4, which never took part in that selection. If part 2's held-out gain is near
zero, part 1's gain is selection noise and should not be deployed.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from road_htr.metrics import competition_error  # noqa: E402
from vote import read_predictions, vote_all  # noqa: E402

MEMBERS = [
    "model134_pseudo3", "model140_pseudo4", "model122_prod", "model122_prod_lora",
    "model130_kraken", "model136_qwen25vl7b", "model141_gotocr2",
    "model143_qwen3vl4b_vision", "model144_qwen25vl7b_vision",
    "model145_qwen3vl4b_vision_pseudo4", "model146_qwen3vl8b_vision",
    "model147_qwen3vl8b_pseudo5", "model149_kraken_pseudo5", "model150_kraken_pseudo4",
    "model151_kraken_pseudo3",
]
ANCHOR = "model134_pseudo3"


def truth() -> dict[str, str]:
    with (ROOT / "Data/processed/train_folds.csv").open(encoding="utf-8") as handle:
        return {row["ID"]: row["Target"] for row in csv.DictReader(handle)}


def find(member: str, fold: int) -> Path | None:
    base = ROOT / "Other" / "outputs" / member
    for pattern in (f"fold{fold}*/oof_final_beam*.csv", f"fold{fold}*/oof_ep*_greedy.csv"):
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[-1]
    return None


VECTORS = {m: {f: read_predictions(p) for f in range(5) if (p := find(m, f))} for m in MEMBERS}
TRUTH = truth()


def score(members: list[str], folds: range | list[int]) -> float:
    merged: dict[str, str] = {}
    for fold in folds:
        anchor = VECTORS[members[0]][fold]
        sources = [VECTORS[m][fold] for m in members[1:] if fold in VECTORS[m]]
        merged.update(vote_all(anchor, sources, 2))
    ids = [i for i in merged if i in TRUTH]
    return competition_error([TRUTH[i] for i in ids], [merged[i] for i in ids])["multi_score"]


def eliminate(members: list[str], folds, label: str) -> list[str]:
    current = list(members)
    best = score(current, folds)
    print(f"{label}: start {len(current)} members, {best:.6f}", flush=True)
    while len(current) > 3:
        candidates = []
        for m in current:
            if m == ANCHOR:
                continue  # the anchor defines the row set
            trial = [x for x in current if x != m]
            candidates.append((score(trial, folds), m, trial))
        candidates.sort(reverse=True)
        gain = candidates[0][0] - best
        if gain <= 0:
            break
        best, dropped, current = candidates[0][0], candidates[0][1], candidates[0][2]
        print(f"{label}: drop {dropped:<34} -> {best:.6f} (+{gain:.6f})", flush=True)
    print(f"{label}: final {len(current)} members, {best:.6f}", flush=True)
    return current


print("=== PART 1: selection on the complete OOF (optimistic)")
full = eliminate(MEMBERS, range(5), "all-folds")
print("kept:", " ".join(full))

print("\n=== PART 2: selection on folds 0-3, scored on held-out fold 4")
sub = eliminate(MEMBERS, [0, 1, 2, 3], "folds0-3")
print("kept:", " ".join(sub))
base14 = [m for m in MEMBERS if m != "model151_kraken_pseudo3"]
print(f"\nheld-out fold 4: deployed 14 = {score(base14, [4]):.6f}")
print(f"held-out fold 4: part-2 subset = {score(sub, [4]):.6f}")
print(f"held-out fold 4: all 15        = {score(MEMBERS, [4]):.6f}")
