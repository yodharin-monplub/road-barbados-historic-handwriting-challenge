#!/usr/bin/env python3
"""Model142: quality-weighted word vote.

Model124 counts every member equally, so Kraken (0.806 out-of-fold) carries the
same weight as the Qwen2.5-VL-7B member (0.902). That is deliberate - diversity
has mattered more than strength - but it is worth measuring whether weighting a
member's vote by how often it is right does better than one-member-one-vote.

Two schemes, both fixed before looking at the result:
  "oof"      weight = member's complete out-of-fold multi_score minus 0.8,
             i.e. proportional to how far above a weak baseline it is
  "logit"    weight = log(s / (1 - s)) on the same score, which spreads strong
             members further apart
An alternative replaces the anchor word when its weighted support exceeds the
anchor's own weighted support by at least `margin`.

  python Model/model142/weighted_vote.py --scheme oof --margin 1.0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from road_htr.metrics import competition_error  # noqa: E402
from ensemble import member_fold_vectors, truth  # noqa: E402
from vote import GAP, align  # noqa: E402

# complete five-fold out-of-fold scores, measured, not guessed
OOF = {
    "model134_pseudo3": 0.902252,
    "model140_pseudo4": 0.902625,
    "model136_qwen25vl7b": 0.902391,
    "model122_prod": 0.896608,
    "model122_prod_lora": 0.892456,
    "model123_qwen3vl4b_lora": 0.836949,
    "model130_kraken": 0.805866,
}
ANCHOR = "model134_pseudo3"
SOURCES = ["model122_prod", "model122_prod_lora", "model123_qwen3vl4b_lora",
           "model130_kraken", "model136_qwen25vl7b"]


def weight_of(member: str, scheme: str) -> float:
    score = OOF[member]
    if scheme == "equal":
        return 1.0
    if scheme == "oof":
        return max(score - 0.80, 0.01) * 10.0
    if scheme == "logit":
        return math.log(score / (1.0 - score))
    raise ValueError(scheme)


def vote_line(anchor_text: str, others: list[tuple[float, str]], margin: float) -> str:
    anchor = anchor_text.split()
    aligned = [(weight, align(anchor, text.split())) for weight, text in others]
    result: list[str] = []
    for index in range(len(anchor) + 1):
        current = " ".join([anchor[index]] if index < len(anchor) else [GAP])
        support: defaultdict[str, float] = defaultdict(float)
        for weight, slots in aligned:
            support[" ".join(slots[index])] += weight
        best = max(support, key=support.get)
        chosen = best if (best != current and support[best] - support[current] >= margin) else current
        result.extend(word for word in chosen.split(" ") if word and word != GAP)
    return " ".join(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", default=ANCHOR)
    parser.add_argument("--sources", nargs="+", default=SOURCES)
    parser.add_argument("--scheme", default="oof", choices=["equal", "oof", "logit"])
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()

    reference = truth()
    anchor_folds = member_fold_vectors(args.anchor, "oof")
    source_folds = {m: member_fold_vectors(m, "oof") for m in args.sources}
    weights = {m: weight_of(m, args.scheme) for m in args.sources}

    merged: dict[str, str] = {}
    for fold in range(5):
        if fold not in anchor_folds:
            continue
        anchor = anchor_folds[fold]
        pairs = [(weights[m], source_folds[m][fold]) for m in args.sources
                 if fold in source_folds[m]]
        for identifier, text in anchor.items():
            others = [(w, vec[identifier]) for w, vec in pairs
                      if identifier in vec and vec[identifier].strip()]
            merged[identifier] = vote_line(text, others, args.margin) if others else text

    ids = [i for i in merged if i in reference]
    score = competition_error([reference[i] for i in ids], [merged[i] for i in ids])
    print(json.dumps({"scheme": args.scheme, "margin": args.margin,
                      "weights": {m: round(w, 3) for m, w in weights.items()},
                      "rows": len(ids), "score": round(score["multi_score"], 6)}))
    out = ROOT / "Other/outputs/model142"
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"oof_{args.scheme}_{args.margin}.csv").open("w", encoding="utf-8",
                                                             newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "prediction"])
        writer.writerows(merged.items())


if __name__ == "__main__":
    main()
