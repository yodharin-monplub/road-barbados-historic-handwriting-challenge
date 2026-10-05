#!/usr/bin/env python3
"""Score an ensemble rule on the folds that are finished, before all five exist.

Used to watch the Qwen2.5-VL-7B member's contribution accumulate while its last
fold is still training. Restricting BOTH rules to the same folds keeps the
comparison honest.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from road_htr.metrics import competition_error  # noqa: E402
from ensemble import member_fold_vectors, truth  # noqa: E402
from vote import vote_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", default="model134_pseudo3")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--min-votes", type=int, default=2)
    args = parser.parse_args()

    reference = truth()
    with (ROOT / "Data/processed/train_folds.csv").open(encoding="utf-8") as handle:
        fold_of = {row["ID"]: int(row["fold"]) for row in csv.DictReader(handle)}

    anchor_folds = member_fold_vectors(args.anchor, "oof")
    source_folds = {m: member_fold_vectors(m, "oof") for m in args.sources}
    merged_anchor, merged_vote = {}, {}
    used = []
    for fold in args.folds:
        if fold not in anchor_folds:
            continue
        available = [m for m in args.sources if fold in source_folds[m]]
        if len(available) != len(args.sources):
            print(f"fold {fold}: only {available} available - skipped")
            continue
        used.append(fold)
        anchor = anchor_folds[fold]
        merged_anchor.update(anchor)
        merged_vote.update(vote_all(anchor, [source_folds[m][fold] for m in available],
                                    args.min_votes))

    ids = [i for i in merged_vote if i in reference and fold_of.get(i) in used]
    def score(predictions):
        return competition_error([reference[i] for i in ids],
                                 [predictions[i] for i in ids])["multi_score"]
    print(json.dumps({"folds": used, "rows": len(ids), "sources": args.sources,
                      "anchor_only": round(score(merged_anchor), 6),
                      "ensemble": round(score(merged_vote), 6)}, indent=2))


if __name__ == "__main__":
    main()
