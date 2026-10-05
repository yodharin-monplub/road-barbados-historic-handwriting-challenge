#!/usr/bin/env python3
"""Model129 round-2 pseudo-labels: per-fold ensemble votes over the test images.

Round 1 (Model125) labeled the test set with a single fold model. Here fold k's
labels come from a 4-member word vote among ONLY the fold-k models of each
member, every one of which excluded fold k from training. The ensemble labels
are more accurate than any single member, and fold k stays unseen, so its
held-out score remains honest.

Using the deployed Model124 test ensemble instead would be wrong: that vector
mixes in models trained on fold k.
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
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from ensemble import member_fold_vectors  # noqa: E402
from vote import vote_all  # noqa: E402

MEMBERS = ["model125_pseudo", "model122_prod", "model122_prod_lora", "model123_qwen3vl4b_lora"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", nargs="+", default=MEMBERS)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--output-dir", default="Other/outputs/pseudo2")
    args = parser.parse_args()

    vectors = {m: member_fold_vectors(m, "test") for m in args.members}
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for fold in range(5):
        available = [m for m in args.members if fold in vectors[m]]
        if not available:
            print(f"fold {fold}: no member has test predictions")
            continue
        anchor = vectors[available[0]][fold]
        sources = [vectors[m][fold] for m in available[1:]]
        voted = vote_all(anchor, sources, args.min_votes)
        kept = [(i, t.strip()) for i, t in voted.items() if t.strip()]
        path = out_dir / f"fold{fold}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ID", "Target"])
            writer.writerows(kept)
        changed = sum(1 for i, t in voted.items() if t != anchor[i])
        manifest[f"fold{fold}"] = {"members": available, "rows": len(kept),
                                   "changed_vs_anchor": changed}
        print(f"fold {fold}: {len(kept)} rows, {changed} changed by the vote")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
