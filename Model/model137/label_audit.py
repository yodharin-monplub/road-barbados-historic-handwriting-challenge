#!/usr/bin/env python3
"""Model137: data-driven hunt for mislabeled training rows.

Two independent participants have reported on the Zindi forum that some
training labels look unreliable. A label error shows a distinctive signature:
the independent recognizers AGREE with each other and all disagree with the
supplied transcription. Ordinary hard handwriting looks different - the models
disagree with the label AND with each other.

For every clean training row this scores:
  consensus   = mean pairwise similarity between the five members' predictions
  label_gap   = mean similarity of the members to the supplied label
A high consensus with a low label gap is the label-error signature.

Nothing is excluded automatically: this writes a ranked audit queue, exactly
the way config/suspect_row_audit.txt handled the 21 community-reported IDs.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from road_htr.metrics import levenshtein  # noqa: E402
from ensemble import member_fold_vectors, truth  # noqa: E402

MEMBERS = ["model134_pseudo3", "model122_prod", "model122_prod_lora",
           "model123_qwen3vl4b_lora", "model130_kraken"]


def similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    distance = levenshtein(tuple(left), tuple(right))
    return 1.0 - distance / max(len(left), len(right), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", nargs="+", default=MEMBERS)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    reference = truth()
    merged = {}
    for member in args.members:
        folds = member_fold_vectors(member, "oof")
        merged[member] = {k: v for fold in folds.values() for k, v in fold.items()}
    ids = [i for i in reference if all(i in merged[m] for m in args.members)]

    rows = []
    for identifier in ids:
        predictions = [merged[m][identifier] for m in args.members]
        consensus = statistics.mean(similarity(a, b) for a, b in combinations(predictions, 2))
        label_gap = statistics.mean(similarity(p, reference[identifier]) for p in predictions)
        rows.append({"id": identifier, "consensus": consensus, "label_agreement": label_gap,
                     "signature": consensus - label_gap,
                     "label": reference[identifier],
                     "consensus_text": max(set(predictions), key=predictions.count)})

    rows.sort(key=lambda r: r["signature"], reverse=True)
    out_dir = ROOT / "Other/outputs/model137"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "label_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    strong = [r for r in rows if r["consensus"] >= 0.85 and r["label_agreement"] <= 0.60]
    print(json.dumps({
        "rows_scored": len(rows),
        "strong_label_error_signature": len(strong),
        "median_consensus": round(statistics.median(r["consensus"] for r in rows), 4),
        "median_label_agreement": round(statistics.median(r["label_agreement"] for r in rows), 4),
    }, indent=2))
    print("\ntop suspects (models agree with each other, not with the label):")
    for row in rows[: args.top]:
        print(f"\n{row['id']}  consensus={row['consensus']:.3f} label={row['label_agreement']:.3f}")
        print(f"  label    : {row['label'][:110]}")
        print(f"  models   : {row['consensus_text'][:110]}")
    (out_dir / "strong_suspects.txt").write_text(
        "\n".join(r["id"] for r in strong) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
