#!/usr/bin/env python3
"""Headroom check: the best score any per-row selection over the members could reach.

This is a diagnostic ceiling, never a deployable prediction - it peeks at the
labels to pick the best member for every row. If the ceiling is far below a
target, no smarter voting rule over these members can reach that target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from road_htr.metrics import competition_error, levenshtein  # noqa: E402
from ensemble import member_fold_vectors, truth  # noqa: E402

MEMBERS = ["model134_pseudo3", "model122_prod", "model122_prod_lora",
           "model123_qwen3vl4b_lora", "model130_kraken"]


def row_cost(reference: str, hypothesis: str) -> float:
    """Weighted edit cost of one row under the challenge metric."""
    ref_words, hyp_words = reference.split(), hypothesis.split()
    word_weight = len(ref_words) ** 0.5
    char_weight = len(reference) ** 0.5
    wer = levenshtein(ref_words, hyp_words) * word_weight
    cer = levenshtein(tuple(reference), tuple(hypothesis)) * char_weight
    return wer / 12.0 + cer / 55.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", nargs="+", default=MEMBERS)
    args = parser.parse_args()

    reference = truth()
    vectors = {m: member_fold_vectors(m, "oof") for m in args.members}
    merged = {m: {k: v for fold in vec.values() for k, v in fold.items()}
              for m, vec in vectors.items()}
    ids = [i for i in reference if all(i in merged[m] for m in args.members)]

    best_rows, per_member_wins = [], {m: 0 for m in args.members}
    for identifier in ids:
        costs = {m: row_cost(reference[identifier], merged[m][identifier]) for m in args.members}
        winner = min(costs, key=costs.get)
        per_member_wins[winner] += 1
        best_rows.append(merged[winner][identifier])

    oracle = competition_error([reference[i] for i in ids], best_rows)
    singles = {m: competition_error([reference[i] for i in ids],
                                    [merged[m][i] for i in ids])["multi_score"]
               for m in args.members}
    exact = sum(1 for i in ids
                if any(merged[m][i] == reference[i] for m in args.members))
    print(json.dumps({
        "rows": len(ids),
        "single_member_scores": {m: round(s, 6) for m, s in singles.items()},
        "oracle_row_selection": round(oracle["multi_score"], 6),
        "rows_where_some_member_is_exactly_right": exact,
        "fraction_exact": round(exact / len(ids), 4),
        "wins_per_member": per_member_wins,
    }, indent=2))


if __name__ == "__main__":
    main()
