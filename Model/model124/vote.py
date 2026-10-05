#!/usr/bin/env python3
"""Model124: anchored word-level majority vote across several transcript sources.

Each source is aligned to the anchor with a word-level Levenshtein alignment;
at every anchor slot the aligned words are collected and the anchor word is
replaced only when at least `min_votes` other sources agree on the same
alternative. Deletions and insertions are voted the same way, so a missing or
extra word can be fixed too.

  python model124/vote.py score --anchor A.csv --sources B.csv C.csv --min-votes 2
  python model124/vote.py apply --anchor A.csv --sources B.csv C.csv --output out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))

from road_htr.metrics import competition_error  # noqa: E402

GAP = "\x00"  # stands for "this source has no word in this slot"


def read_predictions(path: str | Path) -> dict[str, str]:
    with Path(path).open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        column = [c for c in reader.fieldnames if c != "ID"][0]
        return {row["ID"]: row[column] for row in reader}


def align(anchor: list[str], other: list[str]) -> list[list[str]]:
    """Return, for each anchor slot, the words of `other` aligned to it.

    Slot i holds the words aligned to anchor[i]; slot len(anchor) holds words
    appended past the end. A slot is [GAP] when the source aligns nothing.
    """
    rows, columns = len(anchor), len(other)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        cost[i][0] = i
    for j in range(columns + 1):
        cost[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            same = 0 if anchor[i - 1] == other[j - 1] else 1
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + same)
    slots: list[list[str]] = [[] for _ in range(rows + 1)]
    i, j = rows, columns
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (
                0 if anchor[i - 1] == other[j - 1] else 1):
            slots[i - 1].insert(0, other[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            i -= 1  # anchor word with no counterpart -> deletion vote
        else:
            slots[i].insert(0, other[j - 1])  # extra source word in this slot
            j -= 1
    return [slot or [GAP] for slot in slots]


def vote_line(anchor_text: str, other_texts: list[str], min_votes: int) -> str:
    anchor = anchor_text.split()
    others = [align(anchor, text.split()) for text in other_texts]
    result: list[str] = []
    for index in range(len(anchor) + 1):
        current = [anchor[index]] if index < len(anchor) else [GAP]
        votes = Counter(" ".join(slot[index]) for slot in others)
        current_key = " ".join(current)
        best, count = votes.most_common(1)[0]
        if best != current_key and count >= min_votes and votes[current_key] < count:
            chosen = best
        else:
            chosen = current_key
        result.extend(word for word in chosen.split(" ") if word and word != GAP)
    return " ".join(result)


def vote_all(anchor: dict[str, str], sources: list[dict[str, str]], min_votes: int) -> dict[str, str]:
    output = {}
    for identifier, text in anchor.items():
        others = [source[identifier] for source in sources
                  if identifier in source and source[identifier].strip()]
        output[identifier] = vote_line(text, others, min_votes) if others else text
    return output


def truth() -> dict[str, str]:
    with (ROOT / "Data/processed/train_folds.csv").open(encoding="utf-8") as handle:
        return {row["ID"]: row["Target"] for row in csv.DictReader(handle)}


def score(predictions: dict[str, str]) -> dict[str, float]:
    reference = truth()
    ids = [i for i in predictions if i in reference]
    return competition_error([reference[i] for i in ids], [predictions[i] for i in ids])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["score", "apply"])
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()

    anchor = read_predictions(args.anchor)
    sources = [read_predictions(path) for path in args.sources]
    voted = vote_all(anchor, sources, args.min_votes)
    if args.cmd == "score":
        base = score(anchor)
        after = score(voted)
        changed = sum(1 for k in anchor if anchor[k] != voted[k])
        print(json.dumps({"anchor_score": base["multi_score"],
                          "voted_score": after["multi_score"],
                          "gain": after["multi_score"] - base["multi_score"],
                          "changed_rows": changed,
                          "min_votes": args.min_votes,
                          "sources": len(sources)}, indent=2))
    else:
        if not args.output:
            raise SystemExit("--output is required for apply")
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ID", "prediction"])
            for key, value in voted.items():
                writer.writerow([key, value])
        print(f"wrote {path} ({len(voted)} rows)")


if __name__ == "__main__":
    main()
