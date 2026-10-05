#!/usr/bin/env python3
"""Model135: architecture-aware word vote (Model112's "family" rule, rebuilt).

Model124's rule counts members equally, so the four LightOnOCR/Qwen members can
outvote the anchor on their own while the single CTC member is easily ignored.
Model112 recorded that gains came from agreement ACROSS architectures. Here each
member carries a family tag and an alternative replaces the anchor word only
when it is supported by at least `min_families` distinct families (and by more
members than the anchor's own word).

  python model135/family_vote.py oof  --min-families 2
  python model135/family_vote.py test --min-families 2 --output out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from road_htr.metrics import competition_error  # noqa: E402
from ensemble import member_fold_vectors, member_test_consensus, truth  # noqa: E402
from vote import GAP, align  # noqa: E402

# family tags: the architecture whose errors are shared
FAMILIES = {
    "model134_pseudo3": "lighton",
    "model129_pseudo2": "lighton",
    "model125_pseudo": "lighton",
    "model122_prod": "lighton",
    "model122_prod_lora": "lighton",
    "model123_qwen3vl4b_lora": "qwen",
    "model130_kraken": "ctc",
    "model132_kraken128": "ctc",
}
DEFAULT_ANCHOR = "model134_pseudo3"
DEFAULT_SOURCES = ["model122_prod", "model122_prod_lora",
                   "model123_qwen3vl4b_lora", "model130_kraken"]


def vote_line(anchor_text: str, others: list[tuple[str, str]], min_families: int,
              min_votes: int) -> str:
    """others: list of (family, text)."""
    anchor = anchor_text.split()
    aligned = [(family, align(anchor, text.split())) for family, text in others]
    result: list[str] = []
    for index in range(len(anchor) + 1):
        current = " ".join([anchor[index]] if index < len(anchor) else [GAP])
        counts: Counter[str] = Counter()
        families: defaultdict[str, set[str]] = defaultdict(set)
        for family, slots in aligned:
            candidate = " ".join(slots[index])
            counts[candidate] += 1
            families[candidate].add(family)
        best, count = counts.most_common(1)[0]
        if (best != current and count >= min_votes
                and len(families[best]) >= min_families
                and counts[current] < count):
            chosen = best
        else:
            chosen = current
        result.extend(word for word in chosen.split(" ") if word and word != GAP)
    return " ".join(result)


def vote_all(anchor: dict[str, str], others: list[tuple[str, dict[str, str]]],
             min_families: int, min_votes: int) -> dict[str, str]:
    out = {}
    for identifier, text in anchor.items():
        pairs = [(family, vec[identifier]) for family, vec in others
                 if identifier in vec and vec[identifier].strip()]
        out[identifier] = vote_line(text, pairs, min_families, min_votes) if pairs else text
    return out


def score(predictions: dict[str, str]) -> float:
    reference = truth()
    ids = [i for i in predictions if i in reference]
    return competition_error([reference[i] for i in ids],
                             [predictions[i] for i in ids])["multi_score"]


def command_oof(args) -> None:
    anchor_folds = member_fold_vectors(args.anchor, "oof")
    source_folds = {m: member_fold_vectors(m, "oof") for m in args.sources}
    merged: dict[str, str] = {}
    for fold in range(5):
        if fold not in anchor_folds:
            continue
        others = [(FAMILIES[m], source_folds[m][fold]) for m in args.sources
                  if fold in source_folds[m]]
        merged.update(vote_all(anchor_folds[fold], others, args.min_families, args.min_votes))
    report = {"anchor": args.anchor, "sources": args.sources,
              "min_families": args.min_families, "min_votes": args.min_votes,
              "rows": len(merged), "score": score(merged)}
    print(json.dumps(report))
    out = ROOT / "Other/outputs/model135"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"oof_f{args.min_families}_v{args.min_votes}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


def command_test(args) -> None:
    anchor = member_test_consensus(args.anchor, args.fold_min_votes)
    others = [(FAMILIES[m], member_test_consensus(m, args.fold_min_votes))
              for m in args.sources]
    others = [(family, vec) for family, vec in others if vec]
    voted = vote_all(anchor, others, args.min_families, args.min_votes)
    for key, value in list(voted.items()):
        if not value.strip():
            candidates = [anchor.get(key, "")] + [vec.get(key, "") for _, vec in others]
            voted[key] = next((c for c in candidates if c.strip()), "")
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "prediction"])
        writer.writerows(voted.items())
    print(json.dumps({"rows": len(voted), "output": str(path)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("oof", "test"):
        p = sub.add_parser(name)
        p.add_argument("--anchor", default=DEFAULT_ANCHOR)
        p.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
        p.add_argument("--min-families", type=int, default=2)
        p.add_argument("--min-votes", type=int, default=2)
        if name == "test":
            p.add_argument("--fold-min-votes", type=int, default=3)
            p.add_argument("--output", required=True)
        p.set_defaults(func=command_oof if name == "oof" else command_test)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
