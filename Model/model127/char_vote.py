#!/usr/bin/env python3
"""Model127: character-level vote inside aligned words (Model113's old recipe).

Same member alignment as Model124's word vote, but when the members disagree on
a word, the decision is made per character position inside that word instead of
on the whole token. Word boundaries are preserved: a word is never deleted and
no word is split or merged. Only the word-level tie-break changes.

  python model127/char_vote.py oof  --members A B C D --min-votes 2
  python model127/char_vote.py test --members A B C D --min-votes 2 --output out.csv
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
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from road_htr.metrics import competition_error  # noqa: E402
from ensemble import member_fold_vectors, member_test_consensus, truth  # noqa: E402
from vote import GAP, align  # noqa: E402


def vote_word_chars(anchor_word: str, other_words: list[str], min_votes: int) -> str:
    """Vote character by character, aligning each variant to the anchor word."""
    if not other_words:
        return anchor_word
    slots = [align(list(anchor_word), list(word)) for word in other_words]
    result: list[str] = []
    for index in range(len(anchor_word) + 1):
        current = anchor_word[index] if index < len(anchor_word) else GAP
        votes = Counter("".join(slot[index]) for slot in slots)
        best, count = votes.most_common(1)[0]
        chosen = best if (best != current and count >= min_votes
                          and votes[current] < count) else current
        result.extend(ch for ch in chosen if ch != GAP)
    return "".join(result)


def vote_line(anchor_text: str, other_texts: list[str], min_votes: int) -> str:
    anchor = anchor_text.split()
    others = [align(anchor, text.split()) for text in other_texts]
    out: list[str] = []
    for index, word in enumerate(anchor):
        variants = [" ".join(slot[index]) for slot in others]
        variants = [v for v in variants if v and v != GAP]
        if not variants:
            out.append(word)
            continue
        counts = Counter(variants)
        best, count = counts.most_common(1)[0]
        if best != word and count >= min_votes and counts[word] < count:
            out.append(best)  # a clear majority replaces the whole token
        else:
            out.append(vote_word_chars(word, variants, min_votes))
    return " ".join(w for w in out if w)


def vote_all(anchor: dict[str, str], sources: list[dict[str, str]], min_votes: int):
    result = {}
    for identifier, text in anchor.items():
        others = [s[identifier] for s in sources if identifier in s and s[identifier].strip()]
        result[identifier] = vote_line(text, others, min_votes) if others else text
    return result


def score(predictions: dict[str, str]) -> float:
    reference = truth()
    ids = [i for i in predictions if i in reference]
    return competition_error([reference[i] for i in ids],
                             [predictions[i] for i in ids])["multi_score"]


def command_oof(args) -> None:
    members = {m: member_fold_vectors(m, "oof") for m in args.members}
    anchor_name = args.members[0]
    merged: dict[str, str] = {}
    for fold in range(5):
        anchor = members[anchor_name].get(fold)
        if anchor is None:
            continue
        sources = [members[m][fold] for m in args.members[1:] if fold in members[m]]
        merged.update(vote_all(anchor, sources, args.min_votes))
    report = {"members": args.members, "min_votes": args.min_votes,
              "rows": len(merged), "char_vote_score": score(merged)}
    print(json.dumps(report, indent=2))
    out = ROOT / "Other/outputs/model127"
    out.mkdir(parents=True, exist_ok=True)
    (out / "oof_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "oof_char_vote.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "prediction"])
        writer.writerows(merged.items())


def command_test(args) -> None:
    consensus = {}
    for member in args.members:
        vector = member_test_consensus(member, args.fold_min_votes)
        if vector is not None:
            consensus[member] = vector
    names = [m for m in args.members if m in consensus]
    anchor = consensus[names[0]]
    voted = vote_all(anchor, [consensus[m] for m in names[1:]], args.min_votes)
    for key, value in list(voted.items()):
        if not value.strip():
            candidates = [anchor.get(key, "")] + [consensus[m].get(key, "") for m in names[1:]]
            voted[key] = next((c for c in candidates if c.strip()), "")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "prediction"])
        writer.writerows(voted.items())
    print(json.dumps({"members": names, "rows": len(voted), "output": str(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("oof", "test"):
        p = sub.add_parser(name)
        p.add_argument("--members", nargs="+", required=True)
        p.add_argument("--min-votes", type=int, default=2)
        if name == "test":
            p.add_argument("--fold-min-votes", type=int, default=3)
            p.add_argument("--output", required=True)
        p.set_defaults(func=command_oof if name == "oof" else command_test)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
