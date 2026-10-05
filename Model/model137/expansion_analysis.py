#!/usr/bin/env python3
"""How much does the abbreviation-then-expansion convention cost us?

Some supplied transcriptions write an abbreviation AND its expansion in line,
e.g. "Execut:^rs Executors", "und^:r under", "S:^d Said". The recognizers emit
only the abbreviation, so every expansion is a missing word - and word edits
are the expensive half of the metric.

This measures: how common the convention is, how consistent it is (does the
same abbreviation always get expanded?), and how much of the anchor's remaining
word error it accounts for.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from road_htr.metrics import levenshtein  # noqa: E402
from ensemble import member_fold_vectors, truth  # noqa: E402

ABBREV = re.compile(r"[\^:]")


def looks_expanded(abbrev: str, following: str) -> bool:
    """Is `following` plausibly the expansion of `abbrev`?"""
    core = re.sub(r"[^A-Za-z]", "", abbrev).lower()
    word = re.sub(r"[^A-Za-z]", "", following).lower()
    if not core or not word or word == core:
        return False
    return word.startswith(core[:3]) and len(word) >= len(core)


def main() -> None:
    reference = truth()
    anchor_folds = member_fold_vectors("model134_pseudo3", "oof")
    anchor = {k: v for fold in anchor_folds.values() for k, v in fold.items()}

    pairs: Counter[tuple[str, str]] = Counter()
    abbrev_seen: Counter[str] = Counter()
    abbrev_expanded: Counter[str] = Counter()
    rows_with = 0
    for identifier, label in reference.items():
        words = label.split()
        has = False
        for index, word in enumerate(words):
            if not ABBREV.search(word):
                continue
            abbrev_seen[word] += 1
            if index + 1 < len(words) and looks_expanded(word, words[index + 1]):
                abbrev_expanded[word] += 1
                pairs[(word, words[index + 1])] += 1
                has = True
        rows_with += has
    total_rows = len(reference)

    # How much of the anchor's word error sits on these expansion words?
    missing_expansion_edits = 0
    total_word_edits = 0
    for identifier, label in reference.items():
        if identifier not in anchor:
            continue
        label_words, pred_words = label.split(), anchor[identifier].split()
        total_word_edits += levenshtein(label_words, pred_words)
        predicted = set(pred_words)
        for index, word in enumerate(label_words):
            if (ABBREV.search(word) and index + 1 < len(label_words)
                    and looks_expanded(word, label_words[index + 1])
                    and label_words[index + 1] not in predicted):
                missing_expansion_edits += 1

    consistency = {}
    for abbrev, count in abbrev_seen.most_common(15):
        if abbrev_expanded[abbrev]:
            consistency[abbrev] = {"seen": count, "expanded": abbrev_expanded[abbrev],
                                   "rate": round(abbrev_expanded[abbrev] / count, 3)}
    print(json.dumps({
        "training_rows": total_rows,
        "rows_containing_an_expansion": rows_with,
        "fraction_of_rows": round(rows_with / total_rows, 4),
        "distinct_abbrev_expansion_pairs": len(pairs),
        "anchor_total_word_edits": total_word_edits,
        "word_edits_that_are_missing_expansions": missing_expansion_edits,
        "share_of_word_error": round(missing_expansion_edits / max(total_word_edits, 1), 4),
        "most_common_pairs": [f"{a} -> {b} ({n})" for (a, b), n in pairs.most_common(12)],
        "expansion_rate_by_abbreviation": consistency,
    }, indent=2))


if __name__ == "__main__":
    main()
