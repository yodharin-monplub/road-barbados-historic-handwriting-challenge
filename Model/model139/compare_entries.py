#!/usr/bin/env python3
"""Paired row bootstrap between two out-of-fold prediction sets.

Zindi scores two selected submissions for the private leaderboard, so the
choice matters. This answers whether one candidate's CV advantage over another
survives resampling of the rows, rather than trusting a difference of a few
ten-thousandths.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from road_htr.metrics import competition_error, levenshtein, strip_conversation_tokens  # noqa: E402
from vote import read_predictions  # noqa: E402
from ensemble import truth  # noqa: E402


def row_terms(reference: str, hypothesis: str) -> tuple[float, float, float, float]:
    """Per-row numerators and denominators of the weighted WER and CER.

    Mirrors src/road_htr/metrics.py exactly: chat tokens are stripped from both
    sides first and an empty reference still carries unit weight.
    """
    ref_text = strip_conversation_tokens(reference)
    hyp_text = strip_conversation_tokens(hypothesis)
    ref_words, hyp_words = ref_text.split(), hyp_text.split()
    word_weight = max(len(ref_words), 1) ** 0.5
    char_weight = max(len(ref_text), 1) ** 0.5
    return (levenshtein(ref_words, hyp_words) * word_weight, word_weight,
            levenshtein(tuple(ref_text), tuple(hyp_text)) * char_weight, char_weight)


def score_from_terms(terms: list[tuple[float, float, float, float]]) -> float:
    """Rebuild the multi-score from per-row terms; identical to the scorer."""
    wer = sum(t[0] for t in terms) / sum(t[1] for t in terms)
    cer = sum(t[2] for t in terms) / sum(t[3] for t in terms)
    return 1.0 - 0.5 * (wer / 12.0 + cer / 55.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="OOF predictions of candidate A")
    parser.add_argument("--b", required=True, help="OOF predictions of candidate B")
    parser.add_argument("--resamples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    reference = truth()
    first, second = read_predictions(args.a), read_predictions(args.b)
    ids = [i for i in reference if i in first and i in second]
    score_a = competition_error([reference[i] for i in ids], [first[i] for i in ids])
    score_b = competition_error([reference[i] for i in ids], [second[i] for i in ids])

    # Precompute per-row terms once; the metric is a ratio of sums, so
    # resampling rows only re-sums them. Same answer, seconds instead of hours.
    terms_a = [row_terms(reference[i], first[i]) for i in ids]
    terms_b = [row_terms(reference[i], second[i]) for i in ids]
    assert abs(score_from_terms(terms_a) - score_a["multi_score"]) < 1e-9

    rng = random.Random(args.seed)
    gains = []
    for _ in range(args.resamples):
        picks = [rng.randrange(len(ids)) for _ in range(len(ids))]
        gains.append(score_from_terms([terms_b[p] for p in picks])
                     - score_from_terms([terms_a[p] for p in picks]))
    gains.sort()
    lower = gains[int(0.025 * len(gains))]
    upper = gains[int(0.975 * len(gains)) - 1]
    positive = sum(1 for g in gains if g > 0) / len(gains)
    print(json.dumps({
        "rows": len(ids),
        "a": args.a, "a_score": round(score_a["multi_score"], 6),
        "b": args.b, "b_score": round(score_b["multi_score"], 6),
        "observed_gain_b_minus_a": round(score_b["multi_score"] - score_a["multi_score"], 6),
        "bootstrap_95_interval": [round(lower, 6), round(upper, 6)],
        "fraction_of_resamples_positive": round(positive, 4),
        "resamples": args.resamples,
        "differing_rows": sum(1 for i in ids if first[i] != second[i]),
    }, indent=2))


if __name__ == "__main__":
    main()

