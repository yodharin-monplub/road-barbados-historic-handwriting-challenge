#!/usr/bin/env python3
"""Fast paired bootstrap of two complete OOF prediction files.

The competition metric is a ratio of per-row sums (weighted edits / weights, for
words and for characters), so edit distances are computed once per row and each
resample only re-sums four vectors. 3,000 resamples take seconds instead of hours.

  python Model/model124/bootstrap.py --base A.csv --candidate B.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))

from road_htr.metrics import (  # noqa: E402
    CER_NORMALIZER, WER_NORMALIZER, levenshtein, strip_conversation_tokens)


def load(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["ID"]: row["prediction"] for row in csv.DictReader(handle)}


def row_terms(ref: str, hyp: str) -> tuple[float, float, float, float]:
    ref, hyp = strip_conversation_tokens(ref), strip_conversation_tokens(hyp)
    rw, hw = ref.split(), hyp.split()
    ww = math.sqrt(max(len(rw), 1))
    wc = math.sqrt(max(len(ref), 1))
    return levenshtein(rw, hw) * ww, ww, levenshtein(tuple(ref), tuple(hyp)) * wc, wc


def scores(terms: np.ndarray, index: np.ndarray) -> np.ndarray:
    """multi_score for each resample; terms is (rows, 4), index is (resamples, rows)."""
    sums = terms[index].sum(axis=1)
    wer = sums[:, 0] / sums[:, 1]
    cer = sums[:, 2] / sums[:, 3]
    return 1 - (wer / WER_NORMALIZER + cer / CER_NORMALIZER) / 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--resamples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    with (ROOT / "Data/processed/train_folds.csv").open(encoding="utf-8") as handle:
        truth = {row["ID"]: row["Target"] for row in csv.DictReader(handle)}
    base, cand = load(args.base), load(args.candidate)
    ids = sorted(set(base) & set(cand) & set(truth))
    a = np.array([row_terms(truth[i], base[i]) for i in ids])
    b = np.array([row_terms(truth[i], cand[i]) for i in ids])

    full = np.arange(len(ids))[None, :]
    point = scores(b, full)[0] - scores(a, full)[0]
    rng = np.random.default_rng(args.seed)
    gains = []
    for start in range(0, args.resamples, 250):
        index = rng.integers(0, len(ids), size=(min(250, args.resamples - start), len(ids)))
        gains.append(scores(b, index) - scores(a, index))
    gains = np.concatenate(gains)
    low, high = np.percentile(gains, [2.5, 97.5])
    print(f"rows {len(ids)}  rows changed {sum(base[i] != cand[i] for i in ids)}")
    print(f"base {scores(a, full)[0]:.6f}  candidate {scores(b, full)[0]:.6f}  gain {point:+.6f}")
    print(f"bootstrap 95% [{low:+.6f}, {high:+.6f}]  positive {100 * (gains > 0).mean():.1f}%")


if __name__ == "__main__":
    main()
