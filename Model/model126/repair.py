#!/usr/bin/env python3
"""Model126: outer-fold cross-fitted recurring-error repair.

Rebuilds the most productive stage of the lost Models 58/61/66 chain. For each
fold k the substitution table is learned ONLY from the other folds' (prediction,
truth) pairs, then applied to fold k, so the held-out score stays honest. For
the test set the table is learned from all five folds.

  python model126/repair.py oof  --predictions outputs/model124/oof_ensemble.csv
  python model126/repair.py test --predictions outputs/model124/test_ensemble.csv \
      --oof outputs/model124/oof_ensemble.csv --output outputs/model126/test_repaired.csv
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
from vote import read_predictions  # noqa: E402


def rows() -> list[dict[str, str]]:
    with (ROOT / "Data/processed/train_folds.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def align_pairs(predicted: list[str], reference: list[str]) -> list[tuple[str, str]]:
    """Word-level Levenshtein alignment, returning aligned (pred, truth) pairs."""
    n, m = len(predicted), len(reference)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = 0 if predicted[i - 1] == reference[j - 1] else 1
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + same)
    pairs: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if cost[i][j] == cost[i - 1][j - 1] + (0 if predicted[i - 1] == reference[j - 1] else 1):
            pairs.append((predicted[i - 1], reference[j - 1]))
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs


def learn(pairs: list[tuple[str, str]], min_support: int, min_precision: float) -> dict[str, str]:
    seen: Counter[str] = Counter()
    substitutions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for predicted, reference in pairs:
        seen[predicted] += 1
        substitutions[predicted][reference] += 1
    table: dict[str, str] = {}
    for word, targets in substitutions.items():
        target, count = targets.most_common(1)[0]
        if target == word:
            continue
        if count >= min_support and count / seen[word] >= min_precision:
            table[word] = target
    return table


def apply_table(text: str, table: dict[str, str]) -> str:
    return " ".join(table.get(word, word) for word in text.split())


def collect_pairs(ids, predictions, truth):
    pairs = []
    for identifier in ids:
        pairs.extend(align_pairs(predictions[identifier].split(), truth[identifier].split()))
    return pairs


def command_oof(args) -> None:
    table_rows = rows()
    truth = {r["ID"]: r["Target"] for r in table_rows}
    fold_of = {r["ID"]: int(r["fold"]) for r in table_rows}
    predictions = read_predictions(args.predictions)
    ids = [i for i in predictions if i in truth]
    repaired = dict(predictions)
    changed = 0
    sizes = {}
    for fold in range(5):
        outer = [i for i in ids if fold_of[i] != fold]
        table = learn(collect_pairs(outer, predictions, truth), args.min_support, args.min_precision)
        sizes[fold] = len(table)
        for identifier in [i for i in ids if fold_of[i] == fold]:
            new = apply_table(predictions[identifier], table)
            changed += new != predictions[identifier]
            repaired[identifier] = new
    before = competition_error([truth[i] for i in ids], [predictions[i] for i in ids])
    after = competition_error([truth[i] for i in ids], [repaired[i] for i in ids])
    report = {
        "predictions": args.predictions, "min_support": args.min_support,
        "min_precision": args.min_precision, "rules_per_fold": sizes,
        "changed_rows": changed,
        "before": before["multi_score"], "after": after["multi_score"],
        "gain": after["multi_score"] - before["multi_score"],
    }
    print(json.dumps(report, indent=2))
    out = ROOT / "Other/outputs/model126"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"oof_report_s{args.min_support}_p{args.min_precision}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


def command_test(args) -> None:
    table_rows = rows()
    truth = {r["ID"]: r["Target"] for r in table_rows}
    oof = read_predictions(args.oof)
    ids = [i for i in oof if i in truth]
    table = learn(collect_pairs(ids, oof, truth), args.min_support, args.min_precision)
    predictions = read_predictions(args.predictions)
    repaired = {k: apply_table(v, table) for k, v in predictions.items()}
    changed = sum(1 for k in predictions if predictions[k] != repaired[k])
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "prediction"])
        for key, value in repaired.items():
            writer.writerow([key, value])
    print(json.dumps({"rules": len(table), "changed_rows": changed,
                      "output": str(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("oof", "test"):
        p = sub.add_parser(name)
        p.add_argument("--predictions", required=True)
        p.add_argument("--min-support", type=int, default=3)
        p.add_argument("--min-precision", type=float, default=0.9)
        if name == "test":
            p.add_argument("--oof", required=True)
            p.add_argument("--output", required=True)
        p.set_defaults(func=command_oof if name == "oof" else command_test)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
