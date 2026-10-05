#!/usr/bin/env python3
"""Assemble Model122 fold predictions into a complete OOF file and score it.

  python model122/assemble.py oof  --run model122_full_native --epoch 4 [--beams 1]
  python model122/assemble.py score --predictions outputs/.../oof_complete.csv
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
sys.path.insert(0, str(ROOT / "Other" / "src"))

from road_htr.metrics import competition_error  # noqa: E402


def read_predictions(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        column = [c for c in reader.fieldnames if c != "ID"][0]
        return {row["ID"]: row[column] for row in reader}


def read_truth() -> dict[str, str]:
    with (ROOT / "Data/processed/train_folds.csv").open(encoding="utf-8") as handle:
        return {row["ID"]: row["Target"] for row in csv.DictReader(handle)}


def write_predictions(path: Path, predictions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "prediction"])
        for key, value in predictions.items():
            writer.writerow([key, value])


def score(predictions: dict[str, str], truth: dict[str, str]) -> dict[str, float]:
    ids = [i for i in truth if i in predictions]
    return competition_error([truth[i] for i in ids], [predictions[i] for i in ids])


def command_oof(args) -> None:
    run_dir = ROOT / "Other" / "outputs" / args.run
    name = f"oof_final_beam{args.beams}.csv" if args.beams > 1 else f"oof_ep{args.epoch}_greedy.csv"
    merged: dict[str, str] = {}
    folds = []
    for fold in range(5):
        matches = sorted(run_dir.glob(f"fold{fold}*/{name}"))
        if not matches:
            print(f"missing fold {fold}: {name}")
            continue
        folds.append(fold)
        merged.update(read_predictions(matches[0]))
    truth = read_truth()
    metrics = score(merged, truth)
    metrics.update(run=args.run, epoch=args.epoch, beams=args.beams,
                   folds=folds, rows=len(merged))
    out = run_dir / f"oof_complete_ep{args.epoch}_beam{args.beams}.csv"
    write_predictions(out, merged)
    (run_dir / f"oof_report_ep{args.epoch}_beam{args.beams}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def command_score(args) -> None:
    predictions = read_predictions(Path(args.predictions))
    print(json.dumps(score(predictions, read_truth()), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    oof = sub.add_parser("oof")
    oof.add_argument("--run", required=True)
    oof.add_argument("--epoch", type=int, default=4)
    oof.add_argument("--beams", type=int, default=1)
    oof.set_defaults(func=command_oof)
    sc = sub.add_parser("score")
    sc.add_argument("--predictions", required=True)
    sc.set_defaults(func=command_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
