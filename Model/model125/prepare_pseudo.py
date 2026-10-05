#!/usr/bin/env python3
"""Collect each fold model's own test predictions as that fold's pseudo-labels.

Fold k's pseudo-label file must come from the fold-k model, which never saw
fold k, so training on it leaves fold k's held-out score honest.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["model122_prod_beam", "model122_prod"],
                        help="output dirs searched, in order, for each fold's test file")
    parser.add_argument("--output-dir", default="Other/outputs/pseudo")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for fold in range(5):
        source = None
        for run in args.runs:
            matches = sorted((ROOT / "Other" / "outputs" / run).glob(f"fold{fold}*/test_beam*.csv"))
            if matches:
                source = matches[-1]
                break
        if source is None:
            print(f"fold {fold}: no test predictions found")
            continue
        with source.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            column = [c for c in reader.fieldnames if c != "ID"][0]
            rows = [(row["ID"], row[column].strip()) for row in reader]
        kept = [(i, t) for i, t in rows if t]
        target = out_dir / f"fold{fold}.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ID", "Target"])
            writer.writerows(kept)
        manifest[f"fold{fold}"] = {"source": str(source.relative_to(ROOT)),
                                   "rows": len(kept), "dropped_blank": len(rows) - len(kept)}
        print(f"fold {fold}: {len(kept)} pseudo rows from {source.name}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
