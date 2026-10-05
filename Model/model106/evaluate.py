#!/usr/bin/env python3
"""Screen Model92 as a sixth character-ROVER witness."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model99.evaluate import fuse_line, prediction_map, score  # noqa: E402
from road_htr.schema import read_table  # noqa: E402


TRUTH = ROOT / "data/processed/train_folds.csv"
BASE = ROOT / "outputs/model87/word_rover_oof.csv"
MODEL92 = ROOT / "outputs/model104/model92_step400_oof.csv"
OUT = ROOT / "outputs/model106"
THRESHOLDS = (3, 4, 5, 6)


def fold_sources(fold: int, model92: dict[str, str]) -> list[dict[str, str]]:
    paths = (
        ROOT / f"outputs/model73/fold{fold}_oof_step400.csv",
        ROOT / f"outputs/model83/fold{fold}/enhanced.csv",
        ROOT / f"outputs/model83/fold{fold}/center.csv",
        ROOT / f"outputs/model83/fold{fold}/center_enhanced.csv",
        ROOT / f"outputs/model35/fold{fold}_epoch07_verified/predictions/03_confusions_fold{fold}.csv",
    )
    sources = [prediction_map(path) for path in paths]
    ids = set(sources[0])
    sources.append({identifier: model92[identifier] for identifier in ids})
    return sources


def write_csv(path: Path, rows: list[dict[str, str]], values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
        writer.writeheader()
        writer.writerows(
            {"ID": row["ID"], "prediction": values[row["ID"]]} for row in rows
        )


def main() -> None:
    rows = list(read_table(TRUTH, require_target=True).rows)
    expected = {row["ID"] for row in rows}
    base = prediction_map(BASE)
    model92 = prediction_map(MODEL92)
    if set(base) != expected or set(model92) != expected:
        raise ValueError("full OOF IDs do not match truth")

    candidates = {threshold: {} for threshold in THRESHOLDS}
    for fold in range(5):
        fold_rows = [row for row in rows if int(row["fold"]) == fold]
        ids = {row["ID"] for row in fold_rows}
        sources = fold_sources(fold, model92)
        if any(set(source) != ids for source in sources):
            raise ValueError(f"source IDs do not match fold {fold}")
        for row in fold_rows:
            identifier = row["ID"]
            texts = [source[identifier] for source in sources]
            for threshold in THRESHOLDS:
                candidates[threshold][identifier] = fuse_line(
                    base[identifier], texts, threshold
                )

    development = [row for row in rows if int(row["fold"]) == 0]
    confirmation = [row for row in rows if int(row["fold"]) != 0]
    development_scores = {
        str(threshold): score(development, values)
        for threshold, values in candidates.items()
    }
    chosen = max(
        THRESHOLDS,
        key=lambda threshold: (development_scores[str(threshold)]["multi_score"], threshold),
    )
    baseline_development = score(development, base)
    baseline_confirmation = score(confirmation, base)
    chosen_confirmation = score(confirmation, candidates[chosen])
    baseline_all = score(rows, base)
    chosen_all = score(rows, candidates[chosen])

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "six_source_character_rover_oof.csv", rows, candidates[chosen])
    report = {
        "schema_version": 1,
        "policy": "threshold selected on fold 0, frozen on folds 1-4",
        "threshold_candidates": list(THRESHOLDS),
        "chosen_threshold": chosen,
        "baseline_development": baseline_development,
        "development_scores": development_scores,
        "baseline_confirmation": baseline_confirmation,
        "chosen_confirmation": chosen_confirmation,
        "confirmation_delta": chosen_confirmation["multi_score"]
        - baseline_confirmation["multi_score"],
        "baseline_all": baseline_all,
        "chosen_all": chosen_all,
        "all_delta": chosen_all["multi_score"] - baseline_all["multi_score"],
        "changed_rows": sum(
            candidates[chosen][row["ID"]] != base[row["ID"]] for row in rows
        ),
        "promoted": chosen_confirmation["multi_score"]
        > baseline_confirmation["multi_score"],
    }
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
