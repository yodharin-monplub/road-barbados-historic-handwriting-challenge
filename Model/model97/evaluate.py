#!/usr/bin/env python3
"""Reveal labels and score the already-frozen blinded Model97 sample."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.metrics import competition_error  # noqa: E402
from road_htr.schema import read_table  # noqa: E402


def prediction_map(path: Path) -> tuple[list[str], dict[str, str]]:
    table = read_table(path)
    columns = [column for column in table.fieldnames if column != table.id_column]
    if len(columns) != 1:
        raise ValueError(f"expected one prediction column in {path}")
    ids = [row[table.id_column].strip() for row in table.rows]
    return ids, {row[table.id_column].strip(): row[columns[0]] for row in table.rows}


def main() -> None:
    truth_table = read_table(ROOT / "data/processed/train_folds.csv", require_target=True)
    assert truth_table.target_column is not None
    truth = {
        row[truth_table.id_column].strip(): row[truth_table.target_column]
        for row in truth_table.rows
    }
    ids, codex = prediction_map(ROOT / "outputs/model97/codex_blind_predictions.csv")
    _, model88 = prediction_map(
        ROOT / "outputs/model88/word_rover_vs_kraken_selector/crossfit_predictions.csv"
    )
    references = [truth[identifier] for identifier in ids]
    codex_score = competition_error(references, [codex[identifier] for identifier in ids])
    model88_score = competition_error(references, [model88[identifier] for identifier in ids])
    report = {
        "schema_version": 1,
        "kind": "model97_blinded_codex_sample",
        "rows": len(ids),
        "sample_source": "Model95 fold-0 ID/image-only manifest",
        "labels_seen_before_predictions_frozen": False,
        "codex_vision": codex_score,
        "model88_same_rows": model88_score,
        "delta": codex_score["multi_score"] - model88_score["multi_score"],
        "warning": "Small diagnostic sample; not a full-fold OOF estimate.",
    }
    output = ROOT / "outputs/model97"
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["ID", "truth", "codex_vision", "model88"]
        )
        writer.writeheader()
        for identifier in ids:
            writer.writerow(
                {
                    "ID": identifier,
                    "truth": truth[identifier],
                    "codex_vision": codex[identifier],
                    "model88": model88[identifier],
                }
            )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
