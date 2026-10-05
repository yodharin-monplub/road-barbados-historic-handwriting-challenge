#!/usr/bin/env python3
"""Score direct Model95 OCR against Model88 on one held-out fold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.metrics import competition_error, levenshtein  # noqa: E402
from road_htr.schema import read_table  # noqa: E402


def values(path: Path) -> tuple[list[str], dict[str, str]]:
    table = read_table(path)
    columns = [name for name in table.fieldnames if name != table.id_column]
    prediction = table.target_column or (columns[0] if len(columns) == 1 else None)
    if prediction is None:
        raise ValueError(f"cannot identify prediction column in {path}")
    ids = [row[table.id_column].strip() for row in table.rows]
    return ids, {row[table.id_column].strip(): row[prediction] for row in table.rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=Path("data/processed/train_folds.csv"))
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path("outputs/model88/word_rover_vs_kraken_selector/crossfit_predictions.csv"),
    )
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--minimum-gain", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    truth_table = read_table(args.truth, require_target=True)
    assert truth_table.target_column is not None
    selected = [row for row in truth_table.rows if int(row["fold"]) == args.fold]
    ids = [row[truth_table.id_column].strip() for row in selected]
    truth = {row[truth_table.id_column].strip(): row[truth_table.target_column] for row in selected}
    _, primary = values(args.primary)
    direct_ids, direct = values(args.direct)
    if set(direct_ids) != set(ids):
        raise ValueError("direct prediction IDs do not exactly match selected fold")
    details = {}
    for line in args.details.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            details[str(row["id"])] = row
    if set(details) != set(ids):
        raise ValueError("details IDs do not exactly match selected fold")

    valid = {
        identifier: bool(details[identifier]["valid"]) and bool(direct[identifier].strip())
        for identifier in ids
    }
    fallback = {
        identifier: direct[identifier] if valid[identifier] else primary[identifier]
        for identifier in ids
    }
    references = [truth[identifier] for identifier in ids]
    baseline_score = competition_error(references, [primary[identifier] for identifier in ids])
    direct_score = competition_error(references, [direct[identifier] for identifier in ids])
    fallback_score = competition_error(references, [fallback[identifier] for identifier in ids])
    baseline_value = baseline_score["multi_score"]
    report = {
        "schema_version": 1,
        "kind": "model95_fold_evaluation",
        "fold": args.fold,
        "rows": len(ids),
        "invalid_rows": sum(not valid[identifier] for identifier in ids),
        "baseline": baseline_score,
        "direct": direct_score,
        "invalid_fallback": fallback_score,
        "direct_delta": direct_score["multi_score"] - baseline_value,
        "invalid_fallback_delta": fallback_score["multi_score"] - baseline_value,
        "minimum_gain": args.minimum_gain,
        "promoted": fallback_score["multi_score"] - baseline_value >= args.minimum_gain,
        "decision_note": "Fold 0 is development only; folds 1-4 remain confirmation data.",
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ID", "truth", "model88", "model95_direct", "valid",
                "model88_character_edits", "model95_character_edits",
                "model88_word_edits", "model95_word_edits",
            ],
        )
        writer.writeheader()
        for identifier in ids:
            writer.writerow(
                {
                    "ID": identifier,
                    "truth": truth[identifier],
                    "model88": primary[identifier],
                    "model95_direct": direct[identifier],
                    "valid": valid[identifier],
                    "model88_character_edits": levenshtein(truth[identifier], primary[identifier]),
                    "model95_character_edits": levenshtein(truth[identifier], direct[identifier]),
                    "model88_word_edits": levenshtein(
                        truth[identifier].split(), primary[identifier].split()
                    ),
                    "model95_word_edits": levenshtein(
                        truth[identifier].split(), direct[identifier].split()
                    ),
                }
            )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
