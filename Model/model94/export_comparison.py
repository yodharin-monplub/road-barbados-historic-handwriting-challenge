#!/usr/bin/env python3
"""Export human-readable before/after Model93 and Model94 text comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.crossfit_candidate_selector import read_predictions  # noqa: E402
from road_htr.metrics import levenshtein  # noqa: E402
from road_htr.schema import read_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--llm-details", type=Path, required=True)
    parser.add_argument("--model93", type=Path, required=True)
    parser.add_argument("--model94", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--changed-output", type=Path, required=True)
    args = parser.parse_args()

    truth = read_table(args.truth, require_target=True)
    assert truth.target_column is not None
    primary = read_predictions(args.primary)
    model93 = read_predictions(args.model93)
    model94 = read_predictions(args.model94)
    details = {}
    with args.llm_details.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            identifier = str(row.get("id", "")).strip()
            if not identifier or identifier in details:
                raise ValueError(f"blank or duplicate detail ID at line {line_number}")
            details[identifier] = row

    fieldnames = [
        truth.id_column,
        "fold",
        "ground_truth",
        "model88_before_llm",
        "llm_raw_output",
        "model93_before_guard",
        "model94_after_guard",
        "llm_changed_text",
        "guard_rejected_llm",
        "character_edits_model88_to_model93",
        "model93_fallback_reason",
    ]
    records = []
    for truth_row in truth.rows:
        identifier = truth_row[truth.id_column].strip()
        for name, values in (
            ("primary", primary),
            ("model93", model93),
            ("model94", model94),
            ("details", details),
        ):
            if identifier not in values:
                raise ValueError(f"{name} is missing {identifier}")
        before = primary[identifier]
        llm = model93[identifier]
        after = model94[identifier]
        changed = llm != before
        record = {
            truth.id_column: identifier,
            "fold": truth_row["fold"],
            "ground_truth": truth_row[truth.target_column],
            "model88_before_llm": before,
            "llm_raw_output": details[identifier]["raw_output"],
            "model93_before_guard": llm,
            "model94_after_guard": after,
            "llm_changed_text": changed,
            "guard_rejected_llm": changed and after == before,
            "character_edits_model88_to_model93": levenshtein(
                tuple(before), tuple(llm)
            ),
            "model93_fallback_reason": details[identifier].get("fallback_reason") or "",
        }
        records.append(record)

    for output, rows in (
        (args.output, records),
        (args.changed_output, [row for row in records if row["llm_changed_text"]]),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                "all_rows": len(records),
                "changed_rows": sum(row["llm_changed_text"] for row in records),
                "guard_rejected_rows": sum(
                    row["guard_rejected_llm"] for row in records
                ),
                "output": str(args.output.resolve()),
                "changed_output": str(args.changed_output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
