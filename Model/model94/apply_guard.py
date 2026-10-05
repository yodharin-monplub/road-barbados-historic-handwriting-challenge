#!/usr/bin/env python3
"""Apply Model94's frozen absolute-edit safety guard to Model93 output."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.crossfit_candidate_selector import read_predictions  # noqa: E402
from road_htr.metrics import levenshtein  # noqa: E402
from road_htr.schema import read_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    maximum = int(config["maximum_absolute_character_edits"])
    if maximum < 0:
        raise ValueError("maximum edit count must be nonnegative")

    primary_table = read_table(args.primary)
    challenger_table = read_table(args.challenger)
    identifiers = [
        row[challenger_table.id_column].strip() for row in challenger_table.rows
    ]
    primary = read_predictions(args.primary)
    challenger = read_predictions(args.challenger)
    missing = set(identifiers) - set(primary)
    if missing:
        raise ValueError(f"primary is missing IDs: {sorted(missing)[:5]}")
    predictions = {}
    accepted = 0
    rejected = 0
    for identifier in identifiers:
        edits = levenshtein(tuple(primary[identifier]), tuple(challenger[identifier]))
        choose_challenger = edits <= maximum
        predictions[identifier] = challenger[identifier] if choose_challenger else primary[identifier]
        accepted += int(choose_challenger and challenger[identifier] != primary[identifier])
        rejected += int(not choose_challenger)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=[challenger_table.id_column, "prediction"]
            )
            writer.writeheader()
            for identifier in identifiers:
                writer.writerow(
                    {
                        challenger_table.id_column: identifier,
                        "prediction": predictions[identifier],
                    }
                )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "rows": len(identifiers),
                "accepted_changed_rows": accepted,
                "rejected_rows": rejected,
                "maximum_absolute_character_edits": maximum,
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
