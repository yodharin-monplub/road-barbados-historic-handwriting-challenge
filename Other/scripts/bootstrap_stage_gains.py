#!/usr/bin/env python3
"""Paired row bootstrap for an ordered list of OOF prediction files."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.metrics import (  # noqa: E402
    CER_NORMALIZER, WER_NORMALIZER, levenshtein, strip_conversation_tokens,
)
from road_htr.schema import predictions_by_id, read_table  # noqa: E402


def _parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("use NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--fold", type=int)
    selection.add_argument("--exclude-fold", type=int, action="append", default=[])
    parser.add_argument(
        "--prediction", type=_parse_prediction, action="append", required=True
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON artifact path (refuses to overwrite an existing file)",
    )
    args = parser.parse_args()
    if len(args.prediction) < 2:
        parser.error("provide at least two ordered --prediction NAME=PATH values")
    if args.samples < 20:
        parser.error("--samples must be at least 20")
    if len({name for name, _ in args.prediction}) != len(args.prediction):
        parser.error("prediction names must be unique")

    truth = read_table(args.truth, require_target=True)
    assert truth.target_column is not None
    filtered = args.fold is not None or bool(args.exclude_fold)
    if filtered:
        if "fold" not in truth.fieldnames:
            raise ValueError("fold selection requires a truth fold column")
        available = {int(row["fold"]) for row in truth.rows}
        requested = {args.fold} if args.fold is not None else set(args.exclude_fold)
        if not requested <= available:
            raise ValueError("requested fold is absent from truth")
    rows = [
        row
        for row in truth.rows
        if (args.fold is None or int(row.get("fold", -1)) == args.fold)
        and int(row.get("fold", -1)) not in args.exclude_fold
    ]
    if not rows:
        raise ValueError("fold selection leaves no evaluation rows")
    ids = [row[truth.id_column].strip() for row in rows]
    references = [strip_conversation_tokens(row[truth.target_column]) for row in rows]
    components: dict[str, list[tuple[float, float, float, float]]] = {}
    for name, path in args.prediction:
        table = read_table(path)
        prediction_column = table.target_column or next(
            column for column in table.fieldnames if column != table.id_column
        )
        all_values = dict(predictions_by_id(table, prediction_column))
        expected_ids = set(ids)
        missing = expected_ids - set(all_values)
        # A filtered comparison may consume full OOF files, but unrelated IDs
        # outside the canonical truth are still invalid.
        allowed = {row[truth.id_column].strip() for row in truth.rows} if filtered else expected_ids
        unexpected = set(all_values) - allowed
        if missing or unexpected:
            raise ValueError(
                f"ID mismatch for {name}: missing={sorted(missing)[:5]}, "
                f"unexpected={sorted(unexpected)[:5]}"
            )
        values = {identifier: all_values[identifier] for identifier in ids}
        entries = []
        for identifier, reference in zip(ids, references, strict=True):
            hypothesis = strip_conversation_tokens(values[identifier])
            word_weight = math.sqrt(max(len(reference.split()), 1))
            char_weight = math.sqrt(max(len(reference), 1))
            entries.append(
                (
                    levenshtein(reference.split(), hypothesis.split()) * word_weight,
                    word_weight,
                    levenshtein(reference, hypothesis) * char_weight,
                    char_weight,
                )
            )
        components[name] = entries

    def score(name: str, indices: list[int] | range) -> float:
        entries = components[name]
        wer = sum(entries[index][0] for index in indices) / sum(
            entries[index][1] for index in indices
        )
        cer = sum(entries[index][2] for index in indices) / sum(
            entries[index][3] for index in indices
        )
        return 1 - 0.5 * (wer / WER_NORMALIZER + cer / CER_NORMALIZER)

    rng = random.Random(args.seed)
    resamples = [
        [rng.randrange(len(ids)) for _ in ids] for _ in range(args.samples)
    ]
    names = [name for name, _ in args.prediction]
    results = []
    for before, after in zip(names, names[1:], strict=False):
        deltas = sorted(
            score(after, indices) - score(before, indices) for indices in resamples
        )
        lower = deltas[int(0.025 * args.samples)]
        upper = deltas[min(int(0.975 * args.samples), args.samples - 1)]
        results.append(
            {
                "before": before,
                "after": after,
                "observed_delta": score(after, range(len(ids)))
                - score(before, range(len(ids))),
                "percentile_95": [lower, upper],
                "bootstrap_gain_fraction": sum(delta > 0 for delta in deltas)
                / args.samples,
            }
        )
    payload = {
        "rows": len(ids),
        "samples": args.samples,
        "seed": args.seed,
        "included_folds": sorted({int(row["fold"]) for row in rows}) if "fold" in truth.fieldnames else None,
        "excluded_folds": sorted(set(args.exclude_fold)),
        "observed_scores": {name: score(name, range(len(ids))) for name in names},
        "claim_scope": "Paired row uncertainty conditional on these predictions; excludes training randomness, group correlation, model-selection bias, and distribution shift.",
        "comparisons": results,
    }
    serialized = json.dumps(payload, indent=2)
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
