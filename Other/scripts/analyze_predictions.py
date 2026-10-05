#!/usr/bin/env python3
"""Break an OCR validation score down by image geometry and worst examples."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.metrics import (  # noqa: E402
    CER_NORMALIZER, WER_NORMALIZER, competition_error, levenshtein,
    strip_conversation_tokens,
)
from road_htr.schema import (  # noqa: E402
    ImageIndex,
    SchemaError,
    predictions_by_id,
    read_table,
)


def _score_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        **competition_error(
            [str(row["reference"]) for row in rows],
            [str(row["prediction"]) for row in rows],
        ),
    }


def add_score_contributions(rows: list[dict[str, object]]) -> None:
    """Attribute the exact aggregate score loss to rows, using its fixed denominators."""
    if not rows:
        raise ValueError("cannot attribute an empty evaluation")
    for row in rows:
        reference = strip_conversation_tokens(str(row["reference"]))
        prediction = strip_conversation_tokens(str(row["prediction"]))
        row["char_edits"] = levenshtein(reference, prediction)
        row["word_edits"] = levenshtein(reference.split(), prediction.split())
        row["char_error_rate"] = row["char_edits"] / max(len(reference), 1)
        row["char_weight"] = math.sqrt(max(len(reference), 1))
        row["word_weight"] = math.sqrt(max(len(reference.split()), 1))
    char_total = sum(float(row["char_weight"]) for row in rows)
    word_total = sum(float(row["word_weight"]) for row in rows)
    for row in rows:
        row["score_loss_contribution"] = 0.5 * (
            float(row["char_edits"]) * float(row["char_weight"]) / (CER_NORMALIZER * char_total)
            + float(row["word_edits"]) * float(row["word_weight"]) / (WER_NORMALIZER * word_total)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data/raw/images")
    parser.add_argument("--prediction-column")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--worst", type=int, default=30)
    parser.add_argument("--sort-by", choices=("char_error_rate", "score_loss_contribution"),
                        default="char_error_rate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    truth = read_table(args.truth, require_target=True)
    predictions = read_table(args.predictions)
    prediction_column = args.prediction_column
    if prediction_column is None and predictions.target_column is None:
        non_id = [name for name in predictions.fieldnames if name != predictions.id_column]
        if len(non_id) != 1:
            raise SchemaError("pass --prediction-column because predictions are ambiguous")
        prediction_column = non_id[0]
    by_id = predictions_by_id(predictions, prediction_column)
    assert truth.target_column is not None
    truth_rows = [
        row
        for row in truth.rows
        if args.fold is None or ("fold" in truth.fieldnames and int(row["fold"]) == args.fold)
    ]
    if args.fold is not None and "fold" not in truth.fieldnames:
        raise SchemaError("--fold was supplied but truth has no fold column")
    ids = [row[truth.id_column].strip() for row in truth_rows]
    if set(ids) != set(by_id):
        raise SchemaError(
            f"ID mismatch: missing={sorted(set(ids)-set(by_id))[:5]}, "
            f"extra={sorted(set(by_id)-set(ids))[:5]}"
        )

    image_index = ImageIndex(args.image_dir)
    rows: list[dict[str, object]] = []
    for truth_row in truth_rows:
        identifier = truth_row[truth.id_column].strip()
        reference = truth_row[truth.target_column]
        prediction = by_id[identifier]
        with Image.open(image_index.resolve(identifier)) as image:
            width, height = image.size
        rows.append(
            {
                "id": identifier,
                "reference": reference,
                "prediction": prediction,
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
            }
        )

    add_score_contributions(rows)

    aspect_ranges = [
        ("under_8", 0.0, 8.0),
        ("8_to_14", 8.0, 14.0),
        ("14_to_22", 14.0, 22.0),
        ("22_and_over", 22.0, float("inf")),
    ]
    aspect_bins = {
        name: _score_rows(
            [row for row in rows if low <= float(row["aspect_ratio"]) < high]
        )
        for name, low, high in aspect_ranges
        if any(low <= float(row["aspect_ratio"]) < high for row in rows)
    }
    worst = sorted(
        rows,
        key=lambda row: (float(row[args.sort_by]), int(row["char_edits"])),
        reverse=True,
    )[: args.worst]
    report = {
        "truth": str(args.truth),
        "predictions": str(args.predictions),
        "fold": args.fold,
        "overall": _score_rows(rows),
        "aspect_bins": aspect_bins,
        "worst_sort_by": args.sort_by,
        "worst_rows_oracle_score_loss": sum(float(row["score_loss_contribution"]) for row in worst),
        "oracle_note": "Loss removed only if these rows became perfect; not a deployable model gain.",
        "worst": worst,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
