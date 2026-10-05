"""Build and validate submissions against the official sample file."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Callable, Mapping

from .schema import CsvTable, SchemaError, read_table


def clean_prediction(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    # Preserve case and punctuation: both are scored. Only normalize line breaks
    # and surrounding whitespace introduced by generation APIs.
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def clean_prediction_preserving_internal_spaces(value: object) -> str:
    """Strip boundaries and line breaks without collapsing genuine spaces."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\r\n|\r|\n", " ", str(value)).strip()


def write_submission(
    sample_path: str | Path,
    output_path: str | Path,
    predictions: Mapping[str, object],
    cleaner: Callable[[object], str] = clean_prediction,
) -> Path:
    sample = read_table(sample_path)
    prediction_columns = [c for c in sample.fieldnames if c != sample.id_column]
    if not prediction_columns:
        raise SchemaError("sample submission has no prediction column")

    sample_ids = [row[sample.id_column].strip() for row in sample.rows]
    missing = sorted(set(sample_ids) - set(predictions))
    extra = sorted(set(predictions) - set(sample_ids))
    if missing or extra:
        raise SchemaError(
            f"prediction ID mismatch: {len(missing)} missing, {len(extra)} extra; "
            f"examples missing={missing[:5]}, extra={extra[:5]}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample.fieldnames)
        writer.writeheader()
        for identifier in sample_ids:
            value = cleaner(predictions[identifier])
            if not value:
                raise SchemaError(f"empty prediction for {identifier!r}")
            row = {sample.id_column: identifier}
            # If Zindi supplies separate metric columns for the same transcript,
            # populate each one identically while preserving the exact template.
            row.update({column: value for column in prediction_columns})
            writer.writerow(row)
    return output_path


def validate_submission(
    submission_path: str | Path, sample_path: str | Path
) -> dict[str, int]:
    submission = read_table(submission_path)
    sample = read_table(sample_path)
    if submission.fieldnames != sample.fieldnames:
        raise SchemaError(
            f"column/order mismatch: {submission.fieldnames} != {sample.fieldnames}"
        )
    expected = {row[sample.id_column].strip() for row in sample.rows}
    actual = {row[submission.id_column].strip() for row in submission.rows}
    if expected != actual:
        raise SchemaError(
            f"ID mismatch: missing={sorted(expected-actual)[:5]}, "
            f"extra={sorted(actual-expected)[:5]}"
        )
    prediction_columns = [c for c in submission.fieldnames if c != submission.id_column]
    for row in submission.rows:
        for column in prediction_columns:
            if not clean_prediction(row.get(column)):
                raise SchemaError(
                    f"blank prediction for {row[submission.id_column]!r} in {column}"
                )
    return {"rows": len(submission.rows), "prediction_columns": len(prediction_columns)}
