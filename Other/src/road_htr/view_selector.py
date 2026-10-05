"""Validation-tuned confidence gate for two fixed OCR image views."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .metrics import (
    CER_NORMALIZER,
    WER_NORMALIZER,
    competition_error,
    levenshtein,
    strip_conversation_tokens,
)
from .schema import SchemaError, predictions_by_id, read_table
from .submission import clean_prediction, write_submission


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIDENCE_MARGINS = (-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05)
MINIMUM_LENGTH_RATIOS = (0.0, 0.8, 0.9, 1.0)


def absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_prediction_map(path: str | Path) -> tuple[str, dict[str, str]]:
    table = read_table(absolute(path))
    prediction_column = table.target_column
    if prediction_column is None:
        columns = [name for name in table.fieldnames if name != table.id_column]
        if len(columns) != 1:
            raise SchemaError(f"prediction CSV must have one value column: {path}")
        prediction_column = columns[0]
    return table.id_column, dict(predictions_by_id(table, prediction_column))


def read_confident_candidates(path: str | Path) -> dict[str, tuple[str, float]]:
    """Read top prediction and mean confidence from a model JSONL file."""

    result: dict[str, tuple[str, float]] = {}
    with absolute(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row["id"]).strip()
            candidates = row.get("candidates") or []
            scores = row.get("scores") or []
            if not identifier or not candidates:
                raise SchemaError(f"missing candidate at {path}:{line_number}")
            if identifier in result:
                raise SchemaError(f"duplicate ID {identifier!r} in {path}")
            prediction = clean_prediction(candidates[0])
            # Kraken emits no character confidences when it recognizes an empty
            # line. Keep the row and make every non-empty view preferable.
            confidence = (
                -1.0
                if not prediction or not scores or scores[0] is None
                else float(scores[0])
            )
            result[identifier] = (prediction, confidence)
    if not result:
        raise SchemaError(f"no candidates in {path}")
    return result


def choose_alternate(
    primary_text: str,
    primary_confidence: float,
    alternate_text: str,
    alternate_confidence: float,
    confidence_margin: float,
    minimum_length_ratio: float,
) -> bool:
    """Select the alternate solely from OCR output text and confidence."""

    long_enough = len(alternate_text) >= minimum_length_ratio * max(len(primary_text), 1)
    return long_enough and alternate_confidence - primary_confidence > confidence_margin


def candidate_rules() -> list[dict[str, float | str]]:
    rules: list[dict[str, float | str]] = [
        {"kind": "primary"},
        {"kind": "alternate"},
    ]
    rules.extend(
        {
            "kind": "confidence",
            "confidence_margin": margin,
            "minimum_length_ratio": ratio,
        }
        for margin in CONFIDENCE_MARGINS
        for ratio in MINIMUM_LENGTH_RATIOS
    )
    return rules


def _uses_alternate(
    rule: Mapping[str, Any],
    primary: tuple[str, float],
    alternate: tuple[str, float],
) -> bool:
    if rule["kind"] == "primary":
        return False
    if rule["kind"] == "alternate":
        return True
    if rule["kind"] != "confidence":
        raise ValueError(f"unknown rule kind {rule['kind']!r}")
    return choose_alternate(
        primary[0],
        primary[1],
        alternate[0],
        alternate[1],
        float(rule["confidence_margin"]),
        float(rule["minimum_length_ratio"]),
    )


def _write_csv(path: Path, id_column: str, ids: list[str], values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_column, "prediction"])
        writer.writeheader()
        for identifier, value in zip(ids, values, strict=True):
            writer.writerow({id_column: identifier, "prediction": value})


def _validate_ids(
    ids: list[str], name: str, values: Mapping[str, tuple[str, float]]
) -> None:
    expected = set(ids)
    if set(values) != expected:
        raise SchemaError(
            f"{name} ID mismatch: missing={sorted(expected-set(values))[:5]}, "
            f"extra={sorted(set(values)-expected)[:5]}"
        )


def _select(
    ids: list[str],
    primary: Mapping[str, tuple[str, float]],
    alternate: Mapping[str, tuple[str, float]],
    rule: Mapping[str, Any],
) -> tuple[list[str], int]:
    values: list[str] = []
    switched = 0
    for identifier in ids:
        use_alternate = _uses_alternate(rule, primary[identifier], alternate[identifier])
        values.append(alternate[identifier][0] if use_alternate else primary[identifier][0])
        switched += int(use_alternate)
    return values, switched


def _precompute_edit_totals(
    references: list[str], primary: list[str], alternate: list[str]
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float, float]:
    primary_totals: list[tuple[float, float]] = []
    alternate_totals: list[tuple[float, float]] = []
    word_denominator = 0.0
    char_denominator = 0.0
    for reference, primary_value, alternate_value in zip(
        references, primary, alternate, strict=True
    ):
        ref_value = strip_conversation_tokens(reference)
        ref_words = ref_value.split()
        ref_chars = tuple(ref_value)
        word_weight = math.sqrt(max(len(ref_words), 1))
        char_weight = math.sqrt(max(len(ref_chars), 1))
        word_denominator += word_weight
        char_denominator += char_weight

        def totals(value: str) -> tuple[float, float]:
            clean = strip_conversation_tokens(value)
            return (
                levenshtein(ref_words, clean.split()) * word_weight,
                levenshtein(ref_chars, tuple(clean)) * char_weight,
            )

        primary_totals.append(totals(primary_value))
        alternate_totals.append(totals(alternate_value))
    return primary_totals, alternate_totals, word_denominator, char_denominator


def _metric_from_totals(
    totals: list[tuple[float, float]], word_denominator: float, char_denominator: float
) -> dict[str, float]:
    wer = sum(value[0] for value in totals) / word_denominator
    cer = sum(value[1] for value in totals) / char_denominator
    normalized_wer = wer / WER_NORMALIZER
    normalized_cer = cer / CER_NORMALIZER
    mean_error = (normalized_wer + normalized_cer) / 2
    return {
        "weighted_wer": wer,
        "weighted_cer": cer,
        "normalized_wer_error": normalized_wer,
        "normalized_cer_error": normalized_cer,
        "mean_error": mean_error,
        "multi_score": 1 - mean_error,
    }


def tune(
    truth_path: str | Path,
    primary_path: str | Path,
    alternate_path: str | Path,
    output_config: str | Path,
    output_predictions: str | Path,
    fold: int | None = None,
) -> dict[str, Any]:
    truth = read_table(absolute(truth_path), require_target=True)
    if fold is not None and "fold" not in truth.fieldnames:
        raise SchemaError("--fold requires a fold column in truth")
    rows = [row for row in truth.rows if fold is None or int(row["fold"]) == fold]
    ids = [row[truth.id_column].strip() for row in rows]
    assert truth.target_column is not None
    references = [row[truth.target_column] for row in rows]
    primary = read_confident_candidates(primary_path)
    alternate = read_confident_candidates(alternate_path)
    _validate_ids(ids, "primary", primary)
    _validate_ids(ids, "alternate", alternate)

    primary_values = [primary[identifier][0] for identifier in ids]
    alternate_values = [alternate[identifier][0] for identifier in ids]
    primary_totals, alternate_totals, word_denominator, char_denominator = (
        _precompute_edit_totals(references, primary_values, alternate_values)
    )
    trials: list[dict[str, Any]] = []
    best: tuple[float, int, int, dict[str, Any], list[str]] | None = None
    for order, rule in enumerate(candidate_rules()):
        values, switched = _select(ids, primary, alternate, rule)
        selected_totals = [
            alternate_totals[position]
            if _uses_alternate(rule, primary[identifier], alternate[identifier])
            else primary_totals[position]
            for position, identifier in enumerate(ids)
        ]
        score = _metric_from_totals(selected_totals, word_denominator, char_denominator)
        trials.append({"rule": rule, "switched_rows": switched, **score})
        key = (float(score["mean_error"]), switched, order)
        if best is None or key[:3] < best[:3]:
            best = (*key, rule, values)
    assert best is not None
    _, switched, _, rule, selected = best

    fold_scores: dict[str, Any] = {}
    if "fold" in truth.fieldnames:
        row_by_id = {row[truth.id_column].strip(): row for row in rows}
        for fold_value in sorted({int(row["fold"]) for row in rows}):
            positions = [
                position
                for position, identifier in enumerate(ids)
                if int(row_by_id[identifier]["fold"]) == fold_value
            ]
            fold_scores[str(fold_value)] = competition_error(
                [references[position] for position in positions],
                [selected[position] for position in positions],
            )

    result = {
        "rule": rule,
        "fold_filter": fold,
        "rows": len(ids),
        "switched_rows": switched,
        "score": competition_error(references, selected),
        "primary_score": competition_error(references, primary_values),
        "alternate_score": competition_error(references, alternate_values),
        "fold_scores": fold_scores,
        "fixed_confidence_margins": list(CONFIDENCE_MARGINS),
        "fixed_minimum_length_ratios": list(MINIMUM_LENGTH_RATIOS),
        "trial_count": len(trials),
        "trials": sorted(trials, key=lambda item: float(item["mean_error"]))[:10],
        "eligibility_note": (
            "The rule uses only OCR output text and model confidence; image metadata "
            "is not used."
        ),
    }
    config_path = absolute(output_config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(absolute(output_predictions), truth.id_column, ids, selected)
    return result


def apply(
    test_path: str | Path,
    primary_path: str | Path,
    alternate_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> None:
    test = read_table(absolute(test_path))
    ids = [row[test.id_column].strip() for row in test.rows]
    primary = read_confident_candidates(primary_path)
    alternate = read_confident_candidates(alternate_path)
    _validate_ids(ids, "primary", primary)
    _validate_ids(ids, "alternate", alternate)
    config = json.loads(absolute(config_path).read_text(encoding="utf-8"))
    selected, switched = _select(ids, primary, alternate, config["rule"])
    _write_csv(absolute(output_path), test.id_column, ids, selected)
    print(f"selected the alternate view for {switched}/{len(ids)} rows")
    if submission_path:
        if sample_path is None:
            raise ValueError("--sample is required with --submission")
        write_submission(
            absolute(sample_path),
            absolute(submission_path),
            dict(zip(ids, selected, strict=True)),
        )


def tune_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    args = parser.parse_args()
    result = tune(
        args.truth,
        args.primary,
        args.alternate,
        args.output_config,
        args.output_predictions,
        args.fold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def apply_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    apply(
        args.test,
        args.primary,
        args.alternate,
        args.config,
        args.output,
        args.sample,
        args.submission,
    )
