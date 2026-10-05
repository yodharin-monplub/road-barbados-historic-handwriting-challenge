"""Select between OCR views using legal output-confidence distributions."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from .metrics import competition_error
from .schema import SchemaError, read_table
from .submission import clean_prediction, write_submission
from .view_selector import (
    _metric_from_totals,
    _precompute_edit_totals,
    _write_csv,
    absolute,
)


QUALITY_KINDS = (
    "mean",
    "geometric",
    "harmonic",
    "q10",
    "q25",
    "mean_q10_25",
    "mean_q10_50",
    "mean_q25_25",
    "mean_q25_50",
)
QUALITY_MARGINS = (-0.02, -0.01, 0.0, 0.005, 0.01, 0.02, 0.03)
MINIMUM_LENGTH_RATIOS = (0.0, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class ConfidenceCandidate:
    text: str
    confidences: tuple[float, ...]


def read_confidence_candidates(path: str | Path) -> dict[str, ConfidenceCandidate]:
    records: dict[str, ConfidenceCandidate] = {}
    with absolute(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row.get("id", "")).strip()
            candidates = row.get("candidates") or []
            raw_confidences = row.get("character_confidences") or []
            if not identifier or not candidates:
                raise SchemaError(f"missing candidate at {path}:{line_number}")
            if identifier in records:
                raise SchemaError(f"duplicate ID {identifier!r} in {path}")
            prediction = clean_prediction(candidates[0])
            confidences = tuple(float(value) for value in raw_confidences)
            if not confidences and not prediction:
                confidences = (0.0,)
            elif not confidences:
                raise SchemaError(
                    f"missing character confidence at {path}:{line_number}"
                )
            if any(not math.isfinite(value) or value < 0 for value in confidences):
                raise SchemaError(f"invalid confidence at {path}:{line_number}")
            records[identifier] = ConfidenceCandidate(prediction, confidences)
    if not records:
        raise SchemaError(f"no candidates in {path}")
    return records


def _quantile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def confidence_quality(values: tuple[float, ...], kind: str) -> float:
    clipped = tuple(max(value, 1e-6) for value in values)
    mean = fmean(clipped)
    if kind == "mean":
        return mean
    if kind == "geometric":
        return math.exp(fmean(math.log(value) for value in clipped))
    if kind == "harmonic":
        return len(clipped) / sum(1 / value for value in clipped)
    q10 = _quantile(clipped, 0.10)
    q25 = _quantile(clipped, 0.25)
    if kind == "q10":
        return q10
    if kind == "q25":
        return q25
    blends = {
        "mean_q10_25": (q10, 0.25),
        "mean_q10_50": (q10, 0.50),
        "mean_q25_25": (q25, 0.25),
        "mean_q25_50": (q25, 0.50),
    }
    if kind not in blends:
        raise ValueError(f"unknown quality kind {kind!r}")
    tail, weight = blends[kind]
    return (1 - weight) * mean + weight * tail


def candidate_rules() -> list[dict[str, float | str]]:
    rules: list[dict[str, float | str]] = [
        {"kind": "primary"},
        {"kind": "alternate"},
    ]
    rules.extend(
        {
            "kind": "distribution",
            "quality_kind": quality_kind,
            "quality_margin": margin,
            "minimum_length_ratio": length_ratio,
        }
        for quality_kind in QUALITY_KINDS
        for margin in QUALITY_MARGINS
        for length_ratio in MINIMUM_LENGTH_RATIOS
    )
    return rules


def _uses_alternate(
    rule: Mapping[str, Any],
    primary: ConfidenceCandidate,
    alternate: ConfidenceCandidate,
) -> bool:
    if rule["kind"] == "primary":
        return False
    if rule["kind"] == "alternate":
        return True
    if rule["kind"] != "distribution":
        raise ValueError(f"unknown rule kind {rule['kind']!r}")
    long_enough = len(alternate.text) >= float(rule["minimum_length_ratio"]) * max(
        len(primary.text), 1
    )
    quality_kind = str(rule["quality_kind"])
    return long_enough and (
        confidence_quality(alternate.confidences, quality_kind)
        - confidence_quality(primary.confidences, quality_kind)
        > float(rule["quality_margin"])
    )


def _validate_ids(
    ids: list[str], name: str, records: Mapping[str, ConfidenceCandidate]
) -> None:
    expected = set(ids)
    if set(records) != expected:
        raise SchemaError(
            f"{name} ID mismatch: missing={sorted(expected-set(records))[:5]}, "
            f"extra={sorted(set(records)-expected)[:5]}"
        )


def _select(
    ids: list[str],
    primary: Mapping[str, ConfidenceCandidate],
    alternate: Mapping[str, ConfidenceCandidate],
    rule: Mapping[str, Any],
) -> tuple[list[str], int]:
    selected: list[str] = []
    switched = 0
    for identifier in ids:
        use_alternate = _uses_alternate(
            rule, primary[identifier], alternate[identifier]
        )
        selected.append(
            alternate[identifier].text if use_alternate else primary[identifier].text
        )
        switched += int(use_alternate)
    return selected, switched


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
    primary = read_confidence_candidates(primary_path)
    alternate = read_confidence_candidates(alternate_path)
    _validate_ids(ids, "primary", primary)
    _validate_ids(ids, "alternate", alternate)

    primary_values = [primary[identifier].text for identifier in ids]
    alternate_values = [alternate[identifier].text for identifier in ids]
    primary_totals, alternate_totals, word_denominator, char_denominator = (
        _precompute_edit_totals(references, primary_values, alternate_values)
    )
    trials: list[dict[str, Any]] = []
    best: tuple[float, int, int, dict[str, Any], list[str]] | None = None
    for order, rule in enumerate(candidate_rules()):
        selected, switched = _select(ids, primary, alternate, rule)
        totals = [
            alternate_totals[position]
            if _uses_alternate(rule, primary[identifier], alternate[identifier])
            else primary_totals[position]
            for position, identifier in enumerate(ids)
        ]
        score = _metric_from_totals(totals, word_denominator, char_denominator)
        trials.append({"rule": rule, "switched_rows": switched, **score})
        key = (float(score["mean_error"]), switched, order)
        if best is None or key < best[:3]:
            best = (*key, rule, selected)
    assert best is not None
    _, switched, _, rule, selected = best

    fold_scores: dict[str, dict[str, float]] = {}
    if "fold" in truth.fieldnames:
        for fold_value in sorted({int(row["fold"]) for row in rows}):
            positions = [
                index for index, row in enumerate(rows) if int(row["fold"]) == fold_value
            ]
            fold_scores[str(fold_value)] = competition_error(
                [references[index] for index in positions],
                [selected[index] for index in positions],
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
        "fixed_quality_kinds": list(QUALITY_KINDS),
        "fixed_quality_margins": list(QUALITY_MARGINS),
        "fixed_minimum_length_ratios": list(MINIMUM_LENGTH_RATIOS),
        "trial_count": len(trials),
        "trials": sorted(trials, key=lambda item: float(item["mean_error"]))[:10],
        "eligibility_note": (
            "The rule uses only recognized text and per-character OCR confidence; "
            "image metadata is not used."
        ),
    }
    config = absolute(output_config)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
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
    primary = read_confidence_candidates(primary_path)
    alternate = read_confidence_candidates(alternate_path)
    _validate_ids(ids, "primary", primary)
    _validate_ids(ids, "alternate", alternate)
    config = json.loads(absolute(config_path).read_text(encoding="utf-8"))
    selected, switched = _select(ids, primary, alternate, config["rule"])
    output = absolute(output_path)
    _write_csv(output, test.id_column, ids, selected)
    print(f"selected alternate for {switched}/{len(ids)} rows")
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
