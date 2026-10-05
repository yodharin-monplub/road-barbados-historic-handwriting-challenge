"""Fold-safe OCR view selection using confidence and a train-label char LM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from .confidence_distribution_selector import (
    ConfidenceCandidate,
    read_confidence_candidates,
)
from .ctc_decode import CharNGramLM
from .metrics import competition_error
from .schema import SchemaError, read_table
from .submission import write_submission
from .view_selector import (
    _metric_from_totals,
    _precompute_edit_totals,
    _write_csv,
    absolute,
)


LM_ORDERS = (3, 4, 5, 6)
LM_ALPHAS = (
    0.0,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
)
QUALITY_MARGINS = (-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05)
MINIMUM_LENGTH_RATIOS = (0.8, 0.9, 1.0)
LM_SMOOTHING = 0.2


def _language_quality(text: str, model: CharNGramLM) -> float:
    if not text:
        return -100.0
    prefix = ""
    total = 0.0
    for character in text:
        total += model.log_probability(prefix, character)
        prefix += character
    return total / len(text)


def choose_alternate(
    primary: ConfidenceCandidate,
    alternate: ConfidenceCandidate,
    primary_lm_quality: float,
    alternate_lm_quality: float,
    lm_alpha: float,
    quality_margin: float,
    minimum_length_ratio: float,
) -> bool:
    long_enough = len(alternate.text) >= minimum_length_ratio * max(
        len(primary.text), 1
    )
    confidence_delta = fmean(alternate.confidences) - fmean(primary.confidences)
    language_delta = alternate_lm_quality - primary_lm_quality
    return long_enough and confidence_delta + lm_alpha * language_delta > quality_margin


def _validate_ids(
    ids: list[str], name: str, records: Mapping[str, ConfidenceCandidate]
) -> None:
    expected = set(ids)
    if set(records) != expected:
        raise SchemaError(
            f"{name} ID mismatch: missing={sorted(expected-set(records))[:5]}, "
            f"extra={sorted(set(records)-expected)[:5]}"
        )


def tune(
    truth_path: str | Path,
    primary_path: str | Path,
    alternate_path: str | Path,
    output_config: str | Path,
    output_predictions: str | Path,
    fold: int | None = None,
) -> dict[str, Any]:
    truth = read_table(absolute(truth_path), require_target=True)
    if "fold" not in truth.fieldnames:
        raise SchemaError("fold-safe LM selection requires a fold column")
    available_folds = {int(row["fold"]) for row in truth.rows}
    if fold is not None and fold not in available_folds:
        raise ValueError(f"fold {fold} is absent")
    assert truth.target_column is not None
    rows = [row for row in truth.rows if fold is None or int(row["fold"]) == fold]
    ids = [row[truth.id_column].strip() for row in rows]
    references = [row[truth.target_column] for row in rows]
    row_folds = [int(row["fold"]) for row in rows]
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
    order_index = 0
    for lm_order in LM_ORDERS:
        models = {
            fold_value: CharNGramLM(lm_order, LM_SMOOTHING).fit(
                row[truth.target_column]
                for row in truth.rows
                if int(row["fold"]) != fold_value
            )
            for fold_value in sorted(set(row_folds))
        }
        primary_lm = [
            _language_quality(value, models[fold_value])
            for value, fold_value in zip(primary_values, row_folds, strict=True)
        ]
        alternate_lm = [
            _language_quality(value, models[fold_value])
            for value, fold_value in zip(alternate_values, row_folds, strict=True)
        ]
        for lm_alpha in LM_ALPHAS:
            for margin in QUALITY_MARGINS:
                for length_ratio in MINIMUM_LENGTH_RATIOS:
                    use_alternate = [
                        choose_alternate(
                            primary[identifier],
                            alternate[identifier],
                            primary_lm[position],
                            alternate_lm[position],
                            lm_alpha,
                            margin,
                            length_ratio,
                        )
                        for position, identifier in enumerate(ids)
                    ]
                    selected = [
                        alternate_values[position]
                        if use_alternate[position]
                        else primary_values[position]
                        for position in range(len(ids))
                    ]
                    totals = [
                        alternate_totals[position]
                        if use_alternate[position]
                        else primary_totals[position]
                        for position in range(len(ids))
                    ]
                    score = _metric_from_totals(
                        totals, word_denominator, char_denominator
                    )
                    rule = {
                        "lm_order": lm_order,
                        "lm_smoothing": LM_SMOOTHING,
                        "lm_alpha": lm_alpha,
                        "quality_margin": margin,
                        "minimum_length_ratio": length_ratio,
                    }
                    switched = sum(use_alternate)
                    trials.append({"rule": rule, "switched_rows": switched, **score})
                    key = (float(score["mean_error"]), switched, order_index)
                    if best is None or key < best[:3]:
                        best = (*key, rule, selected)
                    order_index += 1
    assert best is not None
    _, switched, _, rule, selected = best
    fold_scores: dict[str, dict[str, float]] = {}
    for fold_value in sorted(set(row_folds)):
        positions = [
            position
            for position, value in enumerate(row_folds)
            if value == fold_value
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
        "fixed_lm_orders": list(LM_ORDERS),
        "fixed_lm_alphas": list(LM_ALPHAS),
        "fixed_quality_margins": list(QUALITY_MARGINS),
        "fixed_minimum_length_ratios": list(MINIMUM_LENGTH_RATIOS),
        "trial_count": len(trials),
        "trials": sorted(trials, key=lambda item: float(item["mean_error"]))[:10],
        "eligibility_note": (
            "The language model uses fold-excluded supplied labels; selection "
            "uses OCR output text/confidence and no image metadata."
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
    train_path: str | Path,
    test_path: str | Path,
    primary_path: str | Path,
    alternate_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> None:
    train = read_table(absolute(train_path), require_target=True)
    test = read_table(absolute(test_path))
    assert train.target_column is not None
    ids = [row[test.id_column].strip() for row in test.rows]
    primary = read_confidence_candidates(primary_path)
    alternate = read_confidence_candidates(alternate_path)
    _validate_ids(ids, "primary", primary)
    _validate_ids(ids, "alternate", alternate)
    config = json.loads(absolute(config_path).read_text(encoding="utf-8"))
    rule = config["rule"]
    model = CharNGramLM(
        int(rule["lm_order"]), float(rule.get("lm_smoothing", LM_SMOOTHING))
    ).fit(row[train.target_column] for row in train.rows)
    selected: list[str] = []
    switched = 0
    for identifier in ids:
        primary_quality = _language_quality(primary[identifier].text, model)
        alternate_quality = _language_quality(alternate[identifier].text, model)
        use_alternate = choose_alternate(
            primary[identifier],
            alternate[identifier],
            primary_quality,
            alternate_quality,
            float(rule["lm_alpha"]),
            float(rule["quality_margin"]),
            float(rule["minimum_length_ratio"]),
        )
        selected.append(
            alternate[identifier].text if use_alternate else primary[identifier].text
        )
        switched += int(use_alternate)
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
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    apply(
        args.train,
        args.test,
        args.primary,
        args.alternate,
        args.config,
        args.output,
        args.sample,
        args.submission,
    )
