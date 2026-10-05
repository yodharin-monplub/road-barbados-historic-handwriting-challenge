"""Fold-safe confidence/character-LM selector for three fixed OCR views."""

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
from .lm_view_selector import LM_SMOOTHING, _language_quality
from .metrics import competition_error
from .schema import SchemaError, read_table
from .submission import write_submission
from .view_selector import (
    _metric_from_totals,
    _precompute_edit_totals,
    _write_csv,
    absolute,
)


LM_ORDERS = (4, 5, 6)
LM_ALPHAS = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0)
VIEW_OFFSETS = (-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.08, 0.1)
MINIMUM_LENGTH_RATIOS = (0.8, 0.9, 1.0)


def _validate_ids(
    ids: list[str], name: str, records: Mapping[str, ConfidenceCandidate]
) -> None:
    expected = set(ids)
    if set(records) != expected:
        raise SchemaError(
            f"{name} ID mismatch: missing={sorted(expected-set(records))[:5]}, "
            f"extra={sorted(set(records)-expected)[:5]}"
        )


def select_index(
    candidates: tuple[ConfidenceCandidate, ConfidenceCandidate, ConfidenceCandidate],
    language_qualities: tuple[float, float, float],
    lm_alpha: float,
    secondary_offset: float,
    tertiary_offset: float,
    minimum_length_ratio: float,
) -> int:
    offsets = (0.0, secondary_offset, tertiary_offset)
    minimum_length = minimum_length_ratio * max(len(candidates[0].text), 1)
    best_index = 0
    best_quality = fmean(candidates[0].confidences) + lm_alpha * language_qualities[0]
    for index in (1, 2):
        if len(candidates[index].text) < minimum_length:
            continue
        quality = (
            fmean(candidates[index].confidences)
            + lm_alpha * language_qualities[index]
            + offsets[index]
        )
        if quality > best_quality:
            best_index = index
            best_quality = quality
    return best_index


def tune(
    truth_path: str | Path,
    primary_path: str | Path,
    secondary_path: str | Path,
    tertiary_path: str | Path,
    output_config: str | Path,
    output_predictions: str | Path,
    fold: int | None = None,
) -> dict[str, Any]:
    truth = read_table(absolute(truth_path), require_target=True)
    if "fold" not in truth.fieldnames:
        raise SchemaError("fold-safe three-view selection requires a fold column")
    available_folds = {int(row["fold"]) for row in truth.rows}
    if fold is not None and fold not in available_folds:
        raise ValueError(f"fold {fold} is absent")
    assert truth.target_column is not None
    rows = [row for row in truth.rows if fold is None or int(row["fold"]) == fold]
    ids = [row[truth.id_column].strip() for row in rows]
    references = [row[truth.target_column] for row in rows]
    row_folds = [int(row["fold"]) for row in rows]
    sources = (
        read_confidence_candidates(primary_path),
        read_confidence_candidates(secondary_path),
        read_confidence_candidates(tertiary_path),
    )
    for name, records in zip(("primary", "secondary", "tertiary"), sources, strict=True):
        _validate_ids(ids, name, records)
    texts = tuple([[records[identifier].text for identifier in ids] for records in sources])
    primary_totals, secondary_totals, word_denominator, char_denominator = (
        _precompute_edit_totals(references, texts[0], texts[1])
    )
    tertiary_totals, _, _, _ = _precompute_edit_totals(
        references, texts[2], texts[2]
    )
    source_totals = (primary_totals, secondary_totals, tertiary_totals)

    trials: list[dict[str, Any]] = []
    best: tuple[float, int, int, dict[str, Any], list[str], list[int]] | None = None
    trial_order = 0
    for lm_order in LM_ORDERS:
        models = {
            fold_value: CharNGramLM(lm_order, LM_SMOOTHING).fit(
                row[truth.target_column]
                for row in truth.rows
                if int(row["fold"]) != fold_value
            )
            for fold_value in sorted(set(row_folds))
        }
        lm_qualities = tuple(
            [
                _language_quality(value, models[fold_value])
                for value, fold_value in zip(source_texts, row_folds, strict=True)
            ]
            for source_texts in texts
        )
        for lm_alpha in LM_ALPHAS:
            for secondary_offset in VIEW_OFFSETS:
                for tertiary_offset in VIEW_OFFSETS:
                    for length_ratio in MINIMUM_LENGTH_RATIOS:
                        indices = [
                            select_index(
                                tuple(records[identifier] for records in sources),
                                tuple(
                                    lm_qualities[source_index][position]
                                    for source_index in range(3)
                                ),
                                lm_alpha,
                                secondary_offset,
                                tertiary_offset,
                                length_ratio,
                            )
                            for position, identifier in enumerate(ids)
                        ]
                        selected = [
                            texts[source_index][position]
                            for position, source_index in enumerate(indices)
                        ]
                        totals = [
                            source_totals[source_index][position]
                            for position, source_index in enumerate(indices)
                        ]
                        score = _metric_from_totals(
                            totals, word_denominator, char_denominator
                        )
                        counts = [indices.count(index) for index in range(3)]
                        rule = {
                            "lm_order": lm_order,
                            "lm_smoothing": LM_SMOOTHING,
                            "lm_alpha": lm_alpha,
                            "secondary_offset": secondary_offset,
                            "tertiary_offset": tertiary_offset,
                            "minimum_length_ratio": length_ratio,
                        }
                        trials.append({"rule": rule, "selected_counts": counts, **score})
                        non_primary = counts[1] + counts[2]
                        key = (float(score["mean_error"]), non_primary, trial_order)
                        if best is None or key < best[:3]:
                            best = (*key, rule, selected, indices)
                        trial_order += 1
    assert best is not None
    _, _, _, rule, selected, indices = best
    selected_counts = [indices.count(index) for index in range(3)]
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
        "selected_counts": {
            "primary": selected_counts[0],
            "secondary": selected_counts[1],
            "tertiary": selected_counts[2],
        },
        "score": competition_error(references, selected),
        "individual_scores": {
            name: competition_error(references, values)
            for name, values in zip(
                ("primary", "secondary", "tertiary"), texts, strict=True
            )
        },
        "fold_scores": fold_scores,
        "trial_count": len(trials),
        "trials": sorted(trials, key=lambda item: float(item["mean_error"]))[:10],
        "eligibility_note": (
            "Uses only OCR output text/confidence and fold-excluded supplied "
            "labels for the character LM; image metadata is not used."
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
    secondary_path: str | Path,
    tertiary_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> None:
    train = read_table(absolute(train_path), require_target=True)
    test = read_table(absolute(test_path))
    assert train.target_column is not None
    ids = [row[test.id_column].strip() for row in test.rows]
    sources = (
        read_confidence_candidates(primary_path),
        read_confidence_candidates(secondary_path),
        read_confidence_candidates(tertiary_path),
    )
    for name, records in zip(("primary", "secondary", "tertiary"), sources, strict=True):
        _validate_ids(ids, name, records)
    config = json.loads(absolute(config_path).read_text(encoding="utf-8"))
    rule = config["rule"]
    model = CharNGramLM(
        int(rule["lm_order"]), float(rule.get("lm_smoothing", LM_SMOOTHING))
    ).fit(row[train.target_column] for row in train.rows)
    selected: list[str] = []
    counts = [0, 0, 0]
    for identifier in ids:
        candidates = tuple(records[identifier] for records in sources)
        qualities = tuple(_language_quality(candidate.text, model) for candidate in candidates)
        index = select_index(
            candidates,
            qualities,
            float(rule["lm_alpha"]),
            float(rule["secondary_offset"]),
            float(rule["tertiary_offset"]),
            float(rule["minimum_length_ratio"]),
        )
        selected.append(candidates[index].text)
        counts[index] += 1
    output = absolute(output_path)
    _write_csv(output, test.id_column, ids, selected)
    print(f"selected primary/secondary/tertiary rows: {counts}")
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
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--tertiary", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    args = parser.parse_args()
    result = tune(
        args.truth,
        args.primary,
        args.secondary,
        args.tertiary,
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
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--tertiary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    apply(
        args.train,
        args.test,
        args.primary,
        args.secondary,
        args.tertiary,
        args.config,
        args.output,
        args.sample,
        args.submission,
    )
