"""Word-level fusion of two OCR views using character confidences."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from .confidence_distribution_selector import (
    ConfidenceCandidate,
    confidence_quality,
    read_confidence_candidates,
)
from .metrics import competition_error, levenshtein
from .schema import SchemaError, read_table
from .submission import write_submission
from .view_selector import (
    _metric_from_totals,
    _precompute_edit_totals,
    _uses_alternate,
    _write_csv,
    absolute,
    candidate_rules,
)


TOKEN_RE = re.compile(r"\S+")
QUALITY_KINDS = ("mean", "geometric", "harmonic")
CONFIDENCE_MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05)
MAXIMUM_DISTANCE_RATIOS = (0.25, 0.5, 1.0)


@dataclass(frozen=True)
class ConfidentToken:
    text: str
    confidences: tuple[float, ...]


def _fit_character_confidences(candidate: ConfidenceCandidate) -> tuple[float, ...]:
    """Align confidences after output cleaning removed at most edge whitespace."""

    target_length = len(candidate.text)
    values = candidate.confidences
    if len(values) >= target_length:
        difference = len(values) - target_length
        start = difference // 2
        return values[start : start + target_length]
    fill = fmean(values)
    return values + (fill,) * (target_length - len(values))


def confident_tokens(candidate: ConfidenceCandidate) -> list[ConfidentToken]:
    values = _fit_character_confidences(candidate)
    return [
        ConfidentToken(match.group(), values[match.start() : match.end()])
        for match in TOKEN_RE.finditer(candidate.text)
    ]


def _alignment(base: list[ConfidentToken], donor: list[ConfidentToken]) -> list[int | None]:
    """Map each base token to its diagonal partner in a minimum word alignment."""

    rows = len(base) + 1
    columns = len(donor) + 1
    costs = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        costs[i][0] = i
    for j in range(columns):
        costs[0][j] = j
    for i in range(1, rows):
        for j in range(1, columns):
            costs[i][j] = min(
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
                costs[i - 1][j - 1]
                + (base[i - 1].text != donor[j - 1].text),
            )
    mapping: list[int | None] = [None] * len(base)
    i, j = len(base), len(donor)
    while i or j:
        if i and j:
            diagonal = costs[i - 1][j - 1] + (
                base[i - 1].text != donor[j - 1].text
            )
            if costs[i][j] == diagonal:
                mapping[i - 1] = j - 1
                i -= 1
                j -= 1
                continue
        if i and costs[i][j] == costs[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return mapping


def fuse_tokens(
    base: ConfidenceCandidate,
    donor: ConfidenceCandidate,
    quality_kind: str,
    confidence_margin: float,
    maximum_distance_ratio: float,
) -> tuple[str, list[dict[str, Any]]]:
    base_tokens = confident_tokens(base)
    donor_tokens = confident_tokens(donor)
    mapping = _alignment(base_tokens, donor_tokens)
    result: list[str] = []
    changes: list[dict[str, Any]] = []
    for index, (token, donor_index) in enumerate(zip(base_tokens, mapping, strict=True)):
        if donor_index is None:
            result.append(token.text)
            continue
        alternative = donor_tokens[donor_index]
        distance_ratio = levenshtein(
            tuple(token.text.casefold()), tuple(alternative.text.casefold())
        ) / max(len(token.text), len(alternative.text), 1)
        base_quality = confidence_quality(token.confidences, quality_kind)
        donor_quality = confidence_quality(alternative.confidences, quality_kind)
        replace = (
            alternative.text != token.text
            and distance_ratio <= maximum_distance_ratio
            and donor_quality - base_quality > confidence_margin
        )
        if replace:
            result.append(alternative.text)
            changes.append(
                {
                    "token_index": index,
                    "before": token.text,
                    "after": alternative.text,
                    "distance_ratio": distance_ratio,
                    "base_quality": base_quality,
                    "donor_quality": donor_quality,
                }
            )
        else:
            result.append(token.text)
    return " ".join(result), changes


def _validate_ids(
    ids: list[str], name: str, records: Mapping[str, ConfidenceCandidate]
) -> None:
    expected = set(ids)
    if set(records) != expected:
        raise SchemaError(
            f"{name} ID mismatch: missing={sorted(expected-set(records))[:5]}, "
            f"extra={sorted(set(records)-expected)[:5]}"
        )


def _mean_pair(candidate: ConfidenceCandidate) -> tuple[str, float]:
    return candidate.text, fmean(candidate.confidences)


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
    primary_pairs = {identifier: _mean_pair(primary[identifier]) for identifier in ids}
    alternate_pairs = {
        identifier: _mean_pair(alternate[identifier]) for identifier in ids
    }

    trials: list[dict[str, Any]] = []
    best: tuple[float, int, int, dict[str, Any], dict[str, Any], list[str]] | None = None
    order = 0
    for quality_kind in QUALITY_KINDS:
        for margin in CONFIDENCE_MARGINS:
            for distance_ratio in MAXIMUM_DISTANCE_RATIOS:
                primary_fused: list[str] = []
                alternate_fused: list[str] = []
                primary_changes: list[int] = []
                alternate_changes: list[int] = []
                for identifier in ids:
                    value, changes = fuse_tokens(
                        primary[identifier],
                        alternate[identifier],
                        quality_kind,
                        margin,
                        distance_ratio,
                    )
                    primary_fused.append(value)
                    primary_changes.append(len(changes))
                    value, changes = fuse_tokens(
                        alternate[identifier],
                        primary[identifier],
                        quality_kind,
                        margin,
                        distance_ratio,
                    )
                    alternate_fused.append(value)
                    alternate_changes.append(len(changes))
                primary_totals, alternate_totals, word_denominator, char_denominator = (
                    _precompute_edit_totals(
                        references, primary_fused, alternate_fused
                    )
                )
                fusion_rule = {
                    "quality_kind": quality_kind,
                    "confidence_margin": margin,
                    "maximum_distance_ratio": distance_ratio,
                }
                for base_rule in candidate_rules():
                    uses_alternate = [
                        _uses_alternate(
                            base_rule,
                            primary_pairs[identifier],
                            alternate_pairs[identifier],
                        )
                        for identifier in ids
                    ]
                    totals = [
                        alternate_totals[index]
                        if use_alternate
                        else primary_totals[index]
                        for index, use_alternate in enumerate(uses_alternate)
                    ]
                    selected = [
                        alternate_fused[index]
                        if use_alternate
                        else primary_fused[index]
                        for index, use_alternate in enumerate(uses_alternate)
                    ]
                    changed_tokens = sum(
                        alternate_changes[index]
                        if use_alternate
                        else primary_changes[index]
                        for index, use_alternate in enumerate(uses_alternate)
                    )
                    switched = sum(uses_alternate)
                    score = _metric_from_totals(
                        totals, word_denominator, char_denominator
                    )
                    trials.append(
                        {
                            "base_rule": base_rule,
                            "fusion_rule": fusion_rule,
                            "switched_rows": switched,
                            "changed_tokens": changed_tokens,
                            **score,
                        }
                    )
                    key = (float(score["mean_error"]), changed_tokens, order)
                    if best is None or key < best[:3]:
                        best = (
                            *key,
                            dict(base_rule),
                            fusion_rule,
                            selected,
                        )
                    order += 1
    assert best is not None
    _, changed_tokens, _, base_rule, fusion_rule, selected = best
    switched = sum(
        _uses_alternate(
            base_rule, primary_pairs[identifier], alternate_pairs[identifier]
        )
        for identifier in ids
    )
    baseline_selected = [
        alternate[identifier].text
        if _uses_alternate(
            base_rule, primary_pairs[identifier], alternate_pairs[identifier]
        )
        else primary[identifier].text
        for identifier in ids
    ]
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
        "base_rule": base_rule,
        "fusion_rule": fusion_rule,
        "fold_filter": fold,
        "rows": len(ids),
        "switched_rows": switched,
        "changed_tokens": changed_tokens,
        "score": competition_error(references, selected),
        "base_rule_score_without_fusion": competition_error(
            references, baseline_selected
        ),
        "fold_scores": fold_scores,
        "fixed_quality_kinds": list(QUALITY_KINDS),
        "fixed_confidence_margins": list(CONFIDENCE_MARGINS),
        "fixed_maximum_distance_ratios": list(MAXIMUM_DISTANCE_RATIOS),
        "base_rule_count": len(candidate_rules()),
        "trial_count": len(trials),
        "trials": sorted(trials, key=lambda item: float(item["mean_error"]))[:10],
        "eligibility_note": (
            "Fusion uses recognized tokens and their character confidences only; "
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
    base_rule = config["base_rule"]
    fusion_rule = config["fusion_rule"]
    selected: list[str] = []
    changed_rows = 0
    changed_tokens = 0
    switched = 0
    for identifier in ids:
        use_alternate = _uses_alternate(
            base_rule, _mean_pair(primary[identifier]), _mean_pair(alternate[identifier])
        )
        base = alternate[identifier] if use_alternate else primary[identifier]
        donor = primary[identifier] if use_alternate else alternate[identifier]
        result, changes = fuse_tokens(
            base,
            donor,
            str(fusion_rule["quality_kind"]),
            float(fusion_rule["confidence_margin"]),
            float(fusion_rule["maximum_distance_ratio"]),
        )
        selected.append(result)
        switched += int(use_alternate)
        changed_rows += int(bool(changes))
        changed_tokens += len(changes)
    output = absolute(output_path)
    _write_csv(output, test.id_column, ids, selected)
    print(
        f"selected alternate base for {switched}/{len(ids)} rows; fused "
        f"{changed_tokens} tokens in {changed_rows} rows"
    )
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
