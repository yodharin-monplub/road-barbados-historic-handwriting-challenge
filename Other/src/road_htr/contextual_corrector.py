"""Fold-safe contextual spelling repair learned only from training labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .lexicon_corrector import (
    TOKEN_PATTERN,
    TrainingLexicon,
    _copy_case,
    _validate_prediction_ids,
    _write_predictions,
    correct_text,
)
from .metrics import competition_error, levenshtein
from .schema import SchemaError, read_table
from .submission import write_submission
from .view_selector import absolute, read_prediction_map


MINIMUM_FREQUENCIES = (1, 2, 3, 5)
CONTEXT_WEIGHTS = (2.0, 4.0, 8.0, 16.0)
DISTANCE_PENALTIES = (8.0, 16.0, 32.0, 64.0)
FALLBACK_FREQUENCIES = (5, 8)
SEED_MINIMUM_FREQUENCY = 3
SEED_AMBIGUITY_RATIO = 1.0
TWO_EDIT_MIN_LENGTH = 6


def _deletion_neighborhood(value: str, depth: int) -> set[str]:
    seen = {value}
    frontier = {value}
    for _ in range(depth):
        frontier = {
            item[:position] + item[position + 1 :]
            for item in frontier
            for position in range(len(item))
        } - seen
        seen.update(frontier)
    return seen


def _core(raw_token: str) -> str:
    match = TOKEN_PATTERN.fullmatch(raw_token)
    return match.group(2).casefold() if match else ""


@dataclass(frozen=True)
class ContextCandidate:
    value: str
    distance: int
    frequency: int


class TrainingLanguageModel:
    """Unigram/bigram counts plus a SymSpell-style two-deletion index."""

    def __init__(self, texts: Iterable[str]):
        materialized = list(texts)
        self.frequencies: Counter[str] = Counter()
        self.bigrams: Counter[tuple[str, str]] = Counter()
        for text in materialized:
            words = [_core(token) for token in text.split()]
            self.frequencies.update(word for word in words if word)
            sequence = ["<s>", *words, "</s>"]
            self.bigrams.update(zip(sequence, sequence[1:], strict=False))

        self.delete_index: dict[str, set[str]] = defaultdict(set)
        for word in self.frequencies:
            for deletion in _deletion_neighborhood(word, 2):
                self.delete_index[deletion].add(word)
        self.seed_lexicon = TrainingLexicon(materialized)
        self._cache: dict[tuple[str, int], tuple[ContextCandidate, ...]] = {}

    def candidates(
        self, query: str, two_edit_min_length: int = TWO_EDIT_MIN_LENGTH
    ) -> tuple[ContextCandidate, ...]:
        query = query.casefold()
        if query in self.frequencies:
            return ()
        maximum_distance = 1 if len(query) < two_edit_min_length else 2
        key = (query, maximum_distance)
        if key not in self._cache:
            possible: set[str] = set()
            for deletion in _deletion_neighborhood(query, maximum_distance):
                possible.update(self.delete_index.get(deletion, ()))
            candidates: list[ContextCandidate] = []
            for value in possible:
                if abs(len(value) - len(query)) > maximum_distance:
                    continue
                distance = levenshtein(query, value)
                if distance <= maximum_distance:
                    candidates.append(
                        ContextCandidate(value, distance, self.frequencies[value])
                    )
            candidates.sort(
                key=lambda item: (item.distance, -item.frequency, item.value)
            )
            self._cache[key] = tuple(candidates)
        return self._cache[key]


def correct_contextually(
    value: str,
    language_model: TrainingLanguageModel,
    minimum_frequency: int,
    context_weight: float,
    distance_penalty: float,
    fallback_frequency: int,
    two_edit_min_length: int = TWO_EDIT_MIN_LENGTH,
    seed_minimum_frequency: int = SEED_MINIMUM_FREQUENCY,
    seed_ambiguity_ratio: float = SEED_AMBIGUITY_RATIO,
) -> tuple[str, list[dict[str, Any]]]:
    """Repair words independently while scoring their fixed seed context."""

    raw_tokens = value.split()
    seed, _ = correct_text(
        value,
        language_model.seed_lexicon,
        seed_minimum_frequency,
        seed_ambiguity_ratio,
    )
    seed_cores = ["<s>", *[_core(token) for token in seed.split()], "</s>"]
    if len(seed_cores) != len(raw_tokens) + 2:
        raise RuntimeError("seed correction changed token count")

    corrected: list[str] = []
    changes: list[dict[str, Any]] = []
    for index, raw_token in enumerate(raw_tokens):
        match = TOKEN_PATTERN.fullmatch(raw_token)
        if not match:
            corrected.append(raw_token)
            continue
        candidates = [
            candidate
            for candidate in language_model.candidates(
                match.group(2), two_edit_min_length
            )
            if candidate.frequency >= minimum_frequency
        ]
        if not candidates:
            corrected.append(raw_token)
            continue

        left_context = seed_cores[index]
        right_context = seed_cores[index + 2]

        def score(candidate: ContextCandidate) -> float:
            left = language_model.bigrams[(left_context, candidate.value)]
            right = language_model.bigrams[(candidate.value, right_context)]
            return (
                math.log(candidate.frequency)
                + context_weight * (math.log1p(left) + math.log1p(right))
                - distance_penalty * (candidate.distance - 1)
            )

        best = min(
            candidates,
            key=lambda candidate: (-score(candidate), -candidate.frequency, candidate.value),
        )
        left_support = language_model.bigrams[(left_context, best.value)]
        right_support = language_model.bigrams[(best.value, right_context)]
        if max(left_support, right_support) < 1 and best.frequency < fallback_frequency:
            corrected.append(raw_token)
            continue

        prefix, source_core, suffix = match.groups()
        replacement = prefix + _copy_case(source_core, best.value) + suffix
        corrected.append(replacement)
        if replacement != raw_token:
            changes.append(
                {
                    "token_index": index,
                    "before": raw_token,
                    "after": replacement,
                    "distance": best.distance,
                    "frequency": best.frequency,
                    "left_support": left_support,
                    "right_support": right_support,
                }
            )
    return " ".join(corrected), changes


def _evaluate_setting(
    source: list[str],
    references: list[str],
    row_folds: list[int],
    language_models: dict[int, TrainingLanguageModel],
    minimum_frequency: int,
    context_weight: float,
    distance_penalty: float,
    fallback_frequency: int,
) -> tuple[dict[str, float], list[str], int, int]:
    corrected: list[str] = []
    changed_tokens = 0
    distance_two_tokens = 0
    for value, fold_value in zip(source, row_folds, strict=True):
        result, changes = correct_contextually(
            value,
            language_models[fold_value],
            minimum_frequency,
            context_weight,
            distance_penalty,
            fallback_frequency,
        )
        corrected.append(result)
        changed_tokens += len(changes)
        distance_two_tokens += sum(change["distance"] == 2 for change in changes)
    return (
        competition_error(references, corrected),
        corrected,
        changed_tokens,
        distance_two_tokens,
    )


def tune(
    truth_path: str | Path,
    predictions_path: str | Path,
    output_config: str | Path,
    output_predictions: str | Path,
    fold: int | None = None,
) -> dict[str, Any]:
    truth = read_table(absolute(truth_path), require_target=True)
    if "fold" not in truth.fieldnames:
        raise SchemaError("fold-safe contextual tuning requires a fold column")
    available_folds = {int(row["fold"]) for row in truth.rows}
    if fold is not None and fold not in available_folds:
        raise ValueError(f"fold {fold} is absent")
    assert truth.target_column is not None
    selected_rows = [
        row for row in truth.rows if fold is None or int(row["fold"]) == fold
    ]
    ids = [row[truth.id_column].strip() for row in selected_rows]
    references = [row[truth.target_column] for row in selected_rows]
    _, prediction_map = read_prediction_map(predictions_path)
    _validate_prediction_ids(ids, prediction_map)
    source = [prediction_map[identifier] for identifier in ids]
    row_folds = [int(row["fold"]) for row in selected_rows]
    folds = sorted(set(row_folds))
    language_models = {
        fold_value: TrainingLanguageModel(
            row[truth.target_column]
            for row in truth.rows
            if int(row["fold"]) != fold_value
        )
        for fold_value in folds
    }

    best_key: tuple[float, int, int] | None = None
    best_values: tuple[int, float, float, int, list[str], int, int] | None = None
    trials: list[dict[str, Any]] = []
    order = 0
    for minimum_frequency in MINIMUM_FREQUENCIES:
        for context_weight in CONTEXT_WEIGHTS:
            for distance_penalty in DISTANCE_PENALTIES:
                for fallback_frequency in FALLBACK_FREQUENCIES:
                    metric, corrected, changes, distance_two = _evaluate_setting(
                        source,
                        references,
                        row_folds,
                        language_models,
                        minimum_frequency,
                        context_weight,
                        distance_penalty,
                        fallback_frequency,
                    )
                    record = {
                        "minimum_frequency": minimum_frequency,
                        "context_weight": context_weight,
                        "distance_penalty": distance_penalty,
                        "fallback_frequency": fallback_frequency,
                        "changed_tokens": changes,
                        "distance_two_tokens": distance_two,
                        **metric,
                    }
                    trials.append(record)
                    key = (float(metric["mean_error"]), changes, order)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_values = (
                            minimum_frequency,
                            context_weight,
                            distance_penalty,
                            fallback_frequency,
                            corrected,
                            changes,
                            distance_two,
                        )
                    order += 1
    assert best_values is not None
    (
        minimum_frequency,
        context_weight,
        distance_penalty,
        fallback_frequency,
        corrected,
        changes,
        distance_two,
    ) = best_values
    result = {
        "fold_filter": fold,
        "rows": len(ids),
        "minimum_frequency": minimum_frequency,
        "context_weight": context_weight,
        "distance_penalty": distance_penalty,
        "fallback_frequency": fallback_frequency,
        "two_edit_min_length": TWO_EDIT_MIN_LENGTH,
        "seed_minimum_frequency": SEED_MINIMUM_FREQUENCY,
        "seed_ambiguity_ratio": SEED_AMBIGUITY_RATIO,
        "changed_tokens": changes,
        "distance_two_tokens": distance_two,
        "score": competition_error(references, corrected),
        "baseline_score": competition_error(references, source),
        "trial_count": len(trials),
        "trials": sorted(trials, key=lambda item: float(item["mean_error"]))[:10],
        "method": "fold-excluded unigram/bigram LM with edit distance <= 2",
    }
    config = absolute(output_config)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output = absolute(output_predictions)
    _write_predictions(output, truth.id_column, ids, corrected)
    with output.with_suffix(".changes.jsonl").open("w", encoding="utf-8") as handle:
        for identifier, before, after in zip(ids, source, corrected, strict=True):
            if before != after:
                handle.write(
                    json.dumps(
                        {"id": identifier, "before": before, "after": after},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return result


def apply(
    train_path: str | Path,
    test_path: str | Path,
    predictions_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> None:
    train = read_table(absolute(train_path), require_target=True)
    test = read_table(absolute(test_path))
    assert train.target_column is not None
    language_model = TrainingLanguageModel(
        row[train.target_column] for row in train.rows
    )
    config = json.loads(absolute(config_path).read_text(encoding="utf-8"))
    _, predictions = read_prediction_map(predictions_path)
    ids = [row[test.id_column].strip() for row in test.rows]
    _validate_prediction_ids(ids, predictions)

    corrected: list[str] = []
    changes: list[dict[str, Any]] = []
    for identifier in ids:
        before = predictions[identifier]
        after, token_changes = correct_contextually(
            before,
            language_model,
            int(config["minimum_frequency"]),
            float(config["context_weight"]),
            float(config["distance_penalty"]),
            int(config["fallback_frequency"]),
            int(config.get("two_edit_min_length", TWO_EDIT_MIN_LENGTH)),
            int(config.get("seed_minimum_frequency", SEED_MINIMUM_FREQUENCY)),
            float(config.get("seed_ambiguity_ratio", SEED_AMBIGUITY_RATIO)),
        )
        corrected.append(after)
        if before != after:
            changes.append(
                {
                    "id": identifier,
                    "before": before,
                    "after": after,
                    "token_changes": token_changes,
                }
            )
    output = absolute(output_path)
    _write_predictions(output, test.id_column, ids, corrected)
    with output.with_suffix(".changes.jsonl").open("w", encoding="utf-8") as handle:
        for row in changes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"corrected {len(changes)}/{len(ids)} predictions")
    if submission_path:
        if sample_path is None:
            raise ValueError("--sample is required with --submission")
        write_submission(
            absolute(sample_path),
            absolute(submission_path),
            dict(zip(ids, corrected, strict=True)),
        )


def tune_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    args = parser.parse_args()
    result = tune(
        args.truth,
        args.predictions,
        args.output_config,
        args.output_predictions,
        args.fold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def apply_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    apply(
        args.train,
        args.test,
        args.predictions,
        args.config,
        args.output,
        args.sample,
        args.submission,
    )
