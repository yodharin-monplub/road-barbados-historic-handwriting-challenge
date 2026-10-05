"""Approximate joined-word repair from frequent training-label bigrams."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .lexicon_corrector import (
    TOKEN_PATTERN,
    _copy_case,
    _validate_prediction_ids,
    _write_predictions,
)
from .metrics import competition_error, levenshtein
from .schema import SchemaError, read_table
from .submission import write_submission
from .view_selector import absolute, read_prediction_map


MAXIMUM_DISTANCES = (1, 2)
MINIMUM_TOKEN_LENGTHS = (6, 8, 10)
MINIMUM_BIGRAM_COUNTS = (2, 3, 5, 8)
CONTEXT_WEIGHTS = (0.0, 4.0)
DISTANCE_PENALTIES = (8.0, 16.0)


def _core(raw_token: str) -> str:
    match = TOKEN_PATTERN.fullmatch(raw_token)
    return match.group(2).casefold() if match else ""


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


@dataclass(frozen=True)
class JoinedCandidate:
    left: str
    right: str
    count: int
    distance: int


class TrainingJoinedWordModel:
    def __init__(self, texts: Iterable[str]):
        self.unigrams: Counter[str] = Counter()
        self.bigrams: Counter[tuple[str, str]] = Counter()
        for text in texts:
            words = [_core(token) for token in text.split()]
            self.unigrams.update(word for word in words if word)
            sequence = ["<s>", *words, "</s>"]
            self.bigrams.update(zip(sequence, sequence[1:], strict=False))

        self.phrases: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        for (left, right), count in self.bigrams.items():
            if left and right and left not in {"<s>"} and right not in {"</s>"}:
                self.phrases[left + right].append((left, right, count))
        self.delete_index: dict[str, set[str]] = defaultdict(set)
        for joined in self.phrases:
            for deletion in _deletion_neighborhood(joined, 2):
                self.delete_index[deletion].add(joined)
        self._cache: dict[tuple[str, int], tuple[JoinedCandidate, ...]] = {}

    def candidates(self, query: str, maximum_distance: int) -> tuple[JoinedCandidate, ...]:
        query = query.casefold()
        key = (query, maximum_distance)
        if key not in self._cache:
            possible: set[str] = set()
            for deletion in _deletion_neighborhood(query, maximum_distance):
                possible.update(self.delete_index.get(deletion, ()))
            candidates: list[JoinedCandidate] = []
            for joined in possible:
                if abs(len(joined) - len(query)) > maximum_distance:
                    continue
                distance = levenshtein(query, joined)
                if not 1 <= distance <= maximum_distance:
                    continue
                candidates.extend(
                    JoinedCandidate(left, right, count, distance)
                    for left, right, count in self.phrases[joined]
                )
            candidates.sort(
                key=lambda item: (item.distance, -item.count, item.left, item.right)
            )
            self._cache[key] = tuple(candidates)
        return self._cache[key]


def repair_approximate_joins(
    value: str,
    model: TrainingJoinedWordModel,
    maximum_distance: int,
    minimum_token_length: int,
    minimum_bigram_count: int,
    context_weight: float,
    distance_penalty: float,
) -> tuple[str, list[dict[str, Any]]]:
    tokens = value.split()
    contexts = ["<s>", *[_core(token) for token in tokens], "</s>"]
    corrected: list[str] = []
    changes: list[dict[str, Any]] = []
    for index, raw_token in enumerate(tokens):
        match = TOKEN_PATTERN.fullmatch(raw_token)
        source = _core(raw_token)
        if (
            not match
            or source in model.unigrams
            or len(source) < minimum_token_length
        ):
            corrected.append(raw_token)
            continue
        candidates = [
            candidate
            for candidate in model.candidates(source, maximum_distance)
            if candidate.count >= minimum_bigram_count
        ]
        if not candidates:
            corrected.append(raw_token)
            continue

        left_context = contexts[index]
        right_context = contexts[index + 2]

        def score(candidate: JoinedCandidate) -> float:
            return (
                math.log1p(candidate.count)
                + context_weight
                * (
                    math.log1p(model.bigrams[(left_context, candidate.left)])
                    + math.log1p(model.bigrams[(candidate.right, right_context)])
                )
                - distance_penalty * (candidate.distance - 1)
            )

        best = min(
            candidates,
            key=lambda candidate: (-score(candidate), -candidate.count, candidate.left, candidate.right),
        )
        prefix, source_core, suffix = match.groups()
        first = prefix + _copy_case(source_core, best.left)
        second_core = best.right.upper() if source_core.isupper() else best.right
        second = second_core + suffix
        corrected.extend((first, second))
        changes.append(
            {
                "operation": "approximate_split",
                "token_index": index,
                "before": [raw_token],
                "after": [first, second],
                "distance": best.distance,
                "bigram_count": best.count,
            }
        )
    return " ".join(corrected), changes


def tune(
    truth_path: str | Path,
    predictions_path: str | Path,
    output_config: str | Path,
    output_predictions: str | Path,
    fold: int | None = None,
) -> dict[str, Any]:
    truth = read_table(absolute(truth_path), require_target=True)
    if "fold" not in truth.fieldnames:
        raise SchemaError("fold-safe join tuning requires a fold column")
    available_folds = {int(row["fold"]) for row in truth.rows}
    if fold is not None and fold not in available_folds:
        raise ValueError(f"fold {fold} is absent")
    assert truth.target_column is not None
    selected_rows = [
        row for row in truth.rows if fold is None or int(row["fold"]) == fold
    ]
    ids = [row[truth.id_column].strip() for row in selected_rows]
    references = [row[truth.target_column] for row in selected_rows]
    row_folds = [int(row["fold"]) for row in selected_rows]
    _, prediction_map = read_prediction_map(predictions_path)
    _validate_prediction_ids(ids, prediction_map)
    source = [prediction_map[identifier] for identifier in ids]
    models = {
        fold_value: TrainingJoinedWordModel(
            row[truth.target_column]
            for row in truth.rows
            if int(row["fold"]) != fold_value
        )
        for fold_value in sorted(set(row_folds))
    }

    trials: list[dict[str, Any]] = []
    best_key: tuple[float, int, int] | None = None
    best_values: tuple[int, int, int, float, float, list[str], int, int] | None = None
    order = 0
    for maximum_distance in MAXIMUM_DISTANCES:
        for minimum_token_length in MINIMUM_TOKEN_LENGTHS:
            for minimum_bigram_count in MINIMUM_BIGRAM_COUNTS:
                for context_weight in CONTEXT_WEIGHTS:
                    for distance_penalty in DISTANCE_PENALTIES:
                        corrected: list[str] = []
                        changed_tokens = 0
                        distance_two_tokens = 0
                        for value, fold_value in zip(source, row_folds, strict=True):
                            result, changes = repair_approximate_joins(
                                value,
                                models[fold_value],
                                maximum_distance,
                                minimum_token_length,
                                minimum_bigram_count,
                                context_weight,
                                distance_penalty,
                            )
                            corrected.append(result)
                            changed_tokens += len(changes)
                            distance_two_tokens += sum(
                                change["distance"] == 2 for change in changes
                            )
                        metric = competition_error(references, corrected)
                        trials.append(
                            {
                                "maximum_distance": maximum_distance,
                                "minimum_token_length": minimum_token_length,
                                "minimum_bigram_count": minimum_bigram_count,
                                "context_weight": context_weight,
                                "distance_penalty": distance_penalty,
                                "changed_tokens": changed_tokens,
                                "distance_two_tokens": distance_two_tokens,
                                **metric,
                            }
                        )
                        key = (float(metric["mean_error"]), changed_tokens, order)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_values = (
                                maximum_distance,
                                minimum_token_length,
                                minimum_bigram_count,
                                context_weight,
                                distance_penalty,
                                corrected,
                                changed_tokens,
                                distance_two_tokens,
                            )
                        order += 1
    assert best_values is not None
    (
        maximum_distance,
        minimum_token_length,
        minimum_bigram_count,
        context_weight,
        distance_penalty,
        corrected,
        changed_tokens,
        distance_two_tokens,
    ) = best_values
    result = {
        "fold_filter": fold,
        "rows": len(ids),
        "maximum_distance": maximum_distance,
        "minimum_token_length": minimum_token_length,
        "minimum_bigram_count": minimum_bigram_count,
        "context_weight": context_weight,
        "distance_penalty": distance_penalty,
        "changed_tokens": changed_tokens,
        "distance_two_tokens": distance_two_tokens,
        "score": competition_error(references, corrected),
        "baseline_score": competition_error(references, source),
        "trial_count": len(trials),
        "trials": sorted(trials, key=lambda item: float(item["mean_error"]))[:10],
        "method": "fold-excluded fuzzy match to frequent joined training bigrams",
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
    model = TrainingJoinedWordModel(row[train.target_column] for row in train.rows)
    config = json.loads(absolute(config_path).read_text(encoding="utf-8"))
    _, predictions = read_prediction_map(predictions_path)
    ids = [row[test.id_column].strip() for row in test.rows]
    _validate_prediction_ids(ids, predictions)

    corrected: list[str] = []
    changes: list[dict[str, Any]] = []
    for identifier in ids:
        before = predictions[identifier]
        after, operations = repair_approximate_joins(
            before,
            model,
            int(config["maximum_distance"]),
            int(config["minimum_token_length"]),
            int(config["minimum_bigram_count"]),
            float(config["context_weight"]),
            float(config["distance_penalty"]),
        )
        corrected.append(after)
        if operations:
            changes.append(
                {
                    "id": identifier,
                    "before": before,
                    "after": after,
                    "operations": operations,
                }
            )
    output = absolute(output_path)
    _write_predictions(output, test.id_column, ids, corrected)
    with output.with_suffix(".changes.jsonl").open("w", encoding="utf-8") as handle:
        for row in changes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"repaired approximate joins in {len(changes)}/{len(ids)} predictions")
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
