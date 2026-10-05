"""Context-gated repair for OCR confusions that are themselves valid words."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
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


CONTEXT_WEIGHTS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SCORE_MARGINS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
CANDIDATE_MINIMUM_FREQUENCY = 2


def _core(raw_token: str) -> str:
    match = TOKEN_PATTERN.fullmatch(raw_token)
    return match.group(2).casefold() if match else ""


def _one_deletions(value: str) -> set[str]:
    return {value, *(value[:index] + value[index + 1 :] for index in range(len(value)))}


class TrainingConfusionModel:
    def __init__(self, texts: Iterable[str]):
        self.unigrams: Counter[str] = Counter()
        self.bigrams: Counter[tuple[str, str]] = Counter()
        for text in texts:
            words = [_core(token) for token in text.split()]
            self.unigrams.update(word for word in words if word)
            sequence = ["<s>", *words, "</s>"]
            self.bigrams.update(zip(sequence, sequence[1:], strict=False))
        self.delete_index: dict[str, set[str]] = defaultdict(set)
        for word in self.unigrams:
            for deletion in _one_deletions(word):
                self.delete_index[deletion].add(word)
        self._cache: dict[str, tuple[str, ...]] = {}

    def alternatives(self, query: str) -> tuple[str, ...]:
        query = query.casefold()
        if query not in self._cache:
            possible: set[str] = set()
            for deletion in _one_deletions(query):
                possible.update(self.delete_index.get(deletion, ()))
            self._cache[query] = tuple(
                sorted(
                    word
                    for word in possible
                    if word != query
                    and abs(len(word) - len(query)) <= 1
                    and levenshtein(query, word) <= 1
                )
            )
        return self._cache[query]


def repair_confusions(
    value: str,
    model: TrainingConfusionModel,
    context_weight: float,
    score_margin: float,
    candidate_minimum_frequency: int = CANDIDATE_MINIMUM_FREQUENCY,
) -> tuple[str, list[dict[str, Any]]]:
    tokens = value.split()
    contexts = ["<s>", *[_core(token) for token in tokens], "</s>"]
    corrected: list[str] = []
    changes: list[dict[str, Any]] = []
    for index, raw_token in enumerate(tokens):
        match = TOKEN_PATTERN.fullmatch(raw_token)
        source = _core(raw_token)
        if not match or not source:
            corrected.append(raw_token)
            continue

        left_context = contexts[index]
        right_context = contexts[index + 2]

        def score(word: str) -> float:
            return math.log1p(model.unigrams[word]) + context_weight * (
                math.log1p(model.bigrams[(left_context, word)])
                + math.log1p(model.bigrams[(word, right_context)])
            )

        source_score = score(source) if source in model.unigrams else 0.0
        candidates = [
            word
            for word in model.alternatives(source)
            if model.unigrams[word] >= candidate_minimum_frequency
        ]
        if not candidates:
            corrected.append(raw_token)
            continue
        best = min(candidates, key=lambda word: (-score(word), -model.unigrams[word], word))
        advantage = score(best) - source_score
        if advantage < score_margin:
            corrected.append(raw_token)
            continue

        prefix, source_core, suffix = match.groups()
        replacement = prefix + _copy_case(source_core, best) + suffix
        corrected.append(replacement)
        changes.append(
            {
                "token_index": index,
                "before": raw_token,
                "after": replacement,
                "score_advantage": advantage,
                "source_frequency": model.unigrams[source],
                "replacement_frequency": model.unigrams[best],
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
        raise SchemaError("fold-safe confusion tuning requires a fold column")
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
        fold_value: TrainingConfusionModel(
            row[truth.target_column]
            for row in truth.rows
            if int(row["fold"]) != fold_value
        )
        for fold_value in sorted(set(row_folds))
    }

    trials: list[dict[str, Any]] = []
    best_key: tuple[float, int, int] | None = None
    best_values: tuple[float, float, list[str], int] | None = None
    order = 0
    for context_weight in CONTEXT_WEIGHTS:
        for score_margin in SCORE_MARGINS:
            corrected: list[str] = []
            changed_tokens = 0
            for value, fold_value in zip(source, row_folds, strict=True):
                result, changes = repair_confusions(
                    value, models[fold_value], context_weight, score_margin
                )
                corrected.append(result)
                changed_tokens += len(changes)
            metric = competition_error(references, corrected)
            trials.append(
                {
                    "context_weight": context_weight,
                    "score_margin": score_margin,
                    "changed_tokens": changed_tokens,
                    **metric,
                }
            )
            key = (float(metric["mean_error"]), changed_tokens, order)
            if best_key is None or key < best_key:
                best_key = key
                best_values = (
                    context_weight,
                    score_margin,
                    corrected,
                    changed_tokens,
                )
            order += 1
    assert best_values is not None
    context_weight, score_margin, corrected, changed_tokens = best_values
    result = {
        "fold_filter": fold,
        "rows": len(ids),
        "context_weight": context_weight,
        "score_margin": score_margin,
        "candidate_minimum_frequency": CANDIDATE_MINIMUM_FREQUENCY,
        "changed_tokens": changed_tokens,
        "score": competition_error(references, corrected),
        "baseline_score": competition_error(references, source),
        "trial_count": len(trials),
        "trials": sorted(trials, key=lambda item: float(item["mean_error"]))[:10],
        "method": "fold-excluded context-gated edit-distance-one confusion repair",
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
    model = TrainingConfusionModel(row[train.target_column] for row in train.rows)
    config = json.loads(absolute(config_path).read_text(encoding="utf-8"))
    _, predictions = read_prediction_map(predictions_path)
    ids = [row[test.id_column].strip() for row in test.rows]
    _validate_prediction_ids(ids, predictions)

    corrected: list[str] = []
    changes: list[dict[str, Any]] = []
    for identifier in ids:
        before = predictions[identifier]
        after, operations = repair_confusions(
            before,
            model,
            float(config["context_weight"]),
            float(config["score_margin"]),
            int(config.get("candidate_minimum_frequency", CANDIDATE_MINIMUM_FREQUENCY)),
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
    print(f"repaired confusions in {len(changes)}/{len(ids)} predictions")
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
