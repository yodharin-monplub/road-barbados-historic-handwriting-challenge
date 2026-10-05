"""Fold-safe joined/split word repair learned from training transcriptions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .lexicon_corrector import (
    TOKEN_PATTERN,
    _copy_case,
    _validate_prediction_ids,
    _write_predictions,
)
from .metrics import competition_error
from .schema import SchemaError, read_table
from .submission import write_submission
from .view_selector import absolute, read_prediction_map


SPLIT_THRESHOLDS = (1, 2, 3, 5, 8, 10, 20)
MERGE_THRESHOLDS = (1, 2, 3, 5, 8, 10, 20)


def _core(raw_token: str) -> str:
    match = TOKEN_PATTERN.fullmatch(raw_token)
    return match.group(2).casefold() if match else ""


class TrainingBoundaryModel:
    def __init__(self, texts: Iterable[str]):
        self.unigrams: Counter[str] = Counter()
        self.bigrams: Counter[tuple[str, str]] = Counter()
        for text in texts:
            words = [_core(token) for token in text.split()]
            self.unigrams.update(word for word in words if word)
            sequence = ["<s>", *words, "</s>"]
            self.bigrams.update(zip(sequence, sequence[1:], strict=False))

    def best_split(self, value: str, minimum_count: int) -> tuple[str, str] | None:
        if value in self.unigrams or len(value) < 4:
            return None
        candidates: list[tuple[int, int, str, str]] = []
        for position in range(1, len(value)):
            left, right = value[:position], value[position:]
            count = self.bigrams[(left, right)]
            if (
                left in self.unigrams
                and right in self.unigrams
                and count >= minimum_count
            ):
                candidates.append(
                    (count, min(self.unigrams[left], self.unigrams[right]), left, right)
                )
        if not candidates:
            return None
        _, _, left, right = max(candidates)
        return left, right


def repair_boundaries(
    value: str,
    model: TrainingBoundaryModel,
    split_minimum_count: int,
    merge_minimum_frequency: int,
) -> tuple[str, list[dict[str, Any]]]:
    tokens = value.split()
    corrected: list[str] = []
    changes: list[dict[str, Any]] = []
    index = 0
    while index < len(tokens):
        raw_token = tokens[index]
        match = TOKEN_PATTERN.fullmatch(raw_token)
        current = _core(raw_token)

        if match and index + 1 < len(tokens):
            next_token = tokens[index + 1]
            next_match = TOKEN_PATTERN.fullmatch(next_token)
            following = _core(next_token)
            joined = current + following
            if (
                next_match
                and not match.group(3)
                and not next_match.group(1)
                and joined in model.unigrams
                and model.unigrams[joined] >= merge_minimum_frequency
                and (current not in model.unigrams or following not in model.unigrams)
            ):
                prefix, source_core, _ = match.groups()
                _, _, suffix = next_match.groups()
                replacement = prefix + _copy_case(source_core, joined) + suffix
                corrected.append(replacement)
                changes.append(
                    {
                        "operation": "merge",
                        "token_index": index,
                        "before": [raw_token, next_token],
                        "after": [replacement],
                        "frequency": model.unigrams[joined],
                    }
                )
                index += 2
                continue

        split = model.best_split(current, split_minimum_count) if match else None
        if split is not None:
            left, right = split
            prefix, source_core, suffix = match.groups()
            first = prefix + _copy_case(source_core, left)
            second_core = right.upper() if source_core.isupper() else right
            second = second_core + suffix
            corrected.extend((first, second))
            changes.append(
                {
                    "operation": "split",
                    "token_index": index,
                    "before": [raw_token],
                    "after": [first, second],
                    "bigram_count": model.bigrams[(left, right)],
                }
            )
            index += 1
            continue

        corrected.append(raw_token)
        index += 1
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
        raise SchemaError("fold-safe boundary tuning requires a fold column")
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
        fold_value: TrainingBoundaryModel(
            row[truth.target_column]
            for row in truth.rows
            if int(row["fold"]) != fold_value
        )
        for fold_value in sorted(set(row_folds))
    }

    trials: list[dict[str, Any]] = []
    best_key: tuple[float, int, int] | None = None
    best_values: tuple[int, int, list[str], int, int] | None = None
    order = 0
    for split_threshold in SPLIT_THRESHOLDS:
        for merge_threshold in MERGE_THRESHOLDS:
            corrected: list[str] = []
            splits = 0
            merges = 0
            for value, fold_value in zip(source, row_folds, strict=True):
                result, changes = repair_boundaries(
                    value,
                    models[fold_value],
                    split_threshold,
                    merge_threshold,
                )
                corrected.append(result)
                splits += sum(change["operation"] == "split" for change in changes)
                merges += sum(change["operation"] == "merge" for change in changes)
            metric = competition_error(references, corrected)
            trials.append(
                {
                    "split_minimum_count": split_threshold,
                    "merge_minimum_frequency": merge_threshold,
                    "splits": splits,
                    "merges": merges,
                    **metric,
                }
            )
            key = (float(metric["mean_error"]), splits + merges, order)
            if best_key is None or key < best_key:
                best_key = key
                best_values = (
                    split_threshold,
                    merge_threshold,
                    corrected,
                    splits,
                    merges,
                )
            order += 1
    assert best_values is not None
    split_threshold, merge_threshold, corrected, splits, merges = best_values
    result = {
        "fold_filter": fold,
        "rows": len(ids),
        "split_minimum_count": split_threshold,
        "merge_minimum_frequency": merge_threshold,
        "splits": splits,
        "merges": merges,
        "score": competition_error(references, corrected),
        "baseline_score": competition_error(references, source),
        "trial_count": len(trials),
        "trials": sorted(trials, key=lambda item: float(item["mean_error"]))[:10],
        "method": "exact train-vocabulary split/merge with fold-excluded counts",
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
    model = TrainingBoundaryModel(row[train.target_column] for row in train.rows)
    config = json.loads(absolute(config_path).read_text(encoding="utf-8"))
    _, predictions = read_prediction_map(predictions_path)
    ids = [row[test.id_column].strip() for row in test.rows]
    _validate_prediction_ids(ids, predictions)

    corrected: list[str] = []
    changes: list[dict[str, Any]] = []
    for identifier in ids:
        before = predictions[identifier]
        after, operations = repair_boundaries(
            before,
            model,
            int(config["split_minimum_count"]),
            int(config["merge_minimum_frequency"]),
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
    print(f"repaired boundaries in {len(changes)}/{len(ids)} predictions")
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
