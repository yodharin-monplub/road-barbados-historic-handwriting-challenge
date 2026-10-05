"""Fold-safe, train-only lexical repair for OCR output."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import competition_error, levenshtein
from .schema import SchemaError, read_table
from .submission import write_submission
from .view_selector import absolute, read_prediction_map


TOKEN_PATTERN = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")
MINIMUM_FREQUENCIES = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50)
AMBIGUITY_RATIOS = (1.0, 2.0, 3.0, 5.0, 10.0)


def _deletions(value: str) -> set[str]:
    return {value, *(value[:position] + value[position + 1 :] for position in range(len(value)))}


@dataclass(frozen=True)
class WordCandidate:
    value: str
    distance: int
    frequency: int


class TrainingLexicon:
    def __init__(self, texts: list[str]):
        frequencies: Counter[str] = Counter()
        for text in texts:
            for raw_token in text.split():
                match = TOKEN_PATTERN.fullmatch(raw_token)
                if match:
                    frequencies[match.group(2).casefold()] += 1
        self.frequencies = frequencies
        self.delete_index: dict[str, set[str]] = defaultdict(set)
        for word in frequencies:
            for deletion in _deletions(word):
                self.delete_index[deletion].add(word)
        self._cache: dict[str, tuple[WordCandidate, ...]] = {}

    def candidates(self, query: str) -> tuple[WordCandidate, ...]:
        query = query.casefold()
        if query in self.frequencies:
            return ()
        if query not in self._cache:
            possible: set[str] = set()
            for deletion in _deletions(query):
                possible.update(self.delete_index.get(deletion, ()))
            candidates = []
            for value in possible:
                if abs(len(value) - len(query)) > 1:
                    continue
                distance = levenshtein(query, value)
                if distance <= 1:
                    candidates.append(
                        WordCandidate(value, distance, self.frequencies[value])
                    )
            candidates.sort(
                key=lambda item: (item.distance, -item.frequency, item.value)
            )
            self._cache[query] = tuple(candidates)
        return self._cache[query]


def _copy_case(source: str, value: str) -> str:
    if source.isupper():
        return value.upper()
    if source[:1].isupper():
        return value[:1].upper() + value[1:]
    return value


def correct_token(
    raw_token: str,
    lexicon: TrainingLexicon,
    minimum_frequency: int,
    ambiguity_ratio: float,
) -> str:
    match = TOKEN_PATTERN.fullmatch(raw_token)
    if not match:
        return raw_token
    prefix, core, suffix = match.groups()
    candidates = [
        candidate
        for candidate in lexicon.candidates(core)
        if candidate.frequency >= minimum_frequency
    ]
    if not candidates:
        return raw_token
    best = candidates[0]
    if (
        len(candidates) > 1
        and candidates[1].distance == best.distance
        and best.frequency < ambiguity_ratio * candidates[1].frequency
    ):
        return raw_token
    return prefix + _copy_case(core, best.value) + suffix


def correct_text(
    value: str,
    lexicon: TrainingLexicon,
    minimum_frequency: int,
    ambiguity_ratio: float,
) -> tuple[str, int]:
    source = value.split()
    corrected = [
        correct_token(token, lexicon, minimum_frequency, ambiguity_ratio)
        for token in source
    ]
    return " ".join(corrected), sum(a != b for a, b in zip(source, corrected, strict=True))


def _write_predictions(
    path: Path, id_column: str, ids: list[str], values: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_column, "prediction"])
        writer.writeheader()
        for identifier, value in zip(ids, values, strict=True):
            writer.writerow({id_column: identifier, "prediction": value})


def _validate_prediction_ids(ids: list[str], predictions: dict[str, str]) -> None:
    expected = set(ids)
    if set(predictions) != expected:
        raise SchemaError(
            f"prediction ID mismatch: missing={sorted(expected-set(predictions))[:5]}, "
            f"extra={sorted(set(predictions)-expected)[:5]}"
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
        raise SchemaError("fold-safe lexicon tuning requires a fold column")
    if fold is not None and fold not in {int(row["fold"]) for row in truth.rows}:
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

    folds = sorted({int(row["fold"]) for row in selected_rows})
    lexicons = {
        fold_value: TrainingLexicon(
            [
                row[truth.target_column]
                for row in truth.rows
                if int(row["fold"]) != fold_value
            ]
        )
        for fold_value in folds
    }
    row_folds = [int(row["fold"]) for row in selected_rows]

    trials: list[dict[str, Any]] = []
    best: tuple[float, int, int, int, list[str]] | None = None
    order = 0
    for minimum_frequency in MINIMUM_FREQUENCIES:
        for ambiguity_ratio in AMBIGUITY_RATIOS:
            corrected: list[str] = []
            changes = 0
            for value, fold_value in zip(source, row_folds, strict=True):
                result, count = correct_text(
                    value,
                    lexicons[fold_value],
                    minimum_frequency,
                    ambiguity_ratio,
                )
                corrected.append(result)
                changes += count
            metric = competition_error(references, corrected)
            record = {
                "minimum_frequency": minimum_frequency,
                "ambiguity_ratio": ambiguity_ratio,
                "changed_tokens": changes,
                **metric,
            }
            trials.append(record)
            key = (float(metric["mean_error"]), changes, order)
            if best is None or key < best[:3]:
                best = (*key, minimum_frequency, corrected)
                best_ratio = ambiguity_ratio
            order += 1
    assert best is not None
    _, changes, _, minimum_frequency, corrected = best

    result = {
        "fold_filter": fold,
        "rows": len(ids),
        "minimum_frequency": minimum_frequency,
        "ambiguity_ratio": best_ratio,
        "changed_tokens": changes,
        "score": competition_error(references, corrected),
        "baseline_score": competition_error(references, source),
        "trial_count": len(trials),
        "trials": sorted(trials, key=lambda item: float(item["mean_error"]))[:10],
        "method": "ASCII alphabetic token, edit distance <= 1, fold-excluded label lexicon",
    }
    config_path = absolute(output_config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_path = absolute(output_predictions)
    _write_predictions(output_path, truth.id_column, ids, corrected)
    with output_path.with_suffix(".changes.jsonl").open("w", encoding="utf-8") as handle:
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
    lexicon = TrainingLexicon([row[train.target_column] for row in train.rows])
    config = json.loads(absolute(config_path).read_text(encoding="utf-8"))
    minimum_frequency = int(config["minimum_frequency"])
    ambiguity_ratio = float(config["ambiguity_ratio"])
    _, predictions = read_prediction_map(predictions_path)
    ids = [row[test.id_column].strip() for row in test.rows]
    _validate_prediction_ids(ids, predictions)

    corrected: list[str] = []
    changes: list[dict[str, str]] = []
    for identifier in ids:
        before = predictions[identifier]
        after, _ = correct_text(
            before, lexicon, minimum_frequency, ambiguity_ratio
        )
        corrected.append(after)
        if before != after:
            changes.append({"id": identifier, "before": before, "after": after})
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

