"""Cross-fitted repair of recurring OCR word substitutions."""

from __future__ import annotations

import argparse
import hashlib
import json
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


MAXIMUM_DISTANCES = (1, 2)
MINIMUM_SUPPORTS = (2, 3, 4, 5)
MINIMUM_PRECISIONS = (0.5, 2 / 3, 0.8, 1.0)
MINIMUM_LENGTHS = (2, 4, 6)
INNER_FOLDS = 5


def _core(raw_token: str) -> str:
    match = TOKEN_PATTERN.fullmatch(raw_token)
    return match.group(2).casefold() if match else ""


def _aligned_token_pairs(
    hypothesis: str, reference: str
) -> list[tuple[str, str]]:
    """Return deterministic diagonal pairs from a minimum word alignment."""

    source = hypothesis.split()
    target = reference.split()
    rows = len(source) + 1
    columns = len(target) + 1
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
                costs[i - 1][j - 1] + (source[i - 1] != target[j - 1]),
            )

    pairs: list[tuple[str, str]] = []
    i, j = len(source), len(target)
    while i or j:
        if i and j:
            diagonal = costs[i - 1][j - 1] + (
                source[i - 1] != target[j - 1]
            )
            if costs[i][j] == diagonal:
                pairs.append((source[i - 1], target[j - 1]))
                i -= 1
                j -= 1
                continue
        if i and costs[i][j] == costs[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


class SupervisedErrorModel:
    """Counts aligned OCR-core to truth-core outcomes."""

    def __init__(self, examples: Iterable[tuple[str, str]]):
        self.outcomes: dict[str, Counter[str]] = defaultdict(Counter)
        for hypothesis, reference in examples:
            for raw_source, raw_target in _aligned_token_pairs(hypothesis, reference):
                source = _core(raw_source)
                target = _core(raw_target)
                if source and target:
                    self.outcomes[source][target] += 1

    def mappings(
        self,
        maximum_distance: int,
        minimum_support: int,
        minimum_precision: float,
        minimum_length: int,
    ) -> dict[str, dict[str, Any]]:
        mappings: dict[str, dict[str, Any]] = {}
        for source, outcomes in self.outcomes.items():
            if len(source) < minimum_length:
                continue
            candidates = []
            for target, count in outcomes.items():
                if target == source:
                    continue
                distance = levenshtein(source, target)
                if 1 <= distance <= maximum_distance:
                    candidates.append((count, -distance, target, distance))
            if not candidates:
                continue
            count, _, target, distance = max(candidates)
            total = sum(outcomes.values())
            precision = count / total
            if count < minimum_support or precision < minimum_precision:
                continue
            mappings[source] = {
                "target": target,
                "support": count,
                "observations": total,
                "precision": precision,
                "distance": distance,
            }
        return mappings


def repair_supervised_errors(
    value: str, mappings: dict[str, dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    corrected: list[str] = []
    changes: list[dict[str, Any]] = []
    for index, raw_token in enumerate(value.split()):
        match = TOKEN_PATTERN.fullmatch(raw_token)
        source = _core(raw_token)
        mapping = mappings.get(source)
        if match is None or mapping is None:
            corrected.append(raw_token)
            continue
        prefix, source_core, suffix = match.groups()
        replacement = (
            prefix + _copy_case(source_core, str(mapping["target"])) + suffix
        )
        corrected.append(replacement)
        if replacement != raw_token:
            changes.append(
                {
                    "token_index": index,
                    "before": raw_token,
                    "after": replacement,
                    **mapping,
                }
            )
    return " ".join(corrected), changes


def _inner_fold(identifier: str, count: int = INNER_FOLDS) -> int:
    digest = hashlib.blake2b(identifier.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % count


def _score_by_fold(
    references: list[str], hypotheses: list[str], folds: list[int]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for fold in sorted(set(folds)):
        indices = [index for index, value in enumerate(folds) if value == fold]
        result[str(fold)] = competition_error(
            [references[index] for index in indices],
            [hypotheses[index] for index in indices],
        )
    return result


def tune(
    truth_path: str | Path,
    predictions_path: str | Path,
    output_config: str | Path,
    output_predictions: str | Path,
    fold: int | None = None,
    inner_folds: int = INNER_FOLDS,
) -> dict[str, Any]:
    truth = read_table(absolute(truth_path), require_target=True)
    if "fold" not in truth.fieldnames:
        raise SchemaError("cross-fitted error tuning requires a fold column")
    available_folds = {int(row["fold"]) for row in truth.rows}
    if fold is not None and fold not in available_folds:
        raise ValueError(f"fold {fold} is absent")
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    assert truth.target_column is not None
    selected_rows = [
        row for row in truth.rows if fold is None or int(row["fold"]) == fold
    ]
    ids = [row[truth.id_column].strip() for row in selected_rows]
    references = [row[truth.target_column] for row in selected_rows]
    outer_folds = [int(row["fold"]) for row in selected_rows]
    inner_assignments = [_inner_fold(identifier, inner_folds) for identifier in ids]
    _, prediction_map = read_prediction_map(predictions_path)
    _validate_prediction_ids(ids, prediction_map)
    source = [prediction_map[identifier] for identifier in ids]

    # Every base prediction within an outer fold was produced without that
    # fold's labels. Learn each error table only from the other inner groups of
    # that same fold, so no evaluated target can influence its own correction.
    models: dict[tuple[int, int], SupervisedErrorModel] = {}
    for outer in sorted(set(outer_folds)):
        for inner in range(inner_folds):
            examples = (
                (source[index], references[index])
                for index in range(len(ids))
                if outer_folds[index] == outer and inner_assignments[index] != inner
            )
            models[(outer, inner)] = SupervisedErrorModel(examples)

    trials: list[dict[str, Any]] = []
    best_key: tuple[float, int, int] | None = None
    best_values: tuple[int, int, float, int, list[str], int, int] | None = None
    order = 0
    for maximum_distance in MAXIMUM_DISTANCES:
        for minimum_support in MINIMUM_SUPPORTS:
            for minimum_precision in MINIMUM_PRECISIONS:
                for minimum_length in MINIMUM_LENGTHS:
                    mapping_cache = {
                        key: model.mappings(
                            maximum_distance,
                            minimum_support,
                            minimum_precision,
                            minimum_length,
                        )
                        for key, model in models.items()
                    }
                    corrected: list[str] = []
                    changed_tokens = 0
                    distance_two_tokens = 0
                    for value, outer, inner in zip(
                        source, outer_folds, inner_assignments, strict=True
                    ):
                        result, changes = repair_supervised_errors(
                            value, mapping_cache[(outer, inner)]
                        )
                        corrected.append(result)
                        changed_tokens += len(changes)
                        distance_two_tokens += sum(
                            int(change["distance"]) == 2 for change in changes
                        )
                    metric = competition_error(references, corrected)
                    trials.append(
                        {
                            "maximum_distance": maximum_distance,
                            "minimum_support": minimum_support,
                            "minimum_precision": minimum_precision,
                            "minimum_length": minimum_length,
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
                            minimum_support,
                            minimum_precision,
                            minimum_length,
                            corrected,
                            changed_tokens,
                            distance_two_tokens,
                        )
                    order += 1
    assert best_values is not None
    (
        maximum_distance,
        minimum_support,
        minimum_precision,
        minimum_length,
        corrected,
        changed_tokens,
        distance_two_tokens,
    ) = best_values
    result = {
        "fold_filter": fold,
        "rows": len(ids),
        "inner_folds": inner_folds,
        "maximum_distance": maximum_distance,
        "minimum_support": minimum_support,
        "minimum_precision": minimum_precision,
        "minimum_length": minimum_length,
        "changed_tokens": changed_tokens,
        "distance_two_tokens": distance_two_tokens,
        "score": competition_error(references, corrected),
        "baseline_score": competition_error(references, source),
        "fold_scores": _score_by_fold(references, corrected, outer_folds),
        "baseline_fold_scores": _score_by_fold(references, source, outer_folds),
        "trial_count": len(trials),
        "trials": sorted(trials, key=lambda item: float(item["mean_error"]))[:10],
        "method": (
            "recurring aligned OCR-to-truth substitutions, trained on other "
            "inner groups within each label-isolated outer fold"
        ),
    }
    config = absolute(output_config)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
    training_predictions_path: str | Path,
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
    _, training_predictions = read_prediction_map(training_predictions_path)
    train_ids = [row[train.id_column].strip() for row in train.rows]
    _validate_prediction_ids(train_ids, training_predictions)
    model = SupervisedErrorModel(
        (training_predictions[identifier], row[train.target_column])
        for identifier, row in zip(train_ids, train.rows, strict=True)
    )
    config = json.loads(absolute(config_path).read_text(encoding="utf-8"))
    mappings = model.mappings(
        int(config["maximum_distance"]),
        int(config["minimum_support"]),
        float(config["minimum_precision"]),
        int(config["minimum_length"]),
    )
    _, predictions = read_prediction_map(predictions_path)
    ids = [row[test.id_column].strip() for row in test.rows]
    _validate_prediction_ids(ids, predictions)

    corrected: list[str] = []
    changes: list[dict[str, Any]] = []
    for identifier in ids:
        before = predictions[identifier]
        after, token_changes = repair_supervised_errors(before, mappings)
        corrected.append(after)
        if token_changes:
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
    print(
        f"learned {len(mappings)} mappings and corrected "
        f"{len(changes)}/{len(ids)} predictions"
    )
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
    parser.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    args = parser.parse_args()
    result = tune(
        args.truth,
        args.predictions,
        args.output_config,
        args.output_predictions,
        args.fold,
        args.inner_folds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def apply_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--training-predictions", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    apply(
        args.train,
        args.training_predictions,
        args.test,
        args.predictions,
        args.config,
        args.output,
        args.sample,
        args.submission,
    )
