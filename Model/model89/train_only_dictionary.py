#!/usr/bin/env python3
"""Conservative, fold-safe dictionary correction for Model88 predictions.

The dictionary and bigram counts are learned exclusively from the supplied
training labels.  An unknown OCR token is changed only when a one-edit
candidate is sufficiently frequent, sufficiently unambiguous, and supported
by its immediate word context.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from road_htr.lexicon_corrector import TOKEN_PATTERN, TrainingLexicon, _copy_case
from road_htr.metrics import (
    CER_NORMALIZER,
    WER_NORMALIZER,
    competition_error,
    levenshtein,
    strip_conversation_tokens,
)
from road_htr.schema import predictions_by_id, read_table
from road_htr.submission import write_submission


@dataclass(frozen=True)
class Configuration:
    minimum_length: int
    minimum_frequency: int
    minimum_context: int
    require_both_contexts: bool
    ambiguity_ratio: float | None
    protect_uppercase: bool


class ContextLexicon:
    def __init__(self, texts: list[str]):
        self.lexicon = TrainingLexicon(texts)
        self.bigrams: Counter[tuple[str, str]] = Counter()
        for text in texts:
            words = [self.core(token) for token in text.split()]
            words = [word for word in words if word]
            sequence = ["<s>", *words, "</s>"]
            self.bigrams.update(zip(sequence, sequence[1:]))

    @staticmethod
    def core(token: str) -> str:
        match = TOKEN_PATTERN.fullmatch(token)
        return match.group(2).casefold() if match else ""


def correction(
    raw_token: str,
    left: str,
    right: str,
    model: ContextLexicon,
    config: Configuration,
) -> tuple[str, dict[str, object] | None]:
    match = TOKEN_PATTERN.fullmatch(raw_token)
    if match is None:
        return raw_token, None
    prefix, source, suffix = match.groups()
    source_key = source.casefold()
    if len(source_key) < config.minimum_length:
        return raw_token, None
    if config.protect_uppercase and source.isupper() and len(source) > 1:
        return raw_token, None

    candidates = list(model.lexicon.candidates(source_key))
    if not candidates:
        return raw_token, None
    best_distance = candidates[0].distance
    nearest = [item for item in candidates if item.distance == best_distance]
    if best_distance != 1:
        return raw_token, None
    nearest.sort(key=lambda item: (-item.frequency, item.value))
    best = nearest[0]
    if best.frequency < config.minimum_frequency:
        return raw_token, None
    if len(nearest) > 1:
        if config.ambiguity_ratio is None:
            return raw_token, None
        if best.frequency < config.ambiguity_ratio * nearest[1].frequency:
            return raw_token, None

    left_support = model.bigrams[(left, best.value)]
    right_support = model.bigrams[(best.value, right)]
    if config.require_both_contexts:
        supported = (
            left_support >= config.minimum_context
            and right_support >= config.minimum_context
        )
    else:
        supported = max(left_support, right_support) >= config.minimum_context
    if not supported:
        return raw_token, None

    replacement = prefix + _copy_case(source, best.value) + suffix
    if replacement == raw_token:
        return raw_token, None
    return replacement, {
        "before": raw_token,
        "after": replacement,
        "candidate_frequency": best.frequency,
        "candidate_count": len(nearest),
        "left_support": left_support,
        "right_support": right_support,
    }


def correct_text(
    value: str, model: ContextLexicon, config: Configuration
) -> tuple[str, list[dict[str, object]]]:
    tokens = value.split()
    cores = [model.core(token) for token in tokens]
    context = ["<s>", *cores, "</s>"]
    output: list[str] = []
    changes: list[dict[str, object]] = []
    for index, token in enumerate(tokens):
        replacement, change = correction(
            token, context[index], context[index + 2], model, config
        )
        output.append(replacement)
        if change is not None:
            changes.append({"token_index": index, **change})
    return " ".join(output), changes


def configurations() -> list[Configuration]:
    result = []
    for minimum_length in (3, 4, 5, 6):
        for minimum_frequency in (2, 3, 5, 8, 12):
            for minimum_context in (1, 2, 3):
                for require_both in (False, True):
                    for ambiguity_ratio in (None, 10.0, 5.0):
                        for protect_uppercase in (True, False):
                            result.append(
                                Configuration(
                                    minimum_length,
                                    minimum_frequency,
                                    minimum_context,
                                    require_both,
                                    ambiguity_ratio,
                                    protect_uppercase,
                                )
                            )
    return result


def score_folds(
    references: list[str], hypotheses: list[str], folds: list[int]
) -> dict[str, dict[str, float]]:
    result = {}
    for fold in sorted(set(folds)):
        selected = [index for index, value in enumerate(folds) if value == fold]
        result[str(fold)] = competition_error(
            [references[index] for index in selected],
            [hypotheses[index] for index in selected],
        )
    return result


class FastMetric:
    """Reuse baseline row distances; only rescore rows a setting changes."""

    def __init__(self, references: list[str], baseline: list[str]):
        self.references = [strip_conversation_tokens(value) for value in references]
        self.baseline = [strip_conversation_tokens(value) for value in baseline]
        self.word_weights = [math.sqrt(max(len(value.split()), 1)) for value in self.references]
        self.char_weights = [math.sqrt(max(len(value), 1)) for value in self.references]
        self.word_total = sum(self.word_weights)
        self.char_total = sum(self.char_weights)
        self.base_word_edits = [
            levenshtein(reference.split(), hypothesis.split())
            for reference, hypothesis in zip(self.references, self.baseline, strict=True)
        ]
        self.base_char_edits = [
            levenshtein(reference, hypothesis)
            for reference, hypothesis in zip(self.references, self.baseline, strict=True)
        ]
        self.base_word_sum = sum(
            weight * edits
            for weight, edits in zip(self.word_weights, self.base_word_edits, strict=True)
        )
        self.base_char_sum = sum(
            weight * edits
            for weight, edits in zip(self.char_weights, self.base_char_edits, strict=True)
        )

    def score(self, hypotheses: list[str]) -> dict[str, float]:
        word_sum = self.base_word_sum
        char_sum = self.base_char_sum
        cleaned = [strip_conversation_tokens(value) for value in hypotheses]
        for index, (before, after) in enumerate(zip(self.baseline, cleaned, strict=True)):
            if before == after:
                continue
            word_sum += self.word_weights[index] * (
                levenshtein(self.references[index].split(), after.split())
                - self.base_word_edits[index]
            )
            char_sum += self.char_weights[index] * (
                levenshtein(self.references[index], after)
                - self.base_char_edits[index]
            )
        wer = word_sum / self.word_total
        cer = char_sum / self.char_total
        normalized_wer = wer / WER_NORMALIZER
        normalized_cer = cer / CER_NORMALIZER
        mean_error = (normalized_wer + normalized_cer) / 2
        return {
            "weighted_wer": wer,
            "weighted_cer": cer,
            "normalized_wer_error": normalized_wer,
            "normalized_cer_error": normalized_cer,
            "mean_error": mean_error,
            "multi_score": 1 - mean_error,
        }


def write_predictions(path: Path, id_column: str, ids: list[str], values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_column, "prediction"])
        writer.writeheader()
        writer.writerows(
            {id_column: identifier, "prediction": value}
            for identifier, value in zip(ids, values, strict=True)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()

    truth = read_table(args.truth, require_target=True)
    oof = read_table(args.oof)
    assert truth.target_column is not None
    prediction_column = oof.target_column or next(
        column for column in oof.fieldnames if column != oof.id_column
    )
    oof_map = predictions_by_id(oof, prediction_column)
    ids = [row[truth.id_column].strip() for row in truth.rows]
    references = [row[truth.target_column] for row in truth.rows]
    folds = [int(row["fold"]) for row in truth.rows]
    source = [oof_map[identifier] for identifier in ids]
    fold_models = {
        fold: ContextLexicon(
            [
                row[truth.target_column]
                for row in truth.rows
                if int(row["fold"]) != fold
            ]
        )
        for fold in sorted(set(folds))
    }

    trials: list[dict[str, object]] = []
    best_key: tuple[float, int, int] | None = None
    best: tuple[Configuration, list[str], list[list[dict[str, object]]]] | None = None
    baseline = competition_error(references, source)
    fast_metric = FastMetric(references, source)
    for order, config in enumerate(configurations()):
        corrected: list[str] = []
        operations: list[list[dict[str, object]]] = []
        for value, fold in zip(source, folds, strict=True):
            result, changes = correct_text(value, fold_models[fold], config)
            corrected.append(result)
            operations.append(changes)
        metric = fast_metric.score(corrected)
        changed_tokens = sum(map(len, operations))
        row = {
            "config": config.__dict__,
            "changed_rows": sum(bool(value) for value in operations),
            "changed_tokens": changed_tokens,
            "score": metric,
        }
        trials.append(row)
        key = (float(metric["mean_error"]), changed_tokens, order)
        if best_key is None or key < best_key:
            best_key = key
            best = config, corrected, operations

    assert best is not None
    config, corrected, operations = best
    selected_score = competition_error(references, corrected)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "crossfit_predictions.csv", truth.id_column, ids, corrected)
    with (output_dir / "crossfit_changes.jsonl").open("w", encoding="utf-8") as handle:
        for identifier, before, after, changes in zip(
            ids, source, corrected, operations, strict=True
        ):
            if changes:
                handle.write(json.dumps({"id": identifier, "before": before, "after": after, "changes": changes}, ensure_ascii=False) + "\n")

    train = read_table(args.train, require_target=True)
    test = read_table(args.test)
    test_predictions = read_table(args.test_predictions)
    assert train.target_column is not None
    test_prediction_column = test_predictions.target_column or next(
        column for column in test_predictions.fieldnames
        if column != test_predictions.id_column
    )
    test_map = predictions_by_id(test_predictions, test_prediction_column)
    final_model = ContextLexicon([row[train.target_column] for row in train.rows])
    test_ids = [row[test.id_column].strip() for row in test.rows]
    test_source = [test_map[identifier] for identifier in test_ids]
    test_corrected: list[str] = []
    test_changes: list[dict[str, object]] = []
    for identifier, value in zip(test_ids, test_source, strict=True):
        result, changes = correct_text(value, final_model, config)
        test_corrected.append(result)
        if changes:
            test_changes.append({"id": identifier, "before": value, "after": result, "changes": changes})
    write_predictions(output_dir / "test.csv", test.id_column, test_ids, test_corrected)
    with (output_dir / "test_changes.jsonl").open("w", encoding="utf-8") as handle:
        for row in test_changes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_submission(args.sample, args.submission, dict(zip(test_ids, test_corrected, strict=True)))

    ranked = sorted(
        trials,
        key=lambda row: (
            float(row["score"]["mean_error"]),
            int(row["changed_tokens"]),
        ),
    )
    report = {
        "method": "training-only, fold-safe one-edit dictionary with immediate bigram evidence",
        "rows": len(ids),
        "trial_count": len(trials),
        "selected_config": config.__dict__,
        "baseline_score": baseline,
        "selected_score": selected_score,
        "multi_score_delta": selected_score["multi_score"] - baseline["multi_score"],
        "baseline_fold_scores": score_folds(references, source, folds),
        "selected_fold_scores": score_folds(references, corrected, folds),
        "changed_rows": sum(bool(value) for value in operations),
        "changed_tokens": sum(map(len, operations)),
        "test_changed_rows": len(test_changes),
        "top_trials": ranked[:20],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
