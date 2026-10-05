"""Leakage-safe selector between two complete OCR transcription candidates.

The selector is deliberately small and dependency free.  It learns a weighted
binary decision (candidate A or candidate B) from prediction-time text,
optional OCR character confidences, and target-corpus language-model scores.
Every reported validation row is excluded from both the selector fit and the
language model used to describe that row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .metrics import (
    CER_NORMALIZER,
    WER_NORMALIZER,
    competition_error,
    levenshtein,
    strip_conversation_tokens,
)
from .schema import SchemaError, predictions_by_id, read_table
from .submission import clean_prediction_preserving_internal_spaces, write_submission


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
MODEL_KIND = "metric_weighted_logistic_selector"
PROTECTED_OUTPUTS = tuple(
    (PROJECT_ROOT / "outputs" / name).resolve() for name in ("model22", "model25")
)
_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class SelectorConfig:
    single_fold_inner_splits: int = 5
    nested_lm_splits: int = 5
    seed: str = "model33-selector-v1"
    char_order: int = 4
    lm_alpha: float = 0.25
    logistic_epochs: int = 900
    logistic_learning_rate: float = 0.035
    logistic_l2: float = 0.08
    minimum_scale: float = 1e-6

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SelectorConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed - {"schema_version", "model_kind", "notes"}
        if unknown:
            raise ValueError(f"unknown selector config keys: {sorted(unknown)}")
        result = cls(**{key: values[key] for key in allowed if key in values})
        if result.single_fold_inner_splits < 2 or result.nested_lm_splits < 2:
            raise ValueError("cross-fitting requires at least two splits")
        if result.char_order < 2 or result.char_order > 6:
            raise ValueError("char_order must be in [2, 6]")
        if result.lm_alpha <= 0 or result.logistic_epochs < 1:
            raise ValueError("invalid language-model/logistic configuration")
        if result.logistic_learning_rate <= 0 or result.logistic_l2 < 0:
            raise ValueError("invalid logistic optimizer configuration")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "single_fold_inner_splits": self.single_fold_inner_splits,
            "nested_lm_splits": self.nested_lm_splits,
            "seed": self.seed,
            "char_order": self.char_order,
            "lm_alpha": self.lm_alpha,
            "logistic_epochs": self.logistic_epochs,
            "logistic_learning_rate": self.logistic_learning_rate,
            "logistic_l2": self.logistic_l2,
            "minimum_scale": self.minimum_scale,
        }


@dataclass(frozen=True)
class LabeledRow:
    identifier: str
    target: str
    fold: int


@dataclass(frozen=True)
class OcrDetails:
    source_text: str
    confidences: tuple[float, ...]


@dataclass(frozen=True)
class RowCost:
    a: float
    b: float

    @property
    def choose_b(self) -> bool:
        return self.b < self.a

    @property
    def importance(self) -> float:
        return abs(self.b - self.a)


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]

    def probability_b(self, values: Sequence[float]) -> float:
        if len(values) != len(self.feature_names):
            raise ValueError("selector feature width mismatch")
        score = self.coefficients[0]
        for value, mean, scale, coefficient in zip(
            values, self.means, self.scales, self.coefficients[1:], strict=True
        ):
            score += coefficient * ((float(value) - mean) / scale)
        return _sigmoid(score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "LogisticModel":
        names = tuple(str(value) for value in values["feature_names"])
        means = tuple(float(value) for value in values["means"])
        scales = tuple(float(value) for value in values["scales"])
        coefficients = tuple(float(value) for value in values["coefficients"])
        if len(names) != len(means) or len(names) != len(scales):
            raise ValueError("corrupt selector standardization state")
        if len(coefficients) != len(names) + 1:
            raise ValueError("corrupt selector coefficient state")
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("corrupt selector scales")
        return cls(names, means, scales, coefficients)


class TextLanguageModel:
    """Smoothed character n-gram and word-unigram target model."""

    def __init__(self, order: int, alpha: float):
        self.order = order
        self.alpha = alpha
        self.ngrams: Counter[str] = Counter()
        self.contexts: Counter[str] = Counter()
        self.words: Counter[str] = Counter()
        self.characters: set[str] = set()
        self.word_total = 0

    @staticmethod
    def normalize(value: str) -> str:
        return " ".join(strip_conversation_tokens(value).split()).casefold()

    @classmethod
    def fit(
        cls, texts: Iterable[str], order: int = 4, alpha: float = 0.25
    ) -> "TextLanguageModel":
        model = cls(order, alpha)
        start = "\x02" * (order - 1)
        for raw in texts:
            text = cls.normalize(raw)
            padded = start + text + "\x03"
            model.characters.update(text)
            model.characters.add("\x03")
            for position in range(order - 1, len(padded)):
                context = padded[position - order + 1 : position]
                ngram = padded[position - order + 1 : position + 1]
                model.contexts[context] += 1
                model.ngrams[ngram] += 1
            tokens = _WORD_RE.findall(text)
            model.words.update(tokens)
            model.word_total += len(tokens)
        return model

    def qualities(self, raw: str) -> tuple[float, float, float]:
        text = self.normalize(raw)
        padded = "\x02" * (self.order - 1) + text + "\x03"
        vocabulary = max(len(self.characters) + 1, 2)
        char_logs: list[float] = []
        for position in range(self.order - 1, len(padded)):
            context = padded[position - self.order + 1 : position]
            ngram = padded[position - self.order + 1 : position + 1]
            numerator = self.ngrams.get(ngram, 0) + self.alpha
            denominator = self.contexts.get(context, 0) + self.alpha * vocabulary
            char_logs.append(math.log(numerator / denominator))
        tokens = _WORD_RE.findall(text)
        word_vocabulary = max(len(self.words) + 1, 2)
        word_denominator = self.word_total + self.alpha * word_vocabulary
        word_logs = [
            math.log((self.words.get(token, 0) + self.alpha) / word_denominator)
            for token in tokens
        ]
        known = sum(token in self.words for token in tokens) / max(len(tokens), 1)
        return (
            fmean(char_logs) if char_logs else math.log(1 / vocabulary),
            fmean(word_logs) if word_logs else math.log(self.alpha / word_denominator),
            known,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "alpha": self.alpha,
            "ngrams": dict(sorted(self.ngrams.items())),
            "contexts": dict(sorted(self.contexts.items())),
            "words": dict(sorted(self.words.items())),
            "characters": sorted(self.characters),
            "word_total": self.word_total,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TextLanguageModel":
        model = cls(int(values["order"]), float(values["alpha"]))
        model.ngrams.update({str(k): int(v) for k, v in values["ngrams"].items()})
        model.contexts.update(
            {str(k): int(v) for k, v in values["contexts"].items()}
        )
        model.words.update({str(k): int(v) for k, v in values["words"].items()})
        model.characters.update(str(value) for value in values["characters"])
        model.word_total = int(values["word_total"])
        return model


def _project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for identifier in sorted(ids):
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: str | Path) -> SelectorConfig:
    config_path = _project_path(path)
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("selector config must be a JSON mapping")
    if values.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("unsupported selector config schema")
    return SelectorConfig.from_mapping(values)


def _prediction_column(table: Any, path: Path) -> str:
    if table.target_column is not None:
        return table.target_column
    candidates = [name for name in table.fieldnames if name != table.id_column]
    if len(candidates) != 1:
        raise SchemaError(f"prediction CSV must have one value column: {path}")
    return candidates[0]


def read_predictions(path: str | Path) -> dict[str, str]:
    resolved = _project_path(path)
    table = read_table(resolved)
    column = _prediction_column(table, resolved)
    return {
        identifier: clean_prediction_preserving_internal_spaces(value)
        for identifier, value in predictions_by_id(table, column).items()
    }


def read_details(path: str | Path | None) -> dict[str, OcrDetails]:
    if path is None:
        return {}
    resolved = _project_path(path)
    records: dict[str, OcrDetails] = {}
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                identifier = str(value.get("id", "")).strip()
                candidates = value.get("candidates") or []
                confidences = tuple(
                    float(item) for item in (value.get("character_confidences") or [])
                )
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise SchemaError(f"invalid OCR details at {resolved}:{line_number}") from error
            if not identifier or not candidates:
                raise SchemaError(f"missing OCR details at {resolved}:{line_number}")
            if identifier in records:
                raise SchemaError(f"duplicate details ID {identifier!r} in {resolved}")
            if any(not math.isfinite(item) or item < 0 or item > 1 for item in confidences):
                raise SchemaError(f"invalid character confidence at {resolved}:{line_number}")
            records[identifier] = OcrDetails(
                clean_prediction_preserving_internal_spaces(candidates[0]), confidences
            )
    if not records:
        raise SchemaError(f"no OCR details in {resolved}")
    return records


def load_labeled_rows(path: str | Path) -> tuple[str, list[LabeledRow]]:
    resolved = _project_path(path)
    table = read_table(resolved, require_target=True)
    if "fold" not in table.fieldnames:
        raise SchemaError("truth CSV must contain a fold column")
    assert table.target_column is not None
    rows: list[LabeledRow] = []
    for row in table.rows:
        try:
            fold = int(row["fold"])
        except (TypeError, ValueError) as error:
            raise SchemaError("truth fold values must be integers") from error
        rows.append(
            LabeledRow(
                row[table.id_column].strip(), row[table.target_column], fold
            )
        )
    if not rows:
        raise SchemaError("truth CSV contains no rows")
    return table.id_column, rows


def _validate_candidate_ids(
    ids: Sequence[str], name: str, predictions: Mapping[str, str]
) -> None:
    expected = set(ids)
    missing = expected - set(predictions)
    if missing:
        raise SchemaError(
            f"{name} is missing {len(missing)} required IDs: {sorted(missing)[:5]}"
        )
    empty = [identifier for identifier in ids if not predictions[identifier]]
    if empty:
        raise SchemaError(f"{name} has empty predictions: {empty[:5]}")


def _validate_detail_ids(
    ids: Sequence[str], name: str, details: Mapping[str, OcrDetails], supplied: bool
) -> None:
    if not supplied:
        return
    missing = set(ids) - set(details)
    if missing:
        raise SchemaError(
            f"{name} is missing {len(missing)} required IDs: {sorted(missing)[:5]}"
        )


def stable_groups(ids: Sequence[str], count: int, seed: str) -> dict[str, int]:
    """Assign deterministic, approximately balanced groups without using labels."""

    if count < 2 or len(ids) < count:
        raise ValueError(f"need at least {count} rows for {count}-way cross-fitting")
    keyed = sorted(
        ids,
        key=lambda identifier: hashlib.sha256(
            f"{seed}\0{identifier}".encode("utf-8")
        ).digest(),
    )
    return {identifier: position % count for position, identifier in enumerate(keyed)}


def row_costs(
    rows: Sequence[LabeledRow],
    candidate_a: Mapping[str, str],
    candidate_b: Mapping[str, str],
) -> dict[str, RowCost]:
    """Return each row's additive contribution to the exact mean error."""

    if not rows:
        raise ValueError("row costs require nonempty rows")
    word_denominator = sum(
        math.sqrt(max(len(strip_conversation_tokens(row.target).split()), 1))
        for row in rows
    )
    char_denominator = sum(
        math.sqrt(max(len(tuple(strip_conversation_tokens(row.target))), 1))
        for row in rows
    )
    result: dict[str, RowCost] = {}
    for row in rows:
        reference = strip_conversation_tokens(row.target)
        ref_words = reference.split()
        ref_chars = tuple(reference)
        word_weight = math.sqrt(max(len(ref_words), 1))
        char_weight = math.sqrt(max(len(ref_chars), 1))

        def contribution(prediction: str) -> float:
            clean = strip_conversation_tokens(prediction)
            word_part = (
                levenshtein(ref_words, clean.split())
                * word_weight
                / word_denominator
                / WER_NORMALIZER
            )
            char_part = (
                levenshtein(ref_chars, tuple(clean))
                * char_weight
                / char_denominator
                / CER_NORMALIZER
            )
            return (word_part + char_part) / 2

        result[row.identifier] = RowCost(
            contribution(candidate_a[row.identifier]),
            contribution(candidate_b[row.identifier]),
        )
    return result


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _confidence_features(prediction: str, details: OcrDetails | None) -> list[float]:
    if details is None or not details.confidences:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    values = tuple(max(value, 1e-8) for value in details.confidences)
    mean = fmean(values)
    variance = fmean((value - mean) ** 2 for value in values)
    source_distance = levenshtein(tuple(prediction), tuple(details.source_text))
    source_ratio = source_distance / max(len(prediction), len(details.source_text), 1)
    return [
        0.0,
        mean,
        math.exp(fmean(math.log(value) for value in values)),
        _quantile(values, 0.10),
        _quantile(values, 0.25),
        min(values),
        math.sqrt(variance),
        source_ratio,
    ]


def _text_shape(text: str) -> list[float]:
    characters = max(len(text), 1)
    words = text.split()
    return [
        math.log1p(len(text)),
        math.log1p(len(words)),
        fmean(len(word) for word in words) if words else 0.0,
        sum(character.isalpha() for character in text) / characters,
        sum(character.isdigit() for character in text) / characters,
        sum(character.isupper() for character in text) / characters,
        sum(not character.isalnum() and not character.isspace() for character in text)
        / characters,
        sum(ord(character) > 127 for character in text) / characters,
        text.count("^") / characters,
    ]


_SHAPE_NAMES = (
    "log_chars",
    "log_words",
    "mean_word_chars",
    "alpha_fraction",
    "digit_fraction",
    "uppercase_fraction",
    "punctuation_fraction",
    "non_ascii_fraction",
    "caret_fraction",
)
_CONFIDENCE_NAMES = (
    "confidence_missing",
    "confidence_mean",
    "confidence_geometric",
    "confidence_q10",
    "confidence_q25",
    "confidence_min",
    "confidence_std",
    "confidence_source_distance",
)


def feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for prefix in ("a", "b"):
        names.extend(f"{prefix}_{name}" for name in _SHAPE_NAMES)
        names.extend(
            (
                f"{prefix}_lm_char_logprob",
                f"{prefix}_lm_word_logprob",
                f"{prefix}_lm_known_word_fraction",
            )
        )
        names.extend(f"{prefix}_{name}" for name in _CONFIDENCE_NAMES)
    names.extend(
        (
            "pair_char_distance_ratio",
            "pair_word_distance_ratio",
            "pair_token_jaccard_distance",
            "pair_log_char_length_b_minus_a",
            "pair_log_word_length_b_minus_a",
            "pair_common_prefix_ratio",
            "pair_common_suffix_ratio",
            "pair_exact_equal",
            "pair_casefold_equal",
            "pair_lm_char_b_minus_a",
            "pair_lm_word_b_minus_a",
            "pair_lm_known_b_minus_a",
            "pair_confidence_mean_b_minus_a",
            "pair_confidence_q10_b_minus_a",
        )
    )
    return tuple(names)


def extract_features(
    text_a: str,
    text_b: str,
    details_a: OcrDetails | None,
    details_b: OcrDetails | None,
    language_model: TextLanguageModel,
) -> list[float]:
    lm_a = language_model.qualities(text_a)
    lm_b = language_model.qualities(text_b)
    confidence_a = _confidence_features(text_a, details_a)
    confidence_b = _confidence_features(text_b, details_b)
    values = [
        *_text_shape(text_a),
        *lm_a,
        *confidence_a,
        *_text_shape(text_b),
        *lm_b,
        *confidence_b,
    ]
    chars_a, chars_b = tuple(text_a), tuple(text_b)
    words_a, words_b = text_a.split(), text_b.split()
    tokens_a = {word.casefold() for word in words_a}
    tokens_b = {word.casefold() for word in words_b}
    union = tokens_a | tokens_b

    prefix = 0
    while prefix < min(len(chars_a), len(chars_b)) and chars_a[prefix] == chars_b[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(chars_a), len(chars_b))
        and chars_a[len(chars_a) - suffix - 1] == chars_b[len(chars_b) - suffix - 1]
    ):
        suffix += 1
    values.extend(
        (
            levenshtein(chars_a, chars_b) / max(len(chars_a), len(chars_b), 1),
            levenshtein(words_a, words_b) / max(len(words_a), len(words_b), 1),
            1 - len(tokens_a & tokens_b) / max(len(union), 1),
            math.log1p(len(text_b)) - math.log1p(len(text_a)),
            math.log1p(len(words_b)) - math.log1p(len(words_a)),
            prefix / max(len(chars_a), len(chars_b), 1),
            suffix / max(len(chars_a), len(chars_b), 1),
            float(text_a == text_b),
            float(text_a.casefold() == text_b.casefold()),
            lm_b[0] - lm_a[0],
            lm_b[1] - lm_a[1],
            lm_b[2] - lm_a[2],
            confidence_b[1] - confidence_a[1],
            confidence_b[3] - confidence_a[3],
        )
    )
    if len(values) != len(feature_names()) or any(
        not math.isfinite(value) for value in values
    ):
        raise RuntimeError("non-finite or malformed selector feature row")
    return values


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-min(value, 60.0))
        return 1 / (1 + inverse)
    exponential = math.exp(max(value, -60.0))
    return exponential / (1 + exponential)


def fit_logistic(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[bool],
    importance: Sequence[float],
    config: SelectorConfig,
) -> LogisticModel:
    if not matrix or len(matrix) != len(labels) or len(matrix) != len(importance):
        raise ValueError("logistic inputs must be equally sized and nonempty")
    width = len(feature_names())
    if any(len(row) != width for row in matrix):
        raise ValueError("logistic matrix has inconsistent feature width")
    means = tuple(fmean(row[column] for row in matrix) for column in range(width))
    scales = tuple(
        max(
            math.sqrt(
                fmean((row[column] - means[column]) ** 2 for row in matrix)
            ),
            config.minimum_scale,
        )
        for column in range(width)
    )
    standardized = [
        [1.0, *((value - mean) / scale for value, mean, scale in zip(row, means, scales, strict=True))]
        for row in matrix
    ]
    # Exact-loss deltas are tiny because they are additive dataset contributions.
    # Their common scale is immaterial; normalize to a mean weight of one while
    # preserving every row's true avoidable-harm ratio.
    total_importance = sum(max(float(value), 0.0) for value in importance)
    if total_importance <= 0:
        return LogisticModel(feature_names(), means, scales, (0.0,) * (width + 1))
    sample_weights = [max(float(value), 0.0) * len(matrix) / total_importance for value in importance]
    positive = sum(weight for weight, label in zip(sample_weights, labels, strict=True) if label)
    negative = sum(weight for weight, label in zip(sample_weights, labels, strict=True) if not label)
    intercept = math.log((positive + 0.5) / (negative + 0.5))
    coefficients = [intercept, *([0.0] * width)]
    first_moment = [0.0] * len(coefficients)
    second_moment = [0.0] * len(coefficients)
    beta1, beta2 = 0.9, 0.999
    weight_sum = sum(sample_weights)
    for epoch in range(1, config.logistic_epochs + 1):
        gradient = [0.0] * len(coefficients)
        for values, label, sample_weight in zip(
            standardized, labels, sample_weights, strict=True
        ):
            probability = _sigmoid(sum(a * b for a, b in zip(coefficients, values, strict=True)))
            residual = sample_weight * (probability - float(label)) / weight_sum
            for column, value in enumerate(values):
                gradient[column] += residual * value
        for column in range(1, len(coefficients)):
            gradient[column] += config.logistic_l2 * coefficients[column]
        for column, value in enumerate(gradient):
            first_moment[column] = beta1 * first_moment[column] + (1 - beta1) * value
            second_moment[column] = beta2 * second_moment[column] + (1 - beta2) * value * value
            corrected_first = first_moment[column] / (1 - beta1**epoch)
            corrected_second = second_moment[column] / (1 - beta2**epoch)
            coefficients[column] -= config.logistic_learning_rate * corrected_first / (
                math.sqrt(corrected_second) + 1e-8
            )
    return LogisticModel(feature_names(), means, scales, tuple(coefficients))


def _details_for(
    details: Mapping[str, OcrDetails], identifier: str
) -> OcrDetails | None:
    return details.get(identifier)


def _feature_matrix(
    rows: Sequence[LabeledRow],
    candidate_a: Mapping[str, str],
    candidate_b: Mapping[str, str],
    details_a: Mapping[str, OcrDetails],
    details_b: Mapping[str, OcrDetails],
    language_model: TextLanguageModel,
) -> list[list[float]]:
    return [
        extract_features(
            candidate_a[row.identifier],
            candidate_b[row.identifier],
            _details_for(details_a, row.identifier),
            _details_for(details_b, row.identifier),
            language_model,
        )
        for row in rows
    ]


def _nested_training_features(
    rows: Sequence[LabeledRow],
    candidate_a: Mapping[str, str],
    candidate_b: Mapping[str, str],
    details_a: Mapping[str, OcrDetails],
    details_b: Mapping[str, OcrDetails],
    config: SelectorConfig,
    seed_suffix: str,
) -> list[list[float]]:
    split_count = min(config.nested_lm_splits, len(rows))
    if split_count < 2:
        raise ValueError("at least two selector-training rows are required")
    groups = stable_groups(
        [row.identifier for row in rows], split_count, f"{config.seed}:{seed_suffix}:lm"
    )
    output: dict[str, list[float]] = {}
    for group in range(split_count):
        fit_rows = [row for row in rows if groups[row.identifier] != group]
        heldout_rows = [row for row in rows if groups[row.identifier] == group]
        language_model = TextLanguageModel.fit(
            (row.target for row in fit_rows), config.char_order, config.lm_alpha
        )
        matrix = _feature_matrix(
            heldout_rows,
            candidate_a,
            candidate_b,
            details_a,
            details_b,
            language_model,
        )
        output.update(
            (row.identifier, features)
            for row, features in zip(heldout_rows, matrix, strict=True)
        )
    return [output[row.identifier] for row in rows]


def _fit_one_split(
    fit_rows: Sequence[LabeledRow],
    evaluation_rows: Sequence[LabeledRow],
    candidate_a: Mapping[str, str],
    candidate_b: Mapping[str, str],
    details_a: Mapping[str, OcrDetails],
    details_b: Mapping[str, OcrDetails],
    config: SelectorConfig,
    split_name: str,
) -> tuple[LogisticModel, TextLanguageModel, list[float]]:
    train_matrix = _nested_training_features(
        fit_rows,
        candidate_a,
        candidate_b,
        details_a,
        details_b,
        config,
        split_name,
    )
    costs = row_costs(fit_rows, candidate_a, candidate_b)
    labels = [costs[row.identifier].choose_b for row in fit_rows]
    importance = [costs[row.identifier].importance for row in fit_rows]
    selector = fit_logistic(train_matrix, labels, importance, config)
    language_model = TextLanguageModel.fit(
        (row.target for row in fit_rows), config.char_order, config.lm_alpha
    )
    evaluation_matrix = _feature_matrix(
        evaluation_rows,
        candidate_a,
        candidate_b,
        details_a,
        details_b,
        language_model,
    )
    probabilities = [selector.probability_b(values) for values in evaluation_matrix]
    return selector, language_model, probabilities


def _prepare_output_dir(path: str | Path) -> Path:
    resolved = _project_path(path)
    if any(resolved == protected or protected in resolved.parents for protected in PROTECTED_OUTPUTS):
        raise ValueError("Model33 may not write into protected Model22/Model25 outputs")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _score_by_partition(
    rows: Sequence[LabeledRow], predictions: Mapping[str, str]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for fold in sorted({row.fold for row in rows}):
        selected = [row for row in rows if row.fold == fold]
        result[str(fold)] = competition_error(
            [row.target for row in selected],
            [predictions[row.identifier] for row in selected],
        )
    return result


def _score_by_assignment(
    rows: Sequence[LabeledRow],
    predictions: Mapping[str, str],
    assignments: Mapping[str, int],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for partition in sorted(set(assignments.values())):
        selected = [row for row in rows if assignments[row.identifier] == partition]
        result[str(partition)] = competition_error(
            [row.target for row in selected],
            [predictions[row.identifier] for row in selected],
        )
    return result


def evaluate_crossfit(
    truth_path: str | Path,
    candidate_a_path: str | Path,
    candidate_b_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    details_a_path: str | Path | None = None,
    details_b_path: str | Path | None = None,
    single_fold: int | None = None,
) -> dict[str, Any]:
    truth = _project_path(truth_path)
    candidate_a_file = _project_path(candidate_a_path)
    candidate_b_file = _project_path(candidate_b_path)
    config_file = _project_path(config_path)
    id_column, all_rows = load_labeled_rows(truth)
    rows = [row for row in all_rows if single_fold is None or row.fold == single_fold]
    if not rows:
        raise ValueError(f"fold {single_fold} has no rows")
    candidate_a = read_predictions(candidate_a_file)
    candidate_b = read_predictions(candidate_b_file)
    ids = [row.identifier for row in rows]
    _validate_candidate_ids(ids, "candidate A", candidate_a)
    _validate_candidate_ids(ids, "candidate B", candidate_b)
    details_a = read_details(details_a_path)
    details_b = read_details(details_b_path)
    _validate_detail_ids(ids, "candidate A details", details_a, details_a_path is not None)
    _validate_detail_ids(ids, "candidate B details", details_b, details_b_path is not None)
    config = load_config(config_file)

    if single_fold is None:
        partitions = {row.identifier: row.fold for row in rows}
        partition_values = sorted(set(partitions.values()))
        if len(partition_values) < 2:
            raise ValueError("complete OOF evaluation requires at least two folds")
        mode = "outer_fold_crossfit"
    else:
        split_count = min(config.single_fold_inner_splits, len(rows))
        partitions = stable_groups(ids, split_count, f"{config.seed}:single-fold:{single_fold}")
        partition_values = list(range(split_count))
        mode = "single_fold_inner_crossfit"

    selected: dict[str, str] = {}
    probabilities: dict[str, float] = {}
    choices: dict[str, str] = {}
    split_audits: dict[str, Any] = {}
    for partition in partition_values:
        fit_rows = [row for row in rows if partitions[row.identifier] != partition]
        evaluation_rows = [row for row in rows if partitions[row.identifier] == partition]
        if len(fit_rows) < 2 or not evaluation_rows:
            raise ValueError("cross-fit partition is too small")
        selector, language_model, split_probabilities = _fit_one_split(
            fit_rows,
            evaluation_rows,
            candidate_a,
            candidate_b,
            details_a,
            details_b,
            config,
            f"{mode}:{partition}",
        )
        for row, probability in zip(evaluation_rows, split_probabilities, strict=True):
            choose_b = probability > 0.5
            selected[row.identifier] = (
                candidate_b[row.identifier] if choose_b else candidate_a[row.identifier]
            )
            probabilities[row.identifier] = probability
            choices[row.identifier] = "b" if choose_b else "a"
        split_audits[str(partition)] = {
            "fit_rows": len(fit_rows),
            "evaluation_rows": len(evaluation_rows),
            "fit_ids_sha256": _ids_sha256(row.identifier for row in fit_rows),
            "evaluation_ids_sha256": _ids_sha256(
                row.identifier for row in evaluation_rows
            ),
            "evaluation_label_excluded_from_selector": True,
            "evaluation_label_excluded_from_language_model": True,
            "selector_coefficient_l2": math.sqrt(
                sum(value * value for value in selector.coefficients[1:])
            ),
            "language_model_target_rows": len(fit_rows),
            "selected_b_rows": sum(choices[row.identifier] == "b" for row in evaluation_rows),
        }

    output = _prepare_output_dir(output_dir)
    ordered_predictions = [selected[row.identifier] for row in rows]
    references = [row.target for row in rows]
    a_values = [candidate_a[row.identifier] for row in rows]
    b_values = [candidate_b[row.identifier] for row in rows]
    prediction_path = output / "crossfit_predictions.csv"
    audit_path = output / "selection_audit.csv"
    _atomic_csv(
        prediction_path,
        (id_column, "prediction"),
        (
            {id_column: row.identifier, "prediction": selected[row.identifier]}
            for row in rows
        ),
    )
    _atomic_csv(
        audit_path,
        (id_column, "partition", "selected_candidate", "probability_b"),
        (
            {
                id_column: row.identifier,
                "partition": partitions[row.identifier],
                "selected_candidate": choices[row.identifier],
                "probability_b": f"{probabilities[row.identifier]:.12g}",
            }
            for row in rows
        ),
    )
    score = competition_error(references, ordered_predictions)
    report = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "mode": mode,
        "single_fold": single_fold,
        "rows": len(rows),
        "partition_count": len(partition_values),
        "selected_a_rows": sum(value == "a" for value in choices.values()),
        "selected_b_rows": sum(value == "b" for value in choices.values()),
        "score": score,
        "candidate_a_score": competition_error(references, a_values),
        "candidate_b_score": competition_error(references, b_values),
        "score_delta_vs_a": score["multi_score"]
        - competition_error(references, a_values)["multi_score"],
        "score_delta_vs_b": score["multi_score"]
        - competition_error(references, b_values)["multi_score"],
        "scores_by_canonical_fold": _score_by_partition(rows, selected),
        "scores_by_crossfit_partition": _score_by_assignment(
            rows, selected, partitions
        ),
        "split_audits": split_audits,
        "prediction_artifact": str(prediction_path),
        "selection_audit_artifact": str(audit_path),
        "inputs": {
            "truth": str(truth),
            "truth_sha256": _sha256(truth),
            "candidate_a": str(candidate_a_file),
            "candidate_a_sha256": _sha256(candidate_a_file),
            "candidate_b": str(candidate_b_file),
            "candidate_b_sha256": _sha256(candidate_b_file),
            "details_a": str(_project_path(details_a_path)) if details_a_path else None,
            "details_a_sha256": _sha256(_project_path(details_a_path)) if details_a_path else None,
            "details_b": str(_project_path(details_b_path)) if details_b_path else None,
            "details_b_sha256": _sha256(_project_path(details_b_path)) if details_b_path else None,
            "config": str(config_file),
            "config_sha256": _sha256(config_file),
        },
        "config": config.to_dict(),
        "feature_names": list(feature_names()),
        "leakage_note": (
            "Every reported row was excluded from its selector fit and target "
            "language model. Selector-training LM features were nested-cross-fitted."
        ),
        "objective_note": (
            "Training labels and importance are the sign and absolute difference "
            "of each candidate's additive WER/CER normalized mean-error contribution."
        ),
    }
    _atomic_json(output / "report.json", report)
    return report


def fit_deployment(
    truth_path: str | Path,
    candidate_a_path: str | Path,
    candidate_b_path: str | Path,
    output_model: str | Path,
    config_path: str | Path,
    details_a_path: str | Path | None = None,
    details_b_path: str | Path | None = None,
) -> dict[str, Any]:
    truth = _project_path(truth_path)
    candidate_a_file = _project_path(candidate_a_path)
    candidate_b_file = _project_path(candidate_b_path)
    config_file = _project_path(config_path)
    _, rows = load_labeled_rows(truth)
    candidate_a = read_predictions(candidate_a_file)
    candidate_b = read_predictions(candidate_b_file)
    ids = [row.identifier for row in rows]
    _validate_candidate_ids(ids, "candidate A", candidate_a)
    _validate_candidate_ids(ids, "candidate B", candidate_b)
    details_a = read_details(details_a_path)
    details_b = read_details(details_b_path)
    _validate_detail_ids(ids, "candidate A details", details_a, details_a_path is not None)
    _validate_detail_ids(ids, "candidate B details", details_b, details_b_path is not None)
    config = load_config(config_file)
    matrix = _nested_training_features(
        rows,
        candidate_a,
        candidate_b,
        details_a,
        details_b,
        config,
        "deployment",
    )
    costs = row_costs(rows, candidate_a, candidate_b)
    selector = fit_logistic(
        matrix,
        [costs[row.identifier].choose_b for row in rows],
        [costs[row.identifier].importance for row in rows],
        config,
    )
    language_model = TextLanguageModel.fit(
        (row.target for row in rows), config.char_order, config.lm_alpha
    )
    output = _project_path(output_model)
    if any(output == protected or protected in output.parents for protected in PROTECTED_OUTPUTS):
        raise ValueError("Model33 may not write into protected Model22/Model25 outputs")
    state = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "config": config.to_dict(),
        "selector": selector.to_dict(),
        "language_model": language_model.to_dict(),
        "training_rows": len(rows),
        "candidate_a_role": "primary",
        "candidate_b_role": "challenger",
        "fit_note": (
            "Selector-training language-model features are target-cross-fitted; "
            "the serialized inference language model uses all supplied training labels."
        ),
        "inputs": {
            "truth_sha256": _sha256(truth),
            "candidate_a_sha256": _sha256(candidate_a_file),
            "candidate_b_sha256": _sha256(candidate_b_file),
            "config_sha256": _sha256(config_file),
            "details_a_sha256": _sha256(_project_path(details_a_path)) if details_a_path else None,
            "details_b_sha256": _sha256(_project_path(details_b_path)) if details_b_path else None,
        },
    }
    _atomic_json(output, state)
    return state


def apply_deployment(
    input_path: str | Path,
    candidate_a_path: str | Path,
    candidate_b_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    details_a_path: str | Path | None = None,
    details_b_path: str | Path | None = None,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> dict[str, Any]:
    input_file = _project_path(input_path)
    table = read_table(input_file)
    ids = [row[table.id_column].strip() for row in table.rows]
    candidate_a = read_predictions(candidate_a_path)
    candidate_b = read_predictions(candidate_b_path)
    _validate_candidate_ids(ids, "candidate A", candidate_a)
    _validate_candidate_ids(ids, "candidate B", candidate_b)
    details_a = read_details(details_a_path)
    details_b = read_details(details_b_path)
    _validate_detail_ids(ids, "candidate A details", details_a, details_a_path is not None)
    _validate_detail_ids(ids, "candidate B details", details_b, details_b_path is not None)
    state = json.loads(_project_path(model_path).read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION or state.get("model_kind") != MODEL_KIND:
        raise ValueError("incompatible Model33 deployment artifact")
    selector = LogisticModel.from_dict(state["selector"])
    if selector.feature_names != feature_names():
        raise ValueError("deployment feature schema differs from this implementation")
    language_model = TextLanguageModel.from_dict(state["language_model"])
    predictions: dict[str, str] = {}
    selected_b = 0
    for identifier in ids:
        features = extract_features(
            candidate_a[identifier],
            candidate_b[identifier],
            _details_for(details_a, identifier),
            _details_for(details_b, identifier),
            language_model,
        )
        choose_b = selector.probability_b(features) > 0.5
        selected_b += int(choose_b)
        predictions[identifier] = candidate_b[identifier] if choose_b else candidate_a[identifier]
    output = _project_path(output_path)
    if any(output == protected or protected in output.parents for protected in PROTECTED_OUTPUTS):
        raise ValueError("Model33 may not write into protected Model22/Model25 outputs")
    _atomic_csv(
        output,
        (table.id_column, "prediction"),
        ({table.id_column: identifier, "prediction": predictions[identifier]} for identifier in ids),
    )
    if submission_path is not None:
        if sample_path is None:
            raise ValueError("--sample is required with --submission")
        submission = _project_path(submission_path)
        if any(
            submission == protected or protected in submission.parents
            for protected in PROTECTED_OUTPUTS
        ):
            raise ValueError("Model33 may not write into protected Model22/Model25 outputs")
        write_submission(
            _project_path(sample_path),
            submission,
            predictions,
            cleaner=clean_prediction_preserving_internal_spaces,
        )
    return {"rows": len(ids), "selected_b_rows": selected_b, "output": str(output)}


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--candidate-a", type=Path, required=True)
    parser.add_argument("--candidate-b", type=Path, required=True)
    parser.add_argument("--details-a", type=Path)
    parser.add_argument("--details-b", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--single-fold", type=int)
    args = parser.parse_args()
    report = evaluate_crossfit(
        args.truth,
        args.candidate_a,
        args.candidate_b,
        args.output_dir,
        args.config,
        args.details_a,
        args.details_b,
        args.single_fold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def fit_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--candidate-a", type=Path, required=True)
    parser.add_argument("--candidate-b", type=Path, required=True)
    parser.add_argument("--details-a", type=Path)
    parser.add_argument("--details-b", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    args = parser.parse_args()
    state = fit_deployment(
        args.truth,
        args.candidate_a,
        args.candidate_b,
        args.output_model,
        args.config,
        args.details_a,
        args.details_b,
    )
    print(json.dumps({"training_rows": state["training_rows"]}, indent=2))


def apply_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidate-a", type=Path, required=True)
    parser.add_argument("--candidate-b", type=Path, required=True)
    parser.add_argument("--details-a", type=Path)
    parser.add_argument("--details-b", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    result = apply_deployment(
        args.input,
        args.candidate_a,
        args.candidate_b,
        args.model,
        args.output,
        args.details_a,
        args.details_b,
        args.sample,
        args.submission,
    )
    print(json.dumps(result, indent=2))
