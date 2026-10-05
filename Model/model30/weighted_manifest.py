"""Deterministic, fold-safe metric-aware Kraken training manifests.

Kraken's PP-OCRv6 CTC criterion uses ``reduction='mean'``. PyTorch therefore
divides each sample's CTC loss by its target length before averaging samples.
The challenge instead gives a sample an error mass proportional to the square
root of its reference word/character length. Repeating manifest entries is a
dependency-free way to change sample frequency without patching Kraken.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from road_htr.kraken_cli import absolute
from road_htr.metrics import CER_NORMALIZER, WER_NORMALIZER, strip_conversation_tokens
from road_htr.schema import read_table
from road_htr.vlm_data import read_id_file


SCHEMA_VERSION = 1
SUPPORTED_MODES = ("uniform", "metric_clipped")


@dataclass(frozen=True)
class TrainingExample:
    identifier: str
    image_path: Path
    target: str
    metric_char_length: int
    metric_word_length: int
    ctc_target_length: int


@dataclass(frozen=True)
class WeightingPolicy:
    mode: str
    clip_min: float
    clip_max: float
    repetition_budget: float
    seed: int


@dataclass(frozen=True)
class WeightingPlan:
    policy: WeightingPolicy
    examples: tuple[TrainingExample, ...]
    raw_importance: tuple[float, ...]
    relative_importance: tuple[float, ...]
    clipped_importance: tuple[float, ...]
    repetitions: tuple[int, ...]
    expanded_paths: tuple[Path, ...]
    training_only_sha256: str


def stable_sha256(path: str | Path) -> str:
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def canonical_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode == "identity":
        mode = "uniform"
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"weighting mode must be one of {SUPPORTED_MODES}, got {value!r}")
    return mode


def policy_from_config(config: dict[str, Any], mode_override: str | None = None) -> WeightingPolicy:
    section = config.get("weighting", {})
    if not isinstance(section, dict):
        raise ValueError("weighting configuration must be a mapping")
    mode = canonical_mode(mode_override or section.get("mode", "metric_clipped"))
    clip_min = float(section.get("clip_min", 0.5))
    clip_max = float(section.get("clip_max", 2.0))
    repetition_budget = float(section.get("repetition_budget", 2.0))
    seed = int(config.get("seed", 20260903))
    if not all(math.isfinite(value) for value in (clip_min, clip_max, repetition_budget)):
        raise ValueError("weighting clip and repetition values must be finite")
    if not 0 < clip_min <= 1 <= clip_max:
        raise ValueError("weighting clips must satisfy 0 < clip_min <= 1 <= clip_max")
    if repetition_budget <= 0:
        raise ValueError("weighting.repetition_budget must be positive")
    if mode == "uniform":
        # Identity control: one manifest occurrence per eligible row.
        repetition_budget = 1.0
    if mode == "metric_clipped" and repetition_budget * clip_min < 1:
        raise ValueError(
            "metric_clipped requires repetition_budget * clip_min >= 1 so every "
            "training row remains present"
        )
    return WeightingPolicy(mode, clip_min, clip_max, repetition_budget, seed)


def _training_rows(config: dict[str, Any], fold: int | None):
    table = read_table(absolute(config["train_csv"]), require_target=True)
    if fold is not None and "fold" not in table.fieldnames:
        raise ValueError("fold weighting requires a fold column")
    excluded = (
        read_id_file(absolute(config["exclude_ids"]))
        if config.get("exclude_ids")
        else set()
    )
    rows = []
    for row in table.rows:
        identifier = row[table.id_column].strip()
        if identifier in excluded:
            continue
        # Filter first. No target value from the held-out fold is read below.
        if fold is not None and int(row["fold"]) == fold:
            continue
        rows.append(row)
    return table, rows


def load_training_examples(
    config: dict[str, Any], fold: int | None, base_training_manifest: str | Path
) -> tuple[TrainingExample, ...]:
    """Load only eligible train-side targets in canonical manifest order."""

    table, rows = _training_rows(config, fold)
    assert table.target_column is not None
    manifest = Path(base_training_manifest)
    paths = tuple(
        Path(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(paths) != len(rows):
        raise ValueError(
            f"base manifest/eligible row mismatch: {len(paths)} != {len(rows)}"
        )
    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("training configuration must be a mapping")
    strip_edges = training.get("strip_target_edges", False)
    if not isinstance(strip_edges, bool):
        raise ValueError("training.strip_target_edges must be a boolean")
    normalization = str(training.get("normalization", "NFD"))
    if normalization not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError("unsupported Unicode normalization")

    examples: list[TrainingExample] = []
    for row, image_path in zip(rows, paths, strict=True):
        identifier = row[table.id_column].strip()
        if image_path.stem != Path(identifier).stem:
            raise ValueError(
                f"manifest order/ID mismatch: {image_path.stem!r} != {identifier!r}"
            )
        target = row[table.target_column]
        if strip_edges:
            target = target.strip()
        gt_path = image_path.with_suffix(".gt.txt")
        if not gt_path.is_file() or gt_path.read_text(encoding="utf-8") != target:
            raise ValueError(f"generated target does not match CSV for {identifier}")
        metric_reference = strip_conversation_tokens(target)
        metric_char_length = len(metric_reference)
        metric_word_length = len(metric_reference.split())
        ctc_target_length = len(unicodedata.normalize(normalization, target))
        if min(metric_char_length, metric_word_length, ctc_target_length) <= 0:
            raise ValueError(f"empty metric/training target for {identifier}")
        examples.append(
            TrainingExample(
                identifier=identifier,
                image_path=image_path,
                target=target,
                metric_char_length=metric_char_length,
                metric_word_length=metric_word_length,
                ctc_target_length=ctc_target_length,
            )
        )
    return tuple(examples)


def _bounded_mean_one(
    values: tuple[float, ...], lower: float, upper: float
) -> tuple[float, ...]:
    """Scale then clip positive values while retaining an arithmetic mean of one."""

    if not values:
        raise ValueError("at least one training example is required")
    low_scale = 0.0
    high_scale = 1.0

    def projected(scale: float) -> tuple[float, ...]:
        return tuple(min(upper, max(lower, scale * value)) for value in values)

    while math.fsum(projected(high_scale)) / len(values) < 1:
        high_scale *= 2
    for _ in range(100):
        middle = (low_scale + high_scale) / 2
        if math.fsum(projected(middle)) / len(values) < 1:
            low_scale = middle
        else:
            high_scale = middle
    return projected((low_scale + high_scale) / 2)


def _tie_key(seed: int, fold: int | None, identifier: str, occurrence: int = 0) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{fold}\0{identifier}\0{occurrence}".encode("utf-8")
    ).digest()


def _integer_repetitions(
    weights: tuple[float, ...],
    examples: tuple[TrainingExample, ...],
    policy: WeightingPolicy,
    fold: int | None,
) -> tuple[int, ...]:
    desired = tuple(policy.repetition_budget * value for value in weights)
    counts = [max(1, math.floor(value + 1e-12)) for value in desired]
    target_total = round(policy.repetition_budget * len(examples))
    difference = target_total - sum(counts)
    if difference > 0:
        order = sorted(
            range(len(examples)),
            key=lambda index: (
                -(desired[index] - math.floor(desired[index])),
                _tie_key(policy.seed, fold, examples[index].identifier),
            ),
        )
        for index in order[:difference]:
            counts[index] += 1
    elif difference < 0:
        order = sorted(
            (index for index, count in enumerate(counts) if count > 1),
            key=lambda index: (
                desired[index] - math.floor(desired[index]),
                _tie_key(policy.seed, fold, examples[index].identifier),
            ),
        )
        if -difference > len(order):
            raise RuntimeError("cannot apportion repetitions while retaining every row")
        for index in order[: -difference]:
            counts[index] -= 1
    if sum(counts) != target_total or min(counts) < 1:
        raise RuntimeError("deterministic repetition apportionment failed")
    return tuple(counts)


def build_weighting_plan(
    examples: tuple[TrainingExample, ...], policy: WeightingPolicy, fold: int | None
) -> WeightingPlan:
    if not examples:
        raise ValueError("no eligible training examples")

    char_root_sum = math.fsum(math.sqrt(item.metric_char_length) for item in examples)
    word_root_sum = math.fsum(math.sqrt(item.metric_word_length) for item in examples)
    raw = tuple(
        item.ctc_target_length
        * 0.5
        * (
            math.sqrt(item.metric_word_length) / (WER_NORMALIZER * word_root_sum)
            + math.sqrt(item.metric_char_length) / (CER_NORMALIZER * char_root_sum)
        )
        for item in examples
    )
    raw_mean = math.fsum(raw) / len(raw)
    relative = tuple(value / raw_mean for value in raw)

    if policy.mode == "uniform":
        clipped = tuple(1.0 for _ in examples)
        repetitions = tuple(1 for _ in examples)
        expanded = tuple(item.image_path for item in examples)
    else:
        clipped = _bounded_mean_one(relative, policy.clip_min, policy.clip_max)
        repetitions = _integer_repetitions(clipped, examples, policy, fold)
        occurrences = [
            (
                _tie_key(policy.seed, fold, item.identifier, occurrence),
                index,
                occurrence,
                item.image_path,
            )
            for index, (item, copies) in enumerate(zip(examples, repetitions, strict=True))
            for occurrence in range(copies)
        ]
        occurrences.sort(key=lambda value: (value[0], value[1], value[2]))
        expanded = tuple(value[3] for value in occurrences)

    training_only = {
        "schema_version": SCHEMA_VERSION,
        "mode": policy.mode,
        "fold": fold,
        "seed": policy.seed,
        "clip_min": policy.clip_min,
        "clip_max": policy.clip_max,
        "repetition_budget": policy.repetition_budget,
        "rows": [
            {
                "id": item.identifier,
                "metric_char_length": item.metric_char_length,
                "metric_word_length": item.metric_word_length,
                "ctc_target_length": item.ctc_target_length,
                "raw_importance": raw[index],
                "relative_importance": relative[index],
                "clipped_importance": clipped[index],
                "repetitions": repetitions[index],
            }
            for index, item in enumerate(examples)
        ],
        "expanded_ids": [
            path.stem for path in expanded
        ],
    }
    training_only_sha256 = hashlib.sha256(
        json.dumps(training_only, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return WeightingPlan(
        policy=policy,
        examples=examples,
        raw_importance=raw,
        relative_importance=relative,
        clipped_importance=clipped,
        repetitions=repetitions,
        expanded_paths=expanded,
        training_only_sha256=training_only_sha256,
    )


def plan_receipt(plan: WeightingPlan, fold: int | None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": plan.policy.mode,
        "fold": fold,
        "seed": plan.policy.seed,
        "derivation": {
            "challenge": (
                "0.5 * (weighted_WER / 12 + weighted_CER / 55), with each "
                "sample weighted by sqrt(reference token length)"
            ),
            "kraken": (
                "CTCLoss(blank=0, reduction='mean') divides each sample loss "
                "by CTC target length before averaging"
            ),
            "raw_importance": (
                "ctc_len * 0.5 * (sqrt(words)/(12*sum_train_sqrt_words) + "
                "sqrt(chars)/(55*sum_train_sqrt_chars))"
            ),
            "disclaimer": (
                "This is a train-side surrogate: CTC frames do not identify word "
                "edits, so OOF evidence is required before promotion."
            ),
        },
        "clip": [plan.policy.clip_min, plan.policy.clip_max],
        "repetition_budget": plan.policy.repetition_budget,
        "eligible_unique_rows": len(plan.examples),
        "weighted_manifest_rows": len(plan.expanded_paths),
        "training_only_sha256": plan.training_only_sha256,
        "statistics": {
            "minimum_repetitions": min(plan.repetitions),
            "maximum_repetitions": max(plan.repetitions),
            "mean_repetitions": math.fsum(plan.repetitions) / len(plan.repetitions),
            "minimum_clipped_importance": min(plan.clipped_importance),
            "maximum_clipped_importance": max(plan.clipped_importance),
            "mean_clipped_importance": math.fsum(plan.clipped_importance)
            / len(plan.clipped_importance),
        },
        "rows": [
            {
                "id": item.identifier,
                "image": str(item.image_path),
                "metric_char_length": item.metric_char_length,
                "metric_word_length": item.metric_word_length,
                "ctc_target_length": item.ctc_target_length,
                "raw_importance": plan.raw_importance[index],
                "relative_importance": plan.relative_importance[index],
                "clipped_importance": plan.clipped_importance[index],
                "repetitions": plan.repetitions[index],
            }
            for index, item in enumerate(plan.examples)
        ],
    }


def write_plan(
    plan: WeightingPlan,
    fold: int | None,
    manifest_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    atomic_text(
        manifest_path,
        "".join(f"{path}\n" for path in plan.expanded_paths),
    )
    receipt = plan_receipt(plan, fold)
    receipt.update(
        {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": stable_sha256(manifest_path),
        }
    )
    atomic_json(receipt_path, receipt)
    return receipt
