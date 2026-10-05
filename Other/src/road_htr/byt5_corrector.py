"""Leakage-safe ByT5 correction of genuine OCR out-of-fold predictions.

Model31 is intentionally stricter than a conventional train/validation script.
For each outer fold, an independent inner fold selects the epoch and a bounded
edit gate.  The selected epoch count is then refit on every non-outer row and
the outer labels are read only for the final score.  Neural dependencies are
imported lazily so the data, leakage, and provenance controls remain testable
without installing or loading the 300M-parameter base model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics import competition_error, levenshtein
from .schema import SchemaError, predictions_by_id, read_table
from .submission import (
    clean_prediction_preserving_internal_spaces,
    validate_submission,
    write_submission,
)


SCHEMA_VERSION = 1
MODEL_ID = "google/byt5-small"
# This immutable Hugging Face commit adds the safetensors conversion to the
# canonical 68377bd weights.  We deliberately avoid the pickle checkpoint.
MODEL_REVISION = "6f07f879d308b7b762708b50c83d41b27e329992"
MODEL_LICENSE = "Apache-2.0"
MODEL_WEIGHT_SHA256 = (
    "1dfcb862fbe12b97e63e1cdb8b3920a97d0ef4ed3a11e431829ebeacac3fc026"
)
MODEL_WEIGHT_BYTES = 1_198_571_496
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
EXPECTED_ATTENTION_TARGETS = 40
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_MODEL22_OUTPUT = (PROJECT_ROOT / "outputs" / "model22").resolve()


class Model31Error(RuntimeError):
    """Raised when a Model31 safety, lineage, or runtime gate fails."""


@dataclass(frozen=True)
class OcrPair:
    identifier: str
    source: str
    target: str | None
    fold: int | None


@dataclass(frozen=True)
class FoldPlan:
    outer_fold: int
    inner_fold: int
    selection_train: tuple[OcrPair, ...]
    selection_validation: tuple[OcrPair, ...]
    refit_train: tuple[OcrPair, ...]
    outer_validation: tuple[OcrPair, ...]


@dataclass(frozen=True)
class GatePolicy:
    enabled: bool
    maximum_edit_ratio: float | None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def reject_model22_destination(path: str | Path) -> None:
    if _inside(Path(path), PROTECTED_MODEL22_OUTPUT):
        raise Model31Error(f"Model31 may not write inside active Model22 outputs: {path}")


def require_new_path(path: str | Path) -> Path:
    path = Path(path)
    reject_model22_destination(path)
    if path.exists():
        if path.is_dir() and not any(path.iterdir()):
            return path
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    return path


def enable_offline_mode() -> None:
    """Force every Hugging Face loader into offline mode before imports."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - project dependency
        raise Model31Error("PyYAML is required to read Model31 config") from error
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with resolved.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Model31 config must be a mapping")
    validate_config(config)
    return config, resolved


def validate_config(config: Mapping[str, Any]) -> None:
    model = config.get("model", {})
    if model.get("id") != MODEL_ID:
        raise ValueError(f"model.id must be {MODEL_ID}")
    if model.get("revision") != MODEL_REVISION:
        raise ValueError(f"model.revision must be immutable {MODEL_REVISION}")
    if model.get("license") != MODEL_LICENSE:
        raise ValueError(f"model.license must be {MODEL_LICENSE}")
    if model.get("local_files_only") is not True:
        raise ValueError("model.local_files_only must be true")
    if model.get("trust_remote_code") is not False:
        raise ValueError("model.trust_remote_code must be false")
    expected = model.get("expected_files_sha256", {})
    if expected.get("model.safetensors") != MODEL_WEIGHT_SHA256:
        raise ValueError("the pinned safetensors SHA256 may not drift")
    if int(model.get("expected_weight_bytes", -1)) != MODEL_WEIGHT_BYTES:
        raise ValueError("the pinned safetensors byte count may not drift")

    folds = tuple(int(value) for value in config.get("cross_validation", {}).get("folds", ()))
    if folds != EXPECTED_FOLDS:
        raise ValueError(f"cross_validation.folds must be {list(EXPECTED_FOLDS)}")
    if config.get("cross_validation", {}).get("inner_policy") != "next_fold":
        raise ValueError("inner_policy must remain deterministic next_fold")

    sequence = config.get("sequence", {})
    if int(sequence.get("maximum_utf8_bytes", -1)) != 120:
        raise ValueError("maximum_utf8_bytes must remain 120 for this dataset")
    if int(sequence.get("max_source_tokens", -1)) != 122:
        raise ValueError("max_source_tokens must remain 122")
    if int(sequence.get("max_target_tokens", -1)) != 122:
        raise ValueError("max_target_tokens must remain 122")
    if sequence.get("training_overlength") != "error":
        raise ValueError("training_overlength must be error")
    if sequence.get("inference_overlength") != "identity":
        raise ValueError("inference_overlength must be identity")

    lora = config.get("lora", {})
    if int(lora.get("rank", -1)) != 8 or int(lora.get("alpha", -1)) != 16:
        raise ValueError("Model31 is locked to rank-8, alpha-16 LoRA")
    if tuple(lora.get("projections", ())) != ("q", "v"):
        raise ValueError("LoRA projections must be exactly q and v")
    if int(lora.get("expected_target_modules", -1)) != EXPECTED_ATTENTION_TARGETS:
        raise ValueError("expected_target_modules must be 40")
    if int(lora.get("max_trainable_parameters", -1)) > 1_000_000:
        raise ValueError("LoRA trainable-parameter cap may not exceed 1,000,000")

    training = config.get("training", {})
    if int(training.get("epochs", 0)) < 1:
        raise ValueError("training.epochs must be positive")
    if int(training.get("batch_size", 0)) < 1:
        raise ValueError("training.batch_size must be positive")
    if int(training.get("gradient_accumulation", 0)) < 1:
        raise ValueError("gradient_accumulation must be positive")
    if training.get("gradient_checkpointing") is not True:
        raise ValueError("gradient checkpointing is required")
    if training.get("precision") != "float16":
        raise ValueError("the measured smoke contract is float16")

    generation = config.get("generation", {})
    if generation.get("do_sample") is not False or int(generation.get("num_beams", 0)) != 1:
        raise ValueError("validation and inference must use deterministic greedy decoding")
    if int(generation.get("max_new_tokens", -1)) != 122:
        raise ValueError("generation.max_new_tokens must remain 122")

    thresholds = tuple(float(value) for value in config.get("gate", {}).get("edit_ratio_thresholds", ()))
    if not thresholds or thresholds != tuple(sorted(set(thresholds))):
        raise ValueError("gate thresholds must be nonempty, unique, and increasing")
    if thresholds[0] < 0 or thresholds[-1] > 1:
        raise ValueError("gate thresholds must lie in [0, 1]")
    if float(config.get("gate", {}).get("minimum_inner_score_gain", -1)) < 0:
        raise ValueError("minimum_inner_score_gain must be nonnegative")

    synthetic = config.get("synthetic", {})
    if not isinstance(synthetic.get("enabled"), bool):
        raise ValueError("synthetic.enabled must be boolean")
    if int(synthetic.get("copies_per_row", 0)) not in (1, 2):
        raise ValueError("synthetic.copies_per_row must be 1 or 2")
    probability = float(synthetic.get("maximum_character_probability", -1))
    if not 0 < probability <= 0.2:
        raise ValueError("synthetic maximum probability must lie in (0, .2]")


def _prediction_column(table: Any, path: Path) -> str:
    if table.target_column is not None:
        return table.target_column
    columns = [column for column in table.fieldnames if column != table.id_column]
    if len(columns) != 1:
        raise SchemaError(f"source CSV must have exactly one prediction column: {path}")
    return columns[0]


def _trim_edges(value: object) -> str:
    return str(value if value is not None else "").strip(" \t\r\n")


def load_oof_pairs(
    truth_path: str | Path,
    source_path: str | Path,
    *,
    expected_folds: Sequence[int] = EXPECTED_FOLDS,
) -> tuple[OcrPair, ...]:
    """Load complete genuine OOF predictions in canonical truth order."""

    truth_path = Path(truth_path)
    source_path = Path(source_path)
    truth = read_table(truth_path, require_target=True)
    source = read_table(source_path)
    if "fold" not in truth.fieldnames:
        raise SchemaError("truth CSV must contain a fold column")
    if not truth.rows:
        raise SchemaError("truth CSV is empty")
    assert truth.target_column is not None
    prediction_column = _prediction_column(source, source_path)
    source_by_id = predictions_by_id(source, prediction_column)
    truth_ids = [row[truth.id_column].strip() for row in truth.rows]
    missing = set(truth_ids) - set(source_by_id)
    extra = set(source_by_id) - set(truth_ids)
    if missing or extra:
        raise SchemaError(
            f"OOF source ID mismatch: {len(missing)} missing, {len(extra)} extra; "
            f"examples missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )

    records: list[OcrPair] = []
    for row in truth.rows:
        identifier = row[truth.id_column].strip()
        try:
            fold = int(row["fold"])
        except (TypeError, ValueError) as error:
            raise SchemaError(f"non-integer fold for {identifier!r}") from error
        source_text = _trim_edges(source_by_id[identifier])
        target = _trim_edges(row[truth.target_column])
        if not source_text:
            raise SchemaError(f"blank source OOF prediction for {identifier!r}")
        if not target:
            raise SchemaError(f"blank target for {identifier!r}")
        records.append(OcrPair(identifier, source_text, target, fold))
    observed = tuple(sorted({record.fold for record in records if record.fold is not None}))
    required = tuple(expected_folds)
    if observed != required:
        raise SchemaError(f"OOF folds must be exactly {required}, found {observed}")
    return tuple(records)


def load_source_pairs(path: str | Path) -> tuple[OcrPair, ...]:
    """Load an unlabeled OCR prediction CSV without altering internal spaces."""

    path = Path(path)
    table = read_table(path)
    column = _prediction_column(table, path)
    records: list[OcrPair] = []
    for row in table.rows:
        identifier = row[table.id_column].strip()
        source = _trim_edges(row.get(column, ""))
        if not source:
            raise SchemaError(f"blank source prediction for {identifier!r}")
        records.append(OcrPair(identifier, source, None, None))
    if not records:
        raise SchemaError("source prediction CSV is empty")
    return tuple(records)


def validate_byte_limits(
    records: Sequence[OcrPair],
    maximum_utf8_bytes: int,
    *,
    require_targets: bool,
) -> dict[str, int]:
    maximum_source = 0
    maximum_target = 0
    for record in records:
        source_bytes = len(record.source.encode("utf-8"))
        maximum_source = max(maximum_source, source_bytes)
        if source_bytes > maximum_utf8_bytes:
            raise Model31Error(
                f"source for {record.identifier!r} is {source_bytes} UTF-8 bytes; "
                f"training limit is {maximum_utf8_bytes} and truncation is forbidden"
            )
        if require_targets:
            if record.target is None:
                raise Model31Error(f"missing target for {record.identifier!r}")
            target_bytes = len(record.target.encode("utf-8"))
            maximum_target = max(maximum_target, target_bytes)
            if target_bytes > maximum_utf8_bytes:
                raise Model31Error(
                    f"target for {record.identifier!r} is {target_bytes} UTF-8 bytes; "
                    f"limit is {maximum_utf8_bytes} and truncation is forbidden"
                )
    return {
        "rows": len(records),
        "maximum_source_utf8_bytes": maximum_source,
        "maximum_target_utf8_bytes": maximum_target,
    }


def build_fold_plan(
    records: Sequence[OcrPair], outer_fold: int, folds: Sequence[int] = EXPECTED_FOLDS
) -> FoldPlan:
    folds = tuple(folds)
    if outer_fold not in folds:
        raise ValueError(f"outer fold {outer_fold} is not in {folds}")
    inner_fold = folds[(folds.index(outer_fold) + 1) % len(folds)]
    selection_train = tuple(
        record for record in records if record.fold not in {outer_fold, inner_fold}
    )
    selection_validation = tuple(record for record in records if record.fold == inner_fold)
    refit_train = tuple(record for record in records if record.fold != outer_fold)
    outer_validation = tuple(record for record in records if record.fold == outer_fold)
    groups = {
        "selection_train": selection_train,
        "selection_validation": selection_validation,
        "outer_validation": outer_validation,
    }
    if any(not group for group in groups.values()):
        raise Model31Error(f"empty fold-plan group: {[name for name, group in groups.items() if not group]}")
    id_sets = {name: {row.identifier for row in group} for name, group in groups.items()}
    for left, right in (
        ("selection_train", "selection_validation"),
        ("selection_train", "outer_validation"),
        ("selection_validation", "outer_validation"),
    ):
        overlap = id_sets[left].intersection(id_sets[right])
        if overlap:
            raise Model31Error(f"fold-plan leakage between {left} and {right}: {sorted(overlap)[:5]}")
    if {row.identifier for row in refit_train}.intersection(id_sets["outer_validation"]):
        raise Model31Error("outer validation IDs leaked into refit training")
    return FoldPlan(
        outer_fold,
        inner_fold,
        selection_train,
        selection_validation,
        refit_train,
        outer_validation,
    )


def fold_plan_receipt(plan: FoldPlan) -> dict[str, Any]:
    def describe(rows: Sequence[OcrPair]) -> dict[str, Any]:
        ids = [row.identifier for row in rows]
        return {
            "rows": len(rows),
            "folds": sorted({row.fold for row in rows if row.fold is not None}),
            "ids_sha256": canonical_sha256(ids),
        }

    return {
        "outer_fold": plan.outer_fold,
        "inner_fold": plan.inner_fold,
        "selection_train": describe(plan.selection_train),
        "selection_validation": describe(plan.selection_validation),
        "refit_train": describe(plan.refit_train),
        "outer_validation": describe(plan.outer_validation),
    }


def _character_edits(reference: str, observed: str) -> list[tuple[str, str]]:
    """Return deterministic single-character reference -> OCR edit pairs."""

    rows, columns = len(reference), len(observed)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        costs[row][0] = row
    for column in range(columns + 1):
        costs[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            costs[row][column] = min(
                costs[row - 1][column - 1]
                + (reference[row - 1] != observed[column - 1]),
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
            )
    edits: list[tuple[str, str]] = []
    row, column = rows, columns
    while row or column:
        if (
            row
            and column
            and reference[row - 1] == observed[column - 1]
            and costs[row][column] == costs[row - 1][column - 1]
        ):
            row -= 1
            column -= 1
        elif row and column and costs[row][column] == costs[row - 1][column - 1] + 1:
            edits.append((reference[row - 1], observed[column - 1]))
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            edits.append((reference[row - 1], ""))
            row -= 1
        elif column and costs[row][column] == costs[row][column - 1] + 1:
            edits.append(("", observed[column - 1]))
            column -= 1
        else:  # pragma: no cover - defensive DP invariant
            raise AssertionError("character alignment backtrace failed")
    edits.reverse()
    return edits


def build_synthetic_corruptions(
    training_records: Sequence[OcrPair],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[tuple[OcrPair, ...], dict[str, Any]]:
    """Learn and apply OCR-like edits from this stage's training rows only."""

    synthetic = config["synthetic"]
    if not synthetic.get("enabled", False):
        return tuple(training_records), {
            "enabled": False,
            "genuine_rows": len(training_records),
            "synthetic_rows": 0,
        }
    copies = int(synthetic["copies_per_row"])
    if copies < 1 or copies > 2:
        raise Model31Error("synthetic copies_per_row must be 1 or 2")
    maximum_probability = float(synthetic["maximum_character_probability"])
    if not 0 < maximum_probability <= 0.2:
        raise Model31Error("synthetic maximum_character_probability must lie in (0, .2]")

    from collections import Counter, defaultdict

    replacements: dict[str, Counter[str]] = defaultdict(Counter)
    insertions: Counter[str] = Counter()
    edit_count = 0
    target_characters = 0
    for record in training_records:
        if record.target is None:
            raise Model31Error("synthetic corruption fitting received an unlabeled row")
        target_characters += len(record.target)
        for clean, noisy in _character_edits(record.target, record.source):
            edit_count += 1
            if clean:
                replacements[clean][noisy] += 1
            elif noisy:
                insertions[noisy] += 1
    if not replacements and not insertions:
        return tuple(training_records), {
            "enabled": True,
            "genuine_rows": len(training_records),
            "synthetic_rows": 0,
            "reason": "no training-side OCR edits",
        }
    empirical_rate = edit_count / max(target_characters, 1)
    probability = min(empirical_rate, maximum_probability)
    insertion_probability = min(sum(insertions.values()) / max(target_characters, 1), probability / 2)
    rng = random.Random(seed)
    generated: list[OcrPair] = []
    maximum_bytes = int(config["sequence"]["maximum_utf8_bytes"])
    for record in training_records:
        assert record.target is not None
        for copy_index in range(copies):
            parts: list[str] = []
            changed = False
            for character in record.target:
                if insertions and rng.random() < insertion_probability:
                    population, weights = zip(*sorted(insertions.items()), strict=True)
                    parts.append(rng.choices(population, weights=weights, k=1)[0])
                    changed = True
                choices = replacements.get(character)
                if choices and rng.random() < probability:
                    population, weights = zip(*sorted(choices.items()), strict=True)
                    replacement = rng.choices(population, weights=weights, k=1)[0]
                    parts.append(replacement)
                    changed = changed or replacement != character
                else:
                    parts.append(character)
            corrupted = "".join(parts)
            if not changed:
                eligible = [index for index, char in enumerate(record.target) if replacements.get(char)]
                if eligible:
                    index = rng.choice(eligible)
                    choices = replacements[record.target[index]]
                    population, weights = zip(*sorted(choices.items()), strict=True)
                    replacement = rng.choices(population, weights=weights, k=1)[0]
                    corrupted = record.target[:index] + replacement + record.target[index + 1 :]
                    changed = corrupted != record.target
            if (
                changed
                and corrupted
                and len(corrupted.encode("utf-8")) <= maximum_bytes
            ):
                generated.append(
                    OcrPair(
                        f"{record.identifier}::synthetic:{copy_index}",
                        corrupted,
                        record.target,
                        record.fold,
                    )
                )
    augmented = tuple(training_records) + tuple(generated)
    learned = {
        clean: dict(sorted(values.items()))
        for clean, values in sorted(replacements.items())
    }
    return augmented, {
        "enabled": True,
        "genuine_rows": len(training_records),
        "synthetic_rows": len(generated),
        "copies_per_row": copies,
        "empirical_edit_rate": empirical_rate,
        "applied_character_probability": probability,
        "applied_insertion_probability": insertion_probability,
        "learned_replacements_sha256": canonical_sha256(learned),
        "learned_insertions_sha256": canonical_sha256(dict(sorted(insertions.items()))),
        "augmented_pairs_sha256": canonical_sha256(
            [(row.identifier, row.source, row.target) for row in augmented]
        ),
    }


def normalized_edit_ratio(source: str, candidate: str) -> float:
    return levenshtein(tuple(source), tuple(candidate)) / max(
        len(source), len(candidate), 1
    )


def apply_gate(
    sources: Sequence[str], generated: Sequence[str], policy: GatePolicy
) -> tuple[list[str], dict[str, int]]:
    if len(sources) != len(generated):
        raise ValueError("source/generated count mismatch")
    selected: list[str] = []
    accepted = 0
    empty_fallbacks = 0
    ratio_fallbacks = 0
    overlength_fallbacks = 0
    for source, raw_candidate in zip(sources, generated, strict=True):
        candidate = _trim_edges(raw_candidate)
        if not policy.enabled:
            selected.append(source)
            continue
        if not candidate:
            selected.append(source)
            empty_fallbacks += 1
            continue
        if len(candidate.encode("utf-8")) > 120:
            selected.append(source)
            overlength_fallbacks += 1
            continue
        assert policy.maximum_edit_ratio is not None
        if normalized_edit_ratio(source, candidate) <= policy.maximum_edit_ratio:
            selected.append(candidate)
            accepted += candidate != source
        else:
            selected.append(source)
            ratio_fallbacks += 1
    return selected, {
        "rows": len(sources),
        "changed_rows": accepted,
        "empty_fallbacks": empty_fallbacks,
        "overlength_fallbacks": overlength_fallbacks,
        "ratio_fallbacks": ratio_fallbacks,
    }


def fit_gate(
    sources: Sequence[str],
    references: Sequence[str],
    generated: Sequence[str],
    thresholds: Sequence[float],
    *,
    minimum_score_gain: float = 0.0,
) -> tuple[GatePolicy, dict[str, Any]]:
    """Fit a source/candidate-only gate on inner labels, never outer labels."""

    if not sources or len(sources) != len(references) or len(sources) != len(generated):
        raise ValueError("gate fitting needs equally sized nonempty sequences")
    policies = [GatePolicy(False, None)] + [
        GatePolicy(True, float(threshold)) for threshold in thresholds
    ]
    trials: list[dict[str, Any]] = []
    for order, policy in enumerate(policies):
        values, audit = apply_gate(sources, generated, policy)
        score = competition_error(references, values)
        trials.append(
            {
                "order": order,
                "policy": asdict(policy),
                "audit": audit,
                "score": score,
            }
        )
    identity_score = trials[0]["score"]
    best = min(
        trials,
        key=lambda item: (
            float(item["score"]["mean_error"]),
            int(item["audit"]["changed_rows"]),
            int(item["order"]),
        ),
    )
    gain = float(best["score"]["multi_score"]) - float(identity_score["multi_score"])
    if best["order"] != 0 and gain + 1e-15 < minimum_score_gain:
        best = trials[0]
        gain = 0.0
    policy = GatePolicy(**best["policy"])
    return policy, {
        "selected_policy": asdict(policy),
        "identity_score": identity_score,
        "selected_score": best["score"],
        "selected_gain": gain,
        "minimum_score_gain": minimum_score_gain,
        "trials": trials,
    }


def verify_local_snapshot(config: Mapping[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    model_config = config["model"]
    local_path = _resolve(root, model_config["local_path"])
    if not local_path.is_dir():
        raise FileNotFoundError(
            f"pinned ByT5 snapshot is absent: {local_path}; no network fallback is allowed"
        )
    required = tuple(model_config["required_files"])
    missing = [name for name in required if not (local_path / name).is_file()]
    if missing:
        raise Model31Error(f"local ByT5 snapshot is incomplete: missing {missing}")
    weight = local_path / "model.safetensors"
    if weight.stat().st_size != MODEL_WEIGHT_BYTES:
        raise Model31Error(
            f"model.safetensors byte count mismatch: {weight.stat().st_size} != {MODEL_WEIGHT_BYTES}"
        )
    hashes = {name: sha256_file(local_path / name) for name in required}
    for name, expected in model_config["expected_files_sha256"].items():
        actual = hashes.get(name)
        if actual != expected:
            raise Model31Error(f"base-model hash mismatch for {name}: {actual} != {expected}")

    raw_config = _load_json(local_path / "config.json")
    expected_architecture = model_config["expected_architecture"]
    observed = {key: raw_config.get(key) for key in expected_architecture}
    if observed != expected_architecture:
        raise Model31Error(
            f"ByT5 architecture drift: observed={observed}, expected={expected_architecture}"
        )
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "local_path": str(local_path),
        "files_sha256": hashes,
        "files_bytes": {name: (local_path / name).stat().st_size for name in required},
        "snapshot_sha256": canonical_sha256(hashes),
        "architecture": observed,
    }


def validate_runtime_versions(config: Mapping[str, Any]) -> dict[str, str]:
    expected = config["runtime"]["expected_versions"]
    found: dict[str, str] = {}
    missing: list[str] = []
    for package, wanted in expected.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"{package}=={wanted}")
            continue
        found[package] = actual
        if actual != str(wanted):
            raise Model31Error(
                f"runtime version mismatch for {package}: {actual} != pinned {wanted}"
            )
    if missing:
        raise Model31Error(
            "missing pinned Model31 dependencies: " + ", ".join(missing)
        )
    return found


_TARGET_RE = re.compile(
    r"^(encoder|decoder)\.block\.(\d+)\.layer\.(\d+)\."
    r"(SelfAttention|EncDecAttention)\.(q|v)$"
)


def discover_lora_targets(model: Any) -> tuple[str, ...]:
    observed = {name for name, _module in model.named_modules() if _TARGET_RE.fullmatch(name)}
    expected: set[str] = set()
    for block in range(12):
        for projection in ("q", "v"):
            expected.add(f"encoder.block.{block}.layer.0.SelfAttention.{projection}")
    for block in range(4):
        for attention_layer, attention_name in ((0, "SelfAttention"), (1, "EncDecAttention")):
            for projection in ("q", "v"):
                expected.add(
                    f"decoder.block.{block}.layer.{attention_layer}.{attention_name}.{projection}"
                )
    if observed != expected:
        raise Model31Error(
            "ByT5 attention architecture drift: "
            f"missing={sorted(expected-observed)[:5]}, extra={sorted(observed-expected)[:5]}"
        )
    return tuple(sorted(observed))


def audit_trainable_parameters(
    model: Any, target_modules: Sequence[str], maximum: int
) -> dict[str, int]:
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    unexpected = [name for name, _value in trainable if "lora_" not in name]
    if unexpected:
        raise Model31Error(f"non-LoRA parameters are trainable: {unexpected[:5]}")
    count = sum(int(value.numel()) for _name, value in trainable)
    if count <= 0 or count > maximum:
        raise Model31Error(f"trainable parameter count {count} is outside (0, {maximum}]")
    return {
        "target_modules": len(target_modules),
        "trainable_tensors": len(trainable),
        "trainable_parameters": count,
    }


def adapter_files(adapter_dir: str | Path) -> dict[str, str]:
    adapter_dir = Path(adapter_dir)
    return {
        str(path.relative_to(adapter_dir)): sha256_file(path)
        for path in sorted(adapter_dir.rglob("*"))
        if path.is_file()
    }


def write_adapter_receipt(
    adapter_dir: str | Path,
    *,
    run_lineage_sha256: str,
    stage: str,
    epoch: int,
    base_snapshot_sha256: str,
) -> Path:
    adapter_dir = Path(adapter_dir)
    hashes = adapter_files(adapter_dir)
    if not hashes or not any(name.endswith(".safetensors") for name in hashes):
        raise Model31Error(f"adapter is incomplete or unsafe: {adapter_dir}")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_lineage_sha256": run_lineage_sha256,
        "base_snapshot_sha256": base_snapshot_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "stage": stage,
        "epoch": epoch,
        "adapter_files_sha256": hashes,
    }
    return _write_json(adapter_dir.with_name(f"{adapter_dir.name}.receipt.json"), receipt)


def verify_adapter_receipt(
    adapter_dir: str | Path,
    receipt_path: str | Path | None = None,
    *,
    expected_base_snapshot_sha256: str | None = None,
    expected_run_lineage_sha256: str | None = None,
) -> dict[str, Any]:
    adapter_dir = Path(adapter_dir).resolve()
    receipt_path = (
        Path(receipt_path).resolve()
        if receipt_path is not None
        else adapter_dir.with_name(f"{adapter_dir.name}.receipt.json")
    )
    if not adapter_dir.is_dir() or not receipt_path.is_file():
        raise FileNotFoundError(f"adapter or receipt missing: {adapter_dir}, {receipt_path}")
    receipt = _load_json(receipt_path)
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise Model31Error("adapter receipt schema mismatch")
    if receipt.get("model_id") != MODEL_ID or receipt.get("model_revision") != MODEL_REVISION:
        raise Model31Error("adapter receipt base-model lineage mismatch")
    if expected_base_snapshot_sha256 is not None and receipt.get("base_snapshot_sha256") != expected_base_snapshot_sha256:
        raise Model31Error("adapter receipt snapshot hash mismatch")
    if expected_run_lineage_sha256 is not None and receipt.get("run_lineage_sha256") != expected_run_lineage_sha256:
        raise Model31Error("adapter receipt run lineage mismatch")
    actual = adapter_files(adapter_dir)
    if actual != receipt.get("adapter_files_sha256"):
        raise Model31Error("adapter bytes differ from the pinned receipt")
    return receipt


def _torch_device(config: Mapping[str, Any], torch: Any) -> tuple[Any, int]:
    raw = str(config["runtime"]["device"])
    match = re.fullmatch(r"cuda:(\d+)", raw)
    if match is None:
        raise Model31Error("Model31 requires an explicit cuda:N device")
    index = int(match.group(1))
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise Model31Error(f"requested CUDA device is unavailable: {raw}")
    # PyTorch 2.13 can report a visible device before its per-device allocator
    # state has been initialized.  Explicit selection makes the subsequent
    # peak-memory reset valid and also prevents an ambient current device from
    # silently disagreeing with the audited cuda:N configuration.
    torch.cuda.set_device(index)
    return torch.device(raw), index


def _load_neural_runtime(
    config: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    adapter_dir: Path | None = None,
    trainable: bool,
) -> tuple[Any, Any, Any, tuple[str, ...], dict[str, int]]:
    """Load only verified local bytes and inject/load the tightly scoped LoRA."""

    enable_offline_mode()
    validate_runtime_versions(config)
    try:
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoTokenizer, T5ForConditionalGeneration
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise Model31Error(
            "Model31 neural runtime is unavailable; install model31/requirements.txt"
        ) from error

    device, _index = _torch_device(config, torch)
    local_path = Path(snapshot["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.__class__.__name__ != "ByT5Tokenizer":
        raise Model31Error(
            f"unexpected tokenizer class {tokenizer.__class__.__name__}; expected ByT5Tokenizer"
        )
    model = T5ForConditionalGeneration.from_pretrained(
        str(local_path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    target_modules = discover_lora_targets(model)
    if adapter_dir is None:
        lora = config["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora["rank"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
                target_modules=list(target_modules),
            ),
        )
    else:
        model = PeftModel.from_pretrained(
            model, str(adapter_dir), is_trainable=trainable, local_files_only=True
        )
    model.to(device)
    if trainable:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        audit = audit_trainable_parameters(
            model,
            target_modules,
            int(config["lora"]["max_trainable_parameters"]),
        )
    else:
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise Model31Error("inference adapter unexpectedly has trainable parameters")
        audit = {
            "target_modules": len(target_modules),
            "trainable_tensors": 0,
            "trainable_parameters": 0,
        }
    return torch, tokenizer, model, target_modules, audit


def _audit_token_lengths(
    tokenizer: Any,
    values: Sequence[str],
    maximum_tokens: int,
    *,
    label: str,
) -> dict[str, int]:
    maximum = 0
    for value in values:
        encoded = tokenizer(value, add_special_tokens=True, truncation=False)
        length = len(encoded["input_ids"])
        maximum = max(maximum, length)
        if length > maximum_tokens:
            raise Model31Error(
                f"{label} token length {length} exceeds {maximum_tokens}; truncation is forbidden"
            )
    return {"rows": len(values), "maximum_tokens": maximum}


def _collate_training(
    tokenizer: Any,
    records: Sequence[OcrPair],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    sources = [record.source for record in records]
    targets = [record.target for record in records]
    if any(target is None for target in targets):
        raise Model31Error("training batch contains an unlabeled record")
    inputs = tokenizer(
        sources,
        padding=True,
        add_special_tokens=True,
        truncation=False,
        return_tensors="pt",
    )
    encoded_targets = tokenizer(
        text_target=targets,
        padding=True,
        add_special_tokens=True,
        truncation=False,
        return_tensors="pt",
    )
    if int(inputs["input_ids"].shape[1]) > int(config["sequence"]["max_source_tokens"]):
        raise Model31Error("source token cap exceeded in collator; no truncation was performed")
    if int(encoded_targets["input_ids"].shape[1]) > int(config["sequence"]["max_target_tokens"]):
        raise Model31Error("target token cap exceeded in collator; no truncation was performed")
    labels = encoded_targets["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return {key: value for key, value in inputs.items() if isinstance(value, torch.Tensor)}


def _record_batches(
    records: Sequence[OcrPair], batch_size: int, *, seed: int | None
) -> Iterable[list[OcrPair]]:
    order = list(range(len(records)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [records[index] for index in order[start : start + batch_size]]


def _generation_batches(
    tokenizer: Any,
    model: Any,
    records: Sequence[OcrPair],
    config: Mapping[str, Any],
    device: Any,
) -> list[str]:
    import torch

    model.eval()
    batch_size = int(config["training"]["eval_batch_size"])
    predictions: list[str] = []
    with torch.inference_mode():
        for batch in _record_batches(records, batch_size, seed=None):
            encoded = tokenizer(
                [record.source for record in batch],
                padding=True,
                add_special_tokens=True,
                truncation=False,
                return_tensors="pt",
            )
            if int(encoded["input_ids"].shape[1]) > int(config["sequence"]["max_source_tokens"]):
                raise Model31Error("generation source exceeded token cap")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                generated = model.generate(
                    **encoded,
                    max_new_tokens=int(config["generation"]["max_new_tokens"]),
                    do_sample=False,
                    num_beams=1,
                )
            decoded = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            predictions.extend(_trim_edges(value) for value in decoded)
    if len(predictions) != len(records):
        raise Model31Error("generation returned the wrong number of rows")
    return predictions


def generate_with_overlength_identity(
    tokenizer: Any,
    model: Any,
    records: Sequence[OcrPair],
    config: Mapping[str, Any],
    device: Any,
) -> tuple[list[str], dict[str, int]]:
    """Generate eligible rows while preserving overlength sources verbatim."""

    limit = int(config["sequence"]["maximum_utf8_bytes"])
    eligible_indices = [
        index
        for index, record in enumerate(records)
        if len(record.source.encode("utf-8")) <= limit
    ]
    eligible = [records[index] for index in eligible_indices]
    eligible_predictions = (
        _generation_batches(tokenizer, model, eligible, config, device) if eligible else []
    )
    output = [record.source for record in records]
    for index, prediction in zip(eligible_indices, eligible_predictions, strict=True):
        output[index] = prediction
    return output, {
        "rows": len(records),
        "generated_rows": len(eligible),
        "overlength_identity_rows": len(records) - len(eligible),
    }


def _cosine_scheduler(optimizer: Any, total_steps: int, warmup_steps: int, minimum_ratio: float):
    from torch.optim.lr_scheduler import LambdaLR

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        span = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / span, 0.0), 1.0)
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, multiplier)


def _seed_everything(seed: int, torch: Any) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _write_prediction_csv(path: Path, records: Sequence[OcrPair], values: Sequence[str]) -> Path:
    if len(records) != len(values):
        raise ValueError("prediction row count mismatch")
    require_new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
        writer.writeheader()
        for record, value in zip(records, values, strict=True):
            cleaned = clean_prediction_preserving_internal_spaces(value)
            if not cleaned:
                raise Model31Error(f"empty prediction for {record.identifier!r}")
            writer.writerow({"ID": record.identifier, "prediction": cleaned})
    return path


def _save_training_state(
    path: Path,
    torch: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    optimizer_steps: int,
) -> str:
    require_new_path(path)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )
    return sha256_file(path)


def _load_training_state(path: Path, expected_sha256: str, torch: Any) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise Model31Error(f"training-state bytes changed: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:  # pragma: no cover - pinned torch supports this
        raise Model31Error("pinned torch with weights_only loading is required") from error
    if not isinstance(state, dict):
        raise Model31Error("invalid training state")
    return state


def _run_stage(
    *,
    stage_dir: Path,
    stage_name: str,
    training_records: Sequence[OcrPair],
    validation_records: Sequence[OcrPair] | None,
    epochs: int,
    seed: int,
    config: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    run_lineage_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    """Train/resume only at hash-pinned epoch boundaries."""

    genuine_training_rows = len(training_records)
    training_records, synthetic_report = build_synthetic_corruptions(
        training_records, config, seed=seed + 100_000
    )
    validate_byte_limits(
        training_records,
        int(config["sequence"]["maximum_utf8_bytes"]),
        require_targets=True,
    )
    progress_path = stage_dir / "progress.json"
    progress: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage_name,
        "stage_lineage_sha256": canonical_sha256(
            {
                "run": run_lineage_sha256,
                "stage": stage_name,
                "training_pairs_sha256": canonical_sha256(
                    [(row.identifier, row.source, row.target) for row in training_records]
                ),
                "validation_ids": [row.identifier for row in validation_records or ()],
                "epochs": epochs,
                "seed": seed,
            }
        ),
        "epochs_requested": epochs,
        "genuine_training_rows": genuine_training_rows,
        "effective_training_rows": len(training_records),
        "synthetic": synthetic_report,
        "epochs": [],
    }
    if progress_path.exists():
        if not resume:
            raise FileExistsError(f"stage already exists; pass --resume: {stage_dir}")
        saved = _load_json(progress_path)
        if saved.get("stage_lineage_sha256") != progress["stage_lineage_sha256"]:
            raise Model31Error(f"resume lineage mismatch for {stage_name}")
        progress = saved
    elif stage_dir.exists() and any(stage_dir.iterdir()):
        raise Model31Error(f"unreceipted partial stage refuses resume: {stage_dir}")

    completed = len(progress["epochs"])
    if completed > epochs:
        raise Model31Error("stage progress exceeds requested epochs")
    if completed == epochs:
        latest = progress["epochs"][-1]
        verify_adapter_receipt(
            latest["adapter"],
            latest["adapter_receipt"],
            expected_base_snapshot_sha256=snapshot["snapshot_sha256"],
            expected_run_lineage_sha256=run_lineage_sha256,
        )
        return progress

    adapter_dir: Path | None = None
    if completed:
        latest = progress["epochs"][-1]
        adapter_dir = Path(latest["adapter"])
        verify_adapter_receipt(
            adapter_dir,
            latest["adapter_receipt"],
            expected_base_snapshot_sha256=snapshot["snapshot_sha256"],
            expected_run_lineage_sha256=run_lineage_sha256,
        )

    torch, tokenizer, model, _targets, _audit = _load_neural_runtime(
        config, snapshot, adapter_dir=adapter_dir, trainable=True
    )
    device, _index = _torch_device(config, torch)
    _audit_token_lengths(
        tokenizer,
        [row.source for row in training_records],
        int(config["sequence"]["max_source_tokens"]),
        label=f"{stage_name} source",
    )
    _audit_token_lengths(
        tokenizer,
        [row.target or "" for row in training_records],
        int(config["sequence"]["max_target_tokens"]),
        label=f"{stage_name} target",
    )

    training = config["training"]
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation"])
    batches_per_epoch = math.ceil(len(training_records) / batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(round(total_steps * float(training["warmup_ratio"])))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = _cosine_scheduler(
        optimizer,
        total_steps,
        warmup_steps,
        float(training["minimum_learning_rate"]) / float(training["learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    optimizer_steps = 0
    if completed:
        latest = progress["epochs"][-1]
        state = _load_training_state(
            Path(latest["training_state"]), latest["training_state_sha256"], torch
        )
        if int(state.get("epoch", -1)) != completed:
            raise Model31Error("training-state epoch does not match progress")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        optimizer_steps = int(state["optimizer_steps"])

    for epoch_index in range(completed, epochs):
        epoch = epoch_index + 1
        _seed_everything(seed + epoch_index, torch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batches = list(
            _record_batches(training_records, batch_size, seed=seed + epoch_index)
        )
        running_loss = 0.0
        for batch_index, batch in enumerate(batches):
            group_start = (batch_index // accumulation) * accumulation
            group_size = min(accumulation, len(batches) - group_start)
            encoded = {
                key: value.to(device)
                for key, value in _collate_training(tokenizer, batch, config).items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**encoded).loss
            running_loss += float(loss.detach().cpu())
            scaler.scale(loss / group_size).backward()
            end_group = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(batches)
            if end_group:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(training["max_grad_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1

        record: dict[str, Any] = {
            "epoch": epoch,
            "mean_training_loss": running_loss / max(len(batches), 1),
            "optimizer_steps": optimizer_steps,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if validation_records is not None:
            generated = _generation_batches(
                tokenizer, model, validation_records, config, device
            )
            sources = [row.source for row in validation_records]
            references = [row.target or "" for row in validation_records]
            policy, gate_report = fit_gate(
                sources,
                references,
                generated,
                [float(value) for value in config["gate"]["edit_ratio_thresholds"]],
                minimum_score_gain=float(config["gate"]["minimum_inner_score_gain"]),
            )
            gated, gate_audit = apply_gate(sources, generated, policy)
            prediction_path = stage_dir / f"inner_epoch_{epoch:03d}.csv"
            _write_prediction_csv(prediction_path, validation_records, gated)
            record.update(
                {
                    "raw_neural_score": competition_error(references, generated),
                    "gated_score": competition_error(references, gated),
                    "gate": gate_report,
                    "gate_audit": gate_audit,
                    "validation_predictions": str(prediction_path.resolve()),
                    "validation_predictions_sha256": sha256_file(prediction_path),
                }
            )

        epoch_adapter = stage_dir / f"epoch_{epoch:03d}_adapter"
        require_new_path(epoch_adapter)
        epoch_adapter.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_adapter, safe_serialization=True)
        adapter_receipt = write_adapter_receipt(
            epoch_adapter,
            run_lineage_sha256=run_lineage_sha256,
            stage=stage_name,
            epoch=epoch,
            base_snapshot_sha256=snapshot["snapshot_sha256"],
        )
        state_path = stage_dir / f"training_state_epoch_{epoch:03d}.pt"
        state_sha256 = _save_training_state(
            state_path,
            torch,
            optimizer,
            scheduler,
            scaler,
            epoch,
            optimizer_steps,
        )
        record.update(
            {
                "adapter": str(epoch_adapter.resolve()),
                "adapter_receipt": str(adapter_receipt.resolve()),
                "adapter_receipt_sha256": sha256_file(adapter_receipt),
                "training_state": str(state_path.resolve()),
                "training_state_sha256": state_sha256,
            }
        )
        progress["epochs"].append(record)
        _write_json(progress_path, progress)
    del model
    torch.cuda.empty_cache()
    return progress


def _smoke_lineage(
    config_path: Path,
    truth_path: Path,
    source_path: Path,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "truth_path": str(truth_path),
        "truth_sha256": sha256_file(truth_path),
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "base_snapshot_sha256": snapshot["snapshot_sha256"],
        "implementation_sha256": sha256_file(Path(__file__)),
    }


def run_smoke(
    config: Mapping[str, Any],
    config_path: Path,
    truth_path: Path,
    source_path: Path,
    records: Sequence[OcrPair],
    snapshot: Mapping[str, Any],
    *,
    resume: bool,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Measure a real LoRA forward/backward before any fold output is created."""

    lineage = _smoke_lineage(config_path, truth_path, source_path, snapshot)
    lineage_sha256 = canonical_sha256(lineage)
    output = _resolve(root, config["smoke"]["receipt"])
    reject_model22_destination(output)
    if output.exists():
        receipt = _load_json(output)
        if not resume:
            raise FileExistsError(f"smoke receipt already exists: {output}")
        if receipt.get("lineage_sha256") != lineage_sha256:
            raise Model31Error("existing smoke receipt has different lineage")
        return receipt

    validate_runtime_versions(config)
    import torch

    _device, device_index = _torch_device(config, torch)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    torch, tokenizer, model, targets, trainable_audit = _load_neural_runtime(
        config, snapshot, trainable=True
    )
    device, _device_index_value = _torch_device(config, torch)
    smoke_batch = sorted(
        records,
        key=lambda row: len(row.source.encode("utf-8"))
        + len((row.target or "").encode("utf-8")),
        reverse=True,
    )[: int(config["training"]["batch_size"])]
    _seed_everything(int(config["seed"]), torch)
    model.train()
    batch = {
        key: value.to(device)
        for key, value in _collate_training(tokenizer, smoke_batch, config).items()
    }
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(**batch).loss
    scaler.scale(loss).backward()
    allocated = int(torch.cuda.max_memory_allocated(device_index))
    reserved = int(torch.cuda.max_memory_reserved(device_index))
    limit = float(config["smoke"]["maximum_reserved_gib"])
    limit_bytes = int(limit * 1024**3)
    if allocated >= limit_bytes or reserved >= limit_bytes:
        raise Model31Error(
            f"Model31 smoke exceeded {limit} GiB: allocated={allocated}, reserved={reserved}"
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "lineage": lineage,
        "lineage_sha256": lineage_sha256,
        "runtime_versions": validate_runtime_versions(config),
        "record_ids": [row.identifier for row in smoke_batch],
        "batch_size": len(smoke_batch),
        "maximum_source_utf8_bytes": max(
            len(row.source.encode("utf-8")) for row in smoke_batch
        ),
        "maximum_target_utf8_bytes": max(
            len((row.target or "").encode("utf-8")) for row in smoke_batch
        ),
        "target_modules": len(targets),
        "trainable_audit": trainable_audit,
        "loss": float(loss.detach().cpu()),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "limit_gib": limit,
    }
    _write_json(output, receipt)
    del model
    torch.cuda.empty_cache()
    return receipt


def verify_smoke_receipt(
    config: Mapping[str, Any],
    config_path: Path,
    truth_path: Path,
    source_path: Path,
    snapshot: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    output = _resolve(root, config["smoke"]["receipt"])
    if not output.is_file():
        raise Model31Error(
            f"smoke receipt missing: {output}; run model31/train.py --smoke-only first"
        )
    receipt = _load_json(output)
    expected = canonical_sha256(
        _smoke_lineage(config_path, truth_path, source_path, snapshot)
    )
    if receipt.get("lineage_sha256") != expected:
        raise Model31Error("smoke receipt does not match current data/config/model/code")
    limit = int(float(config["smoke"]["maximum_reserved_gib"]) * 1024**3)
    if int(receipt.get("peak_allocated_bytes", limit)) >= limit:
        raise Model31Error("smoke allocated-memory gate was not satisfied")
    if int(receipt.get("peak_reserved_bytes", limit)) >= limit:
        raise Model31Error("smoke reserved-memory gate was not satisfied")
    return receipt


def _run_lineage(
    *,
    config_path: Path,
    truth_path: Path,
    source_path: Path,
    output_dir: Path,
    snapshot: Mapping[str, Any],
    plan: FoldPlan,
    runtime_versions: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "truth_path": str(truth_path),
        "truth_sha256": sha256_file(truth_path),
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "output_dir": str(output_dir),
        "base_snapshot": snapshot,
        "fold_plan": fold_plan_receipt(plan),
        "runtime_versions": dict(runtime_versions),
        "implementation_sha256": sha256_file(Path(__file__)),
    }


def _prepare_run_directory(
    output_dir: Path, lineage: Mapping[str, Any], *, resume: bool
) -> tuple[Path, str]:
    reject_model22_destination(output_dir)
    lineage_sha256 = canonical_sha256(lineage)
    manifest_path = output_dir / "run_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"run directory is not empty: {output_dir}")
        if not manifest_path.is_file():
            raise Model31Error("resume directory has no run manifest")
        manifest = _load_json(manifest_path)
        if manifest.get("lineage_sha256") != lineage_sha256 or manifest.get("lineage") != lineage:
            raise Model31Error("resume run lineage differs from current inputs")
    else:
        if resume:
            raise Model31Error("--resume requires an existing receipted run")
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "lineage": lineage,
                "lineage_sha256": lineage_sha256,
            },
        )
    return manifest_path, lineage_sha256


def _best_selection_epoch(progress: Mapping[str, Any]) -> dict[str, Any]:
    epochs = list(progress.get("epochs", ()))
    if not epochs or any("gated_score" not in row for row in epochs):
        raise Model31Error("selection stage has no complete epoch metrics")
    return min(
        epochs,
        key=lambda row: (
            float(row["gated_score"]["mean_error"]),
            int(row["gate_audit"]["changed_rows"]),
            int(row["epoch"]),
        ),
    )


def train_fold(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    fold: int,
    truth_path: Path,
    source_path: Path,
    output_dir: Path,
    resume: bool,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    snapshot = verify_local_snapshot(config, root)
    runtime_versions = validate_runtime_versions(config)
    records = load_oof_pairs(truth_path, source_path)
    byte_audit = validate_byte_limits(
        records,
        int(config["sequence"]["maximum_utf8_bytes"]),
        require_targets=True,
    )
    plan = build_fold_plan(records, fold)
    smoke = verify_smoke_receipt(
        config, config_path, truth_path, source_path, snapshot, root=root
    )
    lineage = _run_lineage(
        config_path=config_path,
        truth_path=truth_path,
        source_path=source_path,
        output_dir=output_dir,
        snapshot=snapshot,
        plan=plan,
        runtime_versions=runtime_versions,
    )
    manifest_path, run_lineage_sha256 = _prepare_run_directory(
        output_dir, lineage, resume=resume
    )
    report_path = output_dir / "report.json"
    if report_path.exists():
        if not resume:
            raise FileExistsError(report_path)
        report = _load_json(report_path)
        if report.get("run_lineage_sha256") != run_lineage_sha256:
            raise Model31Error("completed report lineage mismatch")
        predictions = Path(report["outer_predictions"])
        if sha256_file(predictions) != report.get("outer_predictions_sha256"):
            raise Model31Error("completed outer predictions changed on disk")
        return report

    seed = int(config["seed"])
    selection = _run_stage(
        stage_dir=output_dir / "selection",
        stage_name="selection",
        training_records=plan.selection_train,
        validation_records=plan.selection_validation,
        epochs=int(config["training"]["epochs"]),
        seed=seed + fold * 1000,
        config=config,
        snapshot=snapshot,
        run_lineage_sha256=run_lineage_sha256,
        resume=resume,
    )
    selected = _best_selection_epoch(selection)
    selected_epoch = int(selected["epoch"])
    policy = GatePolicy(**selected["gate"]["selected_policy"])

    refit = _run_stage(
        stage_dir=output_dir / "refit",
        stage_name="refit",
        training_records=plan.refit_train,
        validation_records=None,
        epochs=selected_epoch,
        seed=seed + fold * 1000 + 500,
        config=config,
        snapshot=snapshot,
        run_lineage_sha256=run_lineage_sha256,
        resume=resume,
    )
    final_epoch = refit["epochs"][-1]
    adapter = Path(final_epoch["adapter"])
    verify_adapter_receipt(
        adapter,
        final_epoch["adapter_receipt"],
        expected_base_snapshot_sha256=snapshot["snapshot_sha256"],
        expected_run_lineage_sha256=run_lineage_sha256,
    )
    torch, tokenizer, model, _targets, _audit = _load_neural_runtime(
        config, snapshot, adapter_dir=adapter, trainable=False
    )
    device, _index = _torch_device(config, torch)
    neural, generation_audit = generate_with_overlength_identity(
        tokenizer, model, plan.outer_validation, config, device
    )
    sources = [row.source for row in plan.outer_validation]
    gated, gate_audit = apply_gate(sources, neural, policy)
    outer_predictions = output_dir / "outer_predictions.csv"
    _write_prediction_csv(outer_predictions, plan.outer_validation, gated)
    references = [row.target or "" for row in plan.outer_validation]
    # These fold-level comparisons happen only after the frozen policy has been
    # applied.  They are reporting metrics, never inputs to a row-wise oracle.
    fold_scores = {
        "identity": competition_error(references, sources),
        "raw_neural": competition_error(references, neural),
        "gated": competition_error(references, gated),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_manifest": str(manifest_path.resolve()),
        "run_lineage_sha256": run_lineage_sha256,
        "outer_fold": fold,
        "inner_fold": plan.inner_fold,
        "byte_audit": byte_audit,
        "smoke_receipt_sha256": canonical_sha256(smoke),
        "selected_epoch": selected_epoch,
        "selection_identity_score": selected["gate"]["identity_score"],
        "selection_gated_score": selected["gated_score"],
        "gate_policy": asdict(policy),
        "gate_fit": selected["gate"],
        "final_adapter": str(adapter.resolve()),
        "final_adapter_receipt": final_epoch["adapter_receipt"],
        "generation_audit": generation_audit,
        "gate_audit": gate_audit,
        "fold_scores": fold_scores,
        "outer_predictions": str(outer_predictions.resolve()),
        "outer_predictions_sha256": sha256_file(outer_predictions),
        "outer_ids_sha256": canonical_sha256(
            [row.identifier for row in plan.outer_validation]
        ),
    }
    _write_json(report_path, report)
    del model
    torch.cuda.empty_cache()
    return report


def audit_plan(
    config: Mapping[str, Any], truth_path: Path, source_path: Path, fold: int
) -> dict[str, Any]:
    records = load_oof_pairs(truth_path, source_path)
    byte_audit = validate_byte_limits(
        records,
        int(config["sequence"]["maximum_utf8_bytes"]),
        require_targets=True,
    )
    plan = build_fold_plan(records, fold)
    augmented_selection, selection_synthetic = build_synthetic_corruptions(
        plan.selection_train, config, seed=int(config["seed"]) + fold * 1000 + 100_000
    )
    augmented_refit, refit_synthetic = build_synthetic_corruptions(
        plan.refit_train, config, seed=int(config["seed"]) + fold * 1000 + 100_500
    )
    return {
        "byte_audit": byte_audit,
        "fold_plan": fold_plan_receipt(plan),
        "selection_synthetic": selection_synthetic,
        "refit_synthetic": refit_synthetic,
        "selection_effective_rows": len(augmented_selection),
        "refit_effective_rows": len(augmented_refit),
    }


def train_main(default_config: str | Path) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument("--fold", type=int, required=True, choices=EXPECTED_FOLDS)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only and args.smoke_only:
        parser.error("--audit-only and --smoke-only are mutually exclusive")
    config, config_path = load_config(args.config)
    root = PROJECT_ROOT
    truth_path = _resolve(root, args.truth or config["data"]["truth_csv"])
    source_path = _resolve(root, args.source or config["data"]["train_predictions"])
    output_dir = _resolve(
        root,
        args.output_dir
        or str(config["output_dir"]).format(fold=args.fold),
    )
    if args.audit_only:
        print(json.dumps(audit_plan(config, truth_path, source_path, args.fold), indent=2))
        return
    snapshot = verify_local_snapshot(config, root)
    records = load_oof_pairs(truth_path, source_path)
    validate_byte_limits(
        records,
        int(config["sequence"]["maximum_utf8_bytes"]),
        require_targets=True,
    )
    if args.smoke_only:
        report = run_smoke(
            config,
            config_path,
            truth_path,
            source_path,
            records,
            snapshot,
            resume=args.resume,
            root=root,
        )
    else:
        report = train_fold(
            config,
            config_path,
            fold=args.fold,
            truth_path=truth_path,
            source_path=source_path,
            output_dir=output_dir,
            resume=args.resume,
            root=root,
        )
    print(json.dumps(report, indent=2))


def _load_completed_fold_run(
    run_dir: Path,
    config_path: Path,
    snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, GatePolicy]:
    manifest_path = run_dir / "run_manifest.json"
    report_path = run_dir / "report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"completed Model31 fold run is required: {run_dir}")
    manifest = _load_json(manifest_path)
    report = _load_json(report_path)
    lineage = manifest.get("lineage")
    if not isinstance(lineage, dict):
        raise Model31Error(f"invalid run manifest: {manifest_path}")
    if canonical_sha256(lineage) != manifest.get("lineage_sha256"):
        raise Model31Error(f"run manifest lineage digest mismatch: {manifest_path}")
    if report.get("run_lineage_sha256") != manifest.get("lineage_sha256"):
        raise Model31Error("fold report and run manifest have different lineage")
    if lineage.get("config_sha256") != sha256_file(config_path):
        raise Model31Error("fold adapter was trained with a different config")
    base = lineage.get("base_snapshot", {})
    if base.get("snapshot_sha256") != snapshot["snapshot_sha256"]:
        raise Model31Error("fold adapter base snapshot differs from current verified base")
    adapter = Path(report["final_adapter"])
    verify_adapter_receipt(
        adapter,
        report["final_adapter_receipt"],
        expected_base_snapshot_sha256=snapshot["snapshot_sha256"],
        expected_run_lineage_sha256=manifest["lineage_sha256"],
    )
    try:
        policy = GatePolicy(**report["gate_policy"])
    except (KeyError, TypeError) as error:
        raise Model31Error("fold report has no valid frozen gate policy") from error
    return manifest, report, adapter, policy


def infer_from_fold(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    run_dir: Path,
    input_path: Path,
    output_path: Path,
    sample_path: Path | None,
    submission_path: Path | None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if (sample_path is None) != (submission_path is None):
        raise ValueError("--sample and --submission must be provided together")
    require_new_path(output_path)
    sidecar = output_path.with_suffix(output_path.suffix + ".manifest.json")
    require_new_path(sidecar)
    if submission_path is not None:
        require_new_path(submission_path)
    records = load_source_pairs(input_path)
    if sample_path is not None:
        sample = read_table(sample_path)
        sample_ids = [row[sample.id_column].strip() for row in sample.rows]
        input_ids = [row.identifier for row in records]
        if set(sample_ids) != set(input_ids) or len(sample_ids) != len(input_ids):
            raise SchemaError("test-source IDs do not exactly match SampleSubmission")

    snapshot = verify_local_snapshot(config, root)
    manifest, report, adapter, policy = _load_completed_fold_run(
        run_dir, config_path, snapshot
    )
    torch, tokenizer, model, _targets, _audit = _load_neural_runtime(
        config, snapshot, adapter_dir=adapter, trainable=False
    )
    device, _index = _torch_device(config, torch)
    generated, generation_audit = generate_with_overlength_identity(
        tokenizer, model, records, config, device
    )
    sources = [row.source for row in records]
    selected, gate_audit = apply_gate(sources, generated, policy)
    _write_prediction_csv(output_path, records, selected)
    submission_result: dict[str, Any] | None = None
    if sample_path is not None and submission_path is not None:
        predictions = {
            record.identifier: value
            for record, value in zip(records, selected, strict=True)
        }
        write_submission(
            sample_path,
            submission_path,
            predictions,
            cleaner=clean_prediction_preserving_internal_spaces,
        )
        submission_result = {
            **validate_submission(submission_path, sample_path),
            "path": str(submission_path.resolve()),
            "sha256": sha256_file(submission_path),
        }
    inference_manifest = {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "fold_run": str(run_dir),
        "outer_fold": report["outer_fold"],
        "run_lineage_sha256": manifest["lineage_sha256"],
        "base_snapshot_sha256": snapshot["snapshot_sha256"],
        "adapter": str(adapter),
        "adapter_receipt_sha256": sha256_file(report["final_adapter_receipt"]),
        "gate_policy": asdict(policy),
        "generation_audit": generation_audit,
        "gate_audit": gate_audit,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "submission": submission_result,
    }
    _write_json(sidecar, inference_manifest)
    del model
    torch.cuda.empty_cache()
    return inference_manifest


def infer_main(default_config: str | Path) -> None:
    parser = argparse.ArgumentParser(description="Apply one receipted Model31 fold adapter")
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    root = PROJECT_ROOT
    report = infer_from_fold(
        config,
        config_path,
        run_dir=_resolve(root, args.run),
        input_path=_resolve(root, args.input or config["data"]["test_predictions"]),
        output_path=_resolve(root, args.output),
        sample_path=_resolve(root, args.sample) if args.sample else None,
        submission_path=_resolve(root, args.submission) if args.submission else None,
        root=root,
    )
    print(json.dumps(report, indent=2))


def evaluate_crossfit(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    truth_path: Path,
    source_path: Path,
    run_dirs: Mapping[int, Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Validate and combine exactly one untouched outer prediction per row."""

    require_new_path(output_dir)
    records = load_oof_pairs(truth_path, source_path)
    expected_truth_sha = sha256_file(truth_path)
    expected_source_sha = sha256_file(source_path)
    expected_config_sha = sha256_file(config_path)
    corrected_by_id: dict[str, str] = {}
    fold_reports: dict[str, Any] = {}
    for fold in EXPECTED_FOLDS:
        if fold not in run_dirs:
            raise Model31Error(f"missing run directory for fold {fold}")
        run_dir = run_dirs[fold]
        manifest_path = run_dir / "run_manifest.json"
        report_path = run_dir / "report.json"
        if not manifest_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"incomplete fold run: {run_dir}")
        manifest = _load_json(manifest_path)
        lineage = manifest.get("lineage", {})
        if canonical_sha256(lineage) != manifest.get("lineage_sha256"):
            raise Model31Error(f"fold {fold} run lineage digest mismatch")
        if lineage.get("truth_sha256") != expected_truth_sha:
            raise Model31Error(f"fold {fold} was trained against different truth bytes")
        if lineage.get("source_sha256") != expected_source_sha:
            raise Model31Error(f"fold {fold} was trained against different OOF source bytes")
        if lineage.get("config_sha256") != expected_config_sha:
            raise Model31Error(f"fold {fold} was trained with a different config")
        if int(lineage.get("fold_plan", {}).get("outer_fold", -1)) != fold:
            raise Model31Error(f"run directory for fold {fold} has wrong outer-fold receipt")
        report = _load_json(report_path)
        if report.get("run_lineage_sha256") != manifest.get("lineage_sha256"):
            raise Model31Error(f"fold {fold} report lineage mismatch")
        predictions_path = Path(report["outer_predictions"])
        if sha256_file(predictions_path) != report.get("outer_predictions_sha256"):
            raise Model31Error(f"fold {fold} outer prediction bytes changed")
        table = read_table(predictions_path)
        column = _prediction_column(table, predictions_path)
        values = predictions_by_id(table, column)
        expected_ids = {
            row.identifier for row in records if row.fold == fold
        }
        if set(values) != expected_ids:
            raise SchemaError(
                f"fold {fold} outer IDs differ: missing={sorted(expected_ids-set(values))[:5]}, "
                f"extra={sorted(set(values)-expected_ids)[:5]}"
            )
        overlap = set(corrected_by_id).intersection(values)
        if overlap:
            raise SchemaError(f"duplicate cross-fit IDs: {sorted(overlap)[:5]}")
        corrected_by_id.update(values)
        fold_records = [row for row in records if row.fold == fold]
        references = [row.target or "" for row in fold_records]
        identity = [row.source for row in fold_records]
        corrected = [values[row.identifier] for row in fold_records]
        recomputed = {
            "identity": competition_error(references, identity),
            "gated": competition_error(references, corrected),
        }
        if recomputed["identity"] != report.get("fold_scores", {}).get("identity"):
            raise Model31Error(f"fold {fold} reported identity score is not reproducible")
        if recomputed["gated"] != report.get("fold_scores", {}).get("gated"):
            raise Model31Error(f"fold {fold} reported gated score is not reproducible")
        fold_reports[str(fold)] = {
            "run_dir": str(run_dir),
            "run_lineage_sha256": manifest["lineage_sha256"],
            "selected_epoch": report["selected_epoch"],
            "gate_policy": report["gate_policy"],
            "scores": recomputed,
        }
    expected_all = {row.identifier for row in records}
    if set(corrected_by_id) != expected_all:
        raise SchemaError("combined Model31 OOF IDs are incomplete")
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = [corrected_by_id[row.identifier] for row in records]
    oof_path = output_dir / "allfold_oof.csv"
    _write_prediction_csv(oof_path, records, ordered)
    references = [row.target or "" for row in records]
    identity = [row.source for row in records]
    scores = {
        "identity": competition_error(references, identity),
        "corrected": competition_error(references, ordered),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "rows": len(records),
        "truth_path": str(truth_path),
        "truth_sha256": expected_truth_sha,
        "source_path": str(source_path),
        "source_sha256": expected_source_sha,
        "config_path": str(config_path),
        "config_sha256": expected_config_sha,
        "scores": scores,
        "multi_score_gain": scores["corrected"]["multi_score"]
        - scores["identity"]["multi_score"],
        "folds": fold_reports,
        "oof_path": str(oof_path.resolve()),
        "oof_sha256": sha256_file(oof_path),
    }
    _write_json(output_dir / "report.json", report)
    return report


def evaluate_main(default_config: str | Path) -> None:
    parser = argparse.ArgumentParser(description="Combine and score five Model31 outer folds")
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--run-pattern", default="outputs/model31/fold{fold}")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model31/crossfit"))
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    root = PROJECT_ROOT
    truth_path = _resolve(root, args.truth or config["data"]["truth_csv"])
    source_path = _resolve(root, args.source or config["data"]["train_predictions"])
    run_dirs = {
        fold: _resolve(root, args.run_pattern.format(fold=fold))
        for fold in EXPECTED_FOLDS
    }
    report = evaluate_crossfit(
        config,
        config_path,
        truth_path=truth_path,
        source_path=source_path,
        run_dirs=run_dirs,
        output_dir=_resolve(root, args.output_dir),
    )
    print(json.dumps(report, indent=2))
