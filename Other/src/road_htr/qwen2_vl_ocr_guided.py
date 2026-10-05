"""Receipt-backed OCR-guided Qwen2-VL challenger used by Model32.

The heavy Qwen/PEFT stack is imported only by the explicit training and
inference entry points.  Configuration, draft lineage, prompt construction,
fold comparisons, and output isolation remain testable on CPU.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from . import qwen2_vl_challenger as model27
from .metrics import competition_error
from .schema import ImageIndex, SchemaError, read_table
from .submission import clean_prediction_preserving_internal_spaces, write_submission


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL32_OUTPUT_ROOT = (PROJECT_ROOT / "outputs" / "model32").resolve()
MODEL32_DRAFT_ROOT = (MODEL32_OUTPUT_ROOT / "drafts").resolve()
MODEL32_SUBMISSION_ROOT = (PROJECT_ROOT / "submissions" / "model32").resolve()

SCHEMA_VERSION = 1
DRAFT_RECEIPT_SCHEMA_VERSION = 2
DRAFT_KIND = "model8_zero_shot_unconstrained_full_view"
EMPTY_DRAFT_POLICY = "preserve_blank_between_markers"
MODEL8_CHECKPOINT_SHA256 = (
    "15313b51ace64cbfa81f8f6ef25ad64f04e5a6fb7f7823e67b107527bc081ac9"
)
MODEL27_CONFIG_SHA256 = (
    "ef6df3f4c81451abc55c5a4d9ec4d5ad8667dddbebf224d56b93c00651087caa"
)
MODEL27_IMPLEMENTATION_SHA256 = (
    "148b39afac28d78de32f68acabc590a5cd885465820171c2d0efc2e2d862ee12"
)
MODEL8_INFERENCE_IMPLEMENTATION_SHA256 = (
    "50519a5a354b6c76b41822b93a6729b7399db30a30d14e6a77530b809d726d41"
)
CANONICAL_TRAIN_SHA256 = (
    "1173b87f6415f657d89a13b62f4d3d84bc926e550338ad4499963c89a81a16aa"
)
CANONICAL_TEST_SHA256 = (
    "6cb1ecc1aa8dec912790c173be5bc8f176c14bb3c4eccee54b487ba0b2363deb"
)
SAMPLE_SUBMISSION_SHA256 = (
    "21def37bd4c05488104707a815d972f39cc58fe2d95ee769180268cfcec0581b"
)


@dataclass(frozen=True)
class Model32Config:
    raw: Mapping[str, Any]
    effective_model27: Mapping[str, Any]
    path: Path
    base_config_path: Path
    base_implementation_path: Path
    generator_config_path: Path
    generator_implementation_path: Path


@dataclass(frozen=True)
class GuidedOcrRecord:
    identifier: str
    image_path: Path
    label: str | None
    fold: int | None
    draft: str


@dataclass(frozen=True)
class DraftArtifact:
    predictions: Mapping[str, str]
    audit: Mapping[str, Any]


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def stable_sha256(path: str | Path) -> str:
    """Hash a file and reject a concurrent mutation."""

    resolved = Path(path)
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {resolved}")
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return value


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


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"missing required Model32 config key: {'.'.join(keys)}")
        current = current[key]
    return current


def _below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def require_model32_output(path: str | Path, *, draft: bool = False) -> Path:
    """Restrict mutable artifacts to Model32's own output tree."""

    resolved = _absolute(path)
    root = MODEL32_DRAFT_ROOT if draft else MODEL32_OUTPUT_ROOT
    if not _below(resolved, root):
        raise model27.SafetyGateError(f"Model32 output must remain under {root}: {resolved}")
    return resolved


def require_model32_submission(path: str | Path) -> Path:
    resolved = _absolute(path)
    if not _below(resolved, MODEL32_SUBMISSION_ROOT):
        raise model27.SafetyGateError(
            f"Model32 submission must remain under {MODEL32_SUBMISSION_ROOT}: {resolved}"
        )
    return resolved


def _validate_generator_config(config: Model32Config) -> Mapping[str, Any]:
    try:
        generator = yaml.safe_load(
            config.generator_config_path.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read Model8 draft config: {error}") from error
    if not isinstance(generator, Mapping):
        raise ValueError("Model8 draft configuration must be a mapping")
    exact = {
        ("pretrained_model",): "checkpoints/model8/medium.safetensors",
        ("image_dir",): "data/raw/images",
        ("image_variant",): "full",
        ("inference", "accelerator"): "cpu",
        ("inference", "device"): "auto",
        ("inference", "precision"): "32-true",
        ("inference", "batch_size"): 1,
        ("inference", "line_workers"): 0,
        ("inference", "padding"): 16,
        ("inference", "constrain_charset"): False,
        ("inference", "preserve_internal_spaces"): True,
    }
    for keys, expected in exact.items():
        current: Any = generator
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                raise ValueError(f"Model8 draft config lacks {'.'.join(keys)}")
            current = current[key]
        if current != expected:
            raise ValueError(
                f"Model8 draft config {'.'.join(keys)} must be {expected!r}, "
                f"got {current!r}"
            )
    threads = int(_nested(generator, "inference", "threads"))
    if threads <= 0:
        raise ValueError("Model8 draft inference threads must be positive")
    return generator


def load_config(path: str | Path) -> Model32Config:
    """Load Model32 and verify the exact Model27/Model8 code it reuses."""

    resolved = _absolute(path)
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read Model32 configuration {resolved}: {error}") from error
    if not isinstance(raw, Mapping) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Model32 config must be a schema-version-1 mapping")

    base_config_path = _absolute(_nested(raw, "base_model27", "config"))
    base_implementation_path = _absolute(
        _nested(raw, "base_model27", "implementation")
    )
    generator_config_path = _absolute(
        _nested(raw, "draft_source", "generator_config")
    )
    generator_implementation_path = _absolute(
        _nested(raw, "draft_source", "generator_implementation")
    )
    exact_paths = {
        base_config_path: PROJECT_ROOT / "model27" / "config.yaml",
        base_implementation_path: PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_challenger.py",
        generator_config_path: PROJECT_ROOT / "model32" / "model8_draft_config.yaml",
        generator_implementation_path: PROJECT_ROOT / "src" / "road_htr" / "kraken_cli.py",
    }
    for actual, expected in exact_paths.items():
        if actual != expected.resolve():
            raise ValueError(f"Model32 pinned source path mismatch: {actual} != {expected}")
        if not actual.is_file():
            raise FileNotFoundError(actual)

    pinned = {
        ("base_model27", "config_sha256"): MODEL27_CONFIG_SHA256,
        ("base_model27", "implementation_sha256"): MODEL27_IMPLEMENTATION_SHA256,
        ("draft_source", "generator_implementation_sha256"):
            MODEL8_INFERENCE_IMPLEMENTATION_SHA256,
        ("draft_source", "checkpoint_sha256"): MODEL8_CHECKPOINT_SHA256,
        ("draft_source", "receipt_schema_version"): DRAFT_RECEIPT_SCHEMA_VERSION,
        ("draft_source", "empty_draft_policy"): EMPTY_DRAFT_POLICY,
        ("data", "train_csv_sha256"): CANONICAL_TRAIN_SHA256,
        ("data", "test_csv_sha256"): CANONICAL_TEST_SHA256,
        ("data", "sample_submission_sha256"): SAMPLE_SUBMISSION_SHA256,
        ("draft_source", "kind"): DRAFT_KIND,
    }
    for keys, expected in pinned.items():
        actual = _nested(raw, *keys)
        if actual != expected:
            raise ValueError(
                f"Model32 pin {'.'.join(keys)} must be {expected!r}, got {actual!r}"
            )
    actual_hashes = {
        base_config_path: _nested(raw, "base_model27", "config_sha256"),
        base_implementation_path: _nested(
            raw, "base_model27", "implementation_sha256"
        ),
        generator_config_path: _nested(
            raw, "draft_source", "generator_config_sha256"
        ),
        generator_implementation_path: _nested(
            raw, "draft_source", "generator_implementation_sha256"
        ),
    }
    for source, expected_hash in actual_hashes.items():
        if stable_sha256(source) != expected_hash:
            raise model27.SafetyGateError(
                f"pinned Model32 dependency changed: {source}"
            )

    base_config, _ = model27.load_config(base_config_path)
    effective = copy.deepcopy(base_config)
    effective["train_csv"] = str(_nested(raw, "data", "train_csv"))
    effective["image_dir"] = str(_nested(raw, "data", "image_dir"))
    effective["output_dir"] = str(_nested(raw, "output_dir"))
    effective["prompt"] = str(_nested(raw, "prompt_template"))
    model27.validate_config(effective)

    prompt = str(_nested(raw, "prompt_template"))
    if prompt.count("{draft}") != 1:
        raise ValueError("prompt_template must contain exactly one literal {draft}")
    if "{label}" in prompt or "{target}" in prompt:
        raise ValueError("prompt_template may not interpolate a challenge label")
    if str(_nested(raw, "output_dir")) != "outputs/model32/fold{fold}":
        raise ValueError("Model32 output_dir must remain outputs/model32/fold{fold}")
    if str(_nested(raw, "smoke_report_dir")) != "outputs/model32/smoke":
        raise ValueError("Model32 smoke reports must remain under outputs/model32/smoke")
    exact_draft_paths = {
        ("draft_source", "train_csv"): "outputs/model32/drafts/train_model8_raw.csv",
        ("draft_source", "train_receipt"):
            "outputs/model32/drafts/train_model8_raw.receipt.json",
        ("draft_source", "test_csv"): "outputs/model32/drafts/test_model8_raw.csv",
        ("draft_source", "test_receipt"):
            "outputs/model32/drafts/test_model8_raw.receipt.json",
    }
    for keys, expected in exact_draft_paths.items():
        if _nested(raw, *keys) != expected:
            raise ValueError(
                f"Model32 {'.'.join(keys)} must remain {expected!r}"
            )
    if _nested(raw, "cross_validation", "folds") != 5:
        raise ValueError("Model32 requires five folds")
    if _nested(raw, "cross_validation", "selection_scope") != "whole_fold_report_only":
        raise ValueError("Model32 comparison must be report-only at whole-fold scope")
    if _nested(raw, "cross_validation", "forbid_per_row_oracle") is not True:
        raise ValueError("Model32 must forbid per-row oracle selection")
    safety = {
        "offline": True,
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "double_quant": True,
        "expected_lora_targets": 58,
        "max_allocated_gib_exclusive": 5.3,
        "max_reserved_gib_exclusive": 5.3,
        "preserve_internal_spaces": True,
    }
    if _nested(raw, "safety_assertions") != safety:
        raise ValueError("Model32 safety_assertions may not be weakened")

    result = Model32Config(
        raw=raw,
        effective_model27=effective,
        path=resolved,
        base_config_path=base_config_path,
        base_implementation_path=base_implementation_path,
        generator_config_path=generator_config_path,
        generator_implementation_path=generator_implementation_path,
    )
    _validate_generator_config(result)
    return result


def _input_ids(path: Path) -> tuple[Any, list[str]]:
    table = read_table(path)
    identifiers = [row[table.id_column].strip() for row in table.rows]
    if not identifiers or any(not identifier for identifier in identifiers):
        raise SchemaError(f"input has blank or no IDs: {path}")
    if len(set(identifiers)) != len(identifiers):
        raise SchemaError(f"input contains duplicate IDs: {path}")
    return table, identifiers


def _configured_draft_paths(
    config: Model32Config, role: str
) -> tuple[Path, Path]:
    if role not in {"train", "test"}:
        raise ValueError("draft role must be train or test")
    return (
        _absolute(_nested(config.raw, "draft_source", f"{role}_csv")),
        _absolute(_nested(config.raw, "draft_source", f"{role}_receipt")),
    )


def _images_sha256(config: Model32Config, identifiers: Sequence[str]) -> str:
    """Bind each ordered ID to the exact image bytes used by both recognizers."""

    image_index = ImageIndex(_absolute(_nested(config.raw, "data", "image_dir")))
    entries = []
    for identifier in identifiers:
        image_path = image_index.resolve(identifier).resolve()
        entries.append(
            {
                "id": identifier,
                "path": str(image_path),
                "sha256": stable_sha256(image_path),
            }
        )
    return _json_sha256(entries)


def _read_draft_csv(
    path: Path, expected_id_column: str, expected_ids: Sequence[str]
) -> dict[str, str]:
    table = read_table(path)
    if tuple(table.fieldnames) != (expected_id_column, "prediction"):
        raise SchemaError(
            f"draft CSV columns must be [{expected_id_column!r}, 'prediction'], "
            f"got {table.fieldnames}"
        )
    actual_ids = [row[table.id_column].strip() for row in table.rows]
    if actual_ids != list(expected_ids):
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        raise SchemaError(
            "draft IDs/order differ from source input: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    predictions: dict[str, str] = {}
    for row, identifier in zip(table.rows, expected_ids, strict=True):
        raw = row["prediction"]
        if raw is None:
            raise SchemaError(f"missing zero-shot draft value for {identifier!r}")
        cleaned = clean_prediction_preserving_internal_spaces(raw)
        if cleaned != raw:
            raise SchemaError(
                f"draft {identifier!r} contains unclean edges or line controls"
            )
        predictions[identifier] = raw
    return predictions


def _atomic_id_projection(
    path: Path, id_column: str, identifiers: Sequence[str]
) -> None:
    """Write the only input visible to zero-shot inference: ordered IDs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[id_column])
            writer.writeheader()
            writer.writerows({id_column: identifier} for identifier in identifiers)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_id_projection(
    path: Path, id_column: str, identifiers: Sequence[str]
) -> None:
    projection = read_table(path)
    if tuple(projection.fieldnames) != (id_column,):
        raise model27.SafetyGateError(
            "Model8 inference projection must contain only the ID column"
        )
    actual = [row[projection.id_column].strip() for row in projection.rows]
    if projection.id_column != id_column or actual != list(identifiers):
        raise model27.SafetyGateError("Model8 ID-only projection/order mismatch")


def _validate_confidence_jsonl(
    path: Path, identifiers: Sequence[str], predictions: Mapping[str, str]
) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read Model8 confidence artifact {path}: {error}") from error
    if len(lines) != len(identifiers):
        raise model27.SafetyGateError("Model8 confidence row count mismatch")
    for position, (identifier, line) in enumerate(
        zip(identifiers, lines, strict=True)
    ):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise model27.SafetyGateError(
                f"invalid Model8 confidence JSON at row {position}"
            ) from error
        candidates = payload.get("candidates") if isinstance(payload, Mapping) else None
        if (
            not isinstance(payload, Mapping)
            or payload.get("id") != identifier
            or not isinstance(candidates, list)
            or candidates != [predictions[identifier]]
        ):
            raise model27.SafetyGateError(
                f"Model8 confidence/prediction mismatch for {identifier!r}"
            )


def _validate_source_manifest(
    config: Model32Config,
    inference_input_csv: Path,
    output_csv: Path,
    source_manifest_path: Path,
    row_count: int,
) -> dict[str, Any]:
    source = _read_json(source_manifest_path)
    generator = _validate_generator_config(config)
    checkpoint = _absolute(_nested(config.raw, "draft_source", "checkpoint"))
    exact = {
        "input_csv": str(inference_input_csv),
        "input_sha256": stable_sha256(inference_input_csv),
        "model": str(checkpoint),
        "model_sha256": MODEL8_CHECKPOINT_SHA256,
        "fold": None,
        "image_variant": "full",
        "constrain_charset": False,
        "batch_size": 1,
        "preserve_internal_spaces": True,
        "rows": row_count,
        "config_sha256": stable_sha256(config.generator_config_path),
        "resolved_config": str(config.generator_config_path),
    }
    for key, expected in exact.items():
        if source.get(key) != expected:
            raise model27.SafetyGateError(
                f"Model8 source manifest {key} must be {expected!r}, "
                f"got {source.get(key)!r}"
            )
    if source.get("config") != generator:
        raise model27.SafetyGateError("Model8 source manifest config differs from pin")
    if not output_csv.is_file():
        raise FileNotFoundError(output_csv)
    return source


def _draft_receipt_payload(
    config: Model32Config,
    input_csv: Path,
    output_csv: Path,
    source_manifest_path: Path,
) -> dict[str, Any]:
    table, identifiers = _input_ids(input_csv)
    predictions = _read_draft_csv(output_csv, table.id_column, identifiers)
    inference_input_path = output_csv.with_suffix(".ids.csv")
    _validate_id_projection(inference_input_path, table.id_column, identifiers)
    _validate_source_manifest(
        config,
        inference_input_path,
        output_csv,
        source_manifest_path,
        len(identifiers),
    )
    checkpoint = _absolute(_nested(config.raw, "draft_source", "checkpoint"))
    details_path = output_csv.with_suffix(".jsonl")
    if not details_path.is_file():
        raise FileNotFoundError(details_path)
    _validate_confidence_jsonl(details_path, identifiers, predictions)
    empty_draft_ids = [
        identifier for identifier in identifiers if predictions[identifier] == ""
    ]
    return {
        "schema_version": DRAFT_RECEIPT_SCHEMA_VERSION,
        "kind": DRAFT_KIND,
        "complete": True,
        "label_independent": True,
        "challenge_target_access": "none",
        "postprocessing_uses_challenge_labels": False,
        "input_csv": str(input_csv),
        "input_sha256": stable_sha256(input_csv),
        "input_ids_sha256": _json_sha256(identifiers),
        "images_sha256": _images_sha256(config, identifiers),
        "inference_input_csv": str(inference_input_path),
        "inference_input_sha256": stable_sha256(inference_input_path),
        "inference_input_columns": [table.id_column],
        "id_column": table.id_column,
        "rows": len(identifiers),
        "draft_csv": str(output_csv),
        "draft_csv_sha256": stable_sha256(output_csv),
        "draft_ids_sha256": _json_sha256(list(predictions)),
        "empty_draft_policy": EMPTY_DRAFT_POLICY,
        "empty_draft_count": len(empty_draft_ids),
        "empty_draft_ids": empty_draft_ids,
        "empty_draft_ids_sha256": _json_sha256(empty_draft_ids),
        "confidence_jsonl": str(details_path),
        "confidence_jsonl_sha256": stable_sha256(details_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": stable_sha256(source_manifest_path),
        "generator_config": str(config.generator_config_path),
        "generator_config_sha256": stable_sha256(config.generator_config_path),
        "generator_implementation": str(config.generator_implementation_path),
        "generator_implementation_sha256": stable_sha256(
            config.generator_implementation_path
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": stable_sha256(checkpoint),
        "image_variant": "full",
        "constrain_charset": False,
        "preserve_internal_spaces": True,
        "fold": None,
    }


def verify_draft_artifact(
    config: Model32Config,
    input_csv: str | Path,
    draft_csv: str | Path,
    receipt_path: str | Path,
) -> DraftArtifact:
    """Accept only the label-independent Model8 artifact produced by this model."""

    input_path = _absolute(input_csv)
    draft_path = _absolute(draft_csv)
    receipt_file = _absolute(receipt_path)
    for path in (input_path, draft_path, receipt_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    receipt = _read_json(receipt_file)
    if receipt_file != draft_path.with_suffix(".receipt.json"):
        raise model27.SafetyGateError("draft receipt path is noncanonical")
    details_path = draft_path.with_suffix(".jsonl")
    inference_input_path = draft_path.with_suffix(".ids.csv")
    required = {
        "schema_version": DRAFT_RECEIPT_SCHEMA_VERSION,
        "kind": DRAFT_KIND,
        "complete": True,
        "label_independent": True,
        "challenge_target_access": "none",
        "postprocessing_uses_challenge_labels": False,
        "input_csv": str(input_path),
        "input_sha256": stable_sha256(input_path),
        "draft_csv": str(draft_path),
        "draft_csv_sha256": stable_sha256(draft_path),
        "inference_input_csv": str(inference_input_path),
        "inference_input_sha256": stable_sha256(inference_input_path),
        "confidence_jsonl": str(details_path),
        "confidence_jsonl_sha256": stable_sha256(details_path),
        "generator_config": str(config.generator_config_path),
        "generator_config_sha256": stable_sha256(config.generator_config_path),
        "generator_implementation": str(config.generator_implementation_path),
        "generator_implementation_sha256": MODEL8_INFERENCE_IMPLEMENTATION_SHA256,
        "checkpoint": str(_absolute(_nested(config.raw, "draft_source", "checkpoint"))),
        "checkpoint_sha256": MODEL8_CHECKPOINT_SHA256,
        "image_variant": "full",
        "constrain_charset": False,
        "preserve_internal_spaces": True,
        "empty_draft_policy": EMPTY_DRAFT_POLICY,
        "fold": None,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise model27.SafetyGateError(
                f"draft receipt {key} must be {expected!r}, got {receipt.get(key)!r}"
            )
    if stable_sha256(config.generator_implementation_path) != (
        MODEL8_INFERENCE_IMPLEMENTATION_SHA256
    ):
        raise model27.SafetyGateError("pinned Model8 inference implementation changed")
    checkpoint = _absolute(_nested(config.raw, "draft_source", "checkpoint"))
    if stable_sha256(checkpoint) != MODEL8_CHECKPOINT_SHA256:
        raise model27.SafetyGateError("pinned Model8 checkpoint changed")

    table, identifiers = _input_ids(input_path)
    if receipt.get("rows") != len(identifiers):
        raise model27.SafetyGateError("draft receipt row count mismatch")
    if receipt.get("id_column") != table.id_column:
        raise model27.SafetyGateError("draft receipt ID-column mismatch")
    ids_digest = _json_sha256(identifiers)
    if receipt.get("input_ids_sha256") != ids_digest:
        raise model27.SafetyGateError("draft receipt input-ID hash mismatch")
    images_digest = _images_sha256(config, identifiers)
    if receipt.get("images_sha256") != images_digest:
        raise model27.SafetyGateError("draft receipt source-image hash mismatch")
    predictions = _read_draft_csv(draft_path, table.id_column, identifiers)
    _validate_id_projection(inference_input_path, table.id_column, identifiers)
    if receipt.get("inference_input_columns") != [table.id_column]:
        raise model27.SafetyGateError("draft receipt ID-only column ledger mismatch")
    _validate_confidence_jsonl(details_path, identifiers, predictions)
    if receipt.get("draft_ids_sha256") != _json_sha256(list(predictions)):
        raise model27.SafetyGateError("draft receipt output-ID hash mismatch")
    empty_draft_ids = [
        identifier for identifier in identifiers if predictions[identifier] == ""
    ]
    if receipt.get("empty_draft_count") != len(empty_draft_ids):
        raise model27.SafetyGateError("draft receipt empty-draft count mismatch")
    if receipt.get("empty_draft_ids") != empty_draft_ids:
        raise model27.SafetyGateError("draft receipt ordered empty-ID list mismatch")
    empty_ids_digest = _json_sha256(empty_draft_ids)
    if receipt.get("empty_draft_ids_sha256") != empty_ids_digest:
        raise model27.SafetyGateError("draft receipt empty-ID hash mismatch")

    source_manifest_value = receipt.get("source_manifest")
    if not isinstance(source_manifest_value, str):
        raise model27.SafetyGateError("draft receipt lacks source manifest")
    source_manifest_path = _absolute(source_manifest_value)
    if source_manifest_path != draft_path.with_suffix(".manifest.json"):
        raise model27.SafetyGateError("draft source manifest path is noncanonical")
    if receipt.get("source_manifest_sha256") != stable_sha256(source_manifest_path):
        raise model27.SafetyGateError("draft source manifest checksum mismatch")
    source = _validate_source_manifest(
        config,
        inference_input_path,
        draft_path,
        source_manifest_path,
        len(identifiers),
    )
    audit = {
        "kind": DRAFT_KIND,
        "label_independent": True,
        "challenge_target_access": "none",
        "rows": len(identifiers),
        "input_csv": str(input_path),
        "input_sha256": stable_sha256(input_path),
        "input_ids_sha256": ids_digest,
        "images_sha256": images_digest,
        "inference_input_csv": str(inference_input_path),
        "inference_input_sha256": stable_sha256(inference_input_path),
        "draft_csv": str(draft_path),
        "draft_csv_sha256": stable_sha256(draft_path),
        "receipt_schema_version": DRAFT_RECEIPT_SCHEMA_VERSION,
        "empty_draft_policy": EMPTY_DRAFT_POLICY,
        "empty_draft_count": len(empty_draft_ids),
        "empty_draft_ids": empty_draft_ids,
        "empty_draft_ids_sha256": empty_ids_digest,
        "receipt": str(receipt_file),
        "receipt_sha256": stable_sha256(receipt_file),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": stable_sha256(source_manifest_path),
        "checkpoint_sha256": MODEL8_CHECKPOINT_SHA256,
        "generator_config_sha256": stable_sha256(config.generator_config_path),
        "generator_implementation_sha256": MODEL8_INFERENCE_IMPLEMENTATION_SHA256,
        "source_runtime": {
            "python": source.get("python"),
            "seconds": source.get("seconds"),
        },
    }
    return DraftArtifact(predictions=predictions, audit=audit)


def prepare_drafts(
    config_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run frozen Model8 on CPU with no label-derived decoder constraint."""

    config = load_config(config_path)
    input_path = _absolute(input_csv)
    output_path = require_model32_output(output_csv, draft=True)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    train_input = _absolute(_nested(config.raw, "data", "train_csv"))
    test_input = _absolute(_nested(config.raw, "data", "test_csv"))
    if input_path == train_input and stable_sha256(input_path) == CANONICAL_TRAIN_SHA256:
        role = "train"
    elif input_path == test_input and stable_sha256(input_path) == CANONICAL_TEST_SHA256:
        role = "test"
    else:
        raise model27.SafetyGateError(
            "Model32 draft generation accepts only pinned Train/Test inputs"
        )
    expected_output, _ = _configured_draft_paths(config, role)
    if output_path != expected_output:
        raise model27.SafetyGateError(
            f"Model32 {role} drafts must be written to {expected_output}"
        )
    table, identifiers = _input_ids(input_path)
    images_digest = _images_sha256(config, identifiers)
    checkpoint = _absolute(_nested(config.raw, "draft_source", "checkpoint"))
    if not checkpoint.is_file() or stable_sha256(checkpoint) != MODEL8_CHECKPOINT_SHA256:
        raise model27.SafetyGateError("pinned zero-shot Model8 checkpoint is unavailable")
    source_manifest_path = output_path.with_suffix(".manifest.json")
    receipt_path = output_path.with_suffix(".receipt.json")
    details_path = output_path.with_suffix(".jsonl")
    inference_input_path = output_path.with_suffix(".ids.csv")
    for path in (
        output_path,
        source_manifest_path,
        receipt_path,
        details_path,
        inference_input_path,
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Model32 draft artifact: {path}")
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise FileExistsError(output_path.parent)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "receipt_schema_version": DRAFT_RECEIPT_SCHEMA_VERSION,
        "dry_run": dry_run,
        "kind": DRAFT_KIND,
        "input_csv": str(input_path),
        "input_sha256": stable_sha256(input_path),
        "input_ids_sha256": _json_sha256(identifiers),
        "images_sha256": images_digest,
        "id_column": table.id_column,
        "rows": len(identifiers),
        "output_csv": str(output_path),
        "inference_input_csv": str(inference_input_path),
        "receipt": str(receipt_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": MODEL8_CHECKPOINT_SHA256,
        "runtime": {
            "accelerator": "cpu",
            "precision": "32-true",
            "batch_size": 1,
            "image_variant": "full",
            "constrain_charset": False,
            "preserve_internal_spaces": True,
            "empty_draft_policy": EMPTY_DRAFT_POLICY,
            "challenge_target_access": "none",
        },
    }
    if dry_run:
        return plan

    # The generic inference function reads row IDs and image pixels. With the
    # decoder constraint explicitly false, it never calls the Train.csv
    # character-inventory helper and therefore never reads a Target value.
    from .kraken_cli import infer as kraken_infer

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_id_projection(inference_input_path, table.id_column, identifiers)
    kraken_infer(
        config.generator_config_path,
        inference_input_path,
        output_path,
        checkpoint=checkpoint,
        fold=None,
        image_variant="full",
        constrain_charset=False,
        batch_size_override=1,
    )
    receipt = _draft_receipt_payload(
        config, input_path, output_path, source_manifest_path
    )
    if receipt.get("images_sha256") != images_digest:
        raise RuntimeError("source images changed while generating Model32 drafts")
    _atomic_json(receipt_path, receipt)
    artifact = verify_draft_artifact(
        config, input_path, output_path, receipt_path
    )
    return {**plan, "dry_run": False, "complete": True, "audit": artifact.audit}


def build_guided_prompt(template: str, draft: object) -> str:
    """Insert one present OCR result literally, preserving a genuine blank."""

    if draft is None:
        raise ValueError("OCR-guided prompt requires a present draft value")
    value = clean_prediction_preserving_internal_spaces(draft)
    if template.count("{draft}") != 1:
        raise ValueError("guided prompt template must contain one {draft}")
    return template.replace("{draft}", value)


def build_guided_messages(
    prompt_template: str, draft: object, label: str | None = None
) -> list[dict[str, Any]]:
    prompt = build_guided_prompt(prompt_template, draft)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if label is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": model27.trim_label_edges(label)}
                ],
            }
        )
    return messages


class QwenGuidedCollator(model27.QwenOcrCollator):
    """Model27's hard-gated collator with a per-row, untrusted OCR draft."""

    def __init__(
        self,
        processor: Any,
        prompt_template: str,
        max_image_tokens: int,
        max_sequence_tokens: int,
        spatial_merge_size: int = 2,
    ):
        super().__init__(
            processor,
            prompt_template,
            max_image_tokens,
            max_sequence_tokens,
            spatial_merge_size,
        )
        self.prompt_template = prompt_template

    def _encode_text(self, record: GuidedOcrRecord, include_label: bool) -> str:
        if include_label and record.label is None:
            raise ValueError("training record has no label")
        messages = build_guided_messages(
            self.prompt_template,
            record.draft,
            record.label if include_label else None,
        )
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_label,
        )


def load_guided_records(
    config: Model32Config,
    input_csv: str | Path,
    draft_csv: str | Path,
    draft_receipt: str | Path,
    *,
    require_target: bool,
    require_five_folds: bool,
) -> tuple[tuple[GuidedOcrRecord, ...], Mapping[str, Any]]:
    input_path = _absolute(input_csv)
    artifact = verify_draft_artifact(
        config, input_path, draft_csv, draft_receipt
    )
    base_records = model27.load_records(
        input_path,
        _absolute(_nested(config.raw, "data", "image_dir")),
        require_target=require_target,
        require_five_folds=require_five_folds,
    )
    guided = tuple(
        GuidedOcrRecord(
            record.identifier,
            record.image_path,
            record.label,
            record.fold,
            artifact.predictions[record.identifier],
        )
        for record in base_records
    )
    return guided, artifact.audit


def select_guided_smoke_record(
    records: Sequence[GuidedOcrRecord],
    *,
    min_pixels: int,
    max_pixels: int,
    merged_patch_size: int = 28,
) -> GuidedOcrRecord:
    """Choose the largest image + label + added-draft conservative proxy."""

    labelled = [record for record in records if record.label is not None]
    if not labelled:
        raise ValueError("no labelled guided record is available for smoke")

    def size(record: GuidedOcrRecord) -> tuple[int, int, int, str]:
        image_tokens = model27.estimate_visual_tokens(
            record.image_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            merged_patch_size=merged_patch_size,
        )
        combined = image_tokens + len(record.label or "") + len(record.draft)
        return combined, image_tokens, len(record.draft), record.identifier

    return max(labelled, key=size)


def compare_fold_to_draft(
    records: Sequence[GuidedOcrRecord], predictions: Sequence[str]
) -> dict[str, Any]:
    """Report whole-fold alternatives without constructing a row-wise oracle."""

    if len(records) != len(predictions) or not records:
        raise ValueError("fold comparison requires one prediction per nonempty record")
    references: list[str] = []
    for record in records:
        if record.label is None:
            raise ValueError(f"record {record.identifier!r} has no comparison label")
        references.append(record.label)
    challenger = [model27.trim_label_edges(value) for value in predictions]
    drafts = [record.draft for record in records]
    challenger_metrics = competition_error(references, challenger)
    draft_metrics = competition_error(references, drafts)
    delta = float(challenger_metrics["multi_score"]) - float(
        draft_metrics["multi_score"]
    )
    return {
        "scope": "whole_fold_report_only",
        "rows": len(records),
        "challenger": challenger_metrics,
        "returning_draft": draft_metrics,
        "challenger_multi_score_delta": delta,
        "diagnostic_winner": "challenger" if delta > 0 else "draft",
        "per_row_selection_performed": False,
        "emitted_predictions": "challenger",
    }


def _atomic_predictions(
    path: Path,
    records: Sequence[GuidedOcrRecord],
    predictions: Sequence[str],
) -> Path:
    if len(records) != len(predictions):
        raise ValueError("record/prediction count mismatch")
    require_model32_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "prediction", "fold"])
            writer.writeheader()
            for record, raw_prediction in zip(records, predictions, strict=True):
                writer.writerow(
                    {
                        "ID": record.identifier,
                        "prediction": model27.trim_label_edges(raw_prediction),
                        "fold": "" if record.fold is None else record.fold,
                    }
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _directory_ledger(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    ledger: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise model27.SafetyGateError(f"adapter artifact may not be a symlink: {path}")
        if path.is_file():
            ledger[path.relative_to(directory).as_posix()] = stable_sha256(path)
    if not ledger:
        raise model27.SafetyGateError(f"adapter directory is empty: {directory}")
    return ledger


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "model32" / "train.py",
        PROJECT_ROOT / "model32" / "infer.py",
        PROJECT_ROOT / "model32" / "prepare_drafts.py",
        PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_challenger.py",
        PROJECT_ROOT / "src" / "road_htr" / "metrics.py",
        PROJECT_ROOT / "src" / "road_htr" / "submission.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(stable_sha256(path)))
    return digest.hexdigest()


def _ids_sha256(records: Sequence[GuidedOcrRecord]) -> str:
    return _json_sha256([record.identifier for record in records])


def _validate_canonical_input(
    config: Model32Config, input_path: Path, *, fold: int | None
) -> None:
    train_path = _absolute(_nested(config.raw, "data", "train_csv"))
    test_path = _absolute(_nested(config.raw, "data", "test_csv"))
    if fold is not None:
        if input_path != train_path or stable_sha256(input_path) != CANONICAL_TRAIN_SHA256:
            raise model27.SafetyGateError(
                "Model32 OOF inference requires the pinned canonical training folds"
            )
    elif input_path == train_path:
        raise model27.SafetyGateError(
            "training-data inference must specify --fold; all-row output would not be OOF"
        )
    elif input_path != test_path or stable_sha256(input_path) != CANONICAL_TEST_SHA256:
        raise model27.SafetyGateError(
            "fold-free Model32 inference requires the pinned Test.csv"
        )


def _make_validation_callback(
    processor: Any,
    collator: QwenGuidedCollator,
    validation_records: Sequence[GuidedOcrRecord],
    output_dir: Path,
    run_manifest_path: Path,
    max_new_tokens: int,
) -> Any:
    from transformers import TrainerCallback

    class GuidedExactMetricCallback(TrainerCallback):
        def __init__(self) -> None:
            self.best_score = -math.inf
            self.best_epoch: int | None = None

        def on_epoch_end(
            self,
            args: Any,
            state: Any,
            control: Any,
            model: Any,
            **_: Any,
        ) -> Any:
            if not state.is_world_process_zero:
                return control
            epoch = int(round(float(state.epoch or 0)))
            predictions = model27.greedy_predict(
                model,
                processor,
                collator,
                validation_records,
                max_new_tokens=max_new_tokens,
            )
            comparison = compare_fold_to_draft(validation_records, predictions)
            prediction_path = output_dir / "validation" / f"epoch_{epoch:03d}.csv"
            _atomic_predictions(prediction_path, validation_records, predictions)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "epoch": epoch,
                "rows": len(validation_records),
                "generation": {
                    "do_sample": False,
                    "num_beams": 1,
                    "num_return_sequences": 1,
                    "max_new_tokens": max_new_tokens,
                },
                "comparison": comparison,
                "predictions": str(prediction_path),
                "predictions_sha256": stable_sha256(prediction_path),
                "no_per_row_oracle": True,
            }
            _atomic_json(
                output_dir / "validation" / f"epoch_{epoch:03d}.json", payload
            )
            score = float(comparison["challenger"]["multi_score"])
            if score > self.best_score + 1e-12:
                self.best_score = score
                self.best_epoch = epoch
                best_dir = output_dir / "best_adapter"
                model.save_pretrained(best_dir, safe_serialization=True)
                processor.save_pretrained(best_dir)
                adapter_ledger = _directory_ledger(best_dir)
                _atomic_json(
                    output_dir / "best.json",
                    {
                        **payload,
                        "adapter": str(best_dir),
                        "adapter_files_sha256": adapter_ledger,
                        "run_manifest": str(run_manifest_path),
                        "run_manifest_sha256": stable_sha256(run_manifest_path),
                        "selection": (
                            "maximum full-fold greedy challenger score; draft is "
                            "reported only and never selected per row"
                        ),
                    },
                )
            print(
                f"guided fold validation epoch={epoch} "
                f"challenger={score:.9f} "
                f"draft={comparison['returning_draft']['multi_score']:.9f} "
                f"best_epoch={self.best_epoch}",
                flush=True,
            )
            return control

    return GuidedExactMetricCallback()


def train(
    config_path: str | Path,
    fold: int,
    *,
    smoke_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train one guarded fold; ``dry_run`` imports no CUDA libraries."""

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    model27.enable_offline_mode()
    config = load_config(config_path)
    initial_config_sha256 = stable_sha256(config.path)
    initial_implementation_sha256 = _implementation_sha256()
    if fold not in range(5):
        raise ValueError("fold must be one of 0,1,2,3,4")
    train_csv = _absolute(_nested(config.raw, "data", "train_csv"))
    if stable_sha256(train_csv) != CANONICAL_TRAIN_SHA256:
        raise model27.SafetyGateError("canonical Model32 training folds changed")
    draft_path, receipt_path = _configured_draft_paths(config, "train")
    records, draft_audit = load_guided_records(
        config,
        train_csv,
        draft_path,
        receipt_path,
        require_target=True,
        require_five_folds=True,
    )
    training_records = tuple(record for record in records if record.fold != fold)
    validation_records = tuple(record for record in records if record.fold == fold)
    output_dir = require_model32_output(
        str(_nested(config.raw, "output_dir")).format(fold=fold)
    )
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {output_dir}")
    smoke_path = require_model32_output(
        _absolute(_nested(config.raw, "smoke_report_dir")) / f"fold{fold}.json"
    )
    if smoke_only and smoke_path.exists():
        raise FileExistsError(f"refusing to overwrite Model32 smoke report: {smoke_path}")

    provenance = model27.verify_starter_provenance(config.effective_model27)
    checkpoint = model27.verify_local_checkpoint(config.effective_model27)
    static_audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": "model32_ocr_guided_qwen2_vl",
        "fold": fold,
        "training_excluded_folds": [fold],
        "rows": {
            "all": len(records),
            "train": len(training_records),
            "validation": len(validation_records),
        },
        "ids_sha256": {
            "all": _ids_sha256(records),
            "train": _ids_sha256(training_records),
            "validation": _ids_sha256(validation_records),
        },
        "config": str(config.path),
        "config_sha256": initial_config_sha256,
        "implementation_sha256": initial_implementation_sha256,
        "base_model27_config_sha256": stable_sha256(config.base_config_path),
        "base_model27_implementation_sha256": stable_sha256(
            config.base_implementation_path
        ),
        "train_csv": str(train_csv),
        "train_csv_sha256": stable_sha256(train_csv),
        "draft_artifact": draft_audit,
        "draft_policy": {
            "label_independent": True,
            "outer_fold_safe": True,
            "model25_allowed": False,
            "nested_refit_required": False,
        },
        "starter_provenance": provenance,
        "local_checkpoint": checkpoint,
        "output_dir": str(output_dir),
        "gpu_started": False,
    }
    if dry_run:
        return {**static_audit, "dry_run": True}

    seed = int(_nested(config.effective_model27, "seed"))
    model27.set_seed(seed)
    processor = model27.load_processor(config.effective_model27)
    model, model_audit = model27.load_training_model(config.effective_model27)
    collator = QwenGuidedCollator(
        processor,
        str(_nested(config.raw, "prompt_template")),
        int(_nested(config.effective_model27, "image", "max_image_tokens")),
        int(_nested(config.effective_model27, "image", "max_sequence_tokens")),
    )
    smoke_record = select_guided_smoke_record(
        training_records,
        min_pixels=int(_nested(config.effective_model27, "image", "min_pixels")),
        max_pixels=int(_nested(config.effective_model27, "image", "max_pixels")),
        merged_patch_size=int(
            _nested(config.effective_model27, "image", "merged_patch_size")
        ),
    )
    smoke = model27.run_memory_smoke(
        model,
        collator,
        smoke_record,
        max_allocated_gib=float(
            _nested(config.effective_model27, "smoke", "max_allocated_gib")
        ),
        max_reserved_gib=float(
            _nested(config.effective_model27, "smoke", "max_reserved_gib")
        ),
    )
    runtime_audit = {
        **static_audit,
        "gpu_started": True,
        "model_audit": model_audit,
        "smoke": smoke,
    }
    if smoke_only:
        _atomic_json(smoke_path, runtime_audit)
        return runtime_audit

    import torch
    from transformers import Trainer, TrainingArguments

    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest = {
        **model27._runtime_manifest(config.effective_model27),
        **runtime_audit,
        "resolved_model32_config": str(config.path),
        "effective_model27_config": config.effective_model27,
        "prompt_template": str(_nested(config.raw, "prompt_template")),
    }
    _atomic_json(run_manifest_path, run_manifest)
    _atomic_predictions(
        output_dir / "validation" / "returning_draft.csv",
        validation_records,
        [record.draft for record in validation_records],
    )

    training = _nested(config.effective_model27, "training")
    callback = _make_validation_callback(
        processor,
        collator,
        validation_records,
        output_dir,
        run_manifest_path,
        int(_nested(config.effective_model27, "generation", "max_new_tokens")),
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
        full_determinism=True,
        num_train_epochs=float(_nested(training, "epochs")),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(
            _nested(training, "gradient_accumulation")
        ),
        learning_rate=float(_nested(training, "learning_rate")),
        weight_decay=float(_nested(training, "weight_decay")),
        warmup_ratio=float(_nested(training, "warmup_ratio")),
        lr_scheduler_type=str(_nested(training, "scheduler")),
        max_grad_norm=float(_nested(training, "max_grad_norm")),
        logging_steps=int(_nested(training, "logging_steps")),
        eval_strategy="no",
        save_strategy="no",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        report_to="none",
    )
    model27.set_seed(seed)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=model27.RecordDataset(training_records),
        data_collator=collator,
        callbacks=[callback],
    )
    result = trainer.train()
    if callback.best_epoch is None:
        raise model27.SafetyGateError(
            "Model32 training ended without whole-fold generated validation"
        )
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)
    if stable_sha256(config.path) != static_audit["config_sha256"]:
        raise RuntimeError("Model32 config changed during training")
    if stable_sha256(train_csv) != static_audit["train_csv_sha256"]:
        raise RuntimeError("canonical training CSV changed during Model32 training")
    if _implementation_sha256() != static_audit["implementation_sha256"]:
        raise RuntimeError("Model32 implementation changed during training")
    current_draft = verify_draft_artifact(
        config, train_csv, draft_path, receipt_path
    )
    if current_draft.audit != draft_audit:
        raise RuntimeError("Model32 draft artifact changed during training")
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    model27.enforce_memory_limits(
        peak_allocated,
        peak_reserved,
        max_allocated_gib=float(
            _nested(config.effective_model27, "smoke", "max_allocated_gib")
        ),
        max_reserved_gib=float(
            _nested(config.effective_model27, "smoke", "max_reserved_gib")
        ),
    )
    complete = {
        "schema_version": SCHEMA_VERSION,
        "fold": fold,
        "best_epoch": callback.best_epoch,
        "best_challenger_multi_score": callback.best_score,
        "training_metrics": result.metrics,
        "final_adapter": str(final_dir),
        "final_adapter_files_sha256": _directory_ledger(final_dir),
        "best_adapter": str(output_dir / "best_adapter"),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "allocated_limit_gib_exclusive": 5.3,
        "reserved_limit_gib_exclusive": 5.3,
    }
    _atomic_json(output_dir / "complete.json", complete)
    return complete


def verify_adapter_lineage(
    config: Model32Config, adapter_path: str | Path, *, oof_fold: int | None
) -> dict[str, Any]:
    adapter = _absolute(adapter_path)
    if not _below(adapter, MODEL32_OUTPUT_ROOT) or adapter.name != "best_adapter":
        raise model27.SafetyGateError(
            "Model32 inference accepts only outputs/model32/foldN/best_adapter"
        )
    run_dir = adapter.parent
    run_manifest_path = run_dir / "run_manifest.json"
    best_path = run_dir / "best.json"
    run = _read_json(run_manifest_path)
    best = _read_json(best_path)
    training_fold = run.get("fold")
    if not isinstance(training_fold, int) or training_fold not in range(5):
        raise model27.SafetyGateError("Model32 run manifest has invalid fold")
    if run_dir != require_model32_output(
        str(_nested(config.raw, "output_dir")).format(fold=training_fold)
    ):
        raise model27.SafetyGateError("Model32 adapter is outside its configured fold run")
    if oof_fold is not None and training_fold != oof_fold:
        raise model27.SafetyGateError(
            f"fold {oof_fold} OOF inference cannot use fold {training_fold} adapter"
        )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "training_excluded_folds": [training_fold],
        "config": str(config.path),
        "config_sha256": stable_sha256(config.path),
        "implementation_sha256": _implementation_sha256(),
        "base_model27_config_sha256": MODEL27_CONFIG_SHA256,
        "base_model27_implementation_sha256": MODEL27_IMPLEMENTATION_SHA256,
    }
    for key, value in expected.items():
        if run.get(key) != value:
            raise model27.SafetyGateError(
                f"Model32 adapter run-manifest mismatch for {key}"
            )
    draft_audit = run.get("draft_artifact")
    if not isinstance(draft_audit, Mapping) or (
        draft_audit.get("kind") != DRAFT_KIND
        or draft_audit.get("label_independent") is not True
        or draft_audit.get("challenge_target_access") != "none"
    ):
        raise model27.SafetyGateError("adapter training draft lineage is not outer-safe")
    if best.get("adapter") != str(adapter):
        raise model27.SafetyGateError("best.json names a different adapter")
    if best.get("run_manifest") != str(run_manifest_path) or best.get(
        "run_manifest_sha256"
    ) != stable_sha256(run_manifest_path):
        raise model27.SafetyGateError("best adapter is detached from its run manifest")
    ledger = _directory_ledger(adapter)
    if best.get("adapter_files_sha256") != ledger:
        raise model27.SafetyGateError("best adapter file ledger mismatch")
    smoke = run.get("smoke")
    if not isinstance(smoke, Mapping):
        raise model27.SafetyGateError("adapter run lacks certified memory smoke")
    allocated = smoke.get("peak_allocated_bytes")
    reserved = smoke.get("peak_reserved_bytes")
    if not isinstance(allocated, int) or not isinstance(reserved, int):
        raise model27.SafetyGateError("adapter memory-smoke peaks are invalid")
    model27.enforce_memory_limits(
        allocated,
        reserved,
        max_allocated_gib=5.3,
        max_reserved_gib=5.3,
    )
    return {
        "adapter": str(adapter),
        "adapter_files_sha256": ledger,
        "training_fold": training_fold,
        "training_excluded_folds": [training_fold],
        "run_manifest": str(run_manifest_path),
        "run_manifest_sha256": stable_sha256(run_manifest_path),
        "best": str(best_path),
        "best_sha256": stable_sha256(best_path),
        "training_draft_artifact": draft_audit,
        "memory_smoke": {
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
            "limit_gib_exclusive": 5.3,
        },
    }


def infer(
    config_path: str | Path,
    input_csv: str | Path,
    draft_csv: str | Path,
    draft_receipt: str | Path,
    adapter_path: str | Path,
    output_csv: str | Path,
    *,
    fold: int | None = None,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate the challenger and a separate return-the-draft baseline."""

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    model27.enable_offline_mode()
    config = load_config(config_path)
    initial_config_sha256 = stable_sha256(config.path)
    initial_implementation_sha256 = _implementation_sha256()
    if fold is not None and fold not in range(5):
        raise ValueError("fold must be one of 0,1,2,3,4")
    input_path = _absolute(input_csv)
    _validate_canonical_input(config, input_path, fold=fold)
    role = "train" if fold is not None else "test"
    expected_draft, expected_receipt = _configured_draft_paths(config, role)
    if _absolute(draft_csv) != expected_draft or _absolute(draft_receipt) != expected_receipt:
        raise model27.SafetyGateError(
            f"Model32 {role} inference requires its exact configured draft artifact"
        )
    records, inference_draft_audit = load_guided_records(
        config,
        input_path,
        draft_csv,
        draft_receipt,
        require_target=False,
        require_five_folds=False,
    )
    if fold is not None:
        if not records or records[0].fold is None:
            raise SchemaError("--fold requires the canonical fold column")
        records = tuple(record for record in records if record.fold == fold)
    if not records:
        raise ValueError("Model32 inference selection is empty")
    output = require_model32_output(output_csv)
    inference_root = MODEL32_OUTPUT_ROOT / ("oof" if fold is not None else "test")
    if not _below(output, inference_root):
        raise model27.SafetyGateError(
            f"Model32 inference output must remain under {inference_root}"
        )
    draft_output = output.with_suffix(".draft.csv")
    report_path = output.with_suffix(".metrics.json")
    for path in (output, draft_output, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Model32 inference: {path}")
    submission: Path | None = None
    if submission_path is not None:
        if fold is not None:
            raise ValueError("a held-out-fold inference cannot create a test submission")
        if sample_path is None:
            raise ValueError("--sample is required with --submission")
        sample = _absolute(sample_path)
        expected_sample = _absolute(_nested(config.raw, "data", "sample_submission"))
        if sample != expected_sample or not sample.is_file():
            raise model27.SafetyGateError(
                "Model32 submission requires the pinned SampleSubmission.csv"
            )
        if stable_sha256(sample) != SAMPLE_SUBMISSION_SHA256:
            raise model27.SafetyGateError("pinned SampleSubmission.csv changed")
        submission = require_model32_submission(submission_path)
        if submission.exists():
            raise FileExistsError(f"refusing to overwrite Model32 submission: {submission}")
    elif sample_path is not None:
        raise ValueError("--sample is valid only with --submission")
    adapter_audit = verify_adapter_lineage(config, adapter_path, oof_fold=fold)
    checkpoint_audit = model27.verify_local_checkpoint(config.effective_model27)
    model27.verify_starter_provenance(config.effective_model27)

    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    processor = model27.load_processor(config.effective_model27)
    model, targets = model27._load_inference_model(
        config.effective_model27, _absolute(adapter_path)
    )
    collator = QwenGuidedCollator(
        processor,
        str(_nested(config.raw, "prompt_template")),
        int(_nested(config.effective_model27, "image", "max_image_tokens")),
        int(_nested(config.effective_model27, "image", "max_sequence_tokens")),
    )
    started = time.monotonic()
    predictions = model27.greedy_predict(
        model,
        processor,
        collator,
        records,
        max_new_tokens=int(
            _nested(config.effective_model27, "generation", "max_new_tokens")
        ),
    )
    if stable_sha256(config.path) != initial_config_sha256:
        raise RuntimeError("Model32 config changed during inference")
    if _implementation_sha256() != initial_implementation_sha256:
        raise RuntimeError("Model32 implementation changed during inference")
    current_draft = verify_draft_artifact(
        config, input_path, draft_csv, draft_receipt
    )
    if current_draft.audit != inference_draft_audit:
        raise RuntimeError("Model32 inference draft changed during generation")
    if verify_adapter_lineage(config, adapter_path, oof_fold=fold) != adapter_audit:
        raise RuntimeError("Model32 adapter lineage changed during generation")
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    model27.enforce_memory_limits(
        peak_allocated,
        peak_reserved,
        max_allocated_gib=5.3,
        max_reserved_gib=5.3,
    )
    _atomic_predictions(output, records, predictions)
    _atomic_predictions(draft_output, records, [record.draft for record in records])
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rows": len(records),
        "fold": fold,
        "config": str(config.path),
        "config_sha256": initial_config_sha256,
        "implementation_sha256": initial_implementation_sha256,
        "input_csv": str(input_path),
        "input_sha256": stable_sha256(input_path),
        "inference_draft_artifact": inference_draft_audit,
        "adapter_lineage": adapter_audit,
        "local_checkpoint": checkpoint_audit,
        "output": str(output),
        "output_sha256": stable_sha256(output),
        "returning_draft_output": str(draft_output),
        "returning_draft_output_sha256": stable_sha256(draft_output),
        "target_modules": list(targets),
        "target_module_count": len(targets),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "max_new_tokens": int(
                _nested(config.effective_model27, "generation", "max_new_tokens")
            ),
        },
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "allocated_limit_gib_exclusive": 5.3,
            "reserved_limit_gib_exclusive": 5.3,
        },
        "seconds": time.monotonic() - started,
        "per_row_selection_performed": False,
    }
    if all(record.label is not None for record in records):
        report["comparison"] = compare_fold_to_draft(records, predictions)
    _atomic_json(report_path, report)
    if submission is not None:
        assert sample_path is not None
        by_id = {
            record.identifier: prediction
            for record, prediction in zip(records, predictions, strict=True)
        }
        write_submission(
            _absolute(sample_path),
            submission,
            by_id,
            cleaner=clean_prediction_preserving_internal_spaces,
        )
    return report


def train_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Guarded OCR-guided Model32 training")
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke_only and arguments.dry_run:
        parser.error("--smoke-only and --dry-run are mutually exclusive")
    result = train(
        arguments.config,
        arguments.fold,
        smoke_only=arguments.smoke_only,
        dry_run=arguments.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def infer_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Greedy OCR-guided Model32 inference")
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--draft-receipt", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    arguments = parser.parse_args()
    result = infer(
        arguments.config,
        arguments.input,
        arguments.drafts,
        arguments.draft_receipt,
        arguments.adapter,
        arguments.output,
        fold=arguments.fold,
        sample_path=arguments.sample,
        submission_path=arguments.submission,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def prepare_drafts_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="CPU-only label-independent Model8 draft generation for Model32"
    )
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    result = prepare_drafts(
        arguments.config,
        arguments.input,
        arguments.output,
        dry_run=arguments.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
