"""Model41: Model39's broad QLoRA corrected by Model32's frozen OCR draft.

The module deliberately composes two already audited components without
modifying either one.  CPU structural audits tolerate drafts that have not yet
been generated and report their exact missing paths.  Every operation that
could load a CUDA model requires the complete Model32 receipt and fails closed.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from . import qwen2_vl_broad as broad
from . import qwen2_vl_challenger as challenger
from . import qwen2_vl_ocr_guided as guided
from .submission import clean_prediction_preserving_internal_spaces, write_submission


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL41_OUTPUT_ROOT = (PROJECT_ROOT / "outputs" / "model41").resolve()
MODEL41_SUBMISSION_ROOT = (PROJECT_ROOT / "submissions" / "model41").resolve()

SCHEMA_VERSION = 1
MODEL39_CONFIG_SHA256 = (
    "4e8d28cdbeccd719a76a989a3de29fefbd3d72f9f444c76ec3ec6ba729479ef5"
)
MODEL39_IMPLEMENTATION_SHA256 = (
    "ab8d9b77154bd3c0ef8470b07310c85e6408ca4a98f72b19a7baba478da1a606"
)
MODEL32_CONFIG_SHA256 = (
    "bedf6c4cc4401190e68f4b6094b1b1fdd431a253e21f2dbecbab98a9e3f416e1"
)
MODEL32_IMPLEMENTATION_SHA256 = (
    "015c5b0a8484aa14bf4ea92c5c3890deeefef4d13e1e7d4c7602d07dce78ed68"
)
CANONICAL_TRAIN_SHA256 = guided.CANONICAL_TRAIN_SHA256
CANONICAL_TEST_SHA256 = guided.CANONICAL_TEST_SHA256
SAMPLE_SUBMISSION_SHA256 = guided.SAMPLE_SUBMISSION_SHA256
MODEL8_CHECKPOINT_SHA256 = guided.MODEL8_CHECKPOINT_SHA256


@dataclass(frozen=True)
class Model41Config:
    raw: Mapping[str, Any]
    effective_model39: Mapping[str, Any]
    path: Path
    base_config_path: Path
    base_implementation_path: Path
    draft_config_path: Path
    draft_implementation_path: Path
    draft_config: guided.Model32Config


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"missing required Model41 config key: {'.'.join(keys)}")
        current = current[key]
    return current


def _below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def require_model41_output(path: str | Path) -> Path:
    resolved = _absolute(path)
    if not _below(resolved, MODEL41_OUTPUT_ROOT):
        raise challenger.SafetyGateError(
            f"Model41 output must remain under {MODEL41_OUTPUT_ROOT}: {resolved}"
        )
    return resolved


def require_model41_submission(path: str | Path) -> Path:
    resolved = _absolute(path)
    if not _below(resolved, MODEL41_SUBMISSION_ROOT):
        raise challenger.SafetyGateError(
            f"Model41 submission must remain under {MODEL41_SUBMISSION_ROOT}: {resolved}"
        )
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return value


def _atomic_json(path: str | Path, value: Any) -> Path:
    output = require_model41_output(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _directory_ledger(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    ledger: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise challenger.SafetyGateError(f"adapter artifact may not be a symlink: {path}")
        if path.is_file():
            ledger[path.relative_to(directory).as_posix()] = guided.stable_sha256(path)
    if not ledger:
        raise challenger.SafetyGateError(f"adapter directory is empty: {directory}")
    return ledger


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "model41" / "train.py",
        PROJECT_ROOT / "model41" / "infer.py",
        PROJECT_ROOT / "model41" / "audit.py",
        PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_broad.py",
        PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_ocr_guided.py",
        PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_challenger.py",
        PROJECT_ROOT / "src" / "road_htr" / "metrics.py",
        PROJECT_ROOT / "src" / "road_htr" / "submission.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(guided.stable_sha256(path)))
    return digest.hexdigest()


def load_config(path: str | Path) -> Model41Config:
    """Load Model41 and prove both inherited components are immutable."""

    resolved = _absolute(path)
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read Model41 configuration {resolved}: {error}") from error
    if not isinstance(raw, Mapping) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Model41 config must be a schema-version-1 mapping")

    base_config_path = _absolute(_nested(raw, "base_model39", "config"))
    base_implementation_path = _absolute(
        _nested(raw, "base_model39", "implementation")
    )
    draft_config_path = _absolute(_nested(raw, "base_model32_drafts", "config"))
    draft_implementation_path = _absolute(
        _nested(raw, "base_model32_drafts", "implementation")
    )
    exact_paths = {
        base_config_path: (PROJECT_ROOT / "model39" / "config.yaml").resolve(),
        base_implementation_path: (
            PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_broad.py"
        ).resolve(),
        draft_config_path: (PROJECT_ROOT / "model32" / "config.yaml").resolve(),
        draft_implementation_path: (
            PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_ocr_guided.py"
        ).resolve(),
    }
    for actual, expected in exact_paths.items():
        if actual != expected:
            raise ValueError(f"Model41 pinned dependency path mismatch: {actual} != {expected}")
        if not actual.is_file():
            raise FileNotFoundError(actual)

    configured_pins = {
        ("base_model39", "config_sha256"): MODEL39_CONFIG_SHA256,
        ("base_model39", "implementation_sha256"): MODEL39_IMPLEMENTATION_SHA256,
        ("base_model32_drafts", "config_sha256"): MODEL32_CONFIG_SHA256,
        ("base_model32_drafts", "implementation_sha256"): MODEL32_IMPLEMENTATION_SHA256,
        ("data", "train_csv_sha256"): CANONICAL_TRAIN_SHA256,
        ("data", "test_csv_sha256"): CANONICAL_TEST_SHA256,
        ("data", "sample_submission_sha256"): SAMPLE_SUBMISSION_SHA256,
        ("draft_source", "kind"): guided.DRAFT_KIND,
        ("draft_source", "checkpoint_sha256"): MODEL8_CHECKPOINT_SHA256,
        ("draft_source", "receipt_schema_version"):
            guided.DRAFT_RECEIPT_SCHEMA_VERSION,
        ("draft_source", "empty_draft_policy"): guided.EMPTY_DRAFT_POLICY,
    }
    for keys, expected in configured_pins.items():
        actual = _nested(raw, *keys)
        if actual != expected:
            raise ValueError(
                f"Model41 pin {'.'.join(keys)} must be {expected!r}, got {actual!r}"
            )
    dependency_hashes = {
        base_config_path: MODEL39_CONFIG_SHA256,
        base_implementation_path: MODEL39_IMPLEMENTATION_SHA256,
        draft_config_path: MODEL32_CONFIG_SHA256,
        draft_implementation_path: MODEL32_IMPLEMENTATION_SHA256,
    }
    for source, expected in dependency_hashes.items():
        if guided.stable_sha256(source) != expected:
            raise challenger.SafetyGateError(f"pinned Model41 dependency changed: {source}")

    base, _ = broad.load_config(base_config_path)
    draft_config = guided.load_config(draft_config_path)
    effective = copy.deepcopy(base)
    effective["train_csv"] = str(_nested(raw, "data", "train_csv"))
    effective["image_dir"] = str(_nested(raw, "data", "image_dir"))
    effective["output_dir"] = str(_nested(raw, "output_dir"))
    effective["prompt"] = str(_nested(raw, "prompt_template"))
    effective["smoke"]["report_dir"] = str(_nested(raw, "smoke_report_dir"))
    broad.validate_config(effective)

    prompt = str(_nested(raw, "prompt_template"))
    if prompt.count("{draft}") != 1:
        raise ValueError("prompt_template must contain exactly one literal {draft}")
    lowered_prompt = prompt.casefold()
    if any(marker in lowered_prompt for marker in ("{label}", "{target}")):
        raise ValueError("prompt_template may not interpolate a challenge label")
    exact_strings = {
        ("data", "train_csv"): "data/processed/train_folds.csv",
        ("data", "test_csv"): "data/raw/Test.csv",
        ("data", "sample_submission"): "data/raw/SampleSubmission.csv",
        ("data", "image_dir"): "data/raw/images",
        ("output_dir",): "outputs/model41/fold{fold}",
        ("smoke_report_dir",): "outputs/model41/smoke",
        ("draft_source", "checkpoint"): "checkpoints/model8/medium.safetensors",
        ("draft_source", "train_csv"): "outputs/model32/drafts/train_model8_raw.csv",
        ("draft_source", "train_receipt"): (
            "outputs/model32/drafts/train_model8_raw.receipt.json"
        ),
        ("draft_source", "test_csv"): "outputs/model32/drafts/test_model8_raw.csv",
        ("draft_source", "test_receipt"): (
            "outputs/model32/drafts/test_model8_raw.receipt.json"
        ),
    }
    for keys, expected in exact_strings.items():
        if _nested(raw, *keys) != expected:
            raise ValueError(f"Model41 {'.'.join(keys)} must remain {expected!r}")

    expected_cv = {
        "folds": 5,
        "expected_total_rows": 4095,
        "expected_train_rows_per_fold": 3276,
        "expected_validation_rows_per_fold": 819,
        "selection_scope": "whole_fold_challenger_only",
        "forbid_per_row_oracle": True,
    }
    if _nested(raw, "cross_validation") != expected_cv:
        raise ValueError("Model41 cross-validation contract may not be weakened")
    expected_sequence = {
        "require_every_selected_row": True,
        "max_image_tokens": 320,
        "max_sequence_tokens": 512,
    }
    if _nested(raw, "sequence_preflight") != expected_sequence:
        raise ValueError("Model41 exact sequence preflight may not be weakened")
    expected_safety = {
        "offline": True,
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "double_quant": True,
        "expected_lora_targets": broad.EXPECTED_TARGET_MODULES,
        "expected_trainable_parameters": broad.EXPECTED_TRAINABLE_PARAMETERS,
        "expected_trainable_tensors": broad.EXPECTED_TRAINABLE_TENSORS,
        "expected_updates_per_epoch": 205,
        "expected_total_updates": 615,
        "max_new_tokens": 48,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 4,
        "max_allocated_gib_exclusive": 5.3,
        "max_reserved_gib_exclusive": 5.3,
        "preserve_internal_spaces": True,
    }
    if _nested(raw, "safety_assertions") != expected_safety:
        raise ValueError("Model41 safety assertions may not be weakened")
    inherited = {
        ("image", "max_image_tokens"): 320,
        ("image", "max_sequence_tokens"): 512,
        ("training", "epochs"): 3,
        ("training", "batch_size"): 1,
        ("training", "gradient_accumulation"): 16,
        ("training", "expected_updates_per_epoch"): 205,
        ("training", "expected_total_updates"): 615,
        ("generation", "max_new_tokens"): 48,
        ("generation", "do_sample"): False,
        ("generation", "num_beams"): 1,
        ("generation", "repetition_penalty"): 1.05,
        ("generation", "no_repeat_ngram_size"): 4,
    }
    for keys, expected in inherited.items():
        if challenger._nested(effective, *keys) != expected:
            raise challenger.SafetyGateError(
                f"inherited Model39 contract drifted at {'.'.join(keys)}"
            )
    return Model41Config(
        raw=raw,
        effective_model39=effective,
        path=resolved,
        base_config_path=base_config_path,
        base_implementation_path=base_implementation_path,
        draft_config_path=draft_config_path,
        draft_implementation_path=draft_implementation_path,
        draft_config=draft_config,
    )


def _canonical_path(config: Model41Config, role: str) -> Path:
    if role not in {"train", "test", "sample"}:
        raise ValueError(f"unknown canonical role: {role}")
    key = {"train": "train_csv", "test": "test_csv", "sample": "sample_submission"}[role]
    return _absolute(_nested(config.raw, "data", key))


def _verify_canonical_data(config: Model41Config) -> dict[str, Any]:
    expected = {
        "train": CANONICAL_TRAIN_SHA256,
        "test": CANONICAL_TEST_SHA256,
        "sample": SAMPLE_SUBMISSION_SHA256,
    }
    report: dict[str, Any] = {}
    for role, digest in expected.items():
        path = _canonical_path(config, role)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = guided.stable_sha256(path)
        if actual != digest:
            raise challenger.SafetyGateError(f"canonical {role} data changed: {path}")
        report[role] = {"path": str(path), "sha256": actual}
    checkpoint = _absolute(_nested(config.raw, "draft_source", "checkpoint"))
    if not checkpoint.is_file() or guided.stable_sha256(checkpoint) != MODEL8_CHECKPOINT_SHA256:
        raise challenger.SafetyGateError("pinned Model8 draft checkpoint changed or is absent")
    report["draft_checkpoint"] = {
        "path": str(checkpoint),
        "sha256": MODEL8_CHECKPOINT_SHA256,
    }
    return report


def _draft_paths(config: Model41Config, role: str) -> tuple[Path, Path]:
    if role not in {"train", "test"}:
        raise ValueError("draft role must be train or test")
    return (
        _absolute(_nested(config.raw, "draft_source", f"{role}_csv")),
        _absolute(_nested(config.raw, "draft_source", f"{role}_receipt")),
    )


def inspect_draft_readiness(config: Model41Config, role: str) -> dict[str, Any]:
    """Report missing draft paths, fully verifying a complete artifact if present."""

    input_path = _canonical_path(config, role)
    draft_path, receipt_path = _draft_paths(config, role)
    required = (
        draft_path,
        draft_path.with_suffix(".ids.csv"),
        draft_path.with_suffix(".jsonl"),
        draft_path.with_suffix(".manifest.json"),
        receipt_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "role": role,
            "ready": False,
            "status": "missing_or_incomplete",
            "missing": missing,
            "present": [str(path) for path in required if path.is_file()],
            "required": [str(path) for path in required],
        }
    artifact = guided.verify_draft_artifact(
        config.draft_config, input_path, draft_path, receipt_path
    )
    return {
        "role": role,
        "ready": True,
        "status": "receipt_verified",
        "missing": [],
        "required": [str(path) for path in required],
        "audit": artifact.audit,
    }


def require_draft_artifact(
    config: Model41Config, role: str
) -> tuple[guided.DraftArtifact, dict[str, Any]]:
    readiness = inspect_draft_readiness(config, role)
    if readiness["ready"] is not True:
        raise challenger.SafetyGateError(
            f"Model41 {role} draft receipt is not ready; missing={readiness['missing']}"
        )
    input_path = _canonical_path(config, role)
    draft_path, receipt_path = _draft_paths(config, role)
    artifact = guided.verify_draft_artifact(
        config.draft_config, input_path, draft_path, receipt_path
    )
    return artifact, readiness


def _ids_sha256(records: Sequence[Any]) -> str:
    return _json_sha256([record.identifier for record in records])


def _validate_fold_counts(records: Sequence[Any]) -> None:
    if len(records) != 4095:
        raise challenger.SafetyGateError(f"Model41 requires 4,095 rows, got {len(records)}")
    for fold in range(5):
        training, validation = broad.split_outer_fold(records, fold)
        if len(training) != 3276 or len(validation) != 819:
            raise challenger.SafetyGateError(
                f"fold {fold} must contain 3,276 train and 819 validation rows"
            )


def preflight_sequence_budgets(
    collator: guided.QwenGuidedCollator,
    groups: Sequence[tuple[str, Sequence[guided.GuidedOcrRecord], bool]],
) -> dict[str, Any]:
    """Encode every selected row and record exact maxima before CUDA loading."""

    maxima: dict[str, dict[str, Any]] = {}
    total = 0
    group_reports: dict[str, dict[str, Any]] = {}
    for role, records, include_label in groups:
        if role in group_reports:
            raise ValueError(f"duplicate sequence-preflight role: {role}")
        if not records:
            raise ValueError(f"empty sequence-preflight role: {role}")
        role_maxima: dict[str, dict[str, Any]] = {}
        for record in records:
            _, diagnostics = (
                collator.encode_training(record)
                if include_label
                else collator.encode_prompt(record)
            )
            total += 1
            for key in ("image_tokens", "sequence_tokens"):
                value = diagnostics.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise challenger.SafetyGateError(
                        f"invalid {key} diagnostic for {record.identifier!r}: {value!r}"
                    )
                previous = maxima.get(key)
                candidate = {
                    "value": value,
                    "record_id": record.identifier,
                    "role": role,
                }
                if previous is None or value > int(previous["value"]):
                    maxima[key] = candidate
                role_previous = role_maxima.get(key)
                if role_previous is None or value > int(role_previous["value"]):
                    role_maxima[key] = candidate
            if include_label:
                supervised = diagnostics.get("supervised_tokens")
                if isinstance(supervised, bool) or not isinstance(supervised, int) or supervised <= 0:
                    raise challenger.SafetyGateError(
                        f"invalid supervised-token count for {record.identifier!r}"
                    )
                previous = maxima.get("supervised_tokens")
                candidate = {
                    "value": supervised,
                    "record_id": record.identifier,
                    "role": role,
                }
                if previous is None or supervised > int(previous["value"]):
                    maxima["supervised_tokens"] = candidate
                role_previous = role_maxima.get("supervised_tokens")
                if role_previous is None or supervised > int(role_previous["value"]):
                    role_maxima["supervised_tokens"] = candidate
        group_reports[role] = {"rows": len(records), "maxima": role_maxima}
    if int(maxima["image_tokens"]["value"]) > 320:
        raise challenger.SafetyGateError("exact preflight exceeded 320 image tokens")
    if int(maxima["sequence_tokens"]["value"]) > 512:
        raise challenger.SafetyGateError("exact preflight exceeded 512 sequence tokens")
    return {
        "status": "all_selected_rows_encoded_under_caps",
        "rows": total,
        "groups": group_reports,
        "maxima": maxima,
        "caps": {"image_tokens": 320, "sequence_tokens": 512},
    }


def select_exact_smoke_record(
    records: Sequence[guided.GuidedOcrRecord],
    sequence_preflight: Mapping[str, Any],
    *,
    role: str = "outer_train_with_target",
) -> guided.GuidedOcrRecord:
    """Select the row with the exact largest processor-emitted training sequence."""

    groups = sequence_preflight.get("groups")
    if not isinstance(groups, Mapping) or role not in groups:
        raise challenger.SafetyGateError(f"sequence preflight lacks smoke role {role!r}")
    group = groups[role]
    if not isinstance(group, Mapping):
        raise challenger.SafetyGateError("sequence preflight group is malformed")
    maxima = group.get("maxima")
    sequence = maxima.get("sequence_tokens") if isinstance(maxima, Mapping) else None
    identifier = sequence.get("record_id") if isinstance(sequence, Mapping) else None
    matches = [record for record in records if record.identifier == identifier]
    if len(matches) != 1 or matches[0].label is None:
        raise challenger.SafetyGateError(
            "exact maximum-sequence smoke row is missing, duplicated, or unlabelled"
        )
    return matches[0]


def _atomic_predictions(
    path: str | Path,
    records: Sequence[guided.GuidedOcrRecord],
    predictions: Sequence[str],
) -> Path:
    if len(records) != len(predictions):
        raise ValueError("record/prediction count mismatch")
    output = require_model41_output(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model41 prediction: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    include_fold = any(record.fold is not None for record in records)
    fieldnames = ["ID", "prediction"] + (["fold"] if include_fold else [])
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record, raw in zip(records, predictions, strict=True):
                row: dict[str, Any] = {
                    "ID": record.identifier,
                    "prediction": challenger.trim_label_edges(raw),
                }
                if include_fold:
                    row["fold"] = "" if record.fold is None else record.fold
                writer.writerow(row)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _make_validation_callback(
    processor: Any,
    collator: guided.QwenGuidedCollator,
    validation_records: Sequence[guided.GuidedOcrRecord],
    output_dir: Path,
    run_manifest_path: Path,
    config: Model41Config,
) -> Any:
    from transformers import TrainerCallback

    max_new_tokens = int(_nested(config.effective_model39, "generation", "max_new_tokens"))
    repetition_penalty = float(
        _nested(config.effective_model39, "generation", "repetition_penalty")
    )
    no_repeat_ngram_size = int(
        _nested(config.effective_model39, "generation", "no_repeat_ngram_size")
    )

    class GuidedBroadExactMetricCallback(TrainerCallback):
        def __init__(self) -> None:
            self.best_score = -math.inf
            self.best_epoch: int | None = None

        def on_epoch_end(
            self, args: Any, state: Any, control: Any, model: Any, **_: Any
        ) -> Any:
            if not state.is_world_process_zero:
                return control
            epoch = int(round(float(state.epoch or 0)))
            predictions = challenger.greedy_predict(
                model,
                processor,
                collator,
                validation_records,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            comparison = guided.compare_fold_to_draft(validation_records, predictions)
            prediction_path = _atomic_predictions(
                output_dir / "validation" / f"epoch_{epoch:03d}.csv",
                validation_records,
                predictions,
            )
            payload = {
                "schema_version": SCHEMA_VERSION,
                "epoch": epoch,
                "rows": len(validation_records),
                "generation": {
                    "do_sample": False,
                    "num_beams": 1,
                    "num_return_sequences": 1,
                    "max_new_tokens": max_new_tokens,
                    "repetition_penalty": repetition_penalty,
                    "no_repeat_ngram_size": no_repeat_ngram_size,
                },
                "comparison": comparison,
                "predictions": str(prediction_path),
                "predictions_sha256": guided.stable_sha256(prediction_path),
                "no_per_row_oracle": True,
            }
            _atomic_json(output_dir / "validation" / f"epoch_{epoch:03d}.json", payload)
            score = float(comparison["challenger"]["multi_score"])
            if score > self.best_score + 1e-12:
                self.best_score = score
                self.best_epoch = epoch
                best_dir = output_dir / "best_adapter"
                model.save_pretrained(best_dir, safe_serialization=True)
                processor.save_pretrained(best_dir)
                _atomic_json(
                    output_dir / "best.json",
                    {
                        **payload,
                        "adapter": str(best_dir),
                        "adapter_files_sha256": _directory_ledger(best_dir),
                        "run_manifest": str(run_manifest_path),
                        "run_manifest_sha256": guided.stable_sha256(run_manifest_path),
                        "selection": (
                            "maximum complete-fold challenger multi_score; raw draft "
                            "is report-only and no row oracle is used"
                        ),
                    },
                )
            print(
                f"Model41 fold validation epoch={epoch} challenger={score:.9f} "
                f"draft={comparison['returning_draft']['multi_score']:.9f} "
                f"best_epoch={self.best_epoch}",
                flush=True,
            )
            return control

    return GuidedBroadExactMetricCallback()


def dry_run(
    config_path: str | Path,
    output_path: str | Path,
    *,
    fold: int | None = None,
) -> dict[str, Any]:
    """Run a CUDA-free audit, treating absent draft receipts as an explicit blocker."""

    challenger.enable_offline_mode()
    config = load_config(config_path)
    data = _verify_canonical_data(config)
    provenance = challenger.verify_starter_provenance(config.effective_model39)
    checkpoint = challenger.verify_local_checkpoint(config.effective_model39)
    targets = broad.audit_checkpoint_target_shapes(config.effective_model39)
    records = challenger.load_records(
        _canonical_path(config, "train"),
        _absolute(_nested(config.raw, "data", "image_dir")),
        require_target=True,
        require_five_folds=True,
    )
    _validate_fold_counts(records)
    folds = range(5) if fold is None else (fold,)
    if fold is not None and fold not in range(5):
        raise ValueError("fold must be one of 0,1,2,3,4")
    fold_rows: dict[str, dict[str, int]] = {}
    schedules: dict[str, dict[str, Any]] = {}
    for current_fold in folds:
        training, validation = broad.split_outer_fold(records, current_fold)
        schedule = broad.deterministic_optimizer_schedule(
            len(training),
            int(_nested(config.effective_model39, "training", "gradient_accumulation")),
            int(_nested(config.effective_model39, "training", "epochs")),
        )
        broad.verify_configured_schedule(config.effective_model39, schedule)
        fold_rows[str(current_fold)] = {
            "train": len(training),
            "validation": len(validation),
        }
        schedules[str(current_fold)] = asdict(schedule)
    drafts = {
        role: inspect_draft_readiness(config, role) for role in ("train", "test")
    }
    if drafts["train"]["ready"] and drafts["test"]["ready"]:
        status = "draft_receipts_verified_gpu_smoke_not_run"
    elif drafts["train"]["ready"]:
        status = "train_ready_test_draft_missing_gpu_smoke_not_run"
    else:
        status = "structural_audit_passed_draft_receipts_missing"
    report = {
        "schema_version": SCHEMA_VERSION,
        "model": "model41_broad_qwen2_vl_ocr_guided",
        "status": status,
        "gpu_started": False,
        "resolved_config": str(config.path),
        "config_sha256": guided.stable_sha256(config.path),
        "implementation_sha256": _implementation_sha256(),
        "dependency_hashes": {
            "model39_config": MODEL39_CONFIG_SHA256,
            "model39_implementation": MODEL39_IMPLEMENTATION_SHA256,
            "model32_config": MODEL32_CONFIG_SHA256,
            "model32_implementation": MODEL32_IMPLEMENTATION_SHA256,
        },
        "canonical_data": data,
        "starter_provenance": provenance,
        "local_checkpoint": checkpoint,
        "targets": asdict(targets),
        "targeted_base_parameters": sum(
            inputs * outputs for inputs, outputs in targets.shapes.values()
        ),
        "fold_rows": fold_rows,
        "optimizer_schedules": schedules,
        "drafts": drafts,
        "geometry": {
            "max_image_tokens": 320,
            "max_sequence_tokens": 512,
            "exact_sequence_preflight": "pending verified draft; mandatory before CUDA model load",
        },
        "memory_gate": {
            "allocated_gib_exclusive": 5.3,
            "reserved_gib_exclusive": 5.3,
            "status": "gpu_smoke_not_run",
        },
        "leakage_contract": (
            "frozen Model8 receives ordered IDs and image pixels only; every outer "
            "validation ID is absent from its Qwen training partition"
        ),
    }
    _atomic_json(output_path, report)
    return report


def train(
    config_path: str | Path,
    fold: int,
    *,
    smoke_only: bool = False,
    dry_run_only: bool = False,
) -> dict[str, Any]:
    if fold not in range(5):
        raise ValueError("fold must be one of 0,1,2,3,4")
    if smoke_only and dry_run_only:
        raise ValueError("--smoke-only and --dry-run are mutually exclusive")
    if dry_run_only:
        return dry_run(
            config_path,
            MODEL41_OUTPUT_ROOT / f"dry_run_fold{fold}.json",
            fold=fold,
        )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    challenger.enable_offline_mode()
    config = load_config(config_path)
    initial_config_sha256 = guided.stable_sha256(config.path)
    initial_implementation_sha256 = _implementation_sha256()
    data_audit = _verify_canonical_data(config)
    _, draft_readiness = require_draft_artifact(config, "train")
    draft_path, receipt_path = _draft_paths(config, "train")
    records, draft_audit = guided.load_guided_records(
        config.draft_config,
        _canonical_path(config, "train"),
        draft_path,
        receipt_path,
        require_target=True,
        require_five_folds=True,
    )
    _validate_fold_counts(records)
    training_records, validation_records = broad.split_outer_fold(records, fold)
    schedule = broad.deterministic_optimizer_schedule(
        len(training_records),
        int(_nested(config.effective_model39, "training", "gradient_accumulation")),
        int(_nested(config.effective_model39, "training", "epochs")),
    )
    broad.verify_configured_schedule(config.effective_model39, schedule)
    output_dir = require_model41_output(
        str(_nested(config.raw, "output_dir")).format(fold=fold)
    )
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {output_dir}")
    smoke_path = require_model41_output(
        _absolute(_nested(config.raw, "smoke_report_dir")) / f"fold{fold}.json"
    )
    if smoke_only and smoke_path.exists():
        raise FileExistsError(f"refusing to overwrite Model41 smoke report: {smoke_path}")

    provenance = challenger.verify_starter_provenance(config.effective_model39)
    checkpoint = challenger.verify_local_checkpoint(config.effective_model39)
    header_audit = broad.audit_checkpoint_target_shapes(config.effective_model39)
    processor = challenger.load_processor(config.effective_model39)
    collator = guided.QwenGuidedCollator(
        processor,
        str(_nested(config.raw, "prompt_template")),
        int(_nested(config.effective_model39, "image", "max_image_tokens")),
        int(_nested(config.effective_model39, "image", "max_sequence_tokens")),
    )
    sequence_preflight = preflight_sequence_budgets(
        collator,
        (
            ("outer_train_with_target", training_records, True),
            ("outer_validation_prompt", validation_records, False),
        ),
    )

    seed = int(_nested(config.effective_model39, "seed"))
    challenger.set_seed(seed)
    model, model_audit = broad.load_training_model(config.effective_model39)
    smoke_record = select_exact_smoke_record(
        training_records,
        sequence_preflight,
    )
    smoke = challenger.run_memory_smoke(
        model,
        collator,
        smoke_record,
        max_allocated_gib=float(_nested(config.effective_model39, "smoke", "max_allocated_gib")),
        max_reserved_gib=float(_nested(config.effective_model39, "smoke", "max_reserved_gib")),
    )
    runtime_audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": "model41_broad_qwen2_vl_ocr_guided",
        "model_number": 41,
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
        "base_model39_config_sha256": MODEL39_CONFIG_SHA256,
        "base_model39_implementation_sha256": MODEL39_IMPLEMENTATION_SHA256,
        "base_model32_config_sha256": MODEL32_CONFIG_SHA256,
        "base_model32_implementation_sha256": MODEL32_IMPLEMENTATION_SHA256,
        "canonical_data": data_audit,
        "draft_artifact": draft_audit,
        "draft_readiness": draft_readiness,
        "draft_policy": {
            "label_independent": True,
            "outer_fold_safe": True,
            "label_fitted_postprocessing_allowed": False,
            "per_row_oracle_allowed": False,
        },
        "starter_provenance": provenance,
        "local_checkpoint": checkpoint,
        "checkpoint_target_audit": asdict(header_audit),
        "optimizer_schedule": asdict(schedule),
        "sequence_preflight": sequence_preflight,
        "model_audit": model_audit,
        "smoke": smoke,
        "output_dir": str(output_dir),
        "gpu_started": True,
    }
    if smoke_only:
        _atomic_json(smoke_path, runtime_audit)
        return runtime_audit

    import torch
    from transformers import Trainer, TrainingArguments

    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest = {
        **challenger._runtime_manifest(config.effective_model39),
        **runtime_audit,
        "resolved_model41_config": str(config.path),
        "effective_model39_config": config.effective_model39,
        "prompt_template": str(_nested(config.raw, "prompt_template")),
    }
    _atomic_json(run_manifest_path, run_manifest)
    _atomic_predictions(
        output_dir / "validation" / "returning_draft.csv",
        validation_records,
        [record.draft for record in validation_records],
    )

    class ExactEpochTrainer(broad.ExactEpochScheduleMixin, Trainer):
        pass

    training_config = _nested(config.effective_model39, "training")
    schedule_callback = broad.make_schedule_guard_callback(schedule)
    metric_callback = _make_validation_callback(
        processor,
        collator,
        validation_records,
        output_dir,
        run_manifest_path,
        config,
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
        full_determinism=True,
        num_train_epochs=float(_nested(training_config, "epochs")),
        max_steps=schedule.total_optimizer_updates,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(_nested(training_config, "gradient_accumulation")),
        learning_rate=float(_nested(training_config, "learning_rate")),
        weight_decay=float(_nested(training_config, "weight_decay")),
        warmup_ratio=float(_nested(training_config, "warmup_ratio")),
        lr_scheduler_type=str(_nested(training_config, "scheduler")),
        max_grad_norm=float(_nested(training_config, "max_grad_norm")),
        logging_steps=int(_nested(training_config, "logging_steps")),
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
        dataloader_drop_last=False,
        report_to="none",
    )
    challenger.set_seed(seed)
    trainer = ExactEpochTrainer(
        model=model,
        args=arguments,
        train_dataset=challenger.RecordDataset(training_records),
        data_collator=collator,
        callbacks=[schedule_callback, metric_callback],
    )
    trainer.exact_optimizer_schedule = schedule
    result = trainer.train()
    if int(result.global_step) != schedule.total_optimizer_updates:
        raise challenger.SafetyGateError(
            f"Trainer returned {result.global_step} updates, expected {schedule.total_optimizer_updates}"
        )
    if metric_callback.best_epoch is None:
        raise challenger.SafetyGateError("Model41 ended without complete-fold validation")
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)

    if guided.stable_sha256(config.path) != initial_config_sha256:
        raise RuntimeError("Model41 config changed during training")
    if _implementation_sha256() != initial_implementation_sha256:
        raise RuntimeError("Model41 implementation or a pinned dependency changed during training")
    current_artifact, _ = require_draft_artifact(config, "train")
    if current_artifact.audit != draft_audit:
        raise RuntimeError("Model41 training draft artifact changed during training")
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    challenger.enforce_memory_limits(
        peak_allocated,
        peak_reserved,
        max_allocated_gib=5.3,
        max_reserved_gib=5.3,
    )
    complete = {
        "schema_version": SCHEMA_VERSION,
        "fold": fold,
        "best_epoch": metric_callback.best_epoch,
        "best_challenger_multi_score": metric_callback.best_score,
        "observed_epoch_boundaries": schedule_callback.epoch_boundaries,
        "observed_optimizer_updates": int(result.global_step),
        "expected_optimizer_updates": schedule.total_optimizer_updates,
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
    config: Model41Config, adapter_path: str | Path, *, oof_fold: int | None
) -> dict[str, Any]:
    adapter = _absolute(adapter_path)
    if not _below(adapter, MODEL41_OUTPUT_ROOT) or adapter.name != "best_adapter":
        raise challenger.SafetyGateError(
            "Model41 inference accepts only outputs/model41/foldN/best_adapter"
        )
    run_dir = adapter.parent
    run_path = run_dir / "run_manifest.json"
    best_path = run_dir / "best.json"
    complete_path = run_dir / "complete.json"
    run = _read_json(run_path)
    best = _read_json(best_path)
    complete = _read_json(complete_path)
    training_fold = run.get("fold")
    if not isinstance(training_fold, int) or training_fold not in range(5):
        raise challenger.SafetyGateError("Model41 run manifest has invalid fold")
    expected_dir = require_model41_output(
        str(_nested(config.raw, "output_dir")).format(fold=training_fold)
    )
    if run_dir != expected_dir:
        raise challenger.SafetyGateError("Model41 adapter is outside its configured fold run")
    if oof_fold is not None and training_fold != oof_fold:
        raise challenger.SafetyGateError(
            f"fold {oof_fold} OOF inference cannot use fold {training_fold} adapter"
        )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "model_number": 41,
        "training_excluded_folds": [training_fold],
        "config": str(config.path),
        "config_sha256": guided.stable_sha256(config.path),
        "implementation_sha256": _implementation_sha256(),
        "base_model39_config_sha256": MODEL39_CONFIG_SHA256,
        "base_model39_implementation_sha256": MODEL39_IMPLEMENTATION_SHA256,
        "base_model32_config_sha256": MODEL32_CONFIG_SHA256,
        "base_model32_implementation_sha256": MODEL32_IMPLEMENTATION_SHA256,
    }
    for key, value in expected.items():
        if run.get(key) != value:
            raise challenger.SafetyGateError(f"Model41 run-manifest mismatch for {key}")
    train_artifact, _ = require_draft_artifact(config, "train")
    if run.get("draft_artifact") != train_artifact.audit:
        raise challenger.SafetyGateError("adapter training draft receipt changed or detached")
    target_audit = run.get("checkpoint_target_audit")
    if not isinstance(target_audit, Mapping) or (
        target_audit.get("target_modules") is None
        or len(target_audit["target_modules"]) != broad.EXPECTED_TARGET_MODULES
        or target_audit.get("trainable_parameters") != broad.EXPECTED_TRAINABLE_PARAMETERS
        or target_audit.get("trainable_tensors") != broad.EXPECTED_TRAINABLE_TENSORS
    ):
        raise challenger.SafetyGateError("adapter lacks the exact Model39 target audit")
    if best.get("adapter") != str(adapter):
        raise challenger.SafetyGateError("best.json names a different adapter")
    if best.get("run_manifest") != str(run_path) or best.get(
        "run_manifest_sha256"
    ) != guided.stable_sha256(run_path):
        raise challenger.SafetyGateError("best adapter is detached from its run manifest")
    ledger = _directory_ledger(adapter)
    if best.get("adapter_files_sha256") != ledger:
        raise challenger.SafetyGateError("best adapter file ledger mismatch")
    if complete.get("observed_optimizer_updates") != 615 or complete.get(
        "expected_optimizer_updates"
    ) != 615:
        raise challenger.SafetyGateError("adapter training did not finish 615 optimizer updates")
    boundaries = complete.get("observed_epoch_boundaries")
    if not isinstance(boundaries, list) or [item.get("optimizer_step") for item in boundaries] != [205, 410, 615]:
        raise challenger.SafetyGateError("adapter epoch boundaries do not match the ceil schedule")
    smoke = run.get("smoke")
    if not isinstance(smoke, Mapping):
        raise challenger.SafetyGateError("adapter run lacks a memory smoke")
    allocated = smoke.get("peak_allocated_bytes")
    reserved = smoke.get("peak_reserved_bytes")
    if not isinstance(allocated, int) or not isinstance(reserved, int):
        raise challenger.SafetyGateError("adapter memory-smoke peaks are invalid")
    challenger.enforce_memory_limits(
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
        "run_manifest": str(run_path),
        "run_manifest_sha256": guided.stable_sha256(run_path),
        "best": str(best_path),
        "best_sha256": guided.stable_sha256(best_path),
        "complete": str(complete_path),
        "complete_sha256": guided.stable_sha256(complete_path),
        "training_draft_artifact": train_artifact.audit,
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
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    challenger.enable_offline_mode()
    config = load_config(config_path)
    _verify_canonical_data(config)
    input_path = _absolute(input_csv)
    if fold is not None:
        if fold not in range(5):
            raise ValueError("fold must be one of 0,1,2,3,4")
        role = "train"
        if input_path != _canonical_path(config, role):
            raise challenger.SafetyGateError("Model41 OOF inference requires canonical train folds")
    else:
        role = "test"
        if input_path != _canonical_path(config, role):
            raise challenger.SafetyGateError("fold-free Model41 inference requires canonical Test.csv")
    expected_draft, expected_receipt = _draft_paths(config, role)
    if _absolute(draft_csv) != expected_draft or _absolute(draft_receipt) != expected_receipt:
        raise challenger.SafetyGateError(f"Model41 {role} inference requires canonical draft paths")
    require_draft_artifact(config, role)
    lineage = verify_adapter_lineage(config, adapter_path, oof_fold=fold)
    records, current_draft_audit = guided.load_guided_records(
        config.draft_config,
        input_path,
        expected_draft,
        expected_receipt,
        require_target=fold is not None,
        require_five_folds=fold is not None,
    )
    if fold is not None:
        records = tuple(record for record in records if record.fold == fold)
        if len(records) != 819:
            raise challenger.SafetyGateError(f"OOF fold {fold} must contain 819 rows")
    elif len(records) != 1374:
        raise challenger.SafetyGateError(f"Test.csv must contain 1,374 rows, got {len(records)}")
    output = require_model41_output(output_csv)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model41 inference output: {output}")

    processor = challenger.load_processor(config.effective_model39)
    collator = guided.QwenGuidedCollator(
        processor,
        str(_nested(config.raw, "prompt_template")),
        int(_nested(config.effective_model39, "image", "max_image_tokens")),
        int(_nested(config.effective_model39, "image", "max_sequence_tokens")),
    )
    sequence_preflight = preflight_sequence_budgets(
        collator, ((f"{role}_prompt", records, False),)
    )
    model, target_audit = broad._load_inference_model(
        config.effective_model39, _absolute(adapter_path)
    )
    predictions = challenger.greedy_predict(
        model,
        processor,
        collator,
        records,
        max_new_tokens=int(_nested(config.effective_model39, "generation", "max_new_tokens")),
        repetition_penalty=float(
            _nested(config.effective_model39, "generation", "repetition_penalty")
        ),
        no_repeat_ngram_size=int(
            _nested(config.effective_model39, "generation", "no_repeat_ngram_size")
        ),
    )
    output = _atomic_predictions(output, records, predictions)
    draft_output = _atomic_predictions(
        output.with_suffix(".draft.csv"), records, [record.draft for record in records]
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rows": len(records),
        "role": role,
        "fold": fold,
        "adapter_lineage": lineage,
        "draft_artifact": current_draft_audit,
        "output": str(output),
        "output_sha256": guided.stable_sha256(output),
        "returning_draft_output": str(draft_output),
        "returning_draft_output_sha256": guided.stable_sha256(draft_output),
        "sequence_preflight": sequence_preflight,
        "target_audit": asdict(target_audit),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 48,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 4,
        },
        "per_row_selection_performed": False,
        "emitted_predictions": "challenger",
    }
    if fold is not None:
        report["comparison"] = guided.compare_fold_to_draft(records, predictions)
    _atomic_json(output.with_suffix(".metrics.json"), report)
    if submission_path is not None:
        if sample_path is None:
            raise ValueError("--sample is required with --submission")
        sample = _absolute(sample_path)
        if sample != _canonical_path(config, "sample"):
            raise challenger.SafetyGateError("Model41 submission requires canonical sample")
        submission = require_model41_submission(submission_path)
        if submission.exists():
            raise FileExistsError(f"refusing to overwrite Model41 submission: {submission}")
        by_id = {
            record.identifier: prediction
            for record, prediction in zip(records, predictions, strict=True)
        }
        write_submission(
            sample,
            submission,
            by_id,
            cleaner=clean_prediction_preserving_internal_spaces,
        )
        report["submission"] = str(submission)
        report["submission_sha256"] = guided.stable_sha256(submission)
        _atomic_json(output.with_suffix(".metrics.json"), report)
    return report


def train_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model41 broad OCR-guided Qwen training")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.config,
                args.fold,
                smoke_only=args.smoke_only,
                dry_run_only=args.dry_run,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


def infer_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model41 broad OCR-guided greedy inference")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--draft-receipt", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            infer(
                args.config,
                args.input,
                args.drafts,
                args.draft_receipt,
                args.adapter,
                args.output,
                fold=args.fold,
                sample_path=args.sample,
                submission_path=args.submission,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


def audit_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model41 CUDA-free structural audit")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--output", type=Path, default=MODEL41_OUTPUT_ROOT / "dry_run.json")
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    print(
        json.dumps(
            dry_run(args.config, args.output, fold=args.fold),
            indent=2,
            ensure_ascii=False,
        )
    )
