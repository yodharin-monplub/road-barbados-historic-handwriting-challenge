"""Model45: Model40 full-vision QLoRA with Model32's receipted OCR draft.

This is intentionally an isolated composition.  It imports finalized Model40
for letterbox geometry, architecture, and the corrected epoch schedule, and
Model32 only for its label-independent draft protocol and prompt primitives.
No CUDA model can be loaded until receipt, tokenizer, geometry, and processor
gates have all passed.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image

from . import qwen2_vl_challenger as challenger
from . import qwen2_vl_ocr_guided as guided
from . import qwen2_vl_visual_full as visual
from .submission import clean_prediction_preserving_internal_spaces, write_submission


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL45_OUTPUT_ROOT = (PROJECT_ROOT / "outputs" / "model45").resolve()
MODEL45_SUBMISSION_ROOT = (PROJECT_ROOT / "submissions" / "model45").resolve()

SCHEMA_VERSION = 1
MODEL40_CONFIG_SHA256 = (
    "a409a13e7dc7b88282f7ca460787b9778032b7f8356438bce169f5c0d2c3ec4e"
)
MODEL40_IMPLEMENTATION_SHA256 = (
    "28780ecb6c60bd836570050a666bfb3209246a9e055bdb3494a3fc354c693ca8"
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
class Model45Config:
    raw: Mapping[str, Any]
    effective_model40: Mapping[str, Any]
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
            raise ValueError(f"missing required Model45 config key: {'.'.join(keys)}")
        current = current[key]
    return current


def _below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def require_model45_output(path: str | Path) -> Path:
    resolved = _absolute(path)
    if not _below(resolved, MODEL45_OUTPUT_ROOT):
        raise challenger.SafetyGateError(
            f"Model45 output must remain under {MODEL45_OUTPUT_ROOT}: {resolved}"
        )
    return resolved


def require_model45_submission(path: str | Path) -> Path:
    resolved = _absolute(path)
    if not _below(resolved, MODEL45_SUBMISSION_ROOT):
        raise challenger.SafetyGateError(
            f"Model45 submission must remain under {MODEL45_SUBMISSION_ROOT}: {resolved}"
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
    output = require_model45_output(path)
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
        PROJECT_ROOT / "model45" / "train.py",
        PROJECT_ROOT / "model45" / "infer.py",
        PROJECT_ROOT / "model45" / "audit.py",
        PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_visual_full.py",
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


def load_config(path: str | Path) -> Model45Config:
    """Verify immutable Model40/Model32 dependencies and derive Model45."""

    resolved = _absolute(path)
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read Model45 configuration {resolved}: {error}") from error
    if not isinstance(raw, Mapping) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Model45 config must be a schema-version-1 mapping")

    base_config_path = _absolute(_nested(raw, "base_model40", "config"))
    base_implementation_path = _absolute(_nested(raw, "base_model40", "implementation"))
    draft_config_path = _absolute(_nested(raw, "base_model32_drafts", "config"))
    draft_implementation_path = _absolute(
        _nested(raw, "base_model32_drafts", "implementation")
    )
    exact_paths = {
        base_config_path: (PROJECT_ROOT / "model40" / "config.yaml").resolve(),
        base_implementation_path: (
            PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_visual_full.py"
        ).resolve(),
        draft_config_path: (PROJECT_ROOT / "model32" / "config.yaml").resolve(),
        draft_implementation_path: (
            PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_ocr_guided.py"
        ).resolve(),
    }
    for actual, expected in exact_paths.items():
        if actual != expected:
            raise ValueError(f"Model45 pinned dependency path mismatch: {actual} != {expected}")
        if not actual.is_file():
            raise FileNotFoundError(actual)
    pins = {
        ("base_model40", "config_sha256"): MODEL40_CONFIG_SHA256,
        ("base_model40", "implementation_sha256"): MODEL40_IMPLEMENTATION_SHA256,
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
    for keys, expected in pins.items():
        actual = _nested(raw, *keys)
        if actual != expected:
            raise ValueError(
                f"Model45 pin {'.'.join(keys)} must be {expected!r}, got {actual!r}"
            )
    dependency_hashes = {
        base_config_path: MODEL40_CONFIG_SHA256,
        base_implementation_path: MODEL40_IMPLEMENTATION_SHA256,
        draft_config_path: MODEL32_CONFIG_SHA256,
        draft_implementation_path: MODEL32_IMPLEMENTATION_SHA256,
    }
    for dependency, expected in dependency_hashes.items():
        if guided.stable_sha256(dependency) != expected:
            raise challenger.SafetyGateError(f"pinned Model45 dependency changed: {dependency}")

    base, _ = visual.load_config(base_config_path)
    draft_config = guided.load_config(draft_config_path)
    effective = copy.deepcopy(base)
    effective["train_csv"] = str(_nested(raw, "data", "train_csv"))
    effective["image_dir"] = str(_nested(raw, "data", "image_dir"))
    effective["output_dir"] = str(_nested(raw, "output_dir"))
    effective["prompt"] = str(_nested(raw, "prompt_template"))
    effective["smoke"]["report_dir"] = str(_nested(raw, "smoke_report_dir"))
    visual.validate_config(effective)

    prompt = str(_nested(raw, "prompt_template"))
    if prompt.count("{draft}") != 1:
        raise ValueError("prompt_template must contain exactly one literal {draft}")
    lowered = prompt.casefold()
    if "{label}" in lowered or "{target}" in lowered:
        raise ValueError("prompt_template may not interpolate a challenge label")
    exact_strings = {
        ("data", "train_csv"): "data/processed/train_folds.csv",
        ("data", "test_csv"): "data/raw/Test.csv",
        ("data", "sample_submission"): "data/raw/SampleSubmission.csv",
        ("data", "image_dir"): "data/raw/images",
        ("output_dir",): "outputs/model45/fold{fold}",
        ("smoke_report_dir",): "outputs/model45/smoke",
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
            raise ValueError(f"Model45 {'.'.join(keys)} must remain {expected!r}")
    expected_cv = {
        "folds": 5,
        "expected_total_rows": 4095,
        "expected_train_rows_per_fold": 3276,
        "expected_validation_rows_per_fold": 819,
        "selection_scope": "whole_fold_challenger_only",
        "forbid_per_row_oracle": True,
    }
    if _nested(raw, "cross_validation") != expected_cv:
        raise ValueError("Model45 cross-validation contract may not be weakened")
    cap_decision = {
        "inherited_training_cap": 768,
        "inherited_generation_context_cap": 816,
        "configured_training_cap": 768,
        "configured_generation_context_cap": 816,
        "increased_from_model40": False,
        "require_receipted_all_row_audit_before_increase": True,
        "require_runtime_processor_preflight": True,
    }
    if _nested(raw, "sequence_cap_decision") != cap_decision:
        raise ValueError("Model45 may not raise context without a receipted all-row audit")
    safety = {
        "offline": True,
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "double_quant": True,
        "expected_lora_targets": visual.EXPECTED_TARGET_MODULES,
        "expected_trainable_parameters": visual.EXPECTED_TRAINABLE_PARAMETERS,
        "expected_trainable_tensors": visual.EXPECTED_TRAINABLE_TENSORS,
        "min_effective_image_tokens": 512,
        "max_image_tokens": 640,
        "expected_updates_per_epoch": 205,
        "expected_total_updates": 615,
        "max_new_tokens": 48,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 4,
        "max_allocated_gib_exclusive": 5.3,
        "max_reserved_gib_exclusive": 5.3,
        "preserve_internal_spaces": True,
    }
    if _nested(raw, "safety_assertions") != safety:
        raise ValueError("Model45 safety assertions may not be weakened")
    inherited = {
        ("image", "min_effective_image_tokens"): 512,
        ("image", "max_image_tokens"): 640,
        ("image", "max_sequence_tokens"): 768,
        ("image", "max_generation_context_tokens"): 816,
        ("image", "resize_policy"): "aspect_preserving_letterbox_lanczos",
        ("architecture", "expected_target_modules"): 326,
        ("architecture", "expected_trainable_parameters"): 14_610_432,
        ("training", "expected_updates_per_epoch"): 205,
        ("training", "expected_total_updates"): 615,
        ("generation", "max_new_tokens"): 48,
        ("generation", "repetition_penalty"): 1.05,
        ("generation", "no_repeat_ngram_size"): 4,
    }
    for keys, expected in inherited.items():
        if challenger._nested(effective, *keys) != expected:
            raise challenger.SafetyGateError(
                f"inherited Model40 contract drifted at {'.'.join(keys)}"
            )
    return Model45Config(
        raw=raw,
        effective_model40=effective,
        path=resolved,
        base_config_path=base_config_path,
        base_implementation_path=base_implementation_path,
        draft_config_path=draft_config_path,
        draft_implementation_path=draft_implementation_path,
        draft_config=draft_config,
    )


def _canonical_path(config: Model45Config, role: str) -> Path:
    keys = {"train": "train_csv", "test": "test_csv", "sample": "sample_submission"}
    if role not in keys:
        raise ValueError(f"unknown canonical role: {role}")
    return _absolute(_nested(config.raw, "data", keys[role]))


def _verify_canonical_data(config: Model45Config) -> dict[str, Any]:
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
        raise challenger.SafetyGateError("pinned Model8 checkpoint changed or is absent")
    report["draft_checkpoint"] = {
        "path": str(checkpoint),
        "sha256": MODEL8_CHECKPOINT_SHA256,
    }
    return report


def _draft_paths(config: Model45Config, role: str) -> tuple[Path, Path]:
    if role not in {"train", "test"}:
        raise ValueError("draft role must be train or test")
    return (
        _absolute(_nested(config.raw, "draft_source", f"{role}_csv")),
        _absolute(_nested(config.raw, "draft_source", f"{role}_receipt")),
    )


def inspect_draft_readiness(config: Model45Config, role: str) -> dict[str, Any]:
    input_path = _canonical_path(config, role)
    draft, receipt = _draft_paths(config, role)
    required = (
        draft,
        draft.with_suffix(".ids.csv"),
        draft.with_suffix(".jsonl"),
        draft.with_suffix(".manifest.json"),
        receipt,
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
        config.draft_config, input_path, draft, receipt
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
    config: Model45Config, role: str
) -> tuple[guided.DraftArtifact, dict[str, Any]]:
    readiness = inspect_draft_readiness(config, role)
    if readiness["ready"] is not True:
        raise challenger.SafetyGateError(
            f"Model45 {role} draft receipt is not ready; missing={readiness['missing']}"
        )
    input_path = _canonical_path(config, role)
    draft, receipt = _draft_paths(config, role)
    artifact = guided.verify_draft_artifact(
        config.draft_config, input_path, draft, receipt
    )
    return artifact, readiness


class FullVisualGuidedCollator(visual.FullVisualQwenCollator):
    """Model40's letterbox and hard caps with one literal per-row OCR draft."""

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

    def _encode_text(self, record: guided.GuidedOcrRecord, include_label: bool) -> str:
        if include_label and record.label is None:
            raise ValueError("training record has no label")
        messages = guided.build_guided_messages(
            self.prompt_template,
            record.draft,
            record.label if include_label else None,
        )
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_label,
        )


def _row_image_tokens(record: guided.GuidedOcrRecord, config: Model45Config) -> dict[str, Any]:
    with Image.open(record.image_path) as image:
        width, height = image.size
    minimum = int(_nested(config.effective_model40, "image", "min_effective_image_tokens"))
    maximum = int(_nested(config.effective_model40, "image", "max_image_tokens"))
    plan = visual.plan_aspect_preserving_resize(
        width,
        height,
        minimum_tokens=minimum,
        maximum_tokens=maximum,
    )
    grid_width, grid_height, tokens = visual.qwen_smart_grid(
        plan.canvas_width,
        plan.canvas_height,
        min_pixels=int(_nested(config.effective_model40, "image", "min_pixels")),
        max_pixels=int(_nested(config.effective_model40, "image", "max_pixels")),
    )
    if (grid_width, grid_height) != (plan.canvas_width, plan.canvas_height):
        raise challenger.SafetyGateError("Qwen would alter Model45's planned letterbox")
    if tokens != plan.planned_image_tokens or not minimum <= tokens <= maximum:
        raise challenger.SafetyGateError("Model45 planned and effective image tokens differ")
    return {
        "image_tokens": tokens,
        "canvas_width": plan.canvas_width,
        "canvas_height": plan.canvas_height,
        "content_pixel_fraction": plan.content_pixel_fraction,
        "relative_content_aspect_error": plan.relative_content_aspect_error,
    }


def audit_guided_sequence_budget(
    config: Model45Config,
    records: Sequence[guided.GuidedOcrRecord],
    *,
    require_targets: bool,
    enforce_configured_caps: bool = True,
) -> dict[str, Any]:
    """Tokenize every receipted row at its exact planned letterbox geometry."""

    from transformers import AutoTokenizer

    if not records:
        raise ValueError("guided sequence audit requires at least one row")
    checkpoint = _absolute(_nested(config.effective_model40, "model", "local_path"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), trust_remote_code=False, local_files_only=True
    )
    prompt_template = str(_nested(config.raw, "prompt_template"))
    training_cap = int(_nested(config.effective_model40, "image", "max_sequence_tokens"))
    context_cap = int(
        _nested(config.effective_model40, "image", "max_generation_context_tokens")
    )
    max_new_tokens = int(_nested(config.effective_model40, "generation", "max_new_tokens"))
    maxima: dict[str, dict[str, Any]] = {}
    image_token_values: list[int] = []

    def update(name: str, value: int, identifier: str) -> None:
        previous = maxima.get(name)
        if previous is None or value > int(previous["value"]):
            maxima[name] = {"value": value, "record_id": identifier}

    for record in records:
        if require_targets and record.label is None:
            raise challenger.SafetyGateError(
                f"all-row training audit lacks target for {record.identifier!r}"
            )
        geometry = _row_image_tokens(record, config)
        image_tokens = int(geometry["image_tokens"])
        image_token_values.append(image_tokens)
        update("image_tokens", image_tokens, record.identifier)
        prompt_messages = guided.build_guided_messages(
            prompt_template, record.draft, None
        )
        prompt_tokens, prompt_pads = visual._template_token_count(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
            image_tokens=image_tokens,
        )
        if prompt_pads != 1:
            raise challenger.SafetyGateError("guided prompt must contain one image pad")
        generation_upper = prompt_tokens + max_new_tokens
        update("prompt_tokens", prompt_tokens, record.identifier)
        update("generation_context_tokens", generation_upper, record.identifier)
        if require_targets:
            full_messages = guided.build_guided_messages(
                prompt_template, record.draft, record.label
            )
            training_tokens, training_pads = visual._template_token_count(
                tokenizer,
                full_messages,
                add_generation_prompt=False,
                image_tokens=image_tokens,
            )
            if training_pads != 1:
                raise challenger.SafetyGateError("guided training template must contain one image pad")
            target_tokens = len(
                tokenizer.encode(
                    challenger.trim_label_edges(record.label or ""),
                    add_special_tokens=False,
                )
            )
            update("training_sequence_tokens", training_tokens, record.identifier)
            update("target_tokens", target_tokens, record.identifier)
    ordered = sorted(image_token_values)
    observed_training = int(maxima.get("training_sequence_tokens", {"value": 0})["value"])
    observed_context = int(maxima["generation_context_tokens"]["value"])
    observed_target = int(maxima.get("target_tokens", {"value": 0})["value"])
    training_fits = not require_targets or observed_training <= training_cap
    context_fits = observed_context <= context_cap
    generation_headroom_fits = not require_targets or max_new_tokens >= observed_target + 4
    fits = training_fits and context_fits and generation_headroom_fits
    report = {
        "status": (
            "receipted_every_row_tokenizer_geometry_audit_passed"
            if fits
            else "receipted_every_row_audit_proved_configured_caps_insufficient"
        ),
        "rows": len(records),
        "targets_required": require_targets,
        "exact_geometry_per_row": True,
        "exact_pinned_tokenizer_per_row": True,
        "image_tokens": {
            "minimum": ordered[0],
            "median": ordered[len(ordered) // 2],
            "maximum": ordered[-1],
        },
        "maxima": maxima,
        "configured_caps": {
            "training_sequence_tokens": training_cap,
            "generation_context_tokens": context_cap,
            "max_new_tokens": max_new_tokens,
        },
        "minimum_observed_requirements": {
            "training_sequence_tokens": observed_training if require_targets else None,
            "generation_context_tokens": observed_context,
            "target_tokens_plus_four_headroom": observed_target + 4 if require_targets else None,
        },
        "fits_configured_caps": fits,
        "individual_gates": {
            "training_sequence_fits": training_fits,
            "generation_context_fits": context_fits,
            "generation_target_headroom_fits": generation_headroom_fits,
        },
        "additional_tokens_proved_necessary": {
            "training_sequence_tokens": max(0, observed_training - training_cap),
            "generation_context_tokens": max(0, observed_context - context_cap),
            "max_new_tokens": max(0, observed_target + 4 - max_new_tokens),
        },
        "cap_increase_from_model40": {
            "training_tokens": 0,
            "generation_context_tokens": 0,
            "proved_before_any_increase": True,
        },
        "model_context_tokens": visual.MODEL_CONTEXT_TOKENS,
    }
    if enforce_configured_caps and not fits:
        raise challenger.SafetyGateError(
            "receipted all-row Model45 audit proved configured caps insufficient: "
            f"observed_training={observed_training}/{training_cap}, "
            f"observed_context={observed_context}/{context_cap}, "
            f"target_plus_four={observed_target + 4}/{max_new_tokens}"
        )
    return report


def preflight_processor_sequences(
    collator: FullVisualGuidedCollator,
    groups: Sequence[tuple[str, Sequence[guided.GuidedOcrRecord], bool]],
    *,
    generation_context_cap: int = 816,
    max_new_tokens: int = 48,
) -> dict[str, Any]:
    """Encode every selected row with the real processor before CUDA loading."""

    global_maxima: dict[str, dict[str, Any]] = {}
    group_reports: dict[str, dict[str, Any]] = {}
    total = 0

    def update(container: dict[str, dict[str, Any]], key: str, value: int, record: Any, role: str) -> None:
        previous = container.get(key)
        candidate = {"value": value, "record_id": record.identifier, "role": role}
        if previous is None or value > int(previous["value"]):
            container[key] = candidate

    for role, records, include_label in groups:
        if role in group_reports or not records:
            raise ValueError(f"duplicate or empty processor-preflight group: {role}")
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
                        f"invalid {key} for {record.identifier!r}: {value!r}"
                    )
                update(global_maxima, key, value, record, role)
                update(role_maxima, key, value, record, role)
            prompt_tokens = (
                diagnostics.get("prompt_tokens")
                if include_label
                else diagnostics.get("sequence_tokens")
            )
            if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
                raise challenger.SafetyGateError(
                    f"invalid prompt tokens for {record.identifier!r}"
                )
            generation_upper = prompt_tokens + max_new_tokens
            if generation_upper > generation_context_cap:
                raise challenger.SafetyGateError(
                    f"runtime prompt+generation {generation_upper} exceeds "
                    f"{generation_context_cap} for {record.identifier!r}"
                )
            update(global_maxima, "generation_context_tokens", generation_upper, record, role)
            update(role_maxima, "generation_context_tokens", generation_upper, record, role)
            if include_label:
                supervised = diagnostics.get("supervised_tokens")
                if isinstance(supervised, bool) or not isinstance(supervised, int) or supervised <= 0:
                    raise challenger.SafetyGateError(
                        f"invalid supervised tokens for {record.identifier!r}"
                    )
                update(global_maxima, "supervised_tokens", supervised, record, role)
                update(role_maxima, "supervised_tokens", supervised, record, role)
        group_reports[role] = {"rows": len(records), "maxima": role_maxima}
    if int(global_maxima["image_tokens"]["value"]) > 640:
        raise challenger.SafetyGateError("processor emitted more than 640 image tokens")
    if int(global_maxima["sequence_tokens"]["value"]) > 768:
        raise challenger.SafetyGateError("processor emitted more than 768 sequence tokens")
    return {
        "status": "every_selected_row_real_processor_preflight_passed",
        "rows": total,
        "groups": group_reports,
        "maxima": global_maxima,
        "caps": {
            "image_tokens": 640,
            "training_sequence_tokens": 768,
            "generation_context_tokens": generation_context_cap,
        },
    }


def select_exact_smoke_record(
    records: Sequence[guided.GuidedOcrRecord],
    processor_preflight: Mapping[str, Any],
    *,
    role: str = "outer_train_with_target",
) -> guided.GuidedOcrRecord:
    groups = processor_preflight.get("groups")
    group = groups.get(role) if isinstance(groups, Mapping) else None
    maxima = group.get("maxima") if isinstance(group, Mapping) else None
    sequence = maxima.get("sequence_tokens") if isinstance(maxima, Mapping) else None
    identifier = sequence.get("record_id") if isinstance(sequence, Mapping) else None
    matches = [record for record in records if record.identifier == identifier]
    if len(matches) != 1 or matches[0].label is None:
        raise challenger.SafetyGateError("exact maximum-sequence smoke row is unavailable")
    return matches[0]


def _ids_sha256(records: Sequence[Any]) -> str:
    return _json_sha256([record.identifier for record in records])


def _validate_fold_counts(records: Sequence[Any]) -> None:
    if len(records) != 4095:
        raise challenger.SafetyGateError(f"Model45 requires 4,095 rows, got {len(records)}")
    for fold in range(5):
        training, validation = visual.split_outer_fold(records, fold)
        if len(training) != 3276 or len(validation) != 819:
            raise challenger.SafetyGateError(
                f"fold {fold} must contain 3,276 train and 819 validation rows"
            )


def _atomic_predictions(
    path: str | Path,
    records: Sequence[guided.GuidedOcrRecord],
    predictions: Sequence[str],
) -> Path:
    if len(records) != len(predictions):
        raise ValueError("record/prediction count mismatch")
    output = require_model45_output(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model45 prediction: {output}")
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
    collator: FullVisualGuidedCollator,
    validation_records: Sequence[guided.GuidedOcrRecord],
    output_dir: Path,
    run_manifest_path: Path,
    config: Model45Config,
) -> Any:
    from transformers import TrainerCallback

    generation = _nested(config.effective_model40, "generation")
    max_new_tokens = int(_nested(generation, "max_new_tokens"))
    repetition_penalty = float(_nested(generation, "repetition_penalty"))
    no_repeat_ngram_size = int(_nested(generation, "no_repeat_ngram_size"))

    class FullVisualGuidedMetricCallback(TrainerCallback):
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
                            "maximum complete-fold challenger score; raw draft is "
                            "report-only and no row oracle is used"
                        ),
                    },
                )
            print(
                f"Model45 validation epoch={epoch} challenger={score:.9f} "
                f"draft={comparison['returning_draft']['multi_score']:.9f} "
                f"best_epoch={self.best_epoch}",
                flush=True,
            )
            return control

    return FullVisualGuidedMetricCallback()


def dry_run(
    config_path: str | Path,
    output_path: str | Path,
    *,
    fold: int | None = None,
) -> dict[str, Any]:
    """Run all possible CPU audits and explicitly report missing draft gates."""

    challenger.enable_offline_mode()
    config = load_config(config_path)
    data = _verify_canonical_data(config)
    provenance = challenger.verify_starter_provenance(config.effective_model40)
    checkpoint = challenger.verify_local_checkpoint(config.effective_model40)
    targets = visual.audit_checkpoint_target_shapes(config.effective_model40)
    context = visual.audit_model_context(config.effective_model40)
    base_records = challenger.load_records(
        _canonical_path(config, "train"),
        _absolute(_nested(config.raw, "data", "image_dir")),
        require_target=True,
        require_five_folds=True,
    )
    _validate_fold_counts(base_records)
    geometry = visual.audit_resize_geometry(
        _nested(config.raw, "data", "image_dir"), config.effective_model40
    )
    folds = range(5) if fold is None else (fold,)
    if fold is not None and fold not in range(5):
        raise ValueError("fold must be one of 0,1,2,3,4")
    fold_audit: dict[str, Any] = {}
    for current_fold in folds:
        training, validation = visual.split_outer_fold(base_records, current_fold)
        schedule = visual.deterministic_optimizer_schedule(
            len(training),
            int(_nested(config.effective_model40, "training", "gradient_accumulation")),
            int(_nested(config.effective_model40, "training", "epochs")),
        )
        visual.verify_configured_schedule(config.effective_model40, schedule)
        fold_audit[str(current_fold)] = {
            "train_rows": len(training),
            "validation_rows": len(validation),
            "optimizer_schedule": asdict(schedule),
        }
    drafts = {
        role: inspect_draft_readiness(config, role) for role in ("train", "test")
    }
    sequence_audits: dict[str, Any] = {}
    if drafts["train"]["ready"]:
        train_draft, train_receipt = _draft_paths(config, "train")
        train_records, _ = guided.load_guided_records(
            config.draft_config,
            _canonical_path(config, "train"),
            train_draft,
            train_receipt,
            require_target=True,
            require_five_folds=True,
        )
        sequence_audits["train"] = audit_guided_sequence_budget(
            config,
            train_records,
            require_targets=True,
            enforce_configured_caps=False,
        )
    else:
        sequence_audits["train"] = {
            "status": "blocked_until_canonical_train_draft_receipt_exists",
            "rows_audited": 0,
            "caps_increased": False,
        }
    if drafts["test"]["ready"]:
        test_draft, test_receipt = _draft_paths(config, "test")
        test_records, _ = guided.load_guided_records(
            config.draft_config,
            _canonical_path(config, "test"),
            test_draft,
            test_receipt,
            require_target=False,
            require_five_folds=False,
        )
        sequence_audits["test"] = audit_guided_sequence_budget(
            config,
            test_records,
            require_targets=False,
            enforce_configured_caps=False,
        )
    else:
        sequence_audits["test"] = {
            "status": "blocked_until_canonical_test_draft_receipt_exists",
            "rows_audited": 0,
            "caps_increased": False,
        }
    train_caps_fit = sequence_audits["train"].get("fits_configured_caps") is True
    test_caps_fit = sequence_audits["test"].get("fits_configured_caps") is True
    if drafts["train"]["ready"] and not train_caps_fit:
        status = "receipted_train_sequence_audit_requires_cap_decision_gpu_blocked"
    elif drafts["test"]["ready"] and not test_caps_fit:
        status = "receipted_test_sequence_audit_requires_cap_decision_inference_blocked"
    elif drafts["train"]["ready"] and drafts["test"]["ready"]:
        status = "all_cpu_audits_passed_gpu_smoke_not_run"
    elif drafts["train"]["ready"]:
        status = "train_cpu_audits_passed_test_draft_missing_gpu_smoke_not_run"
    else:
        status = "structural_cpu_audits_passed_draft_receipts_missing"
    report = {
        "schema_version": SCHEMA_VERSION,
        "model": "model45_full_visual_qwen2_vl_ocr_guided",
        "status": status,
        "score": None,
        "score_status": "no_oof_or_validation_run",
        "gpu_started": False,
        "resolved_config": str(config.path),
        "config_sha256": guided.stable_sha256(config.path),
        "implementation_sha256": _implementation_sha256(),
        "dependency_hashes": {
            "model40_config": MODEL40_CONFIG_SHA256,
            "model40_implementation": MODEL40_IMPLEMENTATION_SHA256,
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
        "model_context": context,
        "resize_geometry": geometry,
        "folds": fold_audit,
        "drafts": drafts,
        "guided_sequence_audits": sequence_audits,
        "sequence_cap_decision": _nested(config.raw, "sequence_cap_decision"),
        "memory_gate": {
            "allocated_gib_exclusive": 5.3,
            "reserved_gib_exclusive": 5.3,
            "status": "gpu_smoke_not_run",
        },
        "leakage_contract": (
            "frozen Model8 receives only ordered IDs and image pixels; each Qwen "
            "outer validation ID is absent from its training partition"
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
            MODEL45_OUTPUT_ROOT / f"dry_run_fold{fold}.json",
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
    static_sequence = audit_guided_sequence_budget(
        config, records, require_targets=True
    )
    training_records, validation_records = visual.split_outer_fold(records, fold)
    schedule = visual.deterministic_optimizer_schedule(
        len(training_records),
        int(_nested(config.effective_model40, "training", "gradient_accumulation")),
        int(_nested(config.effective_model40, "training", "epochs")),
    )
    visual.verify_configured_schedule(config.effective_model40, schedule)
    output_dir = require_model45_output(
        str(_nested(config.raw, "output_dir")).format(fold=fold)
    )
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {output_dir}")
    smoke_path = require_model45_output(
        _absolute(_nested(config.raw, "smoke_report_dir")) / f"fold{fold}.json"
    )
    if smoke_only and smoke_path.exists():
        raise FileExistsError(f"refusing to overwrite Model45 smoke report: {smoke_path}")

    provenance = challenger.verify_starter_provenance(config.effective_model40)
    checkpoint = challenger.verify_local_checkpoint(config.effective_model40)
    targets = visual.audit_checkpoint_target_shapes(config.effective_model40)
    context = visual.audit_model_context(config.effective_model40)
    processor = challenger.load_processor(config.effective_model40)
    collator = FullVisualGuidedCollator(
        processor,
        str(_nested(config.raw, "prompt_template")),
        int(_nested(config.effective_model40, "image", "max_image_tokens")),
        int(_nested(config.effective_model40, "image", "max_sequence_tokens")),
    )
    processor_preflight = preflight_processor_sequences(
        collator,
        (
            ("outer_train_with_target", training_records, True),
            ("outer_validation_prompt", validation_records, False),
        ),
        generation_context_cap=816,
        max_new_tokens=48,
    )

    seed = int(_nested(config.effective_model40, "seed"))
    challenger.set_seed(seed)
    model, model_audit = visual.load_training_model(config.effective_model40)
    smoke_record = select_exact_smoke_record(training_records, processor_preflight)
    smoke = challenger.run_memory_smoke(
        model,
        collator,
        smoke_record,
        max_allocated_gib=5.3,
        max_reserved_gib=5.3,
    )
    runtime_audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": "model45_full_visual_qwen2_vl_ocr_guided",
        "model_number": 45,
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
        "base_model40_config_sha256": MODEL40_CONFIG_SHA256,
        "base_model40_implementation_sha256": MODEL40_IMPLEMENTATION_SHA256,
        "base_model32_config_sha256": MODEL32_CONFIG_SHA256,
        "base_model32_implementation_sha256": MODEL32_IMPLEMENTATION_SHA256,
        "canonical_data": data_audit,
        "draft_artifact": draft_audit,
        "draft_readiness": draft_readiness,
        "static_all_row_sequence_audit": static_sequence,
        "processor_preflight": processor_preflight,
        "starter_provenance": provenance,
        "local_checkpoint": checkpoint,
        "target_audit": asdict(targets),
        "model_context": context,
        "optimizer_schedule": asdict(schedule),
        "model_audit": model_audit,
        "smoke": smoke,
        "output_dir": str(output_dir),
        "gpu_started": True,
        "score": None,
    }
    if smoke_only:
        _atomic_json(smoke_path, runtime_audit)
        return runtime_audit

    import torch
    from transformers import Trainer, TrainingArguments

    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "run_manifest.json"
    _atomic_json(
        run_manifest_path,
        {
            **challenger._runtime_manifest(config.effective_model40),
            **runtime_audit,
            "resolved_model45_config": str(config.path),
            "effective_model40_config": config.effective_model40,
            "prompt_template": str(_nested(config.raw, "prompt_template")),
        },
    )
    _atomic_predictions(
        output_dir / "validation" / "returning_draft.csv",
        validation_records,
        [record.draft for record in validation_records],
    )

    class ExactEpochTrainer(visual.ExactEpochScheduleMixin, Trainer):
        pass

    schedule_callback = visual.make_schedule_guard_callback(schedule)
    metric_callback = _make_validation_callback(
        processor,
        collator,
        validation_records,
        output_dir,
        run_manifest_path,
        config,
    )
    training_config = _nested(config.effective_model40, "training")
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
        raise challenger.SafetyGateError("Model45 did not finish exactly 615 updates")
    if metric_callback.best_epoch is None:
        raise challenger.SafetyGateError("Model45 ended without complete-fold validation")
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)
    if guided.stable_sha256(config.path) != initial_config_sha256:
        raise RuntimeError("Model45 config changed during training")
    if _implementation_sha256() != initial_implementation_sha256:
        raise RuntimeError("Model45 implementation or pinned dependency changed during training")
    current_artifact, _ = require_draft_artifact(config, "train")
    if current_artifact.audit != draft_audit:
        raise RuntimeError("Model45 draft artifact changed during training")
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
        "expected_optimizer_updates": 615,
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
    config: Model45Config, adapter_path: str | Path, *, oof_fold: int | None
) -> dict[str, Any]:
    adapter = _absolute(adapter_path)
    if not _below(adapter, MODEL45_OUTPUT_ROOT) or adapter.name != "best_adapter":
        raise challenger.SafetyGateError(
            "Model45 accepts only outputs/model45/foldN/best_adapter"
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
        raise challenger.SafetyGateError("Model45 run manifest has invalid fold")
    expected_dir = require_model45_output(
        str(_nested(config.raw, "output_dir")).format(fold=training_fold)
    )
    if run_dir != expected_dir:
        raise challenger.SafetyGateError("Model45 adapter is outside its fold directory")
    if oof_fold is not None and training_fold != oof_fold:
        raise challenger.SafetyGateError(
            f"fold {oof_fold} OOF cannot use fold {training_fold} adapter"
        )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "model_number": 45,
        "training_excluded_folds": [training_fold],
        "config": str(config.path),
        "config_sha256": guided.stable_sha256(config.path),
        "implementation_sha256": _implementation_sha256(),
        "base_model40_config_sha256": MODEL40_CONFIG_SHA256,
        "base_model40_implementation_sha256": MODEL40_IMPLEMENTATION_SHA256,
        "base_model32_config_sha256": MODEL32_CONFIG_SHA256,
        "base_model32_implementation_sha256": MODEL32_IMPLEMENTATION_SHA256,
    }
    for key, value in expected.items():
        if run.get(key) != value:
            raise challenger.SafetyGateError(f"Model45 run-manifest mismatch for {key}")
    train_artifact, _ = require_draft_artifact(config, "train")
    if run.get("draft_artifact") != train_artifact.audit:
        raise challenger.SafetyGateError("adapter training draft receipt is detached")
    target = run.get("target_audit")
    target_modules = target.get("target_modules") if isinstance(target, Mapping) else None
    if not isinstance(target_modules, list) or (
        len(target_modules) != 326
        or target.get("trainable_parameters") != 14_610_432
        or target.get("trainable_tensors") != 652
    ):
        raise challenger.SafetyGateError("adapter lacks exact full-vision target audit")
    static_sequence = run.get("static_all_row_sequence_audit")
    if not isinstance(static_sequence, Mapping) or (
        static_sequence.get("status")
        != "receipted_every_row_tokenizer_geometry_audit_passed"
        or static_sequence.get("rows") != 4095
        or static_sequence.get("fits_configured_caps") is not True
    ):
        raise challenger.SafetyGateError("adapter lacks passing 4,095-row sequence audit")
    processor_preflight = run.get("processor_preflight")
    if not isinstance(processor_preflight, Mapping) or (
        processor_preflight.get("status")
        != "every_selected_row_real_processor_preflight_passed"
        or processor_preflight.get("rows") != 4095
    ):
        raise challenger.SafetyGateError("adapter lacks passing real-processor preflight")
    if best.get("adapter") != str(adapter):
        raise challenger.SafetyGateError("best.json names a different adapter")
    if best.get("run_manifest") != str(run_path) or best.get(
        "run_manifest_sha256"
    ) != guided.stable_sha256(run_path):
        raise challenger.SafetyGateError("best adapter is detached from run manifest")
    ledger = _directory_ledger(adapter)
    if best.get("adapter_files_sha256") != ledger:
        raise challenger.SafetyGateError("best adapter file ledger mismatch")
    if complete.get("observed_optimizer_updates") != 615 or complete.get(
        "expected_optimizer_updates"
    ) != 615:
        raise challenger.SafetyGateError("adapter did not complete 615 optimizer updates")
    boundaries = complete.get("observed_epoch_boundaries")
    if (
        not isinstance(boundaries, list)
        or not all(isinstance(row, Mapping) for row in boundaries)
        or [row.get("optimizer_step") for row in boundaries] != [205, 410, 615]
    ):
        raise challenger.SafetyGateError("adapter epoch boundaries are not 205/410/615")
    if complete.get("best_adapter") != str(adapter):
        raise challenger.SafetyGateError("completion artifact names a different best adapter")
    smoke = run.get("smoke")
    if not isinstance(smoke, Mapping):
        raise challenger.SafetyGateError("adapter lacks real memory smoke")
    allocated = smoke.get("peak_allocated_bytes")
    reserved = smoke.get("peak_reserved_bytes")
    if not isinstance(allocated, int) or not isinstance(reserved, int):
        raise challenger.SafetyGateError("adapter smoke memory peaks are invalid")
    challenger.enforce_memory_limits(
        allocated, reserved, max_allocated_gib=5.3, max_reserved_gib=5.3
    )
    final_allocated = complete.get("peak_allocated_bytes")
    final_reserved = complete.get("peak_reserved_bytes")
    if not isinstance(final_allocated, int) or not isinstance(final_reserved, int):
        raise challenger.SafetyGateError("completion artifact has invalid CUDA memory peaks")
    challenger.enforce_memory_limits(
        final_allocated,
        final_reserved,
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
        "complete_run_memory": {
            "peak_allocated_bytes": final_allocated,
            "peak_reserved_bytes": final_reserved,
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
            raise challenger.SafetyGateError("Model45 OOF requires canonical train folds")
    else:
        role = "test"
        if input_path != _canonical_path(config, role):
            raise challenger.SafetyGateError("fold-free Model45 inference requires Test.csv")
    expected_draft, expected_receipt = _draft_paths(config, role)
    if _absolute(draft_csv) != expected_draft or _absolute(draft_receipt) != expected_receipt:
        raise challenger.SafetyGateError(f"Model45 {role} requires canonical draft paths")
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
    static_sequence = audit_guided_sequence_budget(
        config, records, require_targets=fold is not None
    )
    if fold is not None:
        records = tuple(record for record in records if record.fold == fold)
        if len(records) != 819:
            raise challenger.SafetyGateError(f"OOF fold {fold} must have 819 rows")
    elif len(records) != 1374:
        raise challenger.SafetyGateError(f"Test.csv must have 1,374 rows, got {len(records)}")
    output = require_model45_output(output_csv)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model45 output: {output}")

    processor = challenger.load_processor(config.effective_model40)
    collator = FullVisualGuidedCollator(
        processor,
        str(_nested(config.raw, "prompt_template")),
        640,
        768,
    )
    processor_preflight = preflight_processor_sequences(
        collator,
        ((f"{role}_prompt", records, False),),
        generation_context_cap=816,
        max_new_tokens=48,
    )
    model, targets = visual._load_inference_model(
        config.effective_model40, _absolute(adapter_path)
    )
    predictions = challenger.greedy_predict(
        model,
        processor,
        collator,
        records,
        max_new_tokens=48,
        repetition_penalty=1.05,
        no_repeat_ngram_size=4,
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
        "static_sequence_audit": static_sequence,
        "processor_preflight": processor_preflight,
        "output": str(output),
        "output_sha256": guided.stable_sha256(output),
        "returning_draft_output": str(draft_output),
        "returning_draft_output_sha256": guided.stable_sha256(draft_output),
        "target_audit": asdict(targets),
        "image_policy": {
            "resize": "aspect_preserving_letterbox_lanczos",
            "effective_image_tokens": "520_to_640_in_current_corpus_audit",
            "max_image_tokens": 640,
        },
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
            raise challenger.SafetyGateError("Model45 submission requires canonical sample")
        submission = require_model45_submission(submission_path)
        if submission.exists():
            raise FileExistsError(f"refusing to overwrite Model45 submission: {submission}")
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

    parser = argparse.ArgumentParser(description="Model45 full-visual OCR-guided training")
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

    parser = argparse.ArgumentParser(description="Model45 full-visual OCR-guided inference")
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

    parser = argparse.ArgumentParser(description="Model45 CPU architecture/geometry audit")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--output", type=Path, default=MODEL45_OUTPUT_ROOT / "dry_run.json")
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    print(
        json.dumps(
            dry_run(args.config, args.output, fold=args.fold),
            indent=2,
            ensure_ascii=False,
        )
    )
