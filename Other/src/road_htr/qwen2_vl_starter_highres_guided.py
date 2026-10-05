"""Model47: receipt-gated Model43 with literal untrusted Model8 drafts.

This module composes two immutable, hash-pinned implementations. Model43 owns
the Qwen architecture, high-resolution letterbox, optimizer schedule, and CUDA
gates. Model32 owns the only accepted label-independent draft receipt. No CUDA
model is loaded until the receipt and every-row text/image preflights pass.
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
from . import qwen2_vl_starter_highres as base
from .submission import clean_prediction_preserving_internal_spaces, write_submission


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL47_OUTPUT_ROOT = (PROJECT_ROOT / "outputs" / "model47").resolve()
MODEL47_SUBMISSION_ROOT = (PROJECT_ROOT / "submissions" / "model47").resolve()

SCHEMA_VERSION = 1
MODEL43_CONFIG_SHA256 = (
    "deb30c37d4362917b30b7e3f03284e40062c05824c0132ab948a135813762e94"
)
MODEL43_IMPLEMENTATION_SHA256 = (
    "4da6449f5a4241deaa803305b53013480012717d0ff1b54100da2d7eb89aebda"
)
MODEL32_CONFIG_SHA256 = (
    "bedf6c4cc4401190e68f4b6094b1b1fdd431a253e21f2dbecbab98a9e3f416e1"
)
MODEL32_IMPLEMENTATION_SHA256 = (
    "015c5b0a8484aa14bf4ea92c5c3890deeefef4d13e1e7d4c7602d07dce78ed68"
)
MODEL32_PRODUCER_SHA256 = (
    "da2a76aa94db0be8ba46c574c4423ecb668b284f5d47aa09531bedb7952c47dd"
)
CANONICAL_TRAIN_SHA256 = guided.CANONICAL_TRAIN_SHA256
CANONICAL_TEST_SHA256 = guided.CANONICAL_TEST_SHA256
SAMPLE_SUBMISSION_SHA256 = guided.SAMPLE_SUBMISSION_SHA256
MODEL8_CHECKPOINT_SHA256 = guided.MODEL8_CHECKPOINT_SHA256
MAX_DRAFT_UTF8_BYTES = 160
MAX_DRAFT_QWEN_TOKENS = 160
BYTE_BOUNDARY_ALLOWANCE = 16
TRAINING_SEQUENCE_CAP = 1344
GENERATION_CONTEXT_CAP = 1392


@dataclass(frozen=True)
class Model47Config:
    raw: Mapping[str, Any]
    effective_model43: Mapping[str, Any]
    path: Path
    base_config_path: Path
    base_implementation_path: Path
    draft_config_path: Path
    draft_implementation_path: Path
    draft_producer_path: Path
    draft_config: guided.Model32Config


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"missing required Model47 config key: {'.'.join(keys)}")
        current = current[key]
    return current


def _below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def require_model47_output(path: str | Path) -> Path:
    resolved = _absolute(path)
    if not _below(resolved, MODEL47_OUTPUT_ROOT):
        raise challenger.SafetyGateError(
            f"Model47 output must remain under {MODEL47_OUTPUT_ROOT}: {resolved}"
        )
    return resolved


def require_model47_submission(path: str | Path) -> Path:
    resolved = _absolute(path)
    if not _below(resolved, MODEL47_SUBMISSION_ROOT):
        raise challenger.SafetyGateError(
            f"Model47 submission must remain under {MODEL47_SUBMISSION_ROOT}: {resolved}"
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
    output = require_model47_output(path)
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
            raise challenger.SafetyGateError(f"adapter may not contain a symlink: {path}")
        if path.is_file():
            ledger[path.relative_to(directory).as_posix()] = guided.stable_sha256(path)
    if not ledger:
        raise challenger.SafetyGateError(f"adapter directory is empty: {directory}")
    return ledger


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "model47" / "train.py",
        PROJECT_ROOT / "model47" / "infer.py",
        PROJECT_ROOT / "model47" / "audit.py",
        PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_starter_highres.py",
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


def load_config(path: str | Path) -> Model47Config:
    """Load Model47 and prove its Model43/Model32 dependencies are immutable."""

    resolved = _absolute(path)
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read Model47 configuration {resolved}: {error}") from error
    if not isinstance(raw, Mapping) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Model47 config must be a schema-version-1 mapping")

    base_config_path = _absolute(_nested(raw, "base_model43", "config"))
    base_implementation_path = _absolute(
        _nested(raw, "base_model43", "implementation")
    )
    draft_config_path = _absolute(_nested(raw, "base_model32_drafts", "config"))
    draft_implementation_path = _absolute(
        _nested(raw, "base_model32_drafts", "implementation")
    )
    draft_producer_path = _absolute(_nested(raw, "base_model32_drafts", "producer"))
    exact_paths = {
        base_config_path: (PROJECT_ROOT / "model43" / "config.yaml").resolve(),
        base_implementation_path: (
            PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_starter_highres.py"
        ).resolve(),
        draft_config_path: (PROJECT_ROOT / "model32" / "config.yaml").resolve(),
        draft_implementation_path: (
            PROJECT_ROOT / "src" / "road_htr" / "qwen2_vl_ocr_guided.py"
        ).resolve(),
        draft_producer_path: (PROJECT_ROOT / "model32" / "prepare_drafts.py").resolve(),
    }
    for actual, expected in exact_paths.items():
        if actual != expected:
            raise ValueError(f"Model47 pinned dependency path mismatch: {actual} != {expected}")
        if not actual.is_file():
            raise FileNotFoundError(actual)

    pins = {
        ("base_model43", "config_sha256"): MODEL43_CONFIG_SHA256,
        ("base_model43", "implementation_sha256"): MODEL43_IMPLEMENTATION_SHA256,
        ("base_model32_drafts", "config_sha256"): MODEL32_CONFIG_SHA256,
        ("base_model32_drafts", "implementation_sha256"): MODEL32_IMPLEMENTATION_SHA256,
        ("base_model32_drafts", "producer_sha256"): MODEL32_PRODUCER_SHA256,
        ("data", "train_csv_sha256"): CANONICAL_TRAIN_SHA256,
        ("data", "test_csv_sha256"): CANONICAL_TEST_SHA256,
        ("data", "sample_submission_sha256"): SAMPLE_SUBMISSION_SHA256,
        ("draft_source", "kind"): guided.DRAFT_KIND,
        ("draft_source", "receipt_schema_version"):
            guided.DRAFT_RECEIPT_SCHEMA_VERSION,
        ("draft_source", "empty_draft_policy"): guided.EMPTY_DRAFT_POLICY,
        ("draft_source", "checkpoint_sha256"): MODEL8_CHECKPOINT_SHA256,
    }
    for keys, expected in pins.items():
        actual = _nested(raw, *keys)
        if actual != expected:
            raise ValueError(
                f"Model47 pin {'.'.join(keys)} must be {expected!r}, got {actual!r}"
            )
    dependency_hashes = {
        base_config_path: MODEL43_CONFIG_SHA256,
        base_implementation_path: MODEL43_IMPLEMENTATION_SHA256,
        draft_config_path: MODEL32_CONFIG_SHA256,
        draft_implementation_path: MODEL32_IMPLEMENTATION_SHA256,
        draft_producer_path: MODEL32_PRODUCER_SHA256,
    }
    for dependency, expected in dependency_hashes.items():
        if guided.stable_sha256(dependency) != expected:
            raise challenger.SafetyGateError(f"pinned Model47 dependency changed: {dependency}")

    original, _ = base.load_config(base_config_path)
    draft_config = guided.load_config(draft_config_path)
    effective = copy.deepcopy(original)
    effective["train_csv"] = str(_nested(raw, "data", "train_csv"))
    effective["image_dir"] = str(_nested(raw, "data", "image_dir"))
    effective["output_dir"] = str(_nested(raw, "output_dir"))
    effective["prompt"] = str(_nested(raw, "prompt_template"))
    effective["smoke"]["report_dir"] = str(_nested(raw, "smoke_report_dir"))
    effective["image"]["max_sequence_tokens"] = TRAINING_SEQUENCE_CAP
    effective["image"]["max_generation_context_tokens"] = GENERATION_CONTEXT_CAP

    prompt = str(_nested(raw, "prompt_template"))
    if prompt.count("{draft}") != 1:
        raise ValueError("prompt_template must contain exactly one literal {draft}")
    lowered = prompt.casefold()
    if "{label}" in lowered or "{target}" in lowered:
        raise ValueError("prompt_template may not interpolate a challenge label")
    if not prompt.startswith("Transcribe EXACTLY as seen."):
        raise ValueError("Model47 must retain the organizer exact-transcription intent")
    if "untrusted" not in lowered or "pixels support it" not in lowered:
        raise ValueError("Model47 must mark the raw OCR draft as untrusted")
    placeholder = prompt.index("{draft}")
    if prompt[placeholder - 1 : placeholder] != "\n" or prompt[
        placeholder + len("{draft}") : placeholder + len("{draft}") + 1
    ] != "\n":
        raise ValueError("draft placeholder must be isolated by newline boundaries")

    exact_strings = {
        ("data", "train_csv"): "data/processed/train_folds.csv",
        ("data", "test_csv"): "data/raw/Test.csv",
        ("data", "sample_submission"): "data/raw/SampleSubmission.csv",
        ("data", "image_dir"): "data/raw/images",
        ("output_dir",): "outputs/model47/fold{fold}",
        ("smoke_report_dir",): "outputs/model47/smoke",
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
            raise ValueError(f"Model47 {'.'.join(keys)} must remain {expected!r}")

    expected_limits = {
        "max_utf8_bytes": MAX_DRAFT_UTF8_BYTES,
        "max_qwen_tokens": MAX_DRAFT_QWEN_TOKENS,
        "byte_fallback_boundary_allowance": BYTE_BOUNDARY_ALLOWANCE,
        "reject_chat_control_substrings": ["<|", "|>"],
        "require_nonempty": False,
        "literal_replacement": True,
    }
    if _nested(raw, "draft_limits") != expected_limits:
        raise ValueError("Model47 draft limits may not be weakened")
    expected_cv = {
        "folds": 5,
        "expected_total_rows": 4095,
        "expected_train_rows_per_fold": 3276,
        "expected_validation_rows_per_fold": 819,
        "selection_scope": "whole_fold_challenger_only",
        "forbid_per_row_oracle": True,
    }
    if _nested(raw, "cross_validation") != expected_cv:
        raise ValueError("Model47 cross-validation contract may not be weakened")
    expected_envelope = {
        "inherited_training_cap": 1152,
        "inherited_generation_context_cap": 1200,
        "configured_training_cap": TRAINING_SEQUENCE_CAP,
        "configured_generation_context_cap": GENERATION_CONTEXT_CAP,
        "model_context_cap": base.MODEL_CONTEXT_TOKENS,
        "min_effective_image_tokens": 896,
        "max_image_tokens": 1024,
        "max_new_tokens": 48,
        "require_byte_fallback_proof": True,
        "require_receipted_all_row_tokenizer_audit": True,
        "require_receipted_all_row_processor_preflight": True,
    }
    if _nested(raw, "sequence_envelope") != expected_envelope:
        raise ValueError("Model47 sequence envelope may not be weakened")
    expected_safety = {
        "offline": True,
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "double_quant": True,
        "visual_tower_frozen": True,
        "merger_frozen": True,
        "expected_lora_targets": base.EXPECTED_TARGET_MODULES,
        "expected_trainable_parameters": base.EXPECTED_TRAINABLE_PARAMETERS,
        "expected_trainable_tensors": base.EXPECTED_TRAINABLE_TENSORS,
        "batch_size": 1,
        "gradient_accumulation": 16,
        "epochs": 3,
        "learning_rate": 0.00003,
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
        raise ValueError("Model47 safety assertions may not be weakened")

    inherited = {
        ("architecture", "expected_target_modules"): 196,
        ("architecture", "expected_trainable_parameters"): 18_464_768,
        ("architecture", "expected_trainable_tensors"): 392,
        ("lora", "rank"): 16,
        ("lora", "include_merger"): False,
        ("lora", "include_visual_blocks"): False,
        ("image", "min_effective_image_tokens"): 896,
        ("image", "max_image_tokens"): 1024,
        ("image", "max_sequence_tokens"): TRAINING_SEQUENCE_CAP,
        ("image", "max_generation_context_tokens"): GENERATION_CONTEXT_CAP,
        ("training", "batch_size"): 1,
        ("training", "gradient_accumulation"): 16,
        ("training", "epochs"): 3,
        ("training", "learning_rate"): 0.00003,
        ("training", "expected_updates_per_epoch"): 205,
        ("training", "expected_total_updates"): 615,
        ("generation", "max_new_tokens"): 48,
        ("generation", "repetition_penalty"): 1.05,
        ("generation", "no_repeat_ngram_size"): 4,
    }
    for keys, expected in inherited.items():
        if challenger._nested(effective, *keys) != expected:
            raise challenger.SafetyGateError(
                f"effective Model43 contract drifted at {'.'.join(keys)}"
            )
    return Model47Config(
        raw=raw,
        effective_model43=effective,
        path=resolved,
        base_config_path=base_config_path,
        base_implementation_path=base_implementation_path,
        draft_config_path=draft_config_path,
        draft_implementation_path=draft_implementation_path,
        draft_producer_path=draft_producer_path,
        draft_config=draft_config,
    )


def _canonical_path(config: Model47Config, role: str) -> Path:
    keys = {"train": "train_csv", "test": "test_csv", "sample": "sample_submission"}
    if role not in keys:
        raise ValueError(f"unknown canonical role: {role}")
    return _absolute(_nested(config.raw, "data", keys[role]))


def _verify_canonical_data(config: Model47Config) -> dict[str, Any]:
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


def _draft_paths(config: Model47Config, role: str) -> tuple[Path, Path]:
    if role not in {"train", "test"}:
        raise ValueError("draft role must be train or test")
    return (
        _absolute(_nested(config.raw, "draft_source", f"{role}_csv")),
        _absolute(_nested(config.raw, "draft_source", f"{role}_receipt")),
    )


def inspect_draft_readiness(config: Model47Config, role: str) -> dict[str, Any]:
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
            "status": "missing_or_incomplete_canonical_receipt",
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
        "status": "strict_model32_receipt_verified",
        "missing": [],
        "required": [str(path) for path in required],
        "audit": artifact.audit,
    }


def require_draft_artifact(
    config: Model47Config, role: str
) -> tuple[guided.DraftArtifact, dict[str, Any]]:
    readiness = inspect_draft_readiness(config, role)
    if readiness["ready"] is not True:
        raise challenger.SafetyGateError(
            f"Model47 {role} draft receipt is not ready; missing={readiness['missing']}"
        )
    draft, receipt = _draft_paths(config, role)
    artifact = guided.verify_draft_artifact(
        config.draft_config, _canonical_path(config, role), draft, receipt
    )
    return artifact, readiness


class StarterHighresGuidedCollator(base.HighResolutionQwenCollator):
    """Model43 letterboxing with one literal, receipted raw draft per row."""

    def __init__(
        self,
        processor: Any,
        prompt_template: str,
        min_image_tokens: int,
        max_image_tokens: int,
        max_sequence_tokens: int,
    ):
        super().__init__(
            processor,
            prompt_template,
            min_image_tokens,
            max_image_tokens,
            max_sequence_tokens,
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


def _collator(config: Model47Config, processor: Any) -> StarterHighresGuidedCollator:
    return StarterHighresGuidedCollator(
        processor,
        str(_nested(config.raw, "prompt_template")),
        int(_nested(config.effective_model43, "image", "min_effective_image_tokens")),
        int(_nested(config.effective_model43, "image", "max_image_tokens")),
        int(_nested(config.effective_model43, "image", "max_sequence_tokens")),
    )


def _row_image_tokens(record: Any, config: Model47Config) -> dict[str, Any]:
    with Image.open(record.image_path) as image:
        width, height = image.size
    minimum = int(
        _nested(config.effective_model43, "image", "min_effective_image_tokens")
    )
    maximum = int(_nested(config.effective_model43, "image", "max_image_tokens"))
    plan = base.plan_letterbox(
        width,
        height,
        minimum_tokens=minimum,
        maximum_tokens=maximum,
    )
    tokens = base.qwen_grid_tokens(plan.canvas_width, plan.canvas_height)
    pixels = plan.canvas_width * plan.canvas_height
    min_pixels = int(_nested(config.effective_model43, "image", "min_pixels"))
    max_pixels = int(_nested(config.effective_model43, "image", "max_pixels"))
    if tokens != plan.planned_image_tokens or not minimum <= tokens <= maximum:
        raise challenger.SafetyGateError("Model47 planned image-token ledger differs")
    if not min_pixels <= pixels <= max_pixels:
        raise challenger.SafetyGateError("Model47 letterbox exceeds processor pixel interval")
    return {
        "image_tokens": tokens,
        "canvas_width": plan.canvas_width,
        "canvas_height": plan.canvas_height,
        "content_pixel_fraction": plan.content_pixel_fraction,
        "relative_content_aspect_error": plan.relative_content_aspect_error,
    }


def _load_tokenizer(config: Model47Config) -> Any:
    from transformers import AutoTokenizer

    checkpoint = _absolute(_nested(config.effective_model43, "model", "local_path"))
    return AutoTokenizer.from_pretrained(
        str(checkpoint), trust_remote_code=False, local_files_only=True
    )


def _verify_byte_fallback(tokenizer: Any) -> dict[str, Any]:
    """Prove that every possible UTF-8 byte has a one-token fallback symbol."""

    from tokenizers.pre_tokenizers import ByteLevel

    alphabet = set(ByteLevel.alphabet())
    vocabulary = tokenizer.get_vocab()
    pretokenizer = repr(tokenizer.backend_tokenizer.pre_tokenizer)
    decoder = repr(tokenizer.backend_tokenizer.decoder)
    if tokenizer.__class__.__name__ != "Qwen2TokenizerFast":
        raise challenger.SafetyGateError("conservative proof requires Qwen2TokenizerFast")
    if len(alphabet) != 256 or not alphabet.issubset(vocabulary):
        raise challenger.SafetyGateError("tokenizer lacks the complete byte fallback alphabet")
    if "ByteLevel" not in pretokenizer or "ByteLevel" not in decoder:
        raise challenger.SafetyGateError("tokenizer is not byte-level at encode/decode")
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "byte_alphabet_symbols": len(alphabet),
        "byte_alphabet_symbols_in_vocabulary": len(alphabet & set(vocabulary)),
        "byte_level_pretokenizer": True,
        "byte_level_decoder": True,
    }


def _scaffold_tokens(
    tokenizer: Any,
    prompt_template: str,
    *,
    label: str | None,
    image_tokens: int,
) -> int:
    """Count fixed template tokens around an isolated synthetic draft slot."""

    sentinel = "MODEL47DRAFTSLOT6E4C19B1"
    if sentinel in prompt_template:
        raise AssertionError("draft sentinel collides with prompt")
    messages = guided.build_guided_messages(prompt_template, sentinel, label)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=label is None,
    )
    if rendered.count(sentinel) != 1:
        raise challenger.SafetyGateError("chat template did not preserve one draft slot")
    prefix, suffix = rendered.split(sentinel)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    image_pad = tokenizer.convert_tokens_to_ids(base.IMAGE_PAD_TOKEN)
    if prefix_ids.count(image_pad) + suffix_ids.count(image_pad) != 1:
        raise challenger.SafetyGateError("guided scaffold must contain one image pad")
    return len(prefix_ids) + len(suffix_ids) - 1 + image_tokens


def prove_conservative_sequence_envelope(
    config: Model47Config, records: Sequence[Any]
) -> dict[str, Any]:
    """Prove caps before drafts exist using a hard UTF-8 byte ceiling.

    The placeholder is isolated between newlines. With the pinned byte-level
    tokenizer, any permitted draft has a valid segmentation using at most one
    token per UTF-8 byte. Sixteen extra tokens are retained as a fail-safe
    boundary allowance even though the newline isolation prevents cross-slot
    pretokenizer coupling.
    """

    if not records or any(record.label is None for record in records):
        raise ValueError("conservative envelope requires all canonical labels")
    tokenizer = _load_tokenizer(config)
    byte_fallback = _verify_byte_fallback(tokenizer)
    template = str(_nested(config.raw, "prompt_template"))
    max_image = int(_nested(config.effective_model43, "image", "max_image_tokens"))
    training_cap = int(
        _nested(config.effective_model43, "image", "max_sequence_tokens")
    )
    context_cap = int(
        _nested(config.effective_model43, "image", "max_generation_context_tokens")
    )
    max_new = int(_nested(config.effective_model43, "generation", "max_new_tokens"))
    max_training_scaffold = (-1, "")
    max_target = (-1, "")
    for record in records:
        scaffold = _scaffold_tokens(
            tokenizer,
            template,
            label=record.label,
            image_tokens=max_image,
        )
        max_training_scaffold = max(
            max_training_scaffold, (scaffold, record.identifier)
        )
        target_tokens = len(
            tokenizer.encode(record.label, add_special_tokens=False)
        )
        max_target = max(max_target, (target_tokens, record.identifier))
    generation_scaffold = _scaffold_tokens(
        tokenizer, template, label=None, image_tokens=max_image
    )
    draft_allowance = MAX_DRAFT_UTF8_BYTES + BYTE_BOUNDARY_ALLOWANCE
    training_upper = max_training_scaffold[0] + draft_allowance
    generation_prompt_upper = generation_scaffold + draft_allowance
    generation_context_upper = generation_prompt_upper + max_new
    fits = (
        training_upper <= training_cap
        and generation_context_upper <= context_cap
        and max_new >= max_target[0] + 4
        and context_cap <= base.MODEL_CONTEXT_TOKENS
    )
    if not fits:
        raise challenger.SafetyGateError(
            "Model47 conservative maximum-draft envelope exceeds configured caps"
        )
    return {
        "status": "conservative_maximum_draft_envelope_proved",
        "records": len(records),
        "proof": {
            **byte_fallback,
            "draft_slot_isolated_by_newlines": True,
            "maximum_draft_utf8_bytes": MAX_DRAFT_UTF8_BYTES,
            "one_fallback_token_per_utf8_byte": True,
            "extra_boundary_token_allowance": BYTE_BOUNDARY_ALLOWANCE,
        },
        "maximum_training_scaffold_tokens": {
            "value": max_training_scaffold[0],
            "record_id": max_training_scaffold[1],
        },
        "generation_scaffold_tokens": generation_scaffold,
        "maximum_target_tokens": {
            "value": max_target[0],
            "record_id": max_target[1],
        },
        "proved_upper_bounds": {
            "training_sequence_tokens": training_upper,
            "generation_prompt_tokens": generation_prompt_upper,
            "generation_context_tokens": generation_context_upper,
        },
        "configured_caps": {
            "training_sequence_tokens": training_cap,
            "generation_context_tokens": context_cap,
            "model_context_tokens": base.MODEL_CONTEXT_TOKENS,
            "max_new_tokens": max_new,
        },
        "headroom": {
            "training_tokens": training_cap - training_upper,
            "generation_context_tokens": context_cap - generation_context_upper,
            "generation_target_tokens": max_new - max_target[0],
        },
    }


def _validate_draft(tokenizer: Any, record: guided.GuidedOcrRecord) -> tuple[int, int]:
    draft = record.draft
    if not isinstance(draft, str):
        raise challenger.SafetyGateError(f"missing/non-string draft for {record.identifier!r}")
    if clean_prediction_preserving_internal_spaces(draft) != draft:
        raise challenger.SafetyGateError(f"draft edges/line controls changed for {record.identifier!r}")
    for forbidden in ("<|", "|>"):
        if forbidden in draft:
            raise challenger.SafetyGateError(
                f"draft contains a chat-control substring for {record.identifier!r}"
            )
    byte_count = len(draft.encode("utf-8"))
    token_count = len(tokenizer.encode(draft, add_special_tokens=False))
    if byte_count > MAX_DRAFT_UTF8_BYTES:
        raise challenger.SafetyGateError(
            f"draft exceeds {MAX_DRAFT_UTF8_BYTES} UTF-8 bytes for {record.identifier!r}"
        )
    if token_count > MAX_DRAFT_QWEN_TOKENS:
        raise challenger.SafetyGateError(
            f"draft exceeds {MAX_DRAFT_QWEN_TOKENS} Qwen tokens for {record.identifier!r}"
        )
    return byte_count, token_count


def audit_receipted_sequences(
    config: Model47Config,
    records: Sequence[guided.GuidedOcrRecord],
    *,
    require_targets: bool,
) -> dict[str, Any]:
    """Audit every receipted row with its exact draft, label, and image grid."""

    if not records:
        raise ValueError("receipted sequence audit requires at least one row")
    tokenizer = _load_tokenizer(config)
    _verify_byte_fallback(tokenizer)
    template = str(_nested(config.raw, "prompt_template"))
    training_cap = int(
        _nested(config.effective_model43, "image", "max_sequence_tokens")
    )
    context_cap = int(
        _nested(config.effective_model43, "image", "max_generation_context_tokens")
    )
    max_new = int(_nested(config.effective_model43, "generation", "max_new_tokens"))
    maxima: dict[str, dict[str, Any]] = {}
    image_values: list[int] = []

    def update(key: str, value: int, record: Any) -> None:
        previous = maxima.get(key)
        if previous is None or value > int(previous["value"]):
            maxima[key] = {"value": value, "record_id": record.identifier}

    for record in records:
        if require_targets and record.label is None:
            raise challenger.SafetyGateError(
                f"receipted training audit lacks target for {record.identifier!r}"
            )
        draft_bytes, draft_tokens = _validate_draft(tokenizer, record)
        update("draft_utf8_bytes", draft_bytes, record)
        update("draft_qwen_tokens", draft_tokens, record)
        geometry = _row_image_tokens(record, config)
        image_tokens = int(geometry["image_tokens"])
        image_values.append(image_tokens)
        update("image_tokens", image_tokens, record)
        prompt_tokens = base._template_tokens(
            tokenizer,
            guided.build_guided_messages(template, record.draft, None),
            generation_prompt=True,
            image_tokens=image_tokens,
        )
        generation_context = prompt_tokens + max_new
        update("prompt_tokens", prompt_tokens, record)
        update("generation_context_tokens", generation_context, record)
        if generation_context > context_cap:
            raise challenger.SafetyGateError(
                f"receipted prompt exceeds generation context for {record.identifier!r}"
            )
        if require_targets:
            training_tokens = base._template_tokens(
                tokenizer,
                guided.build_guided_messages(template, record.draft, record.label),
                generation_prompt=False,
                image_tokens=image_tokens,
            )
            target_tokens = len(
                tokenizer.encode(record.label or "", add_special_tokens=False)
            )
            update("training_sequence_tokens", training_tokens, record)
            update("target_tokens", target_tokens, record)
            if training_tokens > training_cap:
                raise challenger.SafetyGateError(
                    f"receipted training sequence exceeds cap for {record.identifier!r}"
                )
            if target_tokens + 4 > max_new:
                raise challenger.SafetyGateError(
                    f"generation headroom is insufficient for {record.identifier!r}"
                )
    image_values.sort()
    return {
        "status": "receipted_every_row_exact_tokenizer_geometry_audit_passed",
        "rows": len(records),
        "targets_required": require_targets,
        "exact_pinned_tokenizer_per_row": True,
        "exact_receipted_raw_draft_per_row": True,
        "exact_letterbox_geometry_per_row": True,
        "draft_limits": {
            "max_utf8_bytes": MAX_DRAFT_UTF8_BYTES,
            "max_qwen_tokens": MAX_DRAFT_QWEN_TOKENS,
        },
        "image_tokens": {
            "minimum": image_values[0],
            "median": image_values[len(image_values) // 2],
            "maximum": image_values[-1],
        },
        "maxima": maxima,
        "configured_caps": {
            "training_sequence_tokens": training_cap,
            "generation_context_tokens": context_cap,
            "max_new_tokens": max_new,
        },
    }


def preflight_processor_sequences(
    collator: StarterHighresGuidedCollator,
    groups: Sequence[tuple[str, Sequence[guided.GuidedOcrRecord], bool]],
    *,
    expected_rows: int,
    generation_context_cap: int = GENERATION_CONTEXT_CAP,
    max_new_tokens: int = 48,
) -> dict[str, Any]:
    """Encode every intended row with the real processor before CUDA loading."""

    global_maxima: dict[str, dict[str, Any]] = {}
    group_reports: dict[str, dict[str, Any]] = {}
    total = 0

    def update(
        container: dict[str, dict[str, Any]],
        key: str,
        value: int,
        record: Any,
        role: str,
    ) -> None:
        previous = container.get(key)
        if previous is None or value > int(previous["value"]):
            container[key] = {
                "value": value,
                "record_id": record.identifier,
                "role": role,
            }

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
                        f"invalid processor {key} for {record.identifier!r}: {value!r}"
                    )
                update(global_maxima, key, value, record, role)
                update(role_maxima, key, value, record, role)
            prompt_tokens = (
                diagnostics.get("prompt_tokens")
                if include_label
                else diagnostics.get("sequence_tokens")
            )
            if (
                isinstance(prompt_tokens, bool)
                or not isinstance(prompt_tokens, int)
                or prompt_tokens <= 0
            ):
                raise challenger.SafetyGateError(
                    f"invalid processor prompt length for {record.identifier!r}"
                )
            generation_upper = prompt_tokens + max_new_tokens
            if generation_upper > generation_context_cap:
                raise challenger.SafetyGateError(
                    f"processor prompt+generation exceeds cap for {record.identifier!r}"
                )
            update(global_maxima, "generation_context_tokens", generation_upper, record, role)
            update(role_maxima, "generation_context_tokens", generation_upper, record, role)
            if include_label:
                supervised = diagnostics.get("supervised_tokens")
                if (
                    isinstance(supervised, bool)
                    or not isinstance(supervised, int)
                    or supervised <= 0
                ):
                    raise challenger.SafetyGateError(
                        f"invalid supervised tokens for {record.identifier!r}"
                    )
                update(global_maxima, "supervised_tokens", supervised, record, role)
                update(role_maxima, "supervised_tokens", supervised, record, role)
        group_reports[role] = {"rows": len(records), "maxima": role_maxima}
    if total != expected_rows:
        raise challenger.SafetyGateError(
            f"processor preflight covered {total} rows, expected {expected_rows}"
        )
    if int(global_maxima["image_tokens"]["value"]) > 1024:
        raise challenger.SafetyGateError("processor emitted more than 1024 image tokens")
    if int(global_maxima["sequence_tokens"]["value"]) > TRAINING_SEQUENCE_CAP:
        raise challenger.SafetyGateError("processor emitted more than 1344 sequence tokens")
    return {
        "status": "every_selected_row_real_processor_preflight_passed",
        "rows": total,
        "groups": group_reports,
        "maxima": global_maxima,
        "caps": {
            "image_tokens": 1024,
            "training_sequence_tokens": TRAINING_SEQUENCE_CAP,
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
        raise challenger.SafetyGateError(f"Model47 requires 4,095 rows, got {len(records)}")
    for fold in range(5):
        training, validation = base.split_outer_fold(records, fold)
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
    output = require_model47_output(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model47 prediction: {output}")
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
    collator: StarterHighresGuidedCollator,
    validation_records: Sequence[guided.GuidedOcrRecord],
    output_dir: Path,
    run_manifest_path: Path,
    config: Model47Config,
) -> Any:
    from transformers import TrainerCallback

    generation = _nested(config.effective_model43, "generation")
    max_new_tokens = int(_nested(generation, "max_new_tokens"))
    repetition_penalty = float(_nested(generation, "repetition_penalty"))
    no_repeat_ngram_size = int(_nested(generation, "no_repeat_ngram_size"))

    class StarterHighresGuidedMetricCallback(TrainerCallback):
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
                "returning_draft_is_report_only": True,
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
                            "report-only and no per-row oracle is used"
                        ),
                    },
                )
            print(
                f"Model47 validation epoch={epoch} challenger={score:.9f} "
                f"draft={comparison['returning_draft']['multi_score']:.9f} "
                f"best_epoch={self.best_epoch}",
                flush=True,
            )
            return control

    return StarterHighresGuidedMetricCallback()


def dry_run(
    config_path: str | Path,
    output_path: str | Path,
    *,
    fold: int | None = None,
) -> dict[str, Any]:
    """Run CPU audits and report the receipt gate without inventing a score."""

    challenger.enable_offline_mode()
    config = load_config(config_path)
    runtime = base.audit_runtime_versions()
    data = _verify_canonical_data(config)
    provenance = challenger.verify_starter_provenance(config.effective_model43)
    checkpoint = challenger.verify_local_checkpoint(config.effective_model43)
    targets = base.audit_checkpoint_target_shapes(config.effective_model43)
    context = base.audit_model_context(config.effective_model43)
    records = challenger.load_records(
        _canonical_path(config, "train"),
        _absolute(_nested(config.raw, "data", "image_dir")),
        require_target=True,
        require_five_folds=True,
    )
    _validate_fold_counts(records)
    geometry = base.audit_geometry(
        _nested(config.raw, "data", "image_dir"), config.effective_model43
    )
    conservative = prove_conservative_sequence_envelope(config, records)
    folds = range(5) if fold is None else (fold,)
    if fold is not None and fold not in range(5):
        raise ValueError("fold must be one of 0,1,2,3,4")
    fold_audit: dict[str, Any] = {}
    for current_fold in folds:
        training, validation = base.split_outer_fold(records, current_fold)
        schedule = base.deterministic_optimizer_schedule(
            len(training),
            int(_nested(config.effective_model43, "training", "gradient_accumulation")),
            int(_nested(config.effective_model43, "training", "epochs")),
        )
        base.verify_configured_schedule(config.effective_model43, schedule)
        fold_audit[str(current_fold)] = {
            "train_rows": len(training),
            "validation_rows": len(validation),
            "optimizer_schedule": asdict(schedule),
        }

    readiness = {
        role: inspect_draft_readiness(config, role) for role in ("train", "test")
    }
    sequence_audits: dict[str, Any] = {}
    processor_preflights: dict[str, Any] = {}
    processor: Any | None = None
    for role, expected_rows, require_targets in (
        ("train", 4095, True),
        ("test", 1374, False),
    ):
        if not readiness[role]["ready"]:
            sequence_audits[role] = {
                "status": f"blocked_until_strict_model32_{role}_receipt_exists",
                "rows_audited": 0,
            }
            processor_preflights[role] = {
                "status": f"blocked_until_strict_model32_{role}_receipt_exists",
                "rows_preflighted": 0,
            }
            continue
        draft_path, receipt_path = _draft_paths(config, role)
        guided_records, _ = guided.load_guided_records(
            config.draft_config,
            _canonical_path(config, role),
            draft_path,
            receipt_path,
            require_target=require_targets,
            require_five_folds=require_targets,
        )
        if len(guided_records) != expected_rows:
            raise challenger.SafetyGateError(
                f"canonical {role} receipt has {len(guided_records)} rows"
            )
        sequence_audits[role] = audit_receipted_sequences(
            config, guided_records, require_targets=require_targets
        )
        if processor is None:
            processor = base.load_processor(config.effective_model43)
        processor_preflights[role] = preflight_processor_sequences(
            _collator(config, processor),
            ((f"all_{role}_{'with_target' if require_targets else 'prompt'}", guided_records, require_targets),),
            expected_rows=expected_rows,
        )

    if readiness["train"]["ready"] and readiness["test"]["ready"]:
        status = "all_cpu_receipt_and_preflight_audits_passed_gpu_smoke_not_run"
    elif readiness["train"]["ready"]:
        status = "train_cpu_audits_passed_test_receipt_missing_gpu_smoke_not_run"
    else:
        status = "blocked_on_strict_model32_train_draft_receipt"
    report = {
        "schema_version": SCHEMA_VERSION,
        "model": "model47_model43_highres_with_receipted_raw_model8_draft",
        "status": status,
        "score": None,
        "score_status": "no_oof_validation_or_leaderboard_score",
        "gpu_started": False,
        "resolved_config": str(config.path),
        "config_sha256": guided.stable_sha256(config.path),
        "implementation_sha256": _implementation_sha256(),
        "runtime": runtime,
        "dependency_hashes": {
            "model43_config": MODEL43_CONFIG_SHA256,
            "model43_implementation": MODEL43_IMPLEMENTATION_SHA256,
            "model32_config": MODEL32_CONFIG_SHA256,
            "model32_implementation": MODEL32_IMPLEMENTATION_SHA256,
            "model32_producer": MODEL32_PRODUCER_SHA256,
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
        "conservative_sequence_envelope": conservative,
        "folds": fold_audit,
        "draft_readiness": readiness,
        "receipted_sequence_audits": sequence_audits,
        "real_processor_preflights": processor_preflights,
        "memory_gate": {
            "allocated_gib_exclusive": 5.3,
            "reserved_gib_exclusive": 5.3,
            "status": "blocked_before_gpu_smoke_until_train_receipt_passes",
        },
        "leakage_contract": (
            "only Model32's frozen full-view Model8 image-only receipt is accepted; "
            "the raw draft is never selected per row against challenge labels"
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
            MODEL47_OUTPUT_ROOT / f"dry_run_fold{fold}.json",
            fold=fold,
        )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    challenger.enable_offline_mode()
    config = load_config(config_path)
    initial_config_sha256 = guided.stable_sha256(config.path)
    initial_implementation_sha256 = _implementation_sha256()
    runtime = base.audit_runtime_versions()
    data_audit = _verify_canonical_data(config)

    # All receipt, draft-limit, exact-tokenizer, and exact-processor gates run
    # before CUDA model allocation can begin.
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
    conservative = prove_conservative_sequence_envelope(config, records)
    static_sequences = audit_receipted_sequences(
        config, records, require_targets=True
    )
    training_records, validation_records = base.split_outer_fold(records, fold)
    schedule = base.deterministic_optimizer_schedule(
        len(training_records),
        int(_nested(config.effective_model43, "training", "gradient_accumulation")),
        int(_nested(config.effective_model43, "training", "epochs")),
    )
    base.verify_configured_schedule(config.effective_model43, schedule)
    output_dir = require_model47_output(
        str(_nested(config.raw, "output_dir")).format(fold=fold)
    )
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {output_dir}")
    smoke_path = require_model47_output(
        _absolute(_nested(config.raw, "smoke_report_dir")) / f"fold{fold}.json"
    )
    if smoke_only and smoke_path.exists():
        raise FileExistsError(f"refusing to overwrite Model47 smoke report: {smoke_path}")

    provenance = challenger.verify_starter_provenance(config.effective_model43)
    checkpoint = challenger.verify_local_checkpoint(config.effective_model43)
    header_targets = base.audit_checkpoint_target_shapes(config.effective_model43)
    context = base.audit_model_context(config.effective_model43)
    processor = base.load_processor(config.effective_model43)
    collator = _collator(config, processor)
    processor_preflight = preflight_processor_sequences(
        collator,
        (
            ("outer_train_with_target", training_records, True),
            ("outer_validation_prompt", validation_records, False),
        ),
        expected_rows=4095,
    )

    seed = int(_nested(config.effective_model43, "seed"))
    challenger.set_seed(seed)
    model, model_audit = base.load_training_model(config.effective_model43)
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
        "model": "model47_model43_highres_with_receipted_raw_model8_draft",
        "model_number": 47,
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
        "base_model43_config_sha256": MODEL43_CONFIG_SHA256,
        "base_model43_implementation_sha256": MODEL43_IMPLEMENTATION_SHA256,
        "base_model32_config_sha256": MODEL32_CONFIG_SHA256,
        "base_model32_implementation_sha256": MODEL32_IMPLEMENTATION_SHA256,
        "base_model32_producer_sha256": MODEL32_PRODUCER_SHA256,
        "runtime": runtime,
        "canonical_data": data_audit,
        "draft_artifact": draft_audit,
        "draft_readiness": draft_readiness,
        "conservative_sequence_envelope": conservative,
        "static_all_row_sequence_audit": static_sequences,
        "processor_preflight": processor_preflight,
        "starter_provenance": provenance,
        "local_checkpoint": checkpoint,
        "target_audit": asdict(header_targets),
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
            **challenger._runtime_manifest(config.effective_model43),
            **runtime_audit,
            "resolved_model47_config": str(config.path),
            "effective_model43_config": config.effective_model43,
            "prompt_template": str(_nested(config.raw, "prompt_template")),
        },
    )
    _atomic_predictions(
        output_dir / "validation" / "returning_raw_model8_draft.csv",
        validation_records,
        [record.draft for record in validation_records],
    )

    class ExactEpochTrainer(base.ExactEpochScheduleMixin, Trainer):
        pass

    schedule_callback = base.make_schedule_callback(schedule)
    metric_callback = _make_validation_callback(
        processor,
        collator,
        validation_records,
        output_dir,
        run_manifest_path,
        config,
    )
    training_config = _nested(config.effective_model43, "training")
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
        full_determinism=True,
        num_train_epochs=float(_nested(training_config, "epochs")),
        max_steps=schedule.total_optimizer_updates,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=0.00003,
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
        raise challenger.SafetyGateError("Model47 did not finish exactly 615 updates")
    if metric_callback.best_epoch is None:
        raise challenger.SafetyGateError("Model47 ended without complete-fold validation")
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)
    if guided.stable_sha256(config.path) != initial_config_sha256:
        raise RuntimeError("Model47 config changed during training")
    if _implementation_sha256() != initial_implementation_sha256:
        raise RuntimeError("Model47 implementation or dependency changed during training")
    current_artifact, _ = require_draft_artifact(config, "train")
    if current_artifact.audit != draft_audit:
        raise RuntimeError("Model47 train draft artifact changed during training")
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
        "optimizer_schedule": asdict(schedule),
        "observed_epoch_boundaries": schedule_callback.boundaries,
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
    config: Model47Config,
    adapter_path: str | Path,
    *,
    oof_fold: int | None,
) -> dict[str, Any]:
    adapter = _absolute(adapter_path)
    if not _below(adapter, MODEL47_OUTPUT_ROOT) or adapter.name != "best_adapter":
        raise challenger.SafetyGateError(
            "Model47 accepts only outputs/model47/foldN/best_adapter"
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
        raise challenger.SafetyGateError("Model47 run manifest has invalid fold")
    expected_dir = require_model47_output(
        str(_nested(config.raw, "output_dir")).format(fold=training_fold)
    )
    if run_dir != expected_dir:
        raise challenger.SafetyGateError("Model47 adapter is outside its fold directory")
    if oof_fold is not None and training_fold != oof_fold:
        raise challenger.SafetyGateError(
            f"fold {oof_fold} OOF cannot use fold {training_fold} adapter"
        )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "model_number": 47,
        "training_excluded_folds": [training_fold],
        "config": str(config.path),
        "config_sha256": guided.stable_sha256(config.path),
        "implementation_sha256": _implementation_sha256(),
        "base_model43_config_sha256": MODEL43_CONFIG_SHA256,
        "base_model43_implementation_sha256": MODEL43_IMPLEMENTATION_SHA256,
        "base_model32_config_sha256": MODEL32_CONFIG_SHA256,
        "base_model32_implementation_sha256": MODEL32_IMPLEMENTATION_SHA256,
        "base_model32_producer_sha256": MODEL32_PRODUCER_SHA256,
    }
    for key, value in expected.items():
        if run.get(key) != value:
            raise challenger.SafetyGateError(f"Model47 run-manifest mismatch for {key}")
    train_artifact, _ = require_draft_artifact(config, "train")
    if run.get("draft_artifact") != train_artifact.audit:
        raise challenger.SafetyGateError("adapter training draft receipt is detached")
    target = run.get("target_audit")
    modules = target.get("target_modules") if isinstance(target, Mapping) else None
    if not isinstance(modules, list) or (
        len(modules) != 196
        or target.get("language_modules") != 196
        or target.get("visual_modules") != 0
        or target.get("merger_modules") != 0
        or target.get("trainable_parameters") != 18_464_768
        or target.get("trainable_tensors") != 392
    ):
        raise challenger.SafetyGateError("adapter lacks exact Model43 target audit")
    conservative = run.get("conservative_sequence_envelope")
    if not isinstance(conservative, Mapping) or conservative.get("status") != (
        "conservative_maximum_draft_envelope_proved"
    ):
        raise challenger.SafetyGateError("adapter lacks conservative draft envelope")
    static_sequences = run.get("static_all_row_sequence_audit")
    if not isinstance(static_sequences, Mapping) or (
        static_sequences.get("status")
        != "receipted_every_row_exact_tokenizer_geometry_audit_passed"
        or static_sequences.get("rows") != 4095
    ):
        raise challenger.SafetyGateError("adapter lacks passing 4,095-row receipt audit")
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
        raise challenger.SafetyGateError("adapter did not complete 615 updates")
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
        allocated,
        reserved,
        max_allocated_gib=5.3,
        max_reserved_gib=5.3,
    )
    final_allocated = complete.get("peak_allocated_bytes")
    final_reserved = complete.get("peak_reserved_bytes")
    if not isinstance(final_allocated, int) or not isinstance(final_reserved, int):
        raise challenger.SafetyGateError("completion artifact memory peaks are invalid")
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
    runtime = base.audit_runtime_versions()
    canonical = _verify_canonical_data(config)
    input_path = _absolute(input_csv)
    if fold is not None:
        if fold not in range(5):
            raise ValueError("fold must be one of 0,1,2,3,4")
        role = "train"
        if input_path != _canonical_path(config, role):
            raise challenger.SafetyGateError("Model47 OOF requires canonical train folds")
    else:
        role = "test"
        if input_path != _canonical_path(config, role):
            raise challenger.SafetyGateError("fold-free Model47 inference requires Test.csv")
    expected_draft, expected_receipt = _draft_paths(config, role)
    if _absolute(draft_csv) != expected_draft or _absolute(draft_receipt) != expected_receipt:
        raise challenger.SafetyGateError(f"Model47 {role} requires canonical draft paths")
    require_draft_artifact(config, role)
    lineage = verify_adapter_lineage(config, adapter_path, oof_fold=fold)
    all_records, current_draft_audit = guided.load_guided_records(
        config.draft_config,
        input_path,
        expected_draft,
        expected_receipt,
        require_target=fold is not None,
        require_five_folds=fold is not None,
    )
    expected_rows = 4095 if role == "train" else 1374
    if len(all_records) != expected_rows:
        raise challenger.SafetyGateError(
            f"canonical {role} requires {expected_rows} rows, got {len(all_records)}"
        )
    static_sequences = audit_receipted_sequences(
        config, all_records, require_targets=fold is not None
    )

    # Every row authenticated by the current receipt is processor-preflighted
    # before inference model loading, even when emitting only one OOF fold.
    processor = base.load_processor(config.effective_model43)
    collator = _collator(config, processor)
    processor_preflight = preflight_processor_sequences(
        collator,
        ((f"all_{role}_prompt", all_records, False),),
        expected_rows=expected_rows,
    )
    if fold is not None:
        records = tuple(record for record in all_records if record.fold == fold)
        if len(records) != 819:
            raise challenger.SafetyGateError(f"OOF fold {fold} must have 819 rows")
    else:
        records = tuple(all_records)
    output = require_model47_output(output_csv)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Model47 output: {output}")

    model, targets = base._load_inference_model(
        config.effective_model43, _absolute(adapter_path)
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
        "runtime": runtime,
        "canonical_data": canonical,
        "adapter_lineage": lineage,
        "draft_artifact": current_draft_audit,
        "static_all_receipted_row_sequence_audit": static_sequences,
        "processor_preflight": processor_preflight,
        "output": str(output),
        "output_sha256": guided.stable_sha256(output),
        "returning_draft_output": str(draft_output),
        "returning_draft_output_sha256": guided.stable_sha256(draft_output),
        "target_audit": asdict(targets),
        "image_policy": {
            "resize": "aspect_preserving_letterbox_lanczos",
            "min_effective_image_tokens": 896,
            "max_image_tokens": 1024,
        },
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 48,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 4,
        },
        "per_row_selection_performed": False,
        "emitted_predictions": "model47_challenger",
    }
    if fold is not None:
        report["comparison"] = guided.compare_fold_to_draft(records, predictions)
    _atomic_json(output.with_suffix(".metrics.json"), report)
    if submission_path is not None:
        if sample_path is None:
            raise ValueError("--sample is required with --submission")
        sample = _absolute(sample_path)
        if sample != _canonical_path(config, "sample"):
            raise challenger.SafetyGateError("Model47 submission requires canonical sample")
        submission = require_model47_submission(submission_path)
        if submission.exists():
            raise FileExistsError(f"refusing to overwrite Model47 submission: {submission}")
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

    parser = argparse.ArgumentParser(description="Model47 high-resolution OCR-guided training")
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
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

    parser = argparse.ArgumentParser(description="Model47 high-resolution OCR-guided inference")
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

    parser = argparse.ArgumentParser(description="Model47 CPU receipt/sequence audit")
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    parser.add_argument(
        "--output", type=Path, default=MODEL47_OUTPUT_ROOT / "dry_run.json"
    )
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    print(
        json.dumps(
            dry_run(args.config, args.output, fold=args.fold),
            indent=2,
            ensure_ascii=False,
        )
    )
