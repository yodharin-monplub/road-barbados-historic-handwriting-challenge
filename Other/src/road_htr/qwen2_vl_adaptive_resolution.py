"""Model50: label-free adaptive-resolution successor to Model43.

The adapter architecture and optimization protocol retain Model43's exact
organizer-aligned rank-16 language-only contract.  Image resolution is selected
before tokenization using only immutable source geometry: native-detail and
extreme-aspect crops receive 896..1024 visual tokens, while ordinary resized-low
crops receive 512..640.  No target, identifier, prediction, or confidence can
affect the selection.

This module is deliberately separate from Model43.  It reuses its audited,
read-only architecture primitives but does not modify or monkeypatch Model43.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image

from .qwen2_vl_challenger import (
    GIB,
    LOCAL_CHECKPOINT_FILES,
    LOCAL_CHECKPOINT_PATH,
    MODEL_CLASS,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    PROJECT_ROOT,
    QwenOcrCollator,
    RecordDataset,
    SafetyGateError,
    _absolute,
    _json_write,
    _make_validation_callback,
    _nested,
    _runtime_manifest,
    _sha256,
    build_messages,
    enable_offline_mode,
    exact_metrics,
    greedy_predict,
    load_records,
    reject_model22_destination,
    require_new_output_dir,
    run_memory_smoke,
    set_seed,
    verify_local_checkpoint,
    verify_processor_budget,
    verify_starter_provenance,
    write_predictions,
)
from .qwen2_vl_starter_highres import (
    EXPECTED_TARGET_MODULES,
    EXPECTED_TRAINABLE_PARAMETERS,
    EXPECTED_TRAINABLE_TENSORS,
    LANGUAGE_LAYERS,
    LANGUAGE_PROJECTIONS,
    LORA_RANK,
    MERGED_PATCH_SIZE,
    MODEL_CONTEXT_TOKENS,
    ExactEpochScheduleMixin,
    _load_inference_model,
    _template_tokens,
    audit_checkpoint_target_shapes,
    audit_model_context,
    audit_runtime_versions,
    deterministic_optimizer_schedule,
    load_training_model,
    make_schedule_callback,
    plan_letterbox,
    qwen_grid_tokens,
    split_outer_fold,
    verify_configured_schedule,
)
from .schema import SchemaError
from .submission import clean_prediction_preserving_internal_spaces, write_submission


POLICY_NAME = "source_geometry_v1_long1600_aspect6_25"
NATIVE_LONG_SIDE_THRESHOLD_PX = 1600
HARD_COMPACT_ASPECT_MAX = 6.0
HARD_ELONGATED_ASPECT_MIN = 25.0
LOW_MIN_TOKENS = 512
LOW_MAX_TOKENS = 640
HIGH_MIN_TOKENS = 896
HIGH_MAX_TOKENS = 1024
BASELINE_MIN_TOKENS = 896
BASELINE_MAX_TOKENS = 1024


@dataclass(frozen=True)
class ResolutionTier:
    tier: str
    reason: str
    source_width: int
    source_height: int
    long_side_px: int
    short_side_px: int
    long_to_short_aspect: float
    native_high_resolution: bool
    hard_compact_geometry: bool
    hard_elongated_geometry: bool
    minimum_tokens: int
    maximum_tokens: int


def select_resolution_tier(
    width: int,
    height: int,
    *,
    native_long_side_threshold_px: int = NATIVE_LONG_SIDE_THRESHOLD_PX,
    hard_compact_aspect_max: float = HARD_COMPACT_ASPECT_MAX,
    hard_elongated_aspect_min: float = HARD_ELONGATED_ASPECT_MIN,
    low_min_tokens: int = LOW_MIN_TOKENS,
    low_max_tokens: int = LOW_MAX_TOKENS,
    high_min_tokens: int = HIGH_MIN_TOKENS,
    high_max_tokens: int = HIGH_MAX_TOKENS,
) -> ResolutionTier:
    """Choose a visual tier from source geometry, without row metadata.

    The absolute long-side threshold separates the two supplied raster families.
    The aspect guards reserve detail for unusually compact/multiline or extremely
    elongated crops.  The API intentionally accepts no labels, IDs, predictions,
    fold values, or confidence scores.
    """

    dimensions = (width, height, native_long_side_threshold_px)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in dimensions
    ):
        raise ValueError("image dimensions and native threshold must be positive integers")
    token_values = (low_min_tokens, low_max_tokens, high_min_tokens, high_max_tokens)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in token_values
    ):
        raise ValueError("tier token bounds must be positive integers")
    if not low_min_tokens <= low_max_tokens < high_min_tokens <= high_max_tokens:
        raise ValueError("adaptive token tiers must be ordered and disjoint")
    if not 1.0 <= hard_compact_aspect_max < hard_elongated_aspect_min <= 200.0:
        raise ValueError("hard aspect thresholds are inconsistent")

    long_side = max(width, height)
    short_side = min(width, height)
    aspect = long_side / short_side
    if aspect > 200.0:
        raise SafetyGateError("source image exceeds Qwen's 200:1 aspect limit")
    native = long_side >= native_long_side_threshold_px
    compact = aspect <= hard_compact_aspect_max
    elongated = aspect >= hard_elongated_aspect_min
    high = native or compact or elongated
    if native and (compact or elongated):
        reason = "native_and_hard_geometry"
    elif native:
        reason = "native_high_resolution"
    elif compact:
        reason = "hard_compact_geometry"
    elif elongated:
        reason = "hard_elongated_geometry"
    else:
        reason = "ordinary_resized_low"
    return ResolutionTier(
        tier="high" if high else "low",
        reason=reason,
        source_width=width,
        source_height=height,
        long_side_px=long_side,
        short_side_px=short_side,
        long_to_short_aspect=aspect,
        native_high_resolution=native,
        hard_compact_geometry=compact,
        hard_elongated_geometry=elongated,
        minimum_tokens=high_min_tokens if high else low_min_tokens,
        maximum_tokens=high_max_tokens if high else low_max_tokens,
    )


def _policy_kwargs(config: Mapping[str, Any]) -> dict[str, int | float]:
    image = _nested(config, "image")
    return {
        "native_long_side_threshold_px": int(image["native_long_side_threshold_px"]),
        "hard_compact_aspect_max": float(image["hard_compact_aspect_max"]),
        "hard_elongated_aspect_min": float(image["hard_elongated_aspect_min"]),
        "low_min_tokens": int(image["low_min_effective_image_tokens"]),
        "low_max_tokens": int(image["low_max_image_tokens"]),
        "high_min_tokens": int(image["high_min_effective_image_tokens"]),
        "high_max_tokens": int(image["high_max_image_tokens"]),
    }


def plan_adaptive_letterbox(
    width: int, height: int, config: Mapping[str, Any]
) -> tuple[ResolutionTier, Any]:
    tier = select_resolution_tier(width, height, **_policy_kwargs(config))
    plan = plan_letterbox(
        width,
        height,
        minimum_tokens=tier.minimum_tokens,
        maximum_tokens=tier.maximum_tokens,
    )
    return tier, plan


def _median_corner_background(image: Image.Image) -> tuple[int, int, int]:
    corners = (
        image.getpixel((0, 0)),
        image.getpixel((image.width - 1, 0)),
        image.getpixel((0, image.height - 1)),
        image.getpixel((image.width - 1, image.height - 1)),
    )
    return tuple(
        sum(sorted(pixel[channel] for pixel in corners)[1:3]) // 2
        for channel in range(3)
    )


def load_adaptive_letterboxed_rgb(
    path: str | Path, config: Mapping[str, Any]
) -> tuple[Image.Image, dict[str, Any]]:
    with Image.open(path) as source:
        rgb = source.convert("RGB")
    tier, plan = plan_adaptive_letterbox(rgb.width, rgb.height, config)
    content = rgb.resize(
        (plan.content_width, plan.content_height), Image.Resampling.LANCZOS
    )
    canvas = Image.new(
        "RGB", (plan.canvas_width, plan.canvas_height), _median_corner_background(rgb)
    )
    canvas.paste(
        content,
        (
            (plan.canvas_width - plan.content_width) // 2,
            (plan.canvas_height - plan.content_height) // 2,
        ),
    )
    return canvas, {
        **asdict(tier),
        **asdict(plan),
        "policy": POLICY_NAME,
        "padding_pixel_fraction": 1.0 - plan.content_pixel_fraction,
    }


class AdaptiveResolutionQwenCollator(QwenOcrCollator):
    def __init__(
        self,
        processor: Any,
        prompt: str,
        config: Mapping[str, Any],
        max_sequence_tokens: int,
    ):
        super().__init__(
            processor,
            prompt,
            int(_nested(config, "image", "max_image_tokens")),
            max_sequence_tokens,
        )
        self.config = config

    def _image(self, record: Any) -> tuple[Image.Image, dict[str, Any]]:
        return load_adaptive_letterboxed_rgb(record.image_path, self.config)

    def encode_prompt(self, record: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        image, resize = self._image(record)
        batch = dict(self._processor_call(self._encode_text(record, False), image))
        diagnostics: dict[str, Any] = self._diagnostics(batch)
        diagnostics.update(resize)
        if diagnostics["image_tokens"] != diagnostics["planned_image_tokens"]:
            raise SafetyGateError("processor changed the adaptive aligned visual grid")
        return batch, diagnostics

    def encode_training(self, record: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch

        image, resize = self._image(record)
        full = dict(self._processor_call(self._encode_text(record, True), image))
        prompt = dict(self._processor_call(self._encode_text(record, False), image))
        diagnostics: dict[str, Any] = self._diagnostics(full)
        diagnostics.update(resize)
        if diagnostics["image_tokens"] != diagnostics["planned_image_tokens"]:
            raise SafetyGateError("processor changed the adaptive aligned visual grid")
        prompt_tokens = int(prompt["input_ids"].shape[-1])
        full_tokens = int(full["input_ids"].shape[-1])
        if prompt_tokens >= full_tokens:
            raise SafetyGateError("assistant label produced no trainable tokens")
        if not torch.equal(full["input_ids"][0, :prompt_tokens], prompt["input_ids"][0]):
            raise SafetyGateError("prompt is not an exact prefix of training input")
        labels = full["input_ids"].clone()
        labels[:, :prompt_tokens] = -100
        if "attention_mask" in full:
            labels[full["attention_mask"] == 0] = -100
        if not bool((labels != -100).any()):
            raise SafetyGateError("assistant-only mask is empty")
        full["labels"] = labels
        diagnostics["prompt_tokens"] = prompt_tokens
        diagnostics["supervised_tokens"] = int((labels != -100).sum().item())
        return full, diagnostics

    def __call__(self, examples: Sequence[Any]) -> dict[str, Any]:
        if len(examples) != 1:
            raise SafetyGateError(f"Model50 supports batch size 1, got {len(examples)}")
        batch, _ = self.encode_training(examples[0])
        return batch


def validate_config(config: Mapping[str, Any]) -> None:
    immutable = {
        ("model", "id"): MODEL_ID,
        ("model", "revision"): MODEL_REVISION,
        ("model", "license"): MODEL_LICENSE,
        ("model", "class"): MODEL_CLASS,
        ("model", "trust_remote_code"): False,
        ("model", "local_path"): LOCAL_CHECKPOINT_PATH,
        ("model", "local_files_only"): True,
        ("model", "attn_implementation"): "sdpa",
        ("model", "processor_use_fast"): False,
        ("quantization", "load_in_4bit"): True,
        ("quantization", "type"): "nf4",
        ("quantization", "double_quant"): True,
        ("quantization", "compute_dtype"): "float16",
        ("architecture", "expected_language_layers"): LANGUAGE_LAYERS,
        ("architecture", "expected_vision_blocks"): 32,
        ("architecture", "expected_max_position_embeddings"): MODEL_CONTEXT_TOKENS,
        ("architecture", "expected_target_modules"): EXPECTED_TARGET_MODULES,
        ("architecture", "expected_trainable_parameters"): EXPECTED_TRAINABLE_PARAMETERS,
        ("architecture", "expected_trainable_tensors"): EXPECTED_TRAINABLE_TENSORS,
        ("lora", "rank"): LORA_RANK,
        ("lora", "alpha"): 32,
        ("lora", "language_projections"): list(LANGUAGE_PROJECTIONS),
        ("lora", "include_merger"): False,
        ("lora", "include_visual_blocks"): False,
        ("image", "merged_patch_size"): MERGED_PATCH_SIZE,
        ("image", "min_pixels"): LOW_MIN_TOKENS * MERGED_PATCH_SIZE**2,
        ("image", "max_pixels"): HIGH_MAX_TOKENS * MERGED_PATCH_SIZE**2,
        ("image", "max_image_tokens"): HIGH_MAX_TOKENS,
        ("image", "low_min_effective_image_tokens"): LOW_MIN_TOKENS,
        ("image", "low_max_image_tokens"): LOW_MAX_TOKENS,
        ("image", "high_min_effective_image_tokens"): HIGH_MIN_TOKENS,
        ("image", "high_max_image_tokens"): HIGH_MAX_TOKENS,
        ("image", "native_long_side_threshold_px"): NATIVE_LONG_SIDE_THRESHOLD_PX,
        ("image", "hard_compact_aspect_max"): HARD_COMPACT_ASPECT_MAX,
        ("image", "hard_elongated_aspect_min"): HARD_ELONGATED_ASPECT_MIN,
        ("image", "selection_policy"): POLICY_NAME,
        ("image", "resize_policy"): "aspect_preserving_letterbox_lanczos",
        ("image", "max_sequence_tokens"): 1152,
        ("image", "max_generation_context_tokens"): 1200,
        ("training", "epochs"): 3,
        ("training", "batch_size"): 1,
        ("training", "eval_batch_size"): 1,
        ("training", "gradient_accumulation"): 16,
        ("training", "expected_train_rows_per_fold"): 3276,
        ("training", "expected_updates_per_epoch"): 205,
        ("training", "expected_total_updates"): 615,
        ("training", "learning_rate"): 0.00003,
        ("training", "gradient_checkpointing"): True,
        ("generation", "max_new_tokens"): 48,
        ("generation", "do_sample"): False,
        ("generation", "num_beams"): 1,
        ("generation", "repetition_penalty"): 1.05,
        ("generation", "no_repeat_ngram_size"): 4,
        ("cross_validation", "folds"): 5,
        ("smoke", "expected_target_modules"): EXPECTED_TARGET_MODULES,
        ("smoke", "expected_trainable_parameters"): EXPECTED_TRAINABLE_PARAMETERS,
        ("smoke", "minimum_4bit_modules"): 100,
        ("test_csv",): "data/raw/Test.csv",
    }
    for keys, expected in immutable.items():
        actual = _nested(config, *keys)
        if actual != expected:
            raise ValueError(
                f"{'.'.join(keys)} must be {expected!r} for Model50, got {actual!r}"
            )
    if _nested(config, "model", "expected_files_sha256") != LOCAL_CHECKPOINT_FILES:
        raise ValueError("checkpoint hash ledger differs from the pinned snapshot")
    if not 0.0 <= float(_nested(config, "lora", "dropout")) < 1.0:
        raise ValueError("lora.dropout must be in [0,1)")
    if int(_nested(config, "image", "max_generation_context_tokens")) != int(
        _nested(config, "image", "max_sequence_tokens")
    ) + int(_nested(config, "generation", "max_new_tokens")):
        raise ValueError("sequence/generation envelope is inconsistent")
    for key in ("max_allocated_gib", "max_reserved_gib"):
        value = float(_nested(config, "smoke", key))
        if not 0.0 < value <= 5.3:
            raise ValueError(f"smoke.{key} must be positive and <=5.3")
    parameter_cap = int(_nested(config, "smoke", "max_trainable_parameters"))
    if parameter_cap < EXPECTED_TRAINABLE_PARAMETERS or parameter_cap > 19_000_000:
        raise ValueError("trainable parameter cap must cover exact count and stay <=19M")
    if "{fold}" not in str(_nested(config, "output_dir")):
        raise ValueError("output_dir must contain {fold}")
    # Exercise the pure selector so its tier ordering is checked independently.
    select_resolution_tier(1200, 80, **_policy_kwargs(config))


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {resolved}")
    validate_config(config)
    return config, resolved


def load_processor(config: Mapping[str, Any]) -> Any:
    from transformers import AutoProcessor

    minimum = int(_nested(config, "image", "min_pixels"))
    maximum = int(_nested(config, "image", "max_pixels"))
    processor = AutoProcessor.from_pretrained(
        str(_absolute(str(_nested(config, "model", "local_path")))),
        trust_remote_code=False,
        local_files_only=True,
        use_fast=False,
        min_pixels=minimum,
        max_pixels=maximum,
    )
    verify_processor_budget(processor, minimum, maximum)
    return processor


def _training_records(config: Mapping[str, Any]) -> tuple[Any, ...]:
    return load_records(
        _absolute(str(_nested(config, "train_csv"))),
        _absolute(str(_nested(config, "image_dir"))),
        require_target=True,
        require_five_folds=True,
    )


def _record_plan(record: Any, config: Mapping[str, Any]) -> tuple[ResolutionTier, Any]:
    with Image.open(record.image_path) as image:
        return plan_adaptive_letterbox(image.width, image.height, config)


def audit_sequence_budget(config: Mapping[str, Any], records: Sequence[Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(_absolute(str(_nested(config, "model", "local_path")))),
        trust_remote_code=False,
        local_files_only=True,
    )
    prompt = str(_nested(config, "prompt"))
    worst: tuple[int, str, int, str] = (-1, "", -1, "")
    longest_label: tuple[int, str] = (-1, "")
    for record in records:
        if record.label is None:
            continue
        tier, plan = _record_plan(record, config)
        total = _template_tokens(
            tokenizer,
            build_messages(prompt, record.label),
            generation_prompt=False,
            image_tokens=plan.planned_image_tokens,
        )
        worst = max(worst, (total, record.identifier, plan.planned_image_tokens, tier.tier))
        label_tokens = len(tokenizer.encode(record.label, add_special_tokens=False))
        longest_label = max(longest_label, (label_tokens, record.identifier))
    if worst[0] < 0:
        raise SafetyGateError("sequence audit requires labelled records")
    max_image_tokens = int(_nested(config, "image", "max_image_tokens"))
    prompt_tokens = _template_tokens(
        tokenizer,
        build_messages(prompt),
        generation_prompt=True,
        image_tokens=max_image_tokens,
    )
    max_new = int(_nested(config, "generation", "max_new_tokens"))
    generation_upper = prompt_tokens + max_new
    training_cap = int(_nested(config, "image", "max_sequence_tokens"))
    generation_cap = int(_nested(config, "image", "max_generation_context_tokens"))
    if worst[0] > training_cap:
        raise SafetyGateError("adaptive training sequence exceeds cap")
    if generation_upper > generation_cap or generation_cap > MODEL_CONTEXT_TOKENS:
        raise SafetyGateError("generation context envelope is invalid")
    if max_new < longest_label[0] + 4:
        raise SafetyGateError("generation cap leaves insufficient target headroom")
    return {
        "maximum_image_tokens": max_image_tokens,
        "configured_training_sequence_cap": training_cap,
        "observed_max_training_tokens": worst[0],
        "observed_max_training_id": worst[1],
        "observed_max_training_image_tokens": worst[2],
        "observed_max_training_tier": worst[3],
        "observed_max_label_tokens": longest_label[0],
        "observed_max_label_id": longest_label[1],
        "prompt_tokens_at_max_image_budget": prompt_tokens,
        "max_new_tokens": max_new,
        "generation_upper_bound": generation_upper,
        "configured_generation_context_cap": generation_cap,
        "model_context_tokens": MODEL_CONTEXT_TOKENS,
    }


def _distribution(values: Sequence[int | float]) -> dict[str, float | int]:
    if not values:
        raise SafetyGateError("cannot summarize an empty distribution")
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "median": ordered[len(ordered) // 2],
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "sum": sum(ordered),
    }


def audit_geometry(image_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = sorted(_absolute(image_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(image_dir)
    adaptive_tokens: list[int] = []
    baseline_tokens: list[int] = []
    tier_tokens: dict[str, list[int]] = {"high": [], "low": []}
    fractions: list[float] = []
    errors: list[float] = []
    reasons: dict[str, int] = {}
    native = compact = elongated = 0
    upscaled = downscaled = unchanged = 0
    for path in paths:
        with Image.open(path) as image:
            tier, plan = plan_adaptive_letterbox(image.width, image.height, config)
            baseline = plan_letterbox(
                image.width,
                image.height,
                minimum_tokens=BASELINE_MIN_TOKENS,
                maximum_tokens=BASELINE_MAX_TOKENS,
            )
        if qwen_grid_tokens(plan.canvas_width, plan.canvas_height) != plan.planned_image_tokens:
            raise SafetyGateError("adaptive canvas token count is inconsistent")
        adaptive_tokens.append(plan.planned_image_tokens)
        baseline_tokens.append(baseline.planned_image_tokens)
        tier_tokens[tier.tier].append(plan.planned_image_tokens)
        fractions.append(plan.content_pixel_fraction)
        errors.append(plan.relative_content_aspect_error)
        reasons[tier.reason] = reasons.get(tier.reason, 0) + 1
        native += int(tier.native_high_resolution)
        compact += int(tier.hard_compact_geometry)
        elongated += int(tier.hard_elongated_geometry)
        if plan.uniform_scale > 1.0 + 1e-12:
            upscaled += 1
        elif plan.uniform_scale < 1.0 - 1e-12:
            downscaled += 1
        else:
            unchanged += 1
    rows = len(paths)
    high_rows = len(tier_tokens["high"])
    low_rows = len(tier_tokens["low"])
    baseline_sum = sum(baseline_tokens)
    adaptive_sum = sum(adaptive_tokens)
    baseline_square = sum(value * value for value in baseline_tokens)
    adaptive_square = sum(value * value for value in adaptive_tokens)
    return {
        "images": rows,
        "selection_inputs": ["source_width_px", "source_height_px"],
        "forbidden_selection_inputs": [
            "target_text",
            "fold",
            "identifier",
            "model_prediction",
            "confidence",
            "test_membership",
        ],
        "policy": POLICY_NAME,
        "thresholds": {
            "native_long_side_threshold_px": NATIVE_LONG_SIDE_THRESHOLD_PX,
            "hard_compact_aspect_max": HARD_COMPACT_ASPECT_MAX,
            "hard_elongated_aspect_min": HARD_ELONGATED_ASPECT_MIN,
            "low_token_interval": [LOW_MIN_TOKENS, LOW_MAX_TOKENS],
            "high_token_interval": [HIGH_MIN_TOKENS, HIGH_MAX_TOKENS],
        },
        "groups": {
            "high": {
                "rows": high_rows,
                "proportion": high_rows / rows,
                "effective_tokens": _distribution(tier_tokens["high"]),
            },
            "low": {
                "rows": low_rows,
                "proportion": low_rows / rows,
                "effective_tokens": _distribution(tier_tokens["low"]),
            },
            "native_high_resolution_rows": native,
            "hard_compact_rows": compact,
            "hard_elongated_rows": elongated,
            "reason_counts": dict(sorted(reasons.items())),
        },
        "adaptive_effective_tokens": _distribution(adaptive_tokens),
        "model43_baseline_effective_tokens": _distribution(baseline_tokens),
        "estimated_token_compute_reduction": {
            "interpretation": "geometry-only proxy; not a measured wall-clock claim",
            "visual_token_linear_fraction": 1.0 - adaptive_sum / baseline_sum,
            "visual_token_linear_percent": 100.0 * (1.0 - adaptive_sum / baseline_sum),
            "visual_token_square_upper_proxy_fraction": 1.0
            - adaptive_square / baseline_square,
            "visual_token_square_upper_proxy_percent": 100.0
            * (1.0 - adaptive_square / baseline_square),
            "baseline_visual_token_presentations_per_dataset_pass": baseline_sum,
            "adaptive_visual_token_presentations_per_dataset_pass": adaptive_sum,
            "saved_visual_token_presentations_per_dataset_pass": baseline_sum
            - adaptive_sum,
        },
        "content_pixel_fraction": _distribution(fractions),
        "maximum_content_aspect_error": max(errors),
        "scaling": {
            "upscaled": upscaled,
            "downscaled": downscaled,
            "unchanged": unchanged,
        },
    }


def audit_record_partition(
    records: Sequence[Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize tier assignment using image paths only, never record targets."""

    seen_paths: set[Path] = set()
    tokens: dict[str, list[int]] = {"high": [], "low": []}
    native = compact = elongated = 0
    for record in records:
        path = Path(record.image_path).resolve()
        if path in seen_paths:
            raise SafetyGateError("record partition contains a duplicate image path")
        seen_paths.add(path)
        tier, plan = _record_plan(record, config)
        tokens[tier.tier].append(plan.planned_image_tokens)
        native += int(tier.native_high_resolution)
        compact += int(tier.hard_compact_geometry)
        elongated += int(tier.hard_elongated_geometry)
    rows = len(records)
    if rows == 0 or len(seen_paths) != rows:
        raise SafetyGateError("record partition is empty or incomplete")
    high = len(tokens["high"])
    low = len(tokens["low"])
    return {
        "rows": rows,
        "high_rows": high,
        "high_proportion": high / rows,
        "low_rows": low,
        "low_proportion": low / rows,
        "native_high_resolution_rows": native,
        "native_high_resolution_proportion": native / rows,
        "hard_compact_rows": compact,
        "hard_elongated_rows": elongated,
        "high_effective_tokens": _distribution(tokens["high"]),
        "low_effective_tokens": _distribution(tokens["low"]),
    }


def select_smoke_record(
    records: Sequence[Any], config: Mapping[str, Any], tokenizer: Any
) -> Any:
    prompt = str(_nested(config, "prompt"))
    candidates: list[tuple[tuple[int, int, str], Any]] = []
    for record in records:
        if record.label is None:
            continue
        _, plan = _record_plan(record, config)
        workload = _template_tokens(
            tokenizer,
            build_messages(prompt, record.label),
            generation_prompt=False,
            image_tokens=plan.planned_image_tokens,
        )
        candidates.append(((workload, plan.planned_image_tokens, record.identifier), record))
    if not candidates:
        raise SafetyGateError("no labelled adaptive smoke candidate")
    return max(candidates, key=lambda item: item[0])[1]


def _collator(config: Mapping[str, Any], processor: Any) -> AdaptiveResolutionQwenCollator:
    return AdaptiveResolutionQwenCollator(
        processor,
        str(_nested(config, "prompt")),
        config,
        int(_nested(config, "image", "max_sequence_tokens")),
    )


def audit_processor_samples(
    config: Mapping[str, Any], records: Sequence[Any]
) -> dict[str, Any]:
    processor = load_processor(config)
    collator = _collator(config, processor)
    by_tier: dict[str, tuple[tuple[int, int, str], Any]] = {}
    for record in records:
        if record.label is None:
            continue
        tier, plan = _record_plan(record, config)
        key = (plan.planned_image_tokens, len(record.label), record.identifier)
        if tier.tier not in by_tier or key > by_tier[tier.tier][0]:
            by_tier[tier.tier] = (key, record)
    if set(by_tier) != {"high", "low"}:
        raise SafetyGateError("processor audit could not find both adaptive tiers")
    report: dict[str, Any] = {}
    for tier_name, (_, record) in sorted(by_tier.items()):
        _, diagnostics = collator.encode_training(record)
        if diagnostics["image_tokens"] != diagnostics["planned_image_tokens"]:
            raise SafetyGateError("processor sample differs from adaptive plan")
        report[tier_name] = {
            "record_id": record.identifier,
            "canvas_width": int(diagnostics["canvas_width"]),
            "canvas_height": int(diagnostics["canvas_height"]),
            "planned_image_tokens": int(diagnostics["planned_image_tokens"]),
            "processor_image_tokens": int(diagnostics["image_tokens"]),
            "sequence_tokens": int(diagnostics["sequence_tokens"]),
            "prompt_tokens": int(diagnostics["prompt_tokens"]),
            "supervised_tokens": int(diagnostics["supervised_tokens"]),
            "reason": diagnostics["reason"],
        }
    report["processor_class"] = processor.__class__.__name__
    report["image_processor_class"] = processor.image_processor.__class__.__name__
    return report


def dry_run(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    enable_offline_mode()
    config, resolved = load_config(config_path)
    runtime = audit_runtime_versions()
    provenance = verify_starter_provenance(config)
    checkpoint = verify_local_checkpoint(config)
    targets = audit_checkpoint_target_shapes(config)
    context = audit_model_context(config)
    records = _training_records(config)
    test_records = load_records(
        _absolute(str(_nested(config, "test_csv"))),
        _absolute(str(_nested(config, "image_dir"))),
        require_target=False,
        require_five_folds=False,
    )
    train_ids = {record.identifier for record in records}
    test_ids = {record.identifier for record in test_records}
    if train_ids & test_ids:
        raise SafetyGateError("train and test IDs overlap")
    sequence = audit_sequence_budget(config, records)
    geometry = audit_geometry(str(_nested(config, "image_dir")), config)
    train_partition = audit_record_partition(records, config)
    test_partition = audit_record_partition(test_records, config)
    referenced_paths = {
        Path(record.image_path).resolve() for record in (*records, *test_records)
    }
    all_image_paths = {
        path.resolve()
        for path in _absolute(str(_nested(config, "image_dir"))).glob("*.jpg")
    }
    if not referenced_paths <= all_image_paths:
        raise SafetyGateError("a referenced record image is outside the audited image set")
    geometry["record_partitions"] = {
        "train": train_partition,
        "test": test_partition,
        "referenced_rows": len(referenced_paths),
        "unreferenced_image_rows": len(all_image_paths - referenced_paths),
        "train_test_id_overlap": 0,
    }
    processor_samples = audit_processor_samples(config, records)
    folds: dict[str, Any] = {}
    for fold in range(5):
        training, validation = split_outer_fold(records, fold)
        schedule = deterministic_optimizer_schedule(
            len(training),
            int(_nested(config, "training", "gradient_accumulation")),
            int(_nested(config, "training", "epochs")),
        )
        verify_configured_schedule(config, schedule)
        folds[str(fold)] = {
            "training_rows": len(training),
            "validation_rows": len(validation),
            "schedule": asdict(schedule),
        }
    report = {
        "status": "cpu_audits_passed_gpu_smoke_not_run",
        "model_number": 50,
        "resolved_config": str(resolved),
        "config_sha256": _sha256(resolved),
        "implementation_module": __name__,
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "selection_is_label_free_and_pre_inference": True,
        "prediction_selection_or_routing": False,
        "organizer_alignment": {
            "rank": 16,
            "alpha": 32,
            "target_families": list(LANGUAGE_PROJECTIONS),
            "language_target_paths": EXPECTED_TARGET_MODULES,
            "merger_adapter": False,
            "visual_adapter": False,
            "generic_prompt": str(_nested(config, "prompt")),
        },
        "runtime": runtime,
        "provenance": provenance,
        "checkpoint": checkpoint,
        "targets": asdict(targets),
        "targeted_base_parameters": sum(
            inputs * outputs for inputs, outputs in targets.shapes.values()
        ),
        "model_context": context,
        "sequence": sequence,
        "processor_samples": processor_samples,
        "geometry": geometry,
        "folds": folds,
    }
    _json_write(reject_model22_destination(output_path), report)
    return report


def train(config_path: str | Path, fold: int, *, smoke_only: bool = False) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    enable_offline_mode()
    config, resolved = load_config(config_path)
    runtime = audit_runtime_versions()
    provenance = verify_starter_provenance(config)
    checkpoint = verify_local_checkpoint(config)
    header_targets = audit_checkpoint_target_shapes(config)
    context = audit_model_context(config)
    records = _training_records(config)
    sequence = audit_sequence_budget(config, records)
    training_records, validation_records = split_outer_fold(records, fold)
    schedule = deterministic_optimizer_schedule(
        len(training_records),
        int(_nested(config, "training", "gradient_accumulation")),
        int(_nested(config, "training", "epochs")),
    )
    verify_configured_schedule(config, schedule)
    output_dir = require_new_output_dir(str(_nested(config, "output_dir")).format(fold=fold))
    seed = int(_nested(config, "seed"))
    set_seed(seed)
    processor = load_processor(config)
    model, model_audit = load_training_model(config)
    collator = _collator(config, processor)
    smoke_record = select_smoke_record(training_records, config, processor.tokenizer)
    smoke = run_memory_smoke(
        model,
        collator,
        smoke_record,
        max_allocated_gib=float(_nested(config, "smoke", "max_allocated_gib")),
        max_reserved_gib=float(_nested(config, "smoke", "max_reserved_gib")),
    )
    audit = {
        "fold": fold,
        "rows": {
            "all": len(records),
            "training": len(training_records),
            "validation": len(validation_records),
        },
        "runtime": runtime,
        "provenance": provenance,
        "checkpoint": checkpoint,
        "header_targets": asdict(header_targets),
        "model_context": context,
        "sequence": sequence,
        "schedule": asdict(schedule),
        "model": model_audit,
        "smoke": smoke,
        "adaptive_policy": POLICY_NAME,
    }
    if smoke_only:
        directory = reject_model22_destination(str(_nested(config, "smoke", "report_dir")))
        _json_write(directory / f"fold{fold}.json", audit)
        return audit
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_write(
        output_dir / "run_manifest.json",
        {
            **_runtime_manifest(config),
            **audit,
            "model_number": 50,
            "implementation_module": __name__,
            "implementation_sha256": _sha256(Path(__file__).resolve()),
            "resolved_config": str(resolved),
            "train_csv_sha256": _sha256(_absolute(str(_nested(config, "train_csv")))),
        },
    )

    import torch
    from transformers import Trainer, TrainingArguments

    class ExactTrainer(ExactEpochScheduleMixin, Trainer):
        pass

    generation = _nested(config, "generation")
    validation_callback = _make_validation_callback(
        processor,
        collator,
        validation_records,
        output_dir,
        int(_nested(generation, "max_new_tokens")),
        repetition_penalty=float(_nested(generation, "repetition_penalty")),
        no_repeat_ngram_size=int(_nested(generation, "no_repeat_ngram_size")),
    )
    schedule_callback = make_schedule_callback(schedule)
    training = _nested(config, "training")
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
        full_determinism=True,
        num_train_epochs=3.0,
        max_steps=schedule.total_optimizer_updates,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=0.00003,
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
        dataloader_drop_last=False,
        report_to="none",
    )
    set_seed(seed)
    trainer = ExactTrainer(
        model=model,
        args=arguments,
        train_dataset=RecordDataset(training_records),
        data_collator=collator,
        callbacks=[schedule_callback, validation_callback],
    )
    trainer.exact_optimizer_schedule = schedule
    result = trainer.train()
    if int(result.global_step) != schedule.total_optimizer_updates:
        raise SafetyGateError("Trainer stopped off the exact schedule")
    if validation_callback.best_epoch is None:
        raise SafetyGateError("no complete held-fold validation occurred")
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)
    complete = {
        "fold": fold,
        "best_epoch": validation_callback.best_epoch,
        "best_multi_score": validation_callback.best_score,
        "schedule": asdict(schedule),
        "observed_epoch_boundaries": schedule_callback.boundaries,
        "observed_optimizer_updates": int(result.global_step),
        "training_metrics": result.metrics,
        "best_adapter": str(output_dir / "best_adapter"),
        "final_adapter": str(final_dir),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "adaptive_policy": POLICY_NAME,
    }
    _json_write(output_dir / "complete.json", complete)
    return complete


def infer(
    config_path: str | Path,
    input_csv: str | Path,
    adapter_path: str | Path,
    output_csv: str | Path,
    *,
    fold: int | None = None,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    enable_offline_mode()
    config, _ = load_config(config_path)
    runtime = audit_runtime_versions()
    verify_starter_provenance(config)
    checkpoint = verify_local_checkpoint(config)
    audit_model_context(config)
    records = load_records(
        _absolute(input_csv),
        _absolute(str(_nested(config, "image_dir"))),
        require_target=False,
        require_five_folds=False,
    )
    if fold is not None:
        if fold not in range(5):
            raise ValueError("fold must be 0..4")
        if not records or records[0].fold is None:
            raise SchemaError("--fold requires a fold column")
        records = tuple(record for record in records if record.fold == fold)
    if not records:
        raise ValueError("inference selection is empty")
    adapter = _absolute(adapter_path)
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    processor = load_processor(config)
    model, targets = _load_inference_model(config, adapter)
    collator = _collator(config, processor)
    generation = _nested(config, "generation")
    predictions = greedy_predict(
        model,
        processor,
        collator,
        records,
        max_new_tokens=int(_nested(generation, "max_new_tokens")),
        repetition_penalty=float(_nested(generation, "repetition_penalty")),
        no_repeat_ngram_size=int(_nested(generation, "no_repeat_ngram_size")),
    )
    output = reject_model22_destination(output_csv)
    write_predictions(output, records, predictions)
    report: dict[str, Any] = {
        "rows": len(records),
        "fold": fold,
        "adapter": str(adapter),
        "runtime": runtime,
        "checkpoint": checkpoint,
        "targets": asdict(targets),
        "output": str(output),
        "generation": dict(generation),
        "adaptive_policy": POLICY_NAME,
        "prediction_selection_or_routing": False,
        "implementation_sha256": _sha256(Path(__file__).resolve()),
    }
    if all(record.label is not None for record in records):
        report["metrics"] = exact_metrics(records, predictions)
    _json_write(output.with_suffix(".metrics.json"), report)
    if submission_path is not None:
        if sample_path is None:
            raise ValueError("--sample is required with --submission")
        by_id = {
            record.identifier: prediction
            for record, prediction in zip(records, predictions, strict=True)
        }
        write_submission(
            _absolute(sample_path),
            reject_model22_destination(submission_path),
            by_id,
            cleaner=clean_prediction_preserving_internal_spaces,
        )
    return report


def train_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model50 adaptive-resolution QLoRA")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.fold, smoke_only=args.smoke_only), indent=2, ensure_ascii=False))


def infer_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model50 adaptive-resolution inference")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--input", type=Path, required=True)
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

    parser = argparse.ArgumentParser(description="Model50 offline adaptive-resolution audit")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/model50/dry_run.json")
    args = parser.parse_args()
    print(json.dumps(dry_run(args.config, args.output), indent=2, ensure_ascii=False))
