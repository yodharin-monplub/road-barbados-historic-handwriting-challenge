"""Model40: fully visual, broad-language Qwen2-VL QLoRA challenger.

Model40 adapts every verified language projection, all four projections in all
32 vision blocks, and both merger matrices. It fills a 640-token visual budget
with a factor-aligned, aspect-preserving letterbox that Qwen does not rescale.
Architecture, parameter, geometry, context, memory, and epoch schedules all
fail closed.  CUDA dependencies remain lazy so dry audits are CPU-only.
"""

from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
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
    audit_cuda_residency,
    audit_injected_lora_targets,
    audit_nf4_modules,
    build_messages,
    enable_offline_mode,
    exact_lora_target_pattern,
    exact_metrics,
    freeze_vision_backbone,
    greedy_predict,
    load_processor,
    load_records,
    reject_model22_destination,
    require_new_output_dir,
    run_memory_smoke,
    set_seed,
    verify_local_checkpoint,
    verify_starter_provenance,
    write_predictions,
)
from .schema import SchemaError
from .submission import clean_prediction_preserving_internal_spaces, write_submission


LANGUAGE_LAYERS = 28
LANGUAGE_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
VISION_BLOCKS = 32
ADAPTED_VISION_BLOCKS = tuple(range(VISION_BLOCKS))
VISION_PROJECTIONS = ("qkv", "proj", "fc1", "fc2")
MERGER_TARGETS = ("visual.merger.mlp.0", "visual.merger.mlp.2")
LORA_RANK = 8
MERGED_PATCH_SIZE = 28
MODEL_CONTEXT_TOKENS = 32_768
IMAGE_PAD_TOKEN = "<|image_pad|>"

# Audited (in_features, out_features) from the pinned safetensors headers.
LANGUAGE_SHAPES = {
    "q_proj": (1536, 1536),
    "k_proj": (1536, 256),
    "v_proj": (1536, 256),
    "o_proj": (1536, 1536),
    "gate_proj": (1536, 8960),
    "up_proj": (1536, 8960),
    "down_proj": (8960, 1536),
}
VISION_SHAPES = {
    "qkv": (1280, 3840),
    "proj": (1280, 1280),
    "fc1": (1280, 5120),
    "fc2": (5120, 1280),
}
MERGER_SHAPES = {
    "visual.merger.mlp.0": (5120, 5120),
    "visual.merger.mlp.2": (5120, 1536),
}

EXPECTED_LANGUAGE_TARGETS = LANGUAGE_LAYERS * len(LANGUAGE_PROJECTIONS)
EXPECTED_VISION_TARGETS = VISION_BLOCKS * len(VISION_PROJECTIONS)
EXPECTED_TARGET_MODULES = (
    EXPECTED_LANGUAGE_TARGETS + EXPECTED_VISION_TARGETS + len(MERGER_TARGETS)
)
EXPECTED_TRAINABLE_PARAMETERS = 14_610_432
EXPECTED_TRAINABLE_TENSORS = EXPECTED_TARGET_MODULES * 2

LANGUAGE_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\."
    r"(?:(?:self_attn\.(?P<attention>q_proj|k_proj|v_proj|o_proj))|"
    r"(?:mlp\.(?P<mlp>gate_proj|up_proj|down_proj)))$"
)
VISION_PATTERN = re.compile(
    r"^visual\.blocks\.(?P<block>\d+)\."
    r"(?:(?:attn\.(?P<attention>qkv|proj))|(?:mlp\.(?P<mlp>fc1|fc2)))$"
)


@dataclass(frozen=True)
class FullVisualTargetAudit:
    rank: int
    target_modules: tuple[str, ...]
    language_modules: int
    vision_modules: int
    merger_modules: int
    trainable_parameters: int
    trainable_tensors: int
    parameters_by_family: dict[str, int]
    shapes: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class OptimizerSchedule:
    train_rows: int
    gradient_accumulation: int
    epochs: int
    full_accumulation_groups: int
    remainder_micro_batches: int
    optimizer_updates_per_epoch: int
    total_optimizer_updates: int
    transformer_floor_updates_per_epoch: int
    transformer_floor_total_updates: int
    total_row_presentations: int


@dataclass(frozen=True)
class SequenceBudgetAudit:
    max_image_tokens: int
    configured_training_sequence_cap: int
    observed_max_training_tokens: int
    observed_max_training_id: str
    observed_max_label_tokens: int
    observed_max_label_id: str
    prompt_tokens_at_max_image_budget: int
    max_new_tokens: int
    observed_generation_upper_bound: int
    configured_generation_context_cap: int
    model_context_tokens: int
    image_pad_tokens_per_template: int


@dataclass(frozen=True)
class ResizePlan:
    source_width: int
    source_height: int
    content_width: int
    content_height: int
    canvas_width: int
    canvas_height: int
    uniform_scale: float
    planned_image_tokens: int
    content_pixel_fraction: float
    relative_content_aspect_error: float


def _language_path(layer: int, projection: str) -> str:
    family = "self_attn" if projection in LANGUAGE_PROJECTIONS[:4] else "mlp"
    return f"model.layers.{layer}.{family}.{projection}"


def _vision_path(block: int, projection: str) -> str:
    family = "attn" if projection in VISION_PROJECTIONS[:2] else "mlp"
    return f"visual.blocks.{block}.{family}.{projection}"


def expected_target_shapes() -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}
    for layer in range(LANGUAGE_LAYERS):
        for projection in LANGUAGE_PROJECTIONS:
            shapes[_language_path(layer, projection)] = LANGUAGE_SHAPES[projection]
    for block in ADAPTED_VISION_BLOCKS:
        for projection in VISION_PROJECTIONS:
            shapes[_vision_path(block, projection)] = VISION_SHAPES[projection]
    shapes.update(MERGER_SHAPES)
    if len(shapes) != EXPECTED_TARGET_MODULES:
        raise AssertionError("internal Model40 target ledger is not exact")
    return shapes


def _target_family(name: str) -> str:
    if name.startswith("model.layers."):
        return "language"
    if name.startswith("visual.blocks."):
        return "vision"
    if name in MERGER_TARGETS:
        return "merger"
    raise SafetyGateError(f"unclassified Model40 target: {name}")


def calculate_lora_parameters(
    shapes: Mapping[str, tuple[int, int]], rank: int
) -> tuple[int, dict[str, int]]:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    expected = set(expected_target_shapes())
    if set(shapes) != expected:
        missing = sorted(expected - set(shapes))
        extra = sorted(set(shapes) - expected)
        raise SafetyGateError(
            f"target shape ledger mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    by_family = {"language": 0, "vision": 0, "merger": 0}
    for name, (inputs, outputs) in shapes.items():
        if inputs <= 0 or outputs <= 0:
            raise SafetyGateError(f"invalid target shape for {name}: {(inputs, outputs)}")
        by_family[_target_family(name)] += rank * (inputs + outputs)
    return sum(by_family.values()), by_family


def deterministic_optimizer_schedule(
    train_rows: int, gradient_accumulation: int, epochs: int
) -> OptimizerSchedule:
    values = (train_rows, gradient_accumulation, epochs)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("optimizer schedule inputs must be positive integers")
    groups, remainder = divmod(train_rows, gradient_accumulation)
    updates = groups + int(remainder > 0)
    floor_updates = max(groups, 1)
    return OptimizerSchedule(
        train_rows=train_rows,
        gradient_accumulation=gradient_accumulation,
        epochs=epochs,
        full_accumulation_groups=groups,
        remainder_micro_batches=remainder,
        optimizer_updates_per_epoch=updates,
        total_optimizer_updates=updates * epochs,
        transformer_floor_updates_per_epoch=floor_updates,
        transformer_floor_total_updates=floor_updates * epochs,
        total_row_presentations=train_rows * epochs,
    )


def verify_configured_schedule(config: Mapping[str, Any], schedule: OptimizerSchedule) -> None:
    training = _nested(config, "training")
    expected = {
        "expected_train_rows_per_fold": schedule.train_rows,
        "expected_updates_per_epoch": schedule.optimizer_updates_per_epoch,
        "expected_total_updates": schedule.total_optimizer_updates,
    }
    drift = {
        key: {"configured": training.get(key), "actual": value}
        for key, value in expected.items()
        if training.get(key) != value
    }
    if drift:
        raise SafetyGateError(f"configured optimizer schedule mismatch: {drift}")


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {resolved}")
    validate_config(config)
    return config, resolved


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
        ("quantization", "load_in_4bit"): True,
        ("quantization", "type"): "nf4",
        ("quantization", "double_quant"): True,
        ("quantization", "compute_dtype"): "float16",
        ("architecture", "expected_language_layers"): LANGUAGE_LAYERS,
        ("architecture", "expected_vision_blocks"): VISION_BLOCKS,
        ("architecture", "expected_max_position_embeddings"): MODEL_CONTEXT_TOKENS,
        ("architecture", "expected_target_modules"): EXPECTED_TARGET_MODULES,
        ("architecture", "expected_trainable_parameters"): EXPECTED_TRAINABLE_PARAMETERS,
        ("architecture", "expected_trainable_tensors"): EXPECTED_TRAINABLE_TENSORS,
        ("lora", "rank"): LORA_RANK,
        ("lora", "language_projections"): list(LANGUAGE_PROJECTIONS),
        ("lora", "adapted_vision_blocks"): list(ADAPTED_VISION_BLOCKS),
        ("lora", "vision_projections"): list(VISION_PROJECTIONS),
        ("lora", "include_verified_merger"): True,
        ("image", "merged_patch_size"): MERGED_PATCH_SIZE,
        ("image", "max_image_tokens"): 640,
        ("image", "target_image_tokens"): 640,
        ("image", "min_effective_image_tokens"): 512,
        ("image", "max_sequence_tokens"): 768,
        ("image", "max_generation_context_tokens"): 816,
        ("image", "resize_policy"): "aspect_preserving_letterbox_lanczos",
        ("training", "batch_size"): 1,
        ("training", "eval_batch_size"): 1,
        ("training", "gradient_checkpointing"): True,
        ("cross_validation", "folds"): 5,
        ("generation", "do_sample"): False,
        ("generation", "num_beams"): 1,
        ("generation", "max_new_tokens"): 48,
        ("generation", "repetition_penalty"): 1.05,
        ("generation", "no_repeat_ngram_size"): 4,
        ("smoke", "expected_target_modules"): EXPECTED_TARGET_MODULES,
        ("smoke", "expected_trainable_parameters"): EXPECTED_TRAINABLE_PARAMETERS,
        ("smoke", "minimum_4bit_modules"): 100,
    }
    for keys, expected in immutable.items():
        actual = _nested(config, *keys)
        if actual != expected:
            raise ValueError(
                f"{'.'.join(keys)} must be {expected!r} for Model40, got {actual!r}"
            )
    if _nested(config, "model", "expected_files_sha256") != LOCAL_CHECKPOINT_FILES:
        raise ValueError("model.expected_files_sha256 differs from the pinned ledger")
    if int(_nested(config, "lora", "alpha")) != 16:
        raise ValueError("Model40 lora.alpha must be 16")
    if not 0.0 <= float(_nested(config, "lora", "dropout")) < 1.0:
        raise ValueError("lora.dropout must be in [0, 1)")

    minimum = int(_nested(config, "image", "min_pixels"))
    maximum = int(_nested(config, "image", "max_pixels"))
    target_tokens = int(_nested(config, "image", "target_image_tokens"))
    if minimum <= 0 or minimum % MERGED_PATCH_SIZE**2:
        raise ValueError("min_pixels must be a positive merged-patch multiple")
    if maximum != target_tokens * MERGED_PATCH_SIZE**2:
        raise ValueError("max_pixels must exactly encode the 640-token budget")
    training_cap = int(_nested(config, "image", "max_sequence_tokens"))
    generation_cap = int(_nested(config, "image", "max_generation_context_tokens"))
    new_tokens = int(_nested(config, "generation", "max_new_tokens"))
    if training_cap <= target_tokens:
        raise ValueError("training sequence cap must leave room beyond image tokens")
    if generation_cap != training_cap + new_tokens:
        raise ValueError("generation context cap must equal sequence cap + max_new_tokens")
    if generation_cap > MODEL_CONTEXT_TOKENS:
        raise ValueError("configured context exceeds the pinned Qwen context window")

    for key in ("max_allocated_gib", "max_reserved_gib"):
        limit = float(_nested(config, "smoke", key))
        if not 0.0 < limit <= 5.3:
            raise ValueError(f"smoke.{key} must be positive and at most 5.3")
    trainable_cap = int(_nested(config, "smoke", "max_trainable_parameters"))
    if trainable_cap < EXPECTED_TRAINABLE_PARAMETERS or trainable_cap > 15_000_000:
        raise ValueError("Model40 trainable cap must cover the exact count and stay <=15M")
    if "{fold}" not in str(_nested(config, "output_dir")):
        raise ValueError("output_dir must contain {fold}")
    for key in (
        "epochs",
        "gradient_accumulation",
        "expected_train_rows_per_fold",
        "expected_updates_per_epoch",
        "expected_total_updates",
    ):
        value = _nested(config, "training", key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer")


def _linear_shape(module: Any, name: str) -> tuple[int, int]:
    inputs = getattr(module, "in_features", None)
    outputs = getattr(module, "out_features", None)
    if (
        module.__class__.__name__ not in {"Linear", "Linear4bit"}
        or not isinstance(inputs, int)
        or not isinstance(outputs, int)
        or inputs <= 0
        or outputs <= 0
    ):
        raise SafetyGateError(f"LoRA target is not a supported linear: {name}")
    return inputs, outputs


def discover_full_visual_targets(
    model: Any, rank: int = LORA_RANK
) -> FullVisualTargetAudit:
    modules = dict(model.named_modules())
    wanted = expected_target_shapes()
    observed_language = {
        name for name in modules if LANGUAGE_PATTERN.fullmatch(name) is not None
    }
    expected_language = {
        _language_path(layer, projection)
        for layer in range(LANGUAGE_LAYERS)
        for projection in LANGUAGE_PROJECTIONS
    }
    if observed_language != expected_language:
        raise SafetyGateError(
            "language projection layout mismatch: "
            f"missing={sorted(expected_language - observed_language)[:5]}, "
            f"extra={sorted(observed_language - expected_language)[:5]}"
        )
    observed_vision = {
        name for name in modules if VISION_PATTERN.fullmatch(name) is not None
    }
    expected_vision = {
        _vision_path(block, projection)
        for block in range(VISION_BLOCKS)
        for projection in VISION_PROJECTIONS
    }
    if observed_vision != expected_vision:
        raise SafetyGateError(
            "vision projection layout mismatch: "
            f"missing={sorted(expected_vision - observed_vision)[:5]}, "
            f"extra={sorted(observed_vision - expected_vision)[:5]}"
        )
    merger_linears = {
        name
        for name, module in modules.items()
        if name.startswith("visual.merger.")
        and module.__class__.__name__ in {"Linear", "Linear4bit"}
    }
    if merger_linears != set(MERGER_TARGETS):
        raise SafetyGateError(f"visual merger layout mismatch: {sorted(merger_linears)}")
    shapes: dict[str, tuple[int, int]] = {}
    for name, expected in wanted.items():
        if name not in modules:
            raise SafetyGateError(f"required projection is absent: {name}")
        actual = _linear_shape(modules[name], name)
        if actual != expected:
            raise SafetyGateError(f"projection shape mismatch for {name}: {actual} != {expected}")
        shapes[name] = actual
    trainable, families = calculate_lora_parameters(shapes, rank)
    if rank != LORA_RANK or trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise SafetyGateError(
            f"LoRA parameter audit mismatch: rank={rank}, parameters={trainable:,}"
        )
    return FullVisualTargetAudit(
        rank=rank,
        target_modules=tuple(shapes),
        language_modules=EXPECTED_LANGUAGE_TARGETS,
        vision_modules=EXPECTED_VISION_TARGETS,
        merger_modules=len(MERGER_TARGETS),
        trainable_parameters=trainable,
        trainable_tensors=len(shapes) * 2,
        parameters_by_family=families,
        shapes=shapes,
    )


def audit_checkpoint_target_shapes(
    config: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> FullVisualTargetAudit:
    from safetensors import safe_open

    checkpoint = _absolute(str(_nested(config, "model", "local_path")), project_root)
    wanted = expected_target_shapes()
    found: dict[str, tuple[int, int]] = {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for name in wanted:
                weight = f"{name}.weight"
                if weight not in keys:
                    continue
                raw = tuple(int(value) for value in handle.get_slice(weight).get_shape())
                if len(raw) != 2 or name in found:
                    raise SafetyGateError(f"invalid/duplicate checkpoint target: {weight}")
                found[name] = (raw[1], raw[0])
    if found != wanted:
        missing = sorted(set(wanted) - set(found))
        wrong = {
            name: {"actual": found[name], "expected": wanted[name]}
            for name in set(found) & set(wanted)
            if found[name] != wanted[name]
        }
        raise SafetyGateError(
            f"checkpoint target header mismatch: missing={missing[:5]}, "
            f"wrong={dict(list(wrong.items())[:5])}"
        )
    canonical = {name: found[name] for name in wanted}
    trainable, families = calculate_lora_parameters(canonical, LORA_RANK)
    if trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise SafetyGateError(f"checkpoint-derived LoRA count drifted: {trainable:,}")
    return FullVisualTargetAudit(
        rank=LORA_RANK,
        target_modules=tuple(canonical),
        language_modules=EXPECTED_LANGUAGE_TARGETS,
        vision_modules=EXPECTED_VISION_TARGETS,
        merger_modules=len(MERGER_TARGETS),
        trainable_parameters=trainable,
        trainable_tensors=len(canonical) * 2,
        parameters_by_family=families,
        shapes=canonical,
    )


def audit_model_context(config: Mapping[str, Any]) -> dict[str, int]:
    checkpoint = _absolute(str(_nested(config, "model", "local_path")))
    metadata = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    actual = int(metadata.get("max_position_embeddings", -1))
    if actual != MODEL_CONTEXT_TOKENS:
        raise SafetyGateError(
            f"pinned model context drifted: {actual} != {MODEL_CONTEXT_TOKENS}"
        )
    configured = int(_nested(config, "image", "max_generation_context_tokens"))
    if configured > actual:
        raise SafetyGateError("generation context cap exceeds model context")
    return {
        "model_max_position_embeddings": actual,
        "configured_generation_context_tokens": configured,
        "unused_context_tokens": actual - configured,
    }


@lru_cache(maxsize=None)
def _candidate_grids(
    minimum_tokens: int, maximum_tokens: int
) -> tuple[tuple[int, int, int], ...]:
    grids: list[tuple[int, int, int]] = []
    for height_units in range(1, maximum_tokens + 1):
        for width_units in range(1, maximum_tokens // height_units + 1):
            tokens = height_units * width_units
            if tokens < minimum_tokens:
                continue
            ratio = width_units / height_units
            if max(ratio, 1.0 / ratio) <= 200:
                grids.append((width_units, height_units, tokens))
    if not grids:
        raise ValueError("no factor-aligned canvas satisfies the token range")
    return tuple(grids)


def plan_aspect_preserving_resize(
    width: int,
    height: int,
    *,
    minimum_tokens: int,
    maximum_tokens: int,
    factor: int = MERGED_PATCH_SIZE,
) -> ResizePlan:
    """Maximize nonblank content pixels under a factor-aligned token cap."""

    values = (width, height, minimum_tokens, maximum_tokens, factor)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("resize dimensions/budgets must be positive integers")
    if minimum_tokens > maximum_tokens:
        raise ValueError("minimum_tokens may not exceed maximum_tokens")
    if max(width, height) / min(width, height) > 200:
        raise SafetyGateError("source image exceeds Qwen's 200:1 aspect limit")
    source_aspect = width / height

    def score(grid: tuple[int, int, int]) -> tuple[float, float, int]:
        width_units, height_units, tokens = grid
        canvas_aspect = width_units / height_units
        occupancy = min(canvas_aspect / source_aspect, source_aspect / canvas_aspect)
        # occupancy*tokens is the visual-token-equivalent area occupied by
        # real image content. Tie-break toward less padding, then more tokens.
        return occupancy * tokens, occupancy, tokens

    width_units, height_units, tokens = max(
        _candidate_grids(minimum_tokens, maximum_tokens), key=score
    )
    canvas_width = width_units * factor
    canvas_height = height_units * factor
    scale = min(canvas_width / width, canvas_height / height)
    content_width = min(canvas_width, max(1, int(round(width * scale))))
    content_height = min(canvas_height, max(1, int(round(height * scale))))
    aspect_error = abs(
        (content_width / content_height) / source_aspect - 1.0
    )
    if aspect_error > 0.005:
        raise SafetyGateError(
            f"integer content resize aspect error is too large: {aspect_error}"
        )
    content_fraction = (
        content_width * content_height / (canvas_width * canvas_height)
    )
    return ResizePlan(
        source_width=width,
        source_height=height,
        content_width=content_width,
        content_height=content_height,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        uniform_scale=scale,
        planned_image_tokens=tokens,
        content_pixel_fraction=content_fraction,
        relative_content_aspect_error=aspect_error,
    )


def qwen_smart_grid(
    width: int,
    height: int,
    *,
    min_pixels: int,
    max_pixels: int,
    factor: int = MERGED_PATCH_SIZE,
) -> tuple[int, int, int]:
    """Mirror pinned Transformers 4.49 Qwen smart_resize arithmetic."""

    if min(width, height) < factor or max(width, height) / min(width, height) > 200:
        raise SafetyGateError("resized image violates Qwen geometry limits")
    output_height = round(height / factor) * factor
    output_width = round(width / factor) * factor
    if output_height * output_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        output_height = math.floor(height / beta / factor) * factor
        output_width = math.floor(width / beta / factor) * factor
    elif output_height * output_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        output_height = math.ceil(height * beta / factor) * factor
        output_width = math.ceil(width * beta / factor) * factor
    tokens = output_height * output_width // factor**2
    if output_height <= 0 or output_width <= 0 or tokens <= 0:
        raise SafetyGateError("Qwen smart resize produced invalid dimensions")
    return output_width, output_height, tokens


def load_budget_filled_rgb(
    path: str | Path, *, minimum_tokens: int, maximum_tokens: int
) -> tuple[Image.Image, dict[str, int | float]]:
    with Image.open(path) as source:
        rgb = source.convert("RGB")
    plan = plan_aspect_preserving_resize(
        rgb.width,
        rgb.height,
        minimum_tokens=minimum_tokens,
        maximum_tokens=maximum_tokens,
    )
    content = rgb.resize(
        (plan.content_width, plan.content_height), Image.Resampling.LANCZOS
    )
    corners = (
        rgb.getpixel((0, 0)),
        rgb.getpixel((rgb.width - 1, 0)),
        rgb.getpixel((0, rgb.height - 1)),
        rgb.getpixel((rgb.width - 1, rgb.height - 1)),
    )
    background = tuple(
        sum(sorted(pixel[channel] for pixel in corners)[1:3]) // 2
        for channel in range(3)
    )
    canvas = Image.new("RGB", (plan.canvas_width, plan.canvas_height), background)
    left = (plan.canvas_width - plan.content_width) // 2
    top = (plan.canvas_height - plan.content_height) // 2
    canvas.paste(content, (left, top))
    return canvas, {
        **asdict(plan),
        "padding_pixel_fraction": 1.0 - plan.content_pixel_fraction,
    }


class FullVisualQwenCollator(QwenOcrCollator):
    """Assistant-only collator with Model40's uniform budget-fill resize."""

    def __init__(
        self,
        processor: Any,
        prompt: str,
        max_image_tokens: int,
        max_sequence_tokens: int,
        spatial_merge_size: int = 2,
    ):
        super().__init__(
            processor,
            prompt,
            max_image_tokens,
            max_sequence_tokens,
            spatial_merge_size,
        )
        self.minimum_image_tokens = 512

    def encode_prompt(self, record: Any) -> tuple[dict[str, Any], dict[str, int | float]]:
        image, resize = load_budget_filled_rgb(
            record.image_path,
            minimum_tokens=self.minimum_image_tokens,
            maximum_tokens=self.max_image_tokens,
        )
        text = self._encode_text(record, include_label=False)
        batch = dict(self._processor_call(text, image))
        diagnostics: dict[str, int | float] = self._diagnostics(batch)
        diagnostics.update(resize)
        return batch, diagnostics

    def encode_training(self, record: Any) -> tuple[dict[str, Any], dict[str, int | float]]:
        import torch

        image, resize = load_budget_filled_rgb(
            record.image_path,
            minimum_tokens=self.minimum_image_tokens,
            maximum_tokens=self.max_image_tokens,
        )
        full_text = self._encode_text(record, include_label=True)
        prompt_text = self._encode_text(record, include_label=False)
        full = dict(self._processor_call(full_text, image))
        prompt = dict(self._processor_call(prompt_text, image))
        diagnostics: dict[str, int | float] = self._diagnostics(full)
        prompt_tokens = int(prompt["input_ids"].shape[-1])
        full_tokens = int(full["input_ids"].shape[-1])
        if prompt_tokens >= full_tokens:
            raise SafetyGateError("assistant label produced no trainable tokens")
        if not torch.equal(full["input_ids"][0, :prompt_tokens], prompt["input_ids"][0]):
            raise SafetyGateError("chat template prompt is not an exact training prefix")
        labels = full["input_ids"].clone()
        labels[:, :prompt_tokens] = -100
        if "attention_mask" in full:
            labels[full["attention_mask"] == 0] = -100
        if not bool((labels != -100).any()):
            raise SafetyGateError("assistant-only label mask is empty")
        full["labels"] = labels
        diagnostics.update(resize)
        diagnostics["prompt_tokens"] = prompt_tokens
        diagnostics["supervised_tokens"] = int((labels != -100).sum().item())
        return full, diagnostics

    def __call__(self, examples: Sequence[Any]) -> dict[str, Any]:
        if len(examples) != 1:
            raise SafetyGateError(
                f"Model40 is memory-certified only for batch size 1, got {len(examples)}"
            )
        batch, _ = self.encode_training(examples[0])
        return batch


def audit_resize_geometry(
    image_dir: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    directory = _absolute(image_dir)
    paths = sorted(directory.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no JPG images in {directory}")
    minimum_tokens = int(_nested(config, "image", "min_effective_image_tokens"))
    maximum_tokens = int(_nested(config, "image", "max_image_tokens"))
    min_pixels = int(_nested(config, "image", "min_pixels"))
    max_pixels = int(_nested(config, "image", "max_pixels"))
    tokens: list[int] = []
    aspect_errors: list[float] = []
    content_fractions: list[float] = []
    upscaled = downscaled = unchanged = 0
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        plan = plan_aspect_preserving_resize(
            width,
            height,
            minimum_tokens=minimum_tokens,
            maximum_tokens=maximum_tokens,
        )
        grid_width, grid_height, visual_tokens = qwen_smart_grid(
            plan.canvas_width,
            plan.canvas_height,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if (grid_width, grid_height) != (plan.canvas_width, plan.canvas_height):
            raise SafetyGateError("Qwen would resize the pre-aligned letterbox canvas")
        if visual_tokens != plan.planned_image_tokens:
            raise SafetyGateError("planned and Qwen visual token counts differ")
        if visual_tokens > int(_nested(config, "image", "max_image_tokens")):
            raise SafetyGateError(f"resize exceeds visual token cap for {path.name}")
        tokens.append(visual_tokens)
        aspect_errors.append(plan.relative_content_aspect_error)
        content_fractions.append(plan.content_pixel_fraction)
        if plan.uniform_scale > 1.0 + 1e-12:
            upscaled += 1
        elif plan.uniform_scale < 1.0 - 1e-12:
            downscaled += 1
        else:
            unchanged += 1
    ordered = sorted(tokens)
    ordered_content = sorted(content_fractions)
    return {
        "images": len(paths),
        "resize_policy": "aspect_preserving_letterbox_lanczos",
        "minimum_image_tokens": minimum_tokens,
        "maximum_image_tokens": maximum_tokens,
        "upscaled": upscaled,
        "downscaled": downscaled,
        "unchanged": unchanged,
        "effective_image_tokens": {
            "minimum": ordered[0],
            "median": ordered[len(ordered) // 2],
            "maximum": ordered[-1],
        },
        "content_pixel_fraction": {
            "minimum": ordered_content[0],
            "median": ordered_content[len(ordered_content) // 2],
            "maximum": ordered_content[-1],
        },
        "maximum_relative_content_aspect_error": max(aspect_errors),
    }


def select_memory_smoke_record(
    records: Sequence[Any], config: Mapping[str, Any], tokenizer: Any | None = None
) -> Any:
    """Select the largest conservative post-resize image+label workload."""

    minimum_tokens = int(_nested(config, "image", "min_effective_image_tokens"))
    maximum_tokens = int(_nested(config, "image", "max_image_tokens"))
    min_pixels = int(_nested(config, "image", "min_pixels"))
    max_pixels = int(_nested(config, "image", "max_pixels"))
    candidates: list[tuple[tuple[int, int, int, str], Any]] = []
    for record in records:
        if record.label is None:
            continue
        with Image.open(record.image_path) as image:
            source_width, source_height = image.size
        plan = plan_aspect_preserving_resize(
            source_width,
            source_height,
            minimum_tokens=minimum_tokens,
            maximum_tokens=maximum_tokens,
        )
        _, _, image_tokens = qwen_smart_grid(
            plan.canvas_width,
            plan.canvas_height,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if tokenizer is None:
            # Test/offline fallback: character count is conservative for this
            # ASCII-labelled corpus. Runtime supplies the pinned tokenizer.
            sequence_work = image_tokens + len(record.label)
        else:
            sequence_work, _ = _template_token_count(
                tokenizer,
                build_messages(str(_nested(config, "prompt")), record.label),
                add_generation_prompt=False,
                image_tokens=image_tokens,
            )
        # Ties prefer the larger grid, then longer label, then stable ID.
        key = (
            sequence_work,
            image_tokens,
            len(record.label),
            record.identifier,
        )
        candidates.append((key, record))
    if not candidates:
        raise SafetyGateError("no labelled record is available for memory smoke")
    return max(candidates, key=lambda item: item[0])[1]


def _template_token_count(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
    image_tokens: int,
) -> tuple[int, int]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    pads = ids.count(image_pad_id)
    if pads != 1:
        raise SafetyGateError(f"expected one image pad token in template, got {pads}")
    return len(ids) - pads + image_tokens, pads


def audit_sequence_budget(
    config: Mapping[str, Any], records: Sequence[Any]
) -> SequenceBudgetAudit:
    from transformers import AutoTokenizer

    checkpoint = _absolute(str(_nested(config, "model", "local_path")))
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), trust_remote_code=False, local_files_only=True
    )
    prompt = str(_nested(config, "prompt"))
    image_tokens = int(_nested(config, "image", "max_image_tokens"))
    training_cap = int(_nested(config, "image", "max_sequence_tokens"))
    context_cap = int(_nested(config, "image", "max_generation_context_tokens"))
    max_new_tokens = int(_nested(config, "generation", "max_new_tokens"))
    maximum = -1
    maximum_id = ""
    maximum_label_tokens = -1
    maximum_label_id = ""
    pad_count = -1
    for record in records:
        if record.label is None:
            continue
        tokens, pads = _template_token_count(
            tokenizer,
            build_messages(prompt, record.label),
            add_generation_prompt=False,
            image_tokens=image_tokens,
        )
        if tokens > maximum:
            maximum = tokens
            maximum_id = record.identifier
            pad_count = pads
        label_tokens = len(tokenizer.encode(record.label, add_special_tokens=False))
        if label_tokens > maximum_label_tokens:
            maximum_label_tokens = label_tokens
            maximum_label_id = record.identifier
    if maximum < 0:
        raise SafetyGateError("sequence audit requires labelled records")
    prompt_tokens, prompt_pads = _template_token_count(
        tokenizer,
        build_messages(prompt),
        add_generation_prompt=True,
        image_tokens=image_tokens,
    )
    generation_upper = prompt_tokens + max_new_tokens
    if maximum > training_cap:
        raise SafetyGateError(
            f"static training sequence {maximum} exceeds cap {training_cap}"
        )
    if generation_upper > context_cap or context_cap > MODEL_CONTEXT_TOKENS:
        raise SafetyGateError(
            f"generation context budget is invalid: observed={generation_upper}, cap={context_cap}"
        )
    if max_new_tokens < maximum_label_tokens + 4:
        raise SafetyGateError(
            f"generation cap {max_new_tokens} leaves insufficient room for "
            f"{maximum_label_tokens}-token training target"
        )
    return SequenceBudgetAudit(
        max_image_tokens=image_tokens,
        configured_training_sequence_cap=training_cap,
        observed_max_training_tokens=maximum,
        observed_max_training_id=maximum_id,
        observed_max_label_tokens=maximum_label_tokens,
        observed_max_label_id=maximum_label_id,
        prompt_tokens_at_max_image_budget=prompt_tokens,
        max_new_tokens=max_new_tokens,
        observed_generation_upper_bound=generation_upper,
        configured_generation_context_cap=context_cap,
        model_context_tokens=MODEL_CONTEXT_TOKENS,
        image_pad_tokens_per_template=max(pad_count, prompt_pads),
    )


def audit_trainable_parameters(
    model: Any,
    audit: FullVisualTargetAudit,
    *,
    max_trainable_parameters: int,
) -> dict[str, Any]:
    expected = set(audit.target_modules)
    matched: set[str] = set()
    names: list[str] = []
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        count += int(parameter.numel())
        if ".lora_A." not in name and ".lora_B." not in name:
            raise SafetyGateError(f"non-LoRA parameter is unexpectedly trainable: {name}")
        owners = [target for target in expected if f"{target}.lora_" in name]
        if len(owners) != 1:
            raise SafetyGateError(f"cannot map trainable tensor to one exact target: {name}")
        matched.add(owners[0])
    if matched != expected:
        raise SafetyGateError(f"targets without trainable adapters: {sorted(expected - matched)[:5]}")
    if count != audit.trainable_parameters or count > max_trainable_parameters:
        raise SafetyGateError(
            f"trainable parameter count {count:,} != audited {audit.trainable_parameters:,}"
        )
    if len(names) != audit.trainable_tensors:
        raise SafetyGateError(
            f"trainable tensor count {len(names)} != audited {audit.trainable_tensors}"
        )
    return {
        "trainable_parameters": count,
        "trainable_tensors": len(names),
        "target_modules": len(matched),
        "parameters_by_family": audit.parameters_by_family,
        "sample_names": names[:8],
    }


def split_outer_fold(records: Sequence[Any], fold: int) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if fold not in range(5):
        raise ValueError("fold must be one of 0,1,2,3,4")
    training = tuple(record for record in records if record.fold != fold)
    validation = tuple(record for record in records if record.fold == fold)
    if not training or not validation or len(training) + len(validation) != len(records):
        raise SafetyGateError("outer-fold split is incomplete")
    training_ids = {record.identifier for record in training}
    validation_ids = {record.identifier for record in validation}
    if len(training_ids) != len(training) or len(validation_ids) != len(validation):
        raise SafetyGateError("duplicate IDs inside an outer-fold partition")
    if training_ids & validation_ids:
        raise SafetyGateError("training and validation IDs overlap")
    return training, validation


class ExactEpochScheduleMixin:
    exact_optimizer_schedule: OptimizerSchedule | None = None

    def set_initial_training_values(
        self, args: Any, dataloader: Any, total_train_batch_size: int
    ) -> tuple[int, int, int, int, bool, int, int]:
        super().set_initial_training_values(args, dataloader, total_train_batch_size)
        schedule = self.exact_optimizer_schedule
        if schedule is None:
            raise SafetyGateError("exact optimizer schedule was not attached to Trainer")
        rows = len(dataloader)
        examples = int(self.num_examples(dataloader))
        if rows != schedule.train_rows or examples != schedule.train_rows:
            raise SafetyGateError(
                f"training dataloader drift: len={rows}, examples={examples}, "
                f"expected={schedule.train_rows}"
            )
        if int(args.gradient_accumulation_steps) != schedule.gradient_accumulation:
            raise SafetyGateError("Trainer gradient accumulation differs from schedule")
        if int(args.max_steps) != schedule.total_optimizer_updates:
            raise SafetyGateError("Trainer max_steps differs from ceil schedule")
        if bool(args.dataloader_drop_last):
            raise SafetyGateError("Model40 requires dataloader_drop_last=False")
        return (
            schedule.epochs,
            schedule.optimizer_updates_per_epoch,
            examples,
            schedule.total_row_presentations,
            False,
            rows,
            schedule.total_optimizer_updates,
        )


def make_schedule_guard_callback(
    schedule: OptimizerSchedule, callback_base: type[Any] | None = None
) -> Any:
    if callback_base is None:
        from transformers import TrainerCallback

        callback_base = TrainerCallback

    class ScheduleGuardCallback(callback_base):
        def __init__(self) -> None:
            self.epoch_boundaries: list[dict[str, int]] = []

        def on_epoch_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            completed = len(self.epoch_boundaries) + 1
            expected_step = completed * schedule.optimizer_updates_per_epoch
            if (
                completed > schedule.epochs
                or int(state.global_step) != expected_step
                or not math.isclose(float(state.epoch or 0), float(completed), abs_tol=1e-12)
            ):
                raise SafetyGateError(
                    f"partial/extra epoch: epoch={state.epoch}, step={state.global_step}, "
                    f"expected_epoch={completed}, expected_step={expected_step}"
                )
            self.epoch_boundaries.append(
                {
                    "epoch": completed,
                    "optimizer_step": expected_step,
                    "rows_seen_this_epoch": schedule.train_rows,
                }
            )
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            if (
                len(self.epoch_boundaries) != schedule.epochs
                or int(state.global_step) != schedule.total_optimizer_updates
            ):
                raise SafetyGateError("training did not complete the audited epoch schedule")
            return control

    return ScheduleGuardCallback()


def load_training_model(config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SafetyGateError("Model40 requires CUDA; CPU training is disabled")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SafetyGateError("Model40 is certified only for one CUDA process")
    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    checkpoint = _absolute(str(_nested(config, "model", "local_path")))
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(checkpoint),
        trust_remote_code=False,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": device_index},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    if int(model.config.max_position_embeddings) != MODEL_CONTEXT_TOKENS:
        raise SafetyGateError("loaded model context window differs from pinned audit")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    frozen_vision_parameters = freeze_vision_backbone(model)
    targets = discover_full_visual_targets(model, int(_nested(config, "lora", "rank")))
    adapter = LoraConfig(
        r=targets.rank,
        lora_alpha=int(_nested(config, "lora", "alpha")),
        lora_dropout=float(_nested(config, "lora", "dropout")),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=exact_lora_target_pattern(targets.target_modules),
    )
    model = get_peft_model(model, adapter)
    injected = audit_injected_lora_targets(model, targets.target_modules)
    trainable = audit_trainable_parameters(
        model,
        targets,
        max_trainable_parameters=int(_nested(config, "smoke", "max_trainable_parameters")),
    )
    quantized = audit_nf4_modules(model, int(_nested(config, "smoke", "minimum_4bit_modules")))
    residency = audit_cuda_residency(model)
    return model, {
        "device_index": device_index,
        "frozen_vision_parameters": frozen_vision_parameters,
        "lora_targets": asdict(targets),
        "injected_lora": injected,
        "trainable": trainable,
        "quantization": quantized,
        "residency": residency,
    }


def _load_training_records(config: Mapping[str, Any]) -> tuple[Any, ...]:
    return load_records(
        _absolute(str(_nested(config, "train_csv"))),
        _absolute(str(_nested(config, "image_dir"))),
        require_target=True,
        require_five_folds=True,
    )


def dry_run(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    enable_offline_mode()
    config, resolved = load_config(config_path)
    provenance = verify_starter_provenance(config)
    checkpoint = verify_local_checkpoint(config)
    targets = audit_checkpoint_target_shapes(config)
    context = audit_model_context(config)
    records = _load_training_records(config)
    sequence = audit_sequence_budget(config, records)
    geometry = audit_resize_geometry(str(_nested(config, "image_dir")), config)
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
            "train_rows": len(training),
            "validation_rows": len(validation),
            "optimizer_schedule": asdict(schedule),
        }
    report = {
        "status": "cpu_audits_passed_gpu_smoke_not_run",
        "resolved_config": str(resolved),
        "config_sha256": _sha256(resolved),
        "provenance": provenance,
        "checkpoint": checkpoint,
        "targets": asdict(targets),
        "targeted_base_parameters": sum(
            inputs * outputs for inputs, outputs in targets.shapes.values()
        ),
        "model_context": context,
        "sequence_budget": asdict(sequence),
        "resize_geometry": geometry,
        "folds": folds,
        "leakage_contract": "every validation ID is absent from that fold's training IDs",
    }
    _json_write(reject_model22_destination(output_path), report)
    return report


def train(config_path: str | Path, fold: int, *, smoke_only: bool = False) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    enable_offline_mode()
    config, resolved = load_config(config_path)
    provenance = verify_starter_provenance(config)
    checkpoint = verify_local_checkpoint(config)
    targets = audit_checkpoint_target_shapes(config)
    context = audit_model_context(config)
    records = _load_training_records(config)
    sequence = audit_sequence_budget(config, records)
    training_records, validation_records = split_outer_fold(records, fold)
    schedule = deterministic_optimizer_schedule(
        len(training_records),
        int(_nested(config, "training", "gradient_accumulation")),
        int(_nested(config, "training", "epochs")),
    )
    verify_configured_schedule(config, schedule)
    output_dir = require_new_output_dir(str(_nested(config, "output_dir")).format(fold=fold))
    set_seed(int(_nested(config, "seed")))
    processor = load_processor(config)
    model, model_audit = load_training_model(config)
    collator = FullVisualQwenCollator(
        processor,
        str(_nested(config, "prompt")),
        int(_nested(config, "image", "max_image_tokens")),
        int(_nested(config, "image", "max_sequence_tokens")),
    )
    smoke_record = select_memory_smoke_record(
        training_records, config, tokenizer=processor.tokenizer
    )
    smoke = run_memory_smoke(
        model,
        collator,
        smoke_record,
        max_allocated_gib=float(_nested(config, "smoke", "max_allocated_gib")),
        max_reserved_gib=float(_nested(config, "smoke", "max_reserved_gib")),
    )
    audit = {
        "fold": fold,
        "rows": {"all": len(records), "train": len(training_records), "validation": len(validation_records)},
        "provenance": provenance,
        "local_checkpoint": checkpoint,
        "target_audit": asdict(targets),
        "model_context": context,
        "sequence_budget": asdict(sequence),
        "optimizer_schedule": asdict(schedule),
        "model": model_audit,
        "smoke": smoke,
    }
    if smoke_only:
        smoke_dir = reject_model22_destination(str(_nested(config, "smoke", "report_dir")))
        _json_write(smoke_dir / f"fold{fold}.json", audit)
        return audit

    output_dir.mkdir(parents=True, exist_ok=True)
    _json_write(
        output_dir / "run_manifest.json",
        {
            **_runtime_manifest(config),
            **audit,
            "model_number": 40,
            "resolved_config": str(resolved),
            "train_csv_sha256": _sha256(_absolute(str(_nested(config, "train_csv")))),
        },
    )

    import torch
    from transformers import Trainer, TrainingArguments

    class ExactEpochTrainer(ExactEpochScheduleMixin, Trainer):
        pass

    exact_callback = _make_validation_callback(
        processor,
        collator,
        validation_records,
        output_dir,
        int(_nested(config, "generation", "max_new_tokens")),
        repetition_penalty=float(
            _nested(config, "generation", "repetition_penalty")
        ),
        no_repeat_ngram_size=int(
            _nested(config, "generation", "no_repeat_ngram_size")
        ),
    )
    schedule_callback = make_schedule_guard_callback(schedule)
    training = _nested(config, "training")
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        seed=int(_nested(config, "seed")),
        data_seed=int(_nested(config, "seed")),
        full_determinism=True,
        num_train_epochs=float(_nested(training, "epochs")),
        max_steps=schedule.total_optimizer_updates,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(_nested(training, "gradient_accumulation")),
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
        dataloader_drop_last=False,
        report_to="none",
    )
    set_seed(int(_nested(config, "seed")))
    trainer = ExactEpochTrainer(
        model=model,
        args=arguments,
        train_dataset=RecordDataset(training_records),
        data_collator=collator,
        callbacks=[schedule_callback, exact_callback],
    )
    trainer.exact_optimizer_schedule = schedule
    result = trainer.train()
    if int(result.global_step) != schedule.total_optimizer_updates:
        raise SafetyGateError("Trainer did not complete the exact optimizer schedule")
    if exact_callback.best_epoch is None:
        raise SafetyGateError("training ended without complete held-fold validation")
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)
    complete = {
        "fold": fold,
        "best_epoch": exact_callback.best_epoch,
        "best_multi_score": exact_callback.best_score,
        "optimizer_schedule": asdict(schedule),
        "observed_epoch_boundaries": schedule_callback.epoch_boundaries,
        "observed_optimizer_updates": int(result.global_step),
        "training_metrics": result.metrics,
        "final_adapter": str(final_dir),
        "best_adapter": str(output_dir / "best_adapter"),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
    }
    _json_write(output_dir / "complete.json", complete)
    return complete


def _load_inference_model(
    config: Mapping[str, Any], adapter_path: Path
) -> tuple[Any, FullVisualTargetAudit]:
    import torch
    from peft import PeftModel
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SafetyGateError("Model40 inference requires CUDA")
    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    checkpoint = _absolute(str(_nested(config, "model", "local_path")))
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        str(checkpoint),
        trust_remote_code=False,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": device_index},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    targets = discover_full_visual_targets(base)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    audit_injected_lora_targets(model, targets.target_modules)
    audit_cuda_residency(model)
    audit_nf4_modules(model, int(_nested(config, "smoke", "minimum_4bit_modules")))
    model.eval()
    return model, targets


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
            raise ValueError("fold must be one of 0,1,2,3,4")
        if not records or records[0].fold is None:
            raise SchemaError("--fold requires a fold column")
        records = tuple(record for record in records if record.fold == fold)
    if not records:
        raise ValueError("inference selection is empty")
    adapter = _absolute(adapter_path)
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    output = reject_model22_destination(output_csv)
    processor = load_processor(config)
    model, targets = _load_inference_model(config, adapter)
    collator = FullVisualQwenCollator(
        processor,
        str(_nested(config, "prompt")),
        int(_nested(config, "image", "max_image_tokens")),
        int(_nested(config, "image", "max_sequence_tokens")),
    )
    predictions = greedy_predict(
        model,
        processor,
        collator,
        records,
        max_new_tokens=int(_nested(config, "generation", "max_new_tokens")),
        repetition_penalty=float(
            _nested(config, "generation", "repetition_penalty")
        ),
        no_repeat_ngram_size=int(
            _nested(config, "generation", "no_repeat_ngram_size")
        ),
    )
    write_predictions(output, records, predictions)
    report: dict[str, Any] = {
        "rows": len(records),
        "fold": fold,
        "adapter": str(adapter),
        "local_checkpoint": checkpoint,
        "output": str(output),
        "target_audit": asdict(targets),
        "image_policy": {
            "resize": str(_nested(config, "image", "resize_policy")),
            "max_image_tokens": int(_nested(config, "image", "max_image_tokens")),
        },
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": int(_nested(config, "generation", "max_new_tokens")),
            "repetition_penalty": float(
                _nested(config, "generation", "repetition_penalty")
            ),
            "no_repeat_ngram_size": int(
                _nested(config, "generation", "no_repeat_ngram_size")
            ),
        },
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

    parser = argparse.ArgumentParser(description="Model40 full-visual Qwen2-VL QLoRA")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.fold, smoke_only=args.smoke_only), indent=2, ensure_ascii=False))


def infer_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model40 full-visual Qwen2-VL inference")
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

    parser = argparse.ArgumentParser(description="Model40 offline architecture/sequence audit")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/model40/dry_run.json")
    args = parser.parse_args()
    print(json.dumps(dry_run(args.config, args.output), indent=2, ensure_ascii=False))
