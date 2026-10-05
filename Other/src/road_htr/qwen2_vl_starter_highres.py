"""Model43: organizer-aligned rank-16 language QLoRA at high resolution.

The adapter exactly reproduces the starter's seven language projection
families, but binds them to all 196 full module paths in the pinned checkpoint.
The complete visual tower and merger remain frozen. Full images are uniformly
scaled into factor-aligned 896..1024-token canvases without glyph distortion.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
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
LANGUAGE_SHAPES = {
    "q_proj": (1536, 1536),
    "k_proj": (1536, 256),
    "v_proj": (1536, 256),
    "o_proj": (1536, 1536),
    "gate_proj": (1536, 8960),
    "up_proj": (1536, 8960),
    "down_proj": (8960, 1536),
}
MERGER_SHAPES = {
    "visual.merger.mlp.0": (5120, 5120),
    "visual.merger.mlp.2": (5120, 1536),
}
LORA_RANK = 16
EXPECTED_TARGET_MODULES = LANGUAGE_LAYERS * len(LANGUAGE_PROJECTIONS)
EXPECTED_TRAINABLE_PARAMETERS = 18_464_768
EXPECTED_TRAINABLE_TENSORS = EXPECTED_TARGET_MODULES * 2
MERGED_PATCH_SIZE = 28
MODEL_CONTEXT_TOKENS = 32_768
IMAGE_PAD_TOKEN = "<|image_pad|>"
PINNED_RUNTIME = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "4.49.0",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.2",
    "peft": "0.14.0",
    "safetensors": "0.7.0",
    "Pillow": "12.3.0",
    "PyYAML": "6.0.3",
}

LANGUAGE_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\."
    r"(?:(?:self_attn\.(?P<attention>q_proj|k_proj|v_proj|o_proj))|"
    r"(?:mlp\.(?P<mlp>gate_proj|up_proj|down_proj)))$"
)


@dataclass(frozen=True)
class LanguageTargetAudit:
    rank: int
    target_modules: tuple[str, ...]
    language_layers: int
    language_modules: int
    merger_modules: int
    visual_modules: int
    trainable_parameters: int
    trainable_tensors: int
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


@dataclass(frozen=True)
class SequenceAudit:
    max_image_tokens: int
    training_sequence_cap: int
    observed_max_training_tokens: int
    observed_max_training_id: str
    observed_max_label_tokens: int
    observed_max_label_id: str
    prompt_tokens_at_max_image_budget: int
    max_new_tokens: int
    generation_upper_bound: int
    generation_context_cap: int
    model_context_tokens: int


def _language_path(layer: int, projection: str) -> str:
    family = "self_attn" if projection in LANGUAGE_PROJECTIONS[:4] else "mlp"
    return f"model.layers.{layer}.{family}.{projection}"


def expected_target_shapes() -> dict[str, tuple[int, int]]:
    shapes = {
        _language_path(layer, projection): LANGUAGE_SHAPES[projection]
        for layer in range(LANGUAGE_LAYERS)
        for projection in LANGUAGE_PROJECTIONS
    }
    if len(shapes) != EXPECTED_TARGET_MODULES:
        raise AssertionError("internal Model43 language target ledger is not exact")
    return shapes


def calculate_lora_parameters(
    shapes: Mapping[str, tuple[int, int]], rank: int
) -> int:
    expected = set(expected_target_shapes())
    if set(shapes) != expected:
        raise SafetyGateError(
            f"target ledger mismatch: missing={sorted(expected - set(shapes))[:5]}, "
            f"extra={sorted(set(shapes) - expected)[:5]}"
        )
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if any(inputs <= 0 or outputs <= 0 for inputs, outputs in shapes.values()):
        raise SafetyGateError("target ledger contains a nonpositive dimension")
    return sum(rank * (inputs + outputs) for inputs, outputs in shapes.values())


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
        ("image", "min_effective_image_tokens"): 896,
        ("image", "target_image_tokens"): 1024,
        ("image", "max_image_tokens"): 1024,
        ("image", "max_sequence_tokens"): 1152,
        ("image", "max_generation_context_tokens"): 1200,
        ("image", "resize_policy"): "aspect_preserving_letterbox_lanczos",
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
    }
    for keys, expected in immutable.items():
        actual = _nested(config, *keys)
        if actual != expected:
            raise ValueError(
                f"{'.'.join(keys)} must be {expected!r} for Model43, got {actual!r}"
            )
    if _nested(config, "model", "expected_files_sha256") != LOCAL_CHECKPOINT_FILES:
        raise ValueError("checkpoint hash ledger differs from the pinned snapshot")
    if not 0.0 <= float(_nested(config, "lora", "dropout")) < 1.0:
        raise ValueError("lora.dropout must be in [0,1)")
    minimum_pixels = int(_nested(config, "image", "min_pixels"))
    maximum_pixels = int(_nested(config, "image", "max_pixels"))
    if minimum_pixels != 896 * MERGED_PATCH_SIZE**2:
        raise ValueError("min_pixels must exactly encode 896 visual tokens")
    if maximum_pixels != 1024 * MERGED_PATCH_SIZE**2:
        raise ValueError("max_pixels must exactly encode 1024 visual tokens")
    sequence_cap = int(_nested(config, "image", "max_sequence_tokens"))
    generation_cap = int(_nested(config, "image", "max_generation_context_tokens"))
    if sequence_cap <= 1024 or generation_cap != sequence_cap + 48:
        raise ValueError("sequence/generation envelope is inconsistent")
    if generation_cap > MODEL_CONTEXT_TOKENS:
        raise ValueError("generation cap exceeds Qwen context")
    for key in ("max_allocated_gib", "max_reserved_gib"):
        value = float(_nested(config, "smoke", key))
        if not 0.0 < value <= 5.3:
            raise ValueError(f"smoke.{key} must be positive and <=5.3")
    parameter_cap = int(_nested(config, "smoke", "max_trainable_parameters"))
    if parameter_cap < EXPECTED_TRAINABLE_PARAMETERS or parameter_cap > 19_000_000:
        raise ValueError("trainable parameter cap must cover exact count and stay <=19M")
    for key in (
        "expected_train_rows_per_fold",
        "expected_updates_per_epoch",
        "expected_total_updates",
    ):
        value = _nested(config, "training", key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    if "{fold}" not in str(_nested(config, "output_dir")):
        raise ValueError("output_dir must contain {fold}")


def audit_runtime_versions() -> dict[str, str]:
    """Require the exact dependency versions recorded in requirements.txt."""

    from importlib.metadata import PackageNotFoundError, version

    installed: dict[str, str] = {}
    for package, expected in PINNED_RUNTIME.items():
        try:
            actual = version(package)
        except PackageNotFoundError as error:
            raise SafetyGateError(f"required runtime package is absent: {package}") from error
        if actual != expected:
            raise SafetyGateError(
                f"runtime version mismatch for {package}: {actual} != {expected}"
            )
        installed[package] = actual
    return installed


def load_processor(config: Mapping[str, Any]) -> Any:
    """Load the pinned slow processor with the exact aligned pixel interval."""

    from transformers import AutoProcessor

    minimum = int(_nested(config, "image", "min_pixels"))
    maximum = int(_nested(config, "image", "max_pixels"))
    checkpoint = _absolute(str(_nested(config, "model", "local_path")))
    processor = AutoProcessor.from_pretrained(
        str(checkpoint),
        trust_remote_code=False,
        local_files_only=True,
        use_fast=False,
        min_pixels=minimum,
        max_pixels=maximum,
    )
    verify_processor_budget(processor, minimum, maximum)
    return processor


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
        raise SafetyGateError(f"target is not a supported linear: {name}")
    return inputs, outputs


def discover_language_targets(model: Any, rank: int = LORA_RANK) -> LanguageTargetAudit:
    modules = dict(model.named_modules())
    wanted = expected_target_shapes()
    observed = {
        name for name in modules if LANGUAGE_PATTERN.fullmatch(name) is not None
    }
    if observed != set(wanted):
        raise SafetyGateError(
            f"language layout mismatch: missing={sorted(set(wanted) - observed)[:5]}, "
            f"extra={sorted(observed - set(wanted))[:5]}"
        )
    shapes: dict[str, tuple[int, int]] = {}
    for name, expected in wanted.items():
        actual = _linear_shape(modules[name], name)
        if actual != expected:
            raise SafetyGateError(f"projection shape mismatch for {name}: {actual} != {expected}")
        shapes[name] = actual
    merger_linears = {
        name
        for name, module in modules.items()
        if name.startswith("visual.merger.")
        and module.__class__.__name__ in {"Linear", "Linear4bit"}
    }
    if merger_linears != set(MERGER_SHAPES):
        raise SafetyGateError(f"visual merger layout mismatch: {sorted(merger_linears)}")
    for name, expected in MERGER_SHAPES.items():
        if _linear_shape(modules[name], name) != expected:
            raise SafetyGateError(f"frozen merger shape drifted: {name}")
    trainable = calculate_lora_parameters(shapes, rank)
    if rank != LORA_RANK or trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise SafetyGateError(f"LoRA count mismatch: rank={rank}, params={trainable}")
    return LanguageTargetAudit(
        rank=rank,
        target_modules=tuple(shapes),
        language_layers=LANGUAGE_LAYERS,
        language_modules=len(shapes),
        merger_modules=0,
        visual_modules=0,
        trainable_parameters=trainable,
        trainable_tensors=len(shapes) * 2,
        shapes=shapes,
    )


def audit_checkpoint_target_shapes(
    config: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> LanguageTargetAudit:
    from safetensors import safe_open

    checkpoint = _absolute(str(_nested(config, "model", "local_path")), project_root)
    wanted = expected_target_shapes()
    found: dict[str, tuple[int, int]] = {}
    merger_found: dict[str, tuple[int, int]] = {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for ledger, destination in ((wanted, found), (MERGER_SHAPES, merger_found)):
                for name in ledger:
                    weight = f"{name}.weight"
                    if weight not in keys:
                        continue
                    raw = tuple(int(value) for value in handle.get_slice(weight).get_shape())
                    if len(raw) != 2 or name in destination:
                        raise SafetyGateError(f"invalid/duplicate checkpoint matrix: {weight}")
                    destination[name] = (raw[1], raw[0])
    if found != wanted or merger_found != MERGER_SHAPES:
        raise SafetyGateError("checkpoint language or frozen-merger shape ledger mismatch")
    canonical = {name: found[name] for name in wanted}
    trainable = calculate_lora_parameters(canonical, LORA_RANK)
    return LanguageTargetAudit(
        rank=LORA_RANK,
        target_modules=tuple(canonical),
        language_layers=LANGUAGE_LAYERS,
        language_modules=len(canonical),
        merger_modules=0,
        visual_modules=0,
        trainable_parameters=trainable,
        trainable_tensors=len(canonical) * 2,
        shapes=canonical,
    )


def audit_model_context(config: Mapping[str, Any]) -> dict[str, int]:
    checkpoint = _absolute(str(_nested(config, "model", "local_path")))
    metadata = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    actual = int(metadata.get("max_position_embeddings", -1))
    if actual != MODEL_CONTEXT_TOKENS:
        raise SafetyGateError(f"model context drifted: {actual}")
    configured = int(_nested(config, "image", "max_generation_context_tokens"))
    if configured > actual:
        raise SafetyGateError("configured context exceeds model context")
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
            if max(ratio, 1 / ratio) <= 200:
                grids.append((width_units, height_units, tokens))
    if not grids:
        raise ValueError("no aligned grid satisfies the token interval")
    return tuple(grids)


def plan_letterbox(
    width: int,
    height: int,
    *,
    minimum_tokens: int,
    maximum_tokens: int,
    factor: int = MERGED_PATCH_SIZE,
) -> ResizePlan:
    values = (width, height, minimum_tokens, maximum_tokens, factor)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("resize inputs must be positive integers")
    if minimum_tokens > maximum_tokens:
        raise ValueError("minimum visual tokens exceed maximum")
    source_aspect = width / height
    if max(source_aspect, 1 / source_aspect) > 200:
        raise SafetyGateError("source image exceeds Qwen's 200:1 aspect limit")

    def score(grid: tuple[int, int, int]) -> tuple[float, float, int]:
        width_units, height_units, tokens = grid
        canvas_aspect = width_units / height_units
        occupancy = min(canvas_aspect / source_aspect, source_aspect / canvas_aspect)
        return occupancy * tokens, occupancy, tokens

    width_units, height_units, tokens = max(
        _candidate_grids(minimum_tokens, maximum_tokens), key=score
    )
    canvas_width = width_units * factor
    canvas_height = height_units * factor
    scale = min(canvas_width / width, canvas_height / height)
    content_width = min(canvas_width, max(1, round(width * scale)))
    content_height = min(canvas_height, max(1, round(height * scale)))
    aspect_error = abs((content_width / content_height) / source_aspect - 1)
    if aspect_error > 0.005:
        raise SafetyGateError(f"content aspect error exceeds 0.5%: {aspect_error}")
    fraction = content_width * content_height / (canvas_width * canvas_height)
    return ResizePlan(
        source_width=width,
        source_height=height,
        content_width=content_width,
        content_height=content_height,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        uniform_scale=scale,
        planned_image_tokens=tokens,
        content_pixel_fraction=fraction,
        relative_content_aspect_error=aspect_error,
    )


def qwen_grid_tokens(width: int, height: int, factor: int = MERGED_PATCH_SIZE) -> int:
    if width % factor or height % factor:
        raise SafetyGateError("letterbox canvas is not factor aligned")
    return width * height // factor**2


def load_letterboxed_rgb(
    path: str | Path, *, minimum_tokens: int, maximum_tokens: int
) -> tuple[Image.Image, dict[str, int | float]]:
    with Image.open(path) as source:
        rgb = source.convert("RGB")
    plan = plan_letterbox(
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
    canvas.paste(
        content,
        (
            (plan.canvas_width - plan.content_width) // 2,
            (plan.canvas_height - plan.content_height) // 2,
        ),
    )
    return canvas, {
        **asdict(plan),
        "padding_pixel_fraction": 1 - plan.content_pixel_fraction,
    }


class HighResolutionQwenCollator(QwenOcrCollator):
    def __init__(
        self,
        processor: Any,
        prompt: str,
        min_image_tokens: int,
        max_image_tokens: int,
        max_sequence_tokens: int,
    ):
        super().__init__(processor, prompt, max_image_tokens, max_sequence_tokens)
        self.min_image_tokens = min_image_tokens

    def _image(self, record: Any) -> tuple[Image.Image, dict[str, int | float]]:
        return load_letterboxed_rgb(
            record.image_path,
            minimum_tokens=self.min_image_tokens,
            maximum_tokens=self.max_image_tokens,
        )

    def encode_prompt(self, record: Any) -> tuple[dict[str, Any], dict[str, int | float]]:
        image, resize = self._image(record)
        batch = dict(self._processor_call(self._encode_text(record, False), image))
        diagnostics: dict[str, int | float] = self._diagnostics(batch)
        diagnostics.update(resize)
        if diagnostics["image_tokens"] != diagnostics["planned_image_tokens"]:
            raise SafetyGateError("processor changed the aligned visual token grid")
        return batch, diagnostics

    def encode_training(self, record: Any) -> tuple[dict[str, Any], dict[str, int | float]]:
        import torch

        image, resize = self._image(record)
        full = dict(self._processor_call(self._encode_text(record, True), image))
        prompt = dict(self._processor_call(self._encode_text(record, False), image))
        diagnostics: dict[str, int | float] = self._diagnostics(full)
        diagnostics.update(resize)
        if diagnostics["image_tokens"] != diagnostics["planned_image_tokens"]:
            raise SafetyGateError("processor changed the aligned visual token grid")
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
            raise SafetyGateError(f"Model43 supports batch size 1, got {len(examples)}")
        batch, _ = self.encode_training(examples[0])
        return batch


def _template_tokens(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    generation_prompt: bool,
    image_tokens: int,
) -> int:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=generation_prompt
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    image_pad = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    if ids.count(image_pad) != 1:
        raise SafetyGateError("chat template must contain exactly one image pad")
    return len(ids) - 1 + image_tokens


def audit_sequence_budget(config: Mapping[str, Any], records: Sequence[Any]) -> SequenceAudit:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(_absolute(str(_nested(config, "model", "local_path")))),
        trust_remote_code=False,
        local_files_only=True,
    )
    prompt = str(_nested(config, "prompt"))
    image_tokens = int(_nested(config, "image", "max_image_tokens"))
    maximum = (-1, "")
    maximum_label = (-1, "")
    for record in records:
        if record.label is None:
            continue
        total = _template_tokens(
            tokenizer,
            build_messages(prompt, record.label),
            generation_prompt=False,
            image_tokens=image_tokens,
        )
        maximum = max(maximum, (total, record.identifier))
        label_tokens = len(tokenizer.encode(record.label, add_special_tokens=False))
        maximum_label = max(maximum_label, (label_tokens, record.identifier))
    if maximum[0] < 0:
        raise SafetyGateError("sequence audit requires labelled records")
    prompt_tokens = _template_tokens(
        tokenizer,
        build_messages(prompt),
        generation_prompt=True,
        image_tokens=image_tokens,
    )
    max_new = int(_nested(config, "generation", "max_new_tokens"))
    generation_upper = prompt_tokens + max_new
    training_cap = int(_nested(config, "image", "max_sequence_tokens"))
    context_cap = int(_nested(config, "image", "max_generation_context_tokens"))
    if maximum[0] > training_cap:
        raise SafetyGateError("static training sequence exceeds cap")
    if generation_upper > context_cap or context_cap > MODEL_CONTEXT_TOKENS:
        raise SafetyGateError("generation context envelope is invalid")
    if max_new < maximum_label[0] + 4:
        raise SafetyGateError("generation cap leaves insufficient target headroom")
    return SequenceAudit(
        max_image_tokens=image_tokens,
        training_sequence_cap=training_cap,
        observed_max_training_tokens=maximum[0],
        observed_max_training_id=maximum[1],
        observed_max_label_tokens=maximum_label[0],
        observed_max_label_id=maximum_label[1],
        prompt_tokens_at_max_image_budget=prompt_tokens,
        max_new_tokens=max_new,
        generation_upper_bound=generation_upper,
        generation_context_cap=context_cap,
        model_context_tokens=MODEL_CONTEXT_TOKENS,
    )


def audit_geometry(image_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = sorted(_absolute(image_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(image_dir)
    minimum = int(_nested(config, "image", "min_effective_image_tokens"))
    maximum = int(_nested(config, "image", "max_image_tokens"))
    tokens: list[int] = []
    fractions: list[float] = []
    errors: list[float] = []
    upscaled = downscaled = unchanged = 0
    for path in paths:
        with Image.open(path) as image:
            plan = plan_letterbox(
                image.width,
                image.height,
                minimum_tokens=minimum,
                maximum_tokens=maximum,
            )
        actual_tokens = qwen_grid_tokens(plan.canvas_width, plan.canvas_height)
        if actual_tokens != plan.planned_image_tokens:
            raise SafetyGateError("planned canvas token count is inconsistent")
        tokens.append(actual_tokens)
        fractions.append(plan.content_pixel_fraction)
        errors.append(plan.relative_content_aspect_error)
        if plan.uniform_scale > 1 + 1e-12:
            upscaled += 1
        elif plan.uniform_scale < 1 - 1e-12:
            downscaled += 1
        else:
            unchanged += 1
    tokens.sort()
    fractions.sort()
    return {
        "images": len(paths),
        "policy": "aspect_preserving_letterbox_lanczos",
        "minimum_configured_tokens": minimum,
        "maximum_configured_tokens": maximum,
        "upscaled": upscaled,
        "downscaled": downscaled,
        "unchanged": unchanged,
        "effective_tokens": {
            "minimum": tokens[0],
            "median": tokens[len(tokens) // 2],
            "maximum": tokens[-1],
        },
        "content_pixel_fraction": {
            "minimum": fractions[0],
            "median": fractions[len(fractions) // 2],
            "maximum": fractions[-1],
        },
        "maximum_content_aspect_error": max(errors),
    }


def select_smoke_record(
    records: Sequence[Any], config: Mapping[str, Any], tokenizer: Any
) -> Any:
    prompt = str(_nested(config, "prompt"))
    minimum = int(_nested(config, "image", "min_effective_image_tokens"))
    maximum = int(_nested(config, "image", "max_image_tokens"))
    candidates: list[tuple[tuple[int, int, str], Any]] = []
    for record in records:
        if record.label is None:
            continue
        with Image.open(record.image_path) as image:
            plan = plan_letterbox(
                image.width,
                image.height,
                minimum_tokens=minimum,
                maximum_tokens=maximum,
            )
        workload = _template_tokens(
            tokenizer,
            build_messages(prompt, record.label),
            generation_prompt=False,
            image_tokens=plan.planned_image_tokens,
        )
        candidates.append(((workload, plan.planned_image_tokens, record.identifier), record))
    if not candidates:
        raise SafetyGateError("no labelled smoke candidate")
    return max(candidates, key=lambda item: item[0])[1]


def audit_trainable_parameters(
    model: Any, audit: LanguageTargetAudit, *, maximum: int
) -> dict[str, Any]:
    target_set = set(audit.target_modules)
    matched: set[str] = set()
    names: list[str] = []
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        count += int(parameter.numel())
        if ".lora_A." not in name and ".lora_B." not in name:
            raise SafetyGateError(f"non-LoRA tensor is trainable: {name}")
        owners = [target for target in target_set if f"{target}.lora_" in name]
        if len(owners) != 1:
            raise SafetyGateError(f"trainable tensor has no unique language owner: {name}")
        if "visual." in name:
            raise SafetyGateError(f"visual tower tensor is trainable: {name}")
        matched.add(owners[0])
    if matched != target_set:
        raise SafetyGateError("not every language target received LoRA tensors")
    if count != audit.trainable_parameters or count > maximum:
        raise SafetyGateError(f"trainable parameter mismatch: {count}")
    if len(names) != audit.trainable_tensors:
        raise SafetyGateError(f"trainable tensor mismatch: {len(names)}")
    return {
        "trainable_parameters": count,
        "trainable_tensors": len(names),
        "target_modules": len(matched),
        "visual_trainable_tensors": 0,
        "sample_names": names[:8],
    }


def split_outer_fold(records: Sequence[Any], fold: int) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if fold not in range(5):
        raise ValueError("fold must be one of 0,1,2,3,4")
    training = tuple(record for record in records if record.fold != fold)
    validation = tuple(record for record in records if record.fold == fold)
    if not training or not validation or len(training) + len(validation) != len(records):
        raise SafetyGateError("outer fold split is incomplete")
    train_ids = {record.identifier for record in training}
    validation_ids = {record.identifier for record in validation}
    if len(train_ids) != len(training) or len(validation_ids) != len(validation):
        raise SafetyGateError("duplicate IDs inside fold split")
    if train_ids & validation_ids:
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
            raise SafetyGateError("exact schedule is absent")
        rows = len(dataloader)
        examples = int(self.num_examples(dataloader))
        if rows != schedule.train_rows or examples != schedule.train_rows:
            raise SafetyGateError("training dataloader differs from audited schedule")
        if int(args.gradient_accumulation_steps) != schedule.gradient_accumulation:
            raise SafetyGateError("gradient accumulation differs from schedule")
        if int(args.max_steps) != schedule.total_optimizer_updates:
            raise SafetyGateError("max_steps differs from ceil schedule")
        if bool(args.dataloader_drop_last):
            raise SafetyGateError("Model43 requires dataloader_drop_last=False")
        return (
            schedule.epochs,
            schedule.optimizer_updates_per_epoch,
            examples,
            schedule.total_row_presentations,
            False,
            rows,
            schedule.total_optimizer_updates,
        )


def make_schedule_callback(
    schedule: OptimizerSchedule, callback_base: type[Any] | None = None
) -> Any:
    if callback_base is None:
        from transformers import TrainerCallback

        callback_base = TrainerCallback

    class ScheduleCallback(callback_base):
        def __init__(self) -> None:
            self.boundaries: list[dict[str, int]] = []

        def on_epoch_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            epoch = len(self.boundaries) + 1
            step = epoch * schedule.optimizer_updates_per_epoch
            if (
                epoch > schedule.epochs
                or int(state.global_step) != step
                or not math.isclose(float(state.epoch or 0), float(epoch), abs_tol=1e-12)
            ):
                raise SafetyGateError("partial or extra training epoch")
            self.boundaries.append(
                {"epoch": epoch, "optimizer_step": step, "rows_seen": schedule.train_rows}
            )
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            if len(self.boundaries) != schedule.epochs:
                raise SafetyGateError("not every epoch completed")
            if int(state.global_step) != schedule.total_optimizer_updates:
                raise SafetyGateError("final optimizer step differs from schedule")
            return control

    return ScheduleCallback()


def load_training_model(config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SafetyGateError("Model43 requires CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SafetyGateError("Model43 supports one CUDA process")
    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device)
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
        device_map={"": device},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    if int(model.config.max_position_embeddings) != MODEL_CONTEXT_TOKENS:
        raise SafetyGateError("loaded model context differs from pinned context")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    frozen_visual_parameters = freeze_vision_backbone(model)
    targets = discover_language_targets(model, int(_nested(config, "lora", "rank")))
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
        maximum=int(_nested(config, "smoke", "max_trainable_parameters")),
    )
    return model, {
        "device_index": device,
        "frozen_visual_parameters": frozen_visual_parameters,
        "visual_tower_fully_frozen": True,
        "targets": asdict(targets),
        "injected": injected,
        "trainable": trainable,
        "quantization": audit_nf4_modules(
            model, int(_nested(config, "smoke", "minimum_4bit_modules"))
        ),
        "residency": audit_cuda_residency(model),
    }


def _training_records(config: Mapping[str, Any]) -> tuple[Any, ...]:
    return load_records(
        _absolute(str(_nested(config, "train_csv"))),
        _absolute(str(_nested(config, "image_dir"))),
        require_target=True,
        require_five_folds=True,
    )


def audit_processor_sample(
    config: Mapping[str, Any], records: Sequence[Any]
) -> dict[str, Any]:
    """Run the real pinned processor on the largest audited CPU workload."""

    processor = load_processor(config)
    record = select_smoke_record(records, config, processor.tokenizer)
    _, diagnostics = _collator(config, processor).encode_training(record)
    planned = int(diagnostics["planned_image_tokens"])
    actual = int(diagnostics["image_tokens"])
    sequence = int(diagnostics["sequence_tokens"])
    if planned != actual:
        raise SafetyGateError("real processor image grid differs from letterbox plan")
    if sequence > int(_nested(config, "image", "max_sequence_tokens")):
        raise SafetyGateError("real processor sample exceeds training sequence cap")
    return {
        "record_id": record.identifier,
        "processor_class": processor.__class__.__name__,
        "image_processor_class": processor.image_processor.__class__.__name__,
        "canvas_width": int(diagnostics["canvas_width"]),
        "canvas_height": int(diagnostics["canvas_height"]),
        "planned_image_tokens": planned,
        "processor_image_tokens": actual,
        "sequence_tokens": sequence,
        "prompt_tokens": int(diagnostics["prompt_tokens"]),
        "supervised_tokens": int(diagnostics["supervised_tokens"]),
    }


def dry_run(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    enable_offline_mode()
    config, resolved = load_config(config_path)
    runtime = audit_runtime_versions()
    provenance = verify_starter_provenance(config)
    checkpoint = verify_local_checkpoint(config)
    targets = audit_checkpoint_target_shapes(config)
    context = audit_model_context(config)
    records = _training_records(config)
    sequence = audit_sequence_budget(config, records)
    geometry = audit_geometry(str(_nested(config, "image_dir")), config)
    processor_sample = audit_processor_sample(config, records)
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
        "resolved_config": str(resolved),
        "config_sha256": _sha256(resolved),
        "organizer_alignment": {
            "rank": 16,
            "alpha": 32,
            "target_families": list(LANGUAGE_PROJECTIONS),
            "target_scope_difference": "exact 196 language paths instead of suffix matching",
            "merger_adapter": False,
            "visual_adapter": False,
        },
        "runtime": runtime,
        "provenance": provenance,
        "checkpoint": checkpoint,
        "targets": asdict(targets),
        "targeted_base_parameters": sum(
            inputs * outputs for inputs, outputs in targets.shapes.values()
        ),
        "model_context": context,
        "sequence": asdict(sequence),
        "processor_sample": processor_sample,
        "geometry": geometry,
        "folds": folds,
    }
    _json_write(reject_model22_destination(output_path), report)
    return report


def _collator(config: Mapping[str, Any], processor: Any) -> HighResolutionQwenCollator:
    return HighResolutionQwenCollator(
        processor,
        str(_nested(config, "prompt")),
        int(_nested(config, "image", "min_effective_image_tokens")),
        int(_nested(config, "image", "max_image_tokens")),
        int(_nested(config, "image", "max_sequence_tokens")),
    )


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
        "rows": {"all": len(records), "training": len(training_records), "validation": len(validation_records)},
        "runtime": runtime,
        "provenance": provenance,
        "checkpoint": checkpoint,
        "header_targets": asdict(header_targets),
        "model_context": context,
        "sequence": asdict(sequence),
        "schedule": asdict(schedule),
        "model": model_audit,
        "smoke": smoke,
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
            "model_number": 43,
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
    }
    _json_write(output_dir / "complete.json", complete)
    return complete


def _load_inference_model(
    config: Mapping[str, Any], adapter_path: Path
) -> tuple[Any, LanguageTargetAudit]:
    import torch
    from peft import PeftModel
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SafetyGateError("Model43 inference requires CUDA")
    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device)
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
        device_map={"": device},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    targets = discover_language_targets(base)
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

    parser = argparse.ArgumentParser(description="Model43 high-resolution starter-aligned QLoRA")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.fold, smoke_only=args.smoke_only), indent=2, ensure_ascii=False))


def infer_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model43 high-resolution inference")
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

    parser = argparse.ArgumentParser(description="Model43 offline audit")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/model43/dry_run.json")
    args = parser.parse_args()
    print(json.dumps(dry_run(args.config, args.output), indent=2, ensure_ascii=False))
