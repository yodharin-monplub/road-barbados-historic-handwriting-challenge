"""Model39: architecture-bound broad Qwen2-VL QLoRA.

The organizer starter targets every language attention and MLP projection by
suffix.  Model39 keeps that useful capacity, but resolves every target to an
exact path in the pinned Qwen2-VL 2B architecture.  It additionally adapts the
last eight vision blocks and the visual merger.  All target shapes, injected
modules, and trainable parameter counts are fail-closed audits.

Heavy CUDA dependencies remain inside runtime entry points so configuration,
checkpoint-header, and architecture audits can run without loading the model.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

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
    select_smoke_record,
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
ADAPTED_VISION_BLOCKS = tuple(range(24, 32))
VISION_PROJECTIONS = ("qkv", "proj", "fc1", "fc2")
MERGER_TARGETS = ("visual.merger.mlp.0", "visual.merger.mlp.2")
LORA_RANK = 8

# Values are (in_features, out_features), tied to the pinned config and checked
# against both the safetensors headers and the instantiated module graph.
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
EXPECTED_VISION_TARGETS = len(ADAPTED_VISION_BLOCKS) * len(VISION_PROJECTIONS)
EXPECTED_TARGET_MODULES = (
    EXPECTED_LANGUAGE_TARGETS + EXPECTED_VISION_TARGETS + len(MERGER_TARGETS)
)
EXPECTED_TRAINABLE_PARAMETERS = 10_678_272
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
class BroadTargetAudit:
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
    """Exact batch-one epoch/update budget for a finite training fold."""

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


def deterministic_optimizer_schedule(
    train_rows: int, gradient_accumulation: int, epochs: int
) -> OptimizerSchedule:
    """Return a ceil-based schedule that never drops an epoch remainder.

    Model39 is certified for batch size one and one process, so each training
    row is exactly one dataloader micro-batch.  The final partial accumulation
    group is a real optimizer update and must be included in ``max_steps``.
    """

    values = {
        "train_rows": train_rows,
        "gradient_accumulation": gradient_accumulation,
        "epochs": epochs,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values.values()):
        raise ValueError(f"optimizer schedule inputs must be positive integers: {values}")
    full_groups, remainder = divmod(train_rows, gradient_accumulation)
    updates_per_epoch = full_groups + int(remainder > 0)
    floor_updates = max(full_groups, 1)
    return OptimizerSchedule(
        train_rows=train_rows,
        gradient_accumulation=gradient_accumulation,
        epochs=epochs,
        full_accumulation_groups=full_groups,
        remainder_micro_batches=remainder,
        optimizer_updates_per_epoch=updates_per_epoch,
        total_optimizer_updates=updates_per_epoch * epochs,
        transformer_floor_updates_per_epoch=floor_updates,
        transformer_floor_total_updates=floor_updates * epochs,
        total_row_presentations=train_rows * epochs,
    )


def verify_configured_schedule(
    config: Mapping[str, Any], schedule: OptimizerSchedule
) -> None:
    expected = {
        "expected_train_rows_per_fold": schedule.train_rows,
        "expected_updates_per_epoch": schedule.optimizer_updates_per_epoch,
        "expected_total_updates": schedule.total_optimizer_updates,
    }
    training = _nested(config, "training")
    mismatches = {
        key: {"configured": training.get(key), "actual": value}
        for key, value in expected.items()
        if training.get(key) != value
    }
    if mismatches:
        raise SafetyGateError(f"configured optimizer schedule mismatch: {mismatches}")


def _language_path(layer: int, projection: str) -> str:
    family = "self_attn" if projection in LANGUAGE_PROJECTIONS[:4] else "mlp"
    return f"model.layers.{layer}.{family}.{projection}"


def _vision_path(block: int, projection: str) -> str:
    family = "attn" if projection in VISION_PROJECTIONS[:2] else "mlp"
    return f"visual.blocks.{block}.{family}.{projection}"


def expected_target_shapes() -> dict[str, tuple[int, int]]:
    """Return the immutable ordered target-path -> (in, out) ledger."""

    shapes: dict[str, tuple[int, int]] = {}
    for layer in range(LANGUAGE_LAYERS):
        for projection in LANGUAGE_PROJECTIONS:
            shapes[_language_path(layer, projection)] = LANGUAGE_SHAPES[projection]
    for block in ADAPTED_VISION_BLOCKS:
        for projection in VISION_PROJECTIONS:
            shapes[_vision_path(block, projection)] = VISION_SHAPES[projection]
    shapes.update(MERGER_SHAPES)
    if len(shapes) != EXPECTED_TARGET_MODULES:
        raise AssertionError("internal Model39 target ledger is not exact")
    return shapes


def _target_family(name: str) -> str:
    if name.startswith("model.layers."):
        return "language"
    if name.startswith("visual.blocks."):
        return "vision"
    if name in MERGER_TARGETS:
        return "merger"
    raise SafetyGateError(f"unclassified Model39 target: {name}")


def calculate_lora_parameters(
    shapes: Mapping[str, tuple[int, int]], rank: int
) -> tuple[int, dict[str, int]]:
    """Calculate LoRA A+B parameters from audited module dimensions."""

    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    expected_names = set(expected_target_shapes())
    if set(shapes) != expected_names:
        missing = sorted(expected_names - set(shapes))
        extra = sorted(set(shapes) - expected_names)
        raise SafetyGateError(
            f"target shape ledger mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    by_family = {"language": 0, "vision": 0, "merger": 0}
    for name, (in_features, out_features) in shapes.items():
        if in_features <= 0 or out_features <= 0:
            raise SafetyGateError(f"invalid target shape for {name}: {(in_features, out_features)}")
        by_family[_target_family(name)] += rank * (in_features + out_features)
    return sum(by_family.values()), by_family


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
        ("architecture", "expected_target_modules"): EXPECTED_TARGET_MODULES,
        ("architecture", "expected_trainable_parameters"): EXPECTED_TRAINABLE_PARAMETERS,
        ("architecture", "expected_trainable_tensors"): EXPECTED_TRAINABLE_TENSORS,
        ("lora", "rank"): LORA_RANK,
        ("lora", "language_projections"): list(LANGUAGE_PROJECTIONS),
        ("lora", "adapted_vision_blocks"): list(ADAPTED_VISION_BLOCKS),
        ("lora", "vision_projections"): list(VISION_PROJECTIONS),
        ("lora", "include_verified_merger"): True,
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
                f"{'.'.join(keys)} must be {expected!r} for Model39, got {actual!r}"
            )

    if _nested(config, "model", "expected_files_sha256") != LOCAL_CHECKPOINT_FILES:
        raise ValueError("model.expected_files_sha256 differs from the pinned ledger")
    if int(_nested(config, "lora", "alpha")) != 16:
        raise ValueError("Model39 lora.alpha must be 16")
    if not 0.0 <= float(_nested(config, "lora", "dropout")) < 1.0:
        raise ValueError("lora.dropout must be in [0, 1)")

    patch = int(_nested(config, "image", "merged_patch_size"))
    minimum = int(_nested(config, "image", "min_pixels"))
    maximum = int(_nested(config, "image", "max_pixels"))
    image_tokens = int(_nested(config, "image", "max_image_tokens"))
    sequence_tokens = int(_nested(config, "image", "max_sequence_tokens"))
    if patch != 28 or minimum <= 0 or maximum < minimum:
        raise ValueError("invalid Qwen2-VL image geometry contract")
    if minimum % patch**2 or maximum != image_tokens * patch**2:
        raise ValueError("pixel bounds must exactly encode the merged-token budget")
    if image_tokens > 320 or not image_tokens < sequence_tokens <= 512:
        raise ValueError("Model39 requires <=320 image and <=512 sequence tokens")

    for key in ("max_allocated_gib", "max_reserved_gib"):
        limit = float(_nested(config, "smoke", key))
        if not 0.0 < limit <= 5.3:
            raise ValueError(f"smoke.{key} must be positive and at most 5.3")
    if int(_nested(config, "smoke", "max_trainable_parameters")) < EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError("max_trainable_parameters is below the audited exact count")
    if int(_nested(config, "smoke", "max_trainable_parameters")) > 11_000_000:
        raise ValueError("Model39 may not exceed the 11 million parameter safety cap")
    if "{fold}" not in str(_nested(config, "output_dir")):
        raise ValueError("output_dir must contain {fold}")
    if int(_nested(config, "training", "epochs")) < 1:
        raise ValueError("training.epochs must be positive")
    if int(_nested(config, "training", "gradient_accumulation")) < 1:
        raise ValueError("training.gradient_accumulation must be positive")
    for key in (
        "expected_train_rows_per_fold",
        "expected_updates_per_epoch",
        "expected_total_updates",
    ):
        value = _nested(config, "training", key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer")


def _linear_shape(module: Any, name: str) -> tuple[int, int]:
    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if (
        module.__class__.__name__ not in {"Linear", "Linear4bit"}
        or not isinstance(in_features, int)
        or not isinstance(out_features, int)
        or in_features <= 0
        or out_features <= 0
    ):
        raise SafetyGateError(f"LoRA target is not a supported linear: {name}")
    return in_features, out_features


def discover_broad_lora_targets(model: Any, rank: int = LORA_RANK) -> BroadTargetAudit:
    """Audit the full architecture, then return exactly Model39's 230 targets."""

    modules = dict(model.named_modules())
    expected_selected = expected_target_shapes()

    expected_language = {
        _language_path(layer, projection): LANGUAGE_SHAPES[projection]
        for layer in range(LANGUAGE_LAYERS)
        for projection in LANGUAGE_PROJECTIONS
    }
    observed_language = {
        name for name in modules if LANGUAGE_PATTERN.fullmatch(name) is not None
    }
    if observed_language != set(expected_language):
        missing = sorted(set(expected_language) - observed_language)
        extra = sorted(observed_language - set(expected_language))
        raise SafetyGateError(
            f"language projection layout mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    expected_all_vision = {
        _vision_path(block, projection): VISION_SHAPES[projection]
        for block in range(VISION_BLOCKS)
        for projection in VISION_PROJECTIONS
    }
    observed_vision = {
        name for name in modules if VISION_PATTERN.fullmatch(name) is not None
    }
    if observed_vision != set(expected_all_vision):
        missing = sorted(set(expected_all_vision) - observed_vision)
        extra = sorted(observed_vision - set(expected_all_vision))
        raise SafetyGateError(
            f"vision projection layout mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    merger_linears = {
        name
        for name, module in modules.items()
        if name.startswith("visual.merger.")
        and module.__class__.__name__ in {"Linear", "Linear4bit"}
    }
    if merger_linears != set(MERGER_TARGETS):
        raise SafetyGateError(f"visual merger layout mismatch: {sorted(merger_linears)}")

    # Check every language/vision projection in the pinned architecture, not
    # merely the selected last-eight visual subset.
    for name, expected in {**expected_language, **expected_all_vision, **MERGER_SHAPES}.items():
        module = modules.get(name)
        if module is None:
            raise SafetyGateError(f"required projection is absent: {name}")
        actual = _linear_shape(module, name)
        if actual != expected:
            raise SafetyGateError(f"projection shape mismatch for {name}: {actual} != {expected}")

    selected_shapes = {name: _linear_shape(modules[name], name) for name in expected_selected}
    trainable, by_family = calculate_lora_parameters(selected_shapes, rank)
    if rank != LORA_RANK or trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise SafetyGateError(
            f"LoRA parameter audit mismatch: rank={rank}, parameters={trainable:,}"
        )
    return BroadTargetAudit(
        rank=rank,
        target_modules=tuple(selected_shapes),
        language_modules=EXPECTED_LANGUAGE_TARGETS,
        vision_modules=EXPECTED_VISION_TARGETS,
        merger_modules=len(MERGER_TARGETS),
        trainable_parameters=trainable,
        trainable_tensors=len(selected_shapes) * 2,
        parameters_by_family=by_family,
        shapes=selected_shapes,
    )


def audit_checkpoint_target_shapes(
    config: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> BroadTargetAudit:
    """Read only safetensors headers and audit the actual checkpoint matrices."""

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
                raw_shape = tuple(int(value) for value in handle.get_slice(weight).get_shape())
                if len(raw_shape) != 2:
                    raise SafetyGateError(f"checkpoint target is not a matrix: {weight}")
                if name in found:
                    raise SafetyGateError(f"checkpoint target occurs in multiple shards: {weight}")
                found[name] = (raw_shape[1], raw_shape[0])
    if found != wanted:
        missing = sorted(set(wanted) - set(found))
        wrong = {
            name: {"actual": found[name], "expected": wanted[name]}
            for name in set(found) & set(wanted)
            if found[name] != wanted[name]
        }
        raise SafetyGateError(
            f"checkpoint target header mismatch: missing={missing[:5]}, wrong={dict(list(wrong.items())[:5])}"
        )
    # Header iteration follows shard placement, which is not architectural
    # order.  Rebuild the mapping from the immutable ledger for stable reports
    # and stable exact-path regex construction in downstream audits.
    canonical_found = {name: found[name] for name in wanted}
    trainable, by_family = calculate_lora_parameters(canonical_found, LORA_RANK)
    if trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise SafetyGateError(f"checkpoint-derived LoRA count drifted: {trainable:,}")
    return BroadTargetAudit(
        rank=LORA_RANK,
        target_modules=tuple(canonical_found),
        language_modules=EXPECTED_LANGUAGE_TARGETS,
        vision_modules=EXPECTED_VISION_TARGETS,
        merger_modules=len(MERGER_TARGETS),
        trainable_parameters=trainable,
        trainable_tensors=len(canonical_found) * 2,
        parameters_by_family=by_family,
        shapes=canonical_found,
    )


def audit_trainable_parameters(
    model: Any, audit: BroadTargetAudit, *, max_trainable_parameters: int
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
            raise SafetyGateError(f"cannot map trainable parameter to one exact target: {name}")
        matched.add(owners[0])
    if matched != expected:
        raise SafetyGateError(f"targets without trainable LoRA weights: {sorted(expected - matched)[:5]}")
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
    """Make the exact held-out split and prove row/ID separation."""

    if fold not in range(5):
        raise ValueError("fold must be one of 0,1,2,3,4")
    training = tuple(record for record in records if record.fold != fold)
    validation = tuple(record for record in records if record.fold == fold)
    if not training or not validation or len(training) + len(validation) != len(records):
        raise SafetyGateError("outer-fold split is incomplete")
    train_ids = {record.identifier for record in training}
    validation_ids = {record.identifier for record in validation}
    if len(train_ids) != len(training) or len(validation_ids) != len(validation):
        raise SafetyGateError("duplicate IDs inside an outer-fold partition")
    if train_ids & validation_ids:
        raise SafetyGateError("training and validation IDs overlap")
    return training, validation


class ExactEpochScheduleMixin:
    """Correct Transformers 4.49's floor-based initial step calculation.

    The upstream inner loop already performs a final partial accumulation
    update, but ``set_initial_training_values`` uses floor division when it
    derives ``max_steps`` from epochs.  Model39 supplies explicit ``max_steps``
    and this mixin makes epoch/resume bookkeeping use the matching ceil count.
    """

    exact_optimizer_schedule: OptimizerSchedule | None = None

    def set_initial_training_values(
        self, args: Any, dataloader: Any, total_train_batch_size: int
    ) -> tuple[int, int, int, int, bool, int, int]:
        # Run upstream validation first, then replace only the values whose
        # floor arithmetic is wrong for a remainder accumulation group.
        super().set_initial_training_values(args, dataloader, total_train_batch_size)
        schedule = self.exact_optimizer_schedule
        if schedule is None:
            raise SafetyGateError("exact optimizer schedule was not attached to Trainer")
        dataloader_rows = len(dataloader)
        examples = int(self.num_examples(dataloader))
        if dataloader_rows != schedule.train_rows or examples != schedule.train_rows:
            raise SafetyGateError(
                "training dataloader changed after schedule audit: "
                f"len={dataloader_rows}, examples={examples}, expected={schedule.train_rows}"
            )
        if int(args.gradient_accumulation_steps) != schedule.gradient_accumulation:
            raise SafetyGateError("Trainer gradient accumulation differs from audited schedule")
        if int(args.max_steps) != schedule.total_optimizer_updates:
            raise SafetyGateError("Trainer max_steps differs from audited ceil schedule")
        if bool(args.dataloader_drop_last):
            raise SafetyGateError("Model39 requires dataloader_drop_last=False")
        return (
            schedule.epochs,
            schedule.optimizer_updates_per_epoch,
            examples,
            schedule.total_row_presentations,
            False,
            dataloader_rows,
            schedule.total_optimizer_updates,
        )


def make_schedule_guard_callback(
    schedule: OptimizerSchedule, callback_base: type[Any] | None = None
) -> Any:
    """Fail before validation if an epoch ended early or crossed its budget."""

    if callback_base is None:
        from transformers import TrainerCallback

        callback_base = TrainerCallback

    class ScheduleGuardCallback(callback_base):
        def __init__(self) -> None:
            self.epoch_boundaries: list[dict[str, int | float]] = []

        def on_epoch_end(
            self, args: Any, state: Any, control: Any, **_: Any
        ) -> Any:
            completed_epoch = len(self.epoch_boundaries) + 1
            expected_step = completed_epoch * schedule.optimizer_updates_per_epoch
            actual_step = int(state.global_step)
            actual_epoch = float(state.epoch or 0.0)
            if completed_epoch > schedule.epochs:
                raise SafetyGateError("Trainer entered an extra epoch")
            if actual_step != expected_step or not math.isclose(
                actual_epoch, float(completed_epoch), abs_tol=1e-12
            ):
                raise SafetyGateError(
                    "epoch did not finish on its audited optimizer boundary: "
                    f"epoch={actual_epoch}, step={actual_step}, "
                    f"expected_epoch={completed_epoch}, expected_step={expected_step}"
                )
            self.epoch_boundaries.append(
                {
                    "epoch": completed_epoch,
                    "optimizer_step": actual_step,
                    "rows_seen_this_epoch": schedule.train_rows,
                }
            )
            return control

        def on_train_end(
            self, args: Any, state: Any, control: Any, **_: Any
        ) -> Any:
            if len(self.epoch_boundaries) != schedule.epochs:
                raise SafetyGateError(
                    f"observed {len(self.epoch_boundaries)} complete epochs, expected {schedule.epochs}"
                )
            if int(state.global_step) != schedule.total_optimizer_updates:
                raise SafetyGateError(
                    f"training stopped at {state.global_step}, expected {schedule.total_optimizer_updates}"
                )
            return control

    return ScheduleGuardCallback()


def load_training_model(config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Load NF4 base, freeze it, and inject the exact broad rank-8 adapter."""

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SafetyGateError("Model39 requires CUDA; CPU training is disabled")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SafetyGateError("Model39 is certified only for one CUDA process")
    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    dtype = torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    checkpoint = _absolute(str(_nested(config, "model", "local_path")))
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(checkpoint),
        trust_remote_code=False,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": device_index},
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
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
    target_audit = discover_broad_lora_targets(model, int(_nested(config, "lora", "rank")))
    adapter = LoraConfig(
        r=target_audit.rank,
        lora_alpha=int(_nested(config, "lora", "alpha")),
        lora_dropout=float(_nested(config, "lora", "dropout")),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=exact_lora_target_pattern(target_audit.target_modules),
    )
    model = get_peft_model(model, adapter)
    injected = audit_injected_lora_targets(model, target_audit.target_modules)
    trainable = audit_trainable_parameters(
        model,
        target_audit,
        max_trainable_parameters=int(_nested(config, "smoke", "max_trainable_parameters")),
    )
    quantized = audit_nf4_modules(model, int(_nested(config, "smoke", "minimum_4bit_modules")))
    residency = audit_cuda_residency(model)
    return model, {
        "device_index": device_index,
        "frozen_vision_parameters": frozen_vision_parameters,
        "lora_targets": asdict(target_audit),
        "injected_lora": injected,
        "trainable": trainable,
        "quantization": quantized,
        "residency": residency,
    }


def dry_run(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Run an offline CPU/header audit and persist its evidence."""

    enable_offline_mode()
    config, resolved = load_config(config_path)
    provenance = verify_starter_provenance(config)
    checkpoint = verify_local_checkpoint(config)
    targets = audit_checkpoint_target_shapes(config)
    records = load_records(
        _absolute(str(_nested(config, "train_csv"))),
        _absolute(str(_nested(config, "image_dir"))),
        require_target=True,
        require_five_folds=True,
    )
    fold_rows: dict[str, dict[str, int]] = {}
    schedules: dict[str, dict[str, int]] = {}
    for fold in range(5):
        training, validation = split_outer_fold(records, fold)
        fold_rows[str(fold)] = {"train": len(training), "validation": len(validation)}
        schedule = deterministic_optimizer_schedule(
            len(training),
            int(_nested(config, "training", "gradient_accumulation")),
            int(_nested(config, "training", "epochs")),
        )
        verify_configured_schedule(config, schedule)
        schedules[str(fold)] = asdict(schedule)
    report = {
        "status": "cpu_header_audit_passed_gpu_smoke_not_run",
        "resolved_config": str(resolved),
        "config_sha256": _sha256(resolved),
        "provenance": provenance,
        "checkpoint": checkpoint,
        "targets": asdict(targets),
        "targeted_base_parameters": sum(
            in_features * out_features for in_features, out_features in targets.shapes.values()
        ),
        "fold_rows": fold_rows,
        "optimizer_schedules": schedules,
        "leakage_contract": "each validation ID is absent from its fold's training IDs",
    }
    _json_write(reject_model22_destination(output_path), report)
    return report


def train(config_path: str | Path, fold: int, *, smoke_only: bool = False) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    enable_offline_mode()
    config, resolved_config = load_config(config_path)
    provenance = verify_starter_provenance(config)
    checkpoint = verify_local_checkpoint(config)
    header_audit = audit_checkpoint_target_shapes(config)
    seed = int(_nested(config, "seed"))
    set_seed(seed)
    csv_path = _absolute(str(_nested(config, "train_csv")))
    image_dir = _absolute(str(_nested(config, "image_dir")))
    records = load_records(csv_path, image_dir, require_target=True, require_five_folds=True)
    training_records, validation_records = split_outer_fold(records, fold)
    schedule = deterministic_optimizer_schedule(
        len(training_records),
        int(_nested(config, "training", "gradient_accumulation")),
        int(_nested(config, "training", "epochs")),
    )
    verify_configured_schedule(config, schedule)
    output_dir = require_new_output_dir(str(_nested(config, "output_dir")).format(fold=fold))

    processor = load_processor(config)
    model, model_audit = load_training_model(config)
    collator = QwenOcrCollator(
        processor,
        str(_nested(config, "prompt")),
        int(_nested(config, "image", "max_image_tokens")),
        int(_nested(config, "image", "max_sequence_tokens")),
    )
    smoke_record = select_smoke_record(
        training_records,
        min_pixels=int(_nested(config, "image", "min_pixels")),
        max_pixels=int(_nested(config, "image", "max_pixels")),
        merged_patch_size=int(_nested(config, "image", "merged_patch_size")),
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
        "checkpoint_target_audit": asdict(header_audit),
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
            "model_number": 39,
            "resolved_config": str(resolved_config),
            "train_csv": str(csv_path),
            "train_csv_sha256": _sha256(csv_path),
        },
    )

    import torch
    from transformers import Trainer, TrainingArguments

    class ExactEpochTrainer(ExactEpochScheduleMixin, Trainer):
        pass

    training = _nested(config, "training")
    callback = _make_validation_callback(
        processor,
        collator,
        validation_records,
        output_dir,
        int(_nested(config, "generation", "max_new_tokens")),
        repetition_penalty=float(_nested(config, "generation", "repetition_penalty")),
        no_repeat_ngram_size=int(_nested(config, "generation", "no_repeat_ngram_size")),
    )
    schedule_callback = make_schedule_guard_callback(schedule)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
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
    set_seed(seed)
    trainer = ExactEpochTrainer(
        model=model,
        args=arguments,
        train_dataset=RecordDataset(training_records),
        data_collator=collator,
        callbacks=[schedule_callback, callback],
    )
    trainer.exact_optimizer_schedule = schedule
    result = trainer.train()
    if int(result.global_step) != schedule.total_optimizer_updates:
        raise SafetyGateError(
            f"Trainer returned {result.global_step} updates, expected {schedule.total_optimizer_updates}"
        )
    if callback.best_epoch is None:
        raise SafetyGateError("training ended without full-fold generated validation")
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)
    complete = {
        "fold": fold,
        "best_epoch": callback.best_epoch,
        "best_multi_score": callback.best_score,
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


def _load_inference_model(config: Mapping[str, Any], adapter_path: Path) -> tuple[Any, BroadTargetAudit]:
    import torch
    from peft import PeftModel
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SafetyGateError("Model39 inference requires CUDA")
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
    targets = discover_broad_lora_targets(base)
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
    model, target_audit = _load_inference_model(config, adapter)
    collator = QwenOcrCollator(
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
        repetition_penalty=float(_nested(config, "generation", "repetition_penalty")),
        no_repeat_ngram_size=int(_nested(config, "generation", "no_repeat_ngram_size")),
    )
    write_predictions(output, records, predictions)
    report: dict[str, Any] = {
        "rows": len(records),
        "fold": fold,
        "adapter": str(adapter),
        "local_checkpoint": checkpoint,
        "output": str(output),
        "target_audit": asdict(target_audit),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": int(_nested(config, "generation", "max_new_tokens")),
            "repetition_penalty": float(_nested(config, "generation", "repetition_penalty")),
            "no_repeat_ngram_size": int(_nested(config, "generation", "no_repeat_ngram_size")),
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

    parser = argparse.ArgumentParser(description="Model39 broad Qwen2-VL QLoRA training")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.fold, smoke_only=args.smoke_only), indent=2, ensure_ascii=False))


def infer_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model39 broad Qwen2-VL greedy inference")
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

    parser = argparse.ArgumentParser(description="Model39 offline target/config audit")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/model39/dry_run.json")
    args = parser.parse_args()
    print(json.dumps(dry_run(args.config, args.output), indent=2, ensure_ascii=False))
