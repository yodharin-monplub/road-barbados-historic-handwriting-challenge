"""Model51: exact tail-logit training for Model43's high-resolution QLoRA.

Transformers 4.49.0's Qwen2-VL causal-LM head projects every sequence
position to the 151,936-word vocabulary before its shifted cross entropy
ignores the prompt positions.  Model51 keeps Model43's image, adapter,
optimizer, checkpointing, fold, and validation policies unchanged, while a
transient pre-hook projects only a rigorously bounded trailing window during
training loss computation.  The hook is absent during validation/inference.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import statistics
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

from . import qwen2_vl_starter_highres as highres
from .qwen2_vl_challenger import (
    SafetyGateError,
    _model_device,
    _to_device,
    enforce_memory_limits,
)
from .schema import SchemaError


MODEL_NUMBER = 51
PARENT_MODEL = 43
MODEL_LABEL = "Model51"
MODEL43_SOURCE_SHA256 = "4da6449f5a4241deaa803305b53013480012717d0ff1b54100da2d7eb89aebda"
MAX_SUPERVISED_TOKENS = 36
PROJECTED_HIDDEN_TOKENS = MAX_SUPERVISED_TOKENS + 1
VOCAB_SIZE = 151_936
HIDDEN_SIZE = 1_536


@dataclass(frozen=True)
class TailLabelReceipt:
    sequence_tokens: int
    supervised_tokens: int
    first_supervised_position: int
    maximum_supervised_tokens: int
    projected_hidden_tokens: int


@dataclass(frozen=True)
class HeadProjectionReceipt:
    original_hidden_tokens: int
    projected_hidden_tokens: int
    head_calls: int


def _require_exact(
    mapping: Mapping[str, Any], expected: Mapping[str, Any], scope: str
) -> None:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{scope} must be a mapping")
    extra = sorted(set(mapping) - set(expected))
    if extra:
        raise ValueError(f"{scope} contains unsupported keys for {MODEL_LABEL}: {extra}")
    for key, value in expected.items():
        actual = mapping.get(key)
        if actual != value:
            raise ValueError(
                f"{scope}.{key} must be {value!r} for {MODEL_LABEL}, got {actual!r}"
            )


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail closed unless the one controlled Model51 experiment is requested."""

    if not isinstance(config, Mapping):
        raise ValueError("configuration must be a mapping")
    highres.validate_config(config)
    _require_exact(
        config.get("experiment", {}),
        {
            "model_number": MODEL_NUMBER,
            "profile": "starter_rank16_language_highres_tail_logits",
            "parent_model": PARENT_MODEL,
        },
        "experiment",
    )
    _require_exact(
        config.get("efficiency", {}),
        {
            "strategy": "fixed_trailing_lm_head_projection",
            "maximum_supervised_tokens": MAX_SUPERVISED_TOKENS,
            "projected_hidden_tokens": PROJECTED_HIDDEN_TOKENS,
            "require_batch_size_one": True,
            "require_contiguous_supervised_suffix": True,
            "native_logits_to_keep": False,
            "gradient_checkpointing_unchanged": True,
            "full_determinism_unchanged": True,
            "model43_source_sha256": MODEL43_SOURCE_SHA256,
        },
        "efficiency",
    )
    if config.get("output_dir") != "outputs/model51/fold{fold}":
        raise ValueError("output_dir must be the isolated Model51 fold path")
    if highres._nested(config, "smoke", "report_dir") != "outputs/model51/smoke":
        raise ValueError("smoke report_dir must be isolated under outputs/model51")
    source = Path(highres.__file__).resolve()
    if highres._sha256(source) != MODEL43_SOURCE_SHA256:
        raise ValueError("Model43 source changed from the parent pinned by Model51")


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {resolved}")
    validate_config(config)
    return config, resolved


def audit_installed_qwen_forward() -> dict[str, Any]:
    """Pin the absence of a native logits-window argument in Transformers 4.49."""

    from transformers import Qwen2VLForConditionalGeneration

    signature = inspect.signature(Qwen2VLForConditionalGeneration.forward)
    parameters = tuple(signature.parameters)
    native_names = tuple(
        name for name in ("logits_to_keep", "num_logits_to_keep") if name in parameters
    )
    source = inspect.getsource(Qwen2VLForConditionalGeneration.forward)
    if native_names:
        raise SafetyGateError(
            f"pinned Qwen forward unexpectedly exposes a native logit window: {native_names}"
        )
    required_fragments = (
        "logits = self.lm_head(hidden_states)",
        "logits = logits.float()",
        "shift_logits = logits[..., :-1, :].contiguous()",
        "shift_labels = labels[..., 1:].contiguous()",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise SafetyGateError(f"pinned Qwen loss implementation drifted: {missing}")
    source_file = Path(inspect.getsourcefile(Qwen2VLForConditionalGeneration) or "")
    if not source_file.is_file():
        raise SafetyGateError("cannot resolve the installed Qwen2-VL implementation")
    return {
        "class": Qwen2VLForConditionalGeneration.__name__,
        "forward_parameters": list(parameters),
        "supports_logits_to_keep": False,
        "supports_num_logits_to_keep": False,
        "installed_source": str(source_file.resolve()),
        "installed_source_sha256": highres._sha256(source_file),
        "verified_stock_loss": "full lm_head, float32 upcast, shifted cross entropy",
    }


def _encoded_template_ids(
    tokenizer: Any,
    prompt: str,
    label: str | None,
) -> list[int]:
    messages = highres.build_messages(prompt, label)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=label is None,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def audit_supervised_suffix(
    config: Mapping[str, Any], records: Sequence[Any]
) -> dict[str, Any]:
    """Prove every target occupies a bounded suffix under the pinned template."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(highres._absolute(str(highres._nested(config, "model", "local_path")))),
        trust_remote_code=False,
        local_files_only=True,
    )
    prompt = str(highres._nested(config, "prompt"))
    prompt_ids = _encoded_template_ids(tokenizer, prompt, None)
    if not prompt_ids:
        raise SafetyGateError("generation template is empty")
    counts: list[int] = []
    for record in records:
        if record.label is None:
            raise SafetyGateError("suffix audit requires every record to have a label")
        full_ids = _encoded_template_ids(tokenizer, prompt, record.label)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise SafetyGateError(
                f"assistant template is not a prompt-prefix for {record.identifier}"
            )
        supervised = len(full_ids) - len(prompt_ids)
        if supervised <= 0 or supervised > MAX_SUPERVISED_TOKENS:
            raise SafetyGateError(
                f"supervised suffix for {record.identifier} is {supervised}, "
                f"outside 1..{MAX_SUPERVISED_TOKENS}"
            )
        counts.append(supervised)
    if not counts:
        raise SafetyGateError("suffix audit selection is empty")
    ordered = sorted(counts)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "rows": len(counts),
        "all_templates_have_exact_prompt_prefix": True,
        "supervised_tokens": {
            "minimum": min(counts),
            "median": statistics.median(counts),
            "mean": statistics.mean(counts),
            "p95": p95,
            "maximum": max(counts),
        },
        "configured_maximum_supervised_tokens": MAX_SUPERVISED_TOKENS,
        "projected_hidden_tokens": PROJECTED_HIDDEN_TOKENS,
        "causal_shift_reason": "N supervised suffix labels require N+1 trailing logits positions under stock shifted CE",
    }


def prepare_trailing_labels(
    labels: Any,
    *,
    maximum_supervised_tokens: int = MAX_SUPERVISED_TOKENS,
) -> tuple[Any, TailLabelReceipt]:
    """Validate a batch-one contiguous suffix and return its fixed label window.

    This function is intentionally called by the CPU collator.  The hot GPU
    loss path performs shape-only checks and therefore introduces no device
    synchronization per micro-batch.
    """

    import torch

    if maximum_supervised_tokens != MAX_SUPERVISED_TOKENS:
        raise SafetyGateError("Model51's supervised-token bound is immutable")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise SafetyGateError("labels must be a rank-two tensor")
    if int(labels.shape[0]) != 1:
        raise SafetyGateError("Model51 requires batch size one")
    sequence_tokens = int(labels.shape[1])
    projected = maximum_supervised_tokens + 1
    if sequence_tokens <= projected:
        raise SafetyGateError("training sequence is too short for the fixed tail window")
    positions = torch.nonzero(labels[0].ne(-100), as_tuple=False).flatten()
    supervised = int(positions.numel())
    if supervised <= 0:
        raise SafetyGateError("assistant-only label suffix is empty")
    if supervised > maximum_supervised_tokens:
        raise SafetyGateError(
            f"supervised suffix has {supervised} tokens; cap is {maximum_supervised_tokens}"
        )
    first = int(positions[0].item())
    expected = torch.arange(first, sequence_tokens, device=labels.device)
    if not torch.equal(positions, expected):
        raise SafetyGateError("supervised labels must be one contiguous terminal suffix")
    tail = labels[:, -projected:]
    if bool(tail[:, 0].ne(-100).any().item()):
        raise SafetyGateError("tail window lacks the ignored predecessor required by shift")
    return tail, TailLabelReceipt(
        sequence_tokens=sequence_tokens,
        supervised_tokens=supervised,
        first_supervised_position=first,
        maximum_supervised_tokens=maximum_supervised_tokens,
        projected_hidden_tokens=projected,
    )


class EfficientHighResolutionQwenCollator(highres.HighResolutionQwenCollator):
    """Model43 collator plus strict CPU-side suffix validation/cropping."""

    def encode_training(self, record: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        batch, diagnostics = super().encode_training(record)
        labels, receipt = prepare_trailing_labels(batch["labels"])
        batch["labels"] = labels
        diagnostics = dict(diagnostics)
        diagnostics.update(
            {
                "original_label_sequence_tokens": receipt.sequence_tokens,
                "projected_label_tokens": int(labels.shape[-1]),
                "maximum_supervised_tokens": receipt.maximum_supervised_tokens,
            }
        )
        return batch, diagnostics


def _collator(
    config: Mapping[str, Any], processor: Any
) -> EfficientHighResolutionQwenCollator:
    return EfficientHighResolutionQwenCollator(
        processor,
        str(highres._nested(config, "prompt")),
        int(highres._nested(config, "image", "min_effective_image_tokens")),
        int(highres._nested(config, "image", "max_image_tokens")),
        int(highres._nested(config, "image", "max_sequence_tokens")),
    )


@contextmanager
def trailing_head_projection(
    model: Any,
    *,
    projected_hidden_tokens: int = PROJECTED_HIDDEN_TOKENS,
) -> Iterator[dict[str, int]]:
    """Temporarily slice the single LM-head input without changing model state."""

    if projected_hidden_tokens != PROJECTED_HIDDEN_TOKENS:
        raise SafetyGateError("Model51's projected hidden window is immutable")
    head = model.get_output_embeddings()
    if head is None or not hasattr(head, "register_forward_pre_hook"):
        raise SafetyGateError("model does not expose a hookable output embedding head")
    receipt = {
        "original_hidden_tokens": -1,
        "projected_hidden_tokens": projected_hidden_tokens,
        "head_calls": 0,
    }

    def project(_: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(args) != 1:
            raise SafetyGateError("Qwen LM head received an unexpected argument layout")
        hidden = args[0]
        if not hasattr(hidden, "ndim") or hidden.ndim != 3:
            raise SafetyGateError("Qwen LM head input must be rank three")
        tokens = int(hidden.shape[-2])
        if tokens <= projected_hidden_tokens:
            raise SafetyGateError("Qwen hidden sequence is too short for tail projection")
        receipt["head_calls"] += 1
        receipt["original_hidden_tokens"] = tokens
        return (hidden[:, -projected_hidden_tokens:, :],)

    handle = head.register_forward_pre_hook(project)
    try:
        yield receipt
    finally:
        handle.remove()


def tail_projected_forward(
    model: Any,
    inputs: Mapping[str, Any],
) -> tuple[Any, HeadProjectionReceipt]:
    """Run one stock Qwen forward with only the transient tail-head hook."""

    if "labels" not in inputs:
        raise SafetyGateError("tail-projected training forward requires labels")
    labels = inputs["labels"]
    if not hasattr(labels, "shape") or tuple(labels.shape)[:1] != (1,):
        raise SafetyGateError("tail-projected forward requires batch-one labels")
    if int(labels.shape[-1]) != PROJECTED_HIDDEN_TOKENS:
        raise SafetyGateError(
            f"collator must supply exactly {PROJECTED_HIDDEN_TOKENS} trailing labels"
        )
    with trailing_head_projection(model) as mutable:
        outputs = model(**dict(inputs))
    if mutable["head_calls"] != 1:
        raise SafetyGateError(f"Qwen LM head call count was {mutable['head_calls']}, expected 1")
    logits = outputs["logits"] if isinstance(outputs, Mapping) else outputs.logits
    if int(logits.shape[-2]) != PROJECTED_HIDDEN_TOKENS:
        raise SafetyGateError("tail-projected Qwen logits have the wrong sequence length")
    loss = outputs["loss"] if isinstance(outputs, Mapping) else outputs.loss
    if loss is None:
        raise SafetyGateError("tail-projected Qwen forward returned no loss")
    receipt = HeadProjectionReceipt(**mutable)
    return outputs, receipt


class TailProjectionTrainerMixin:
    """Trainer loss override that leaves evaluation and generation untouched."""

    tail_projection_calls: int = 0

    def compute_loss(
        self,
        model: Any,
        inputs: Mapping[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        del num_items_in_batch
        if getattr(self, "label_smoother", None) is not None:
            raise SafetyGateError("Model51 does not permit a second label-smoothing loss")
        if getattr(self, "compute_loss_func", None) is not None:
            raise SafetyGateError("Model51 does not permit a second custom loss")
        outputs, _ = tail_projected_forward(model, inputs)
        self.tail_projection_calls += 1
        loss = outputs["loss"] if isinstance(outputs, Mapping) else outputs.loss
        return (loss, outputs) if return_outputs else loss


def estimate_logit_materialization(sequence_tokens: int) -> dict[str, Any]:
    """Report tensor byte counts only; allocator peaks/runtime require CUDA smoke."""

    if sequence_tokens <= PROJECTED_HIDDEN_TOKENS:
        raise ValueError("sequence_tokens must exceed the projected tail")

    def bytes_for(positions: int, element_bytes: int) -> int:
        return positions * VOCAB_SIZE * element_bytes

    stock_shift = sequence_tokens - 1
    efficient_shift = PROJECTED_HIDDEN_TOKENS - 1
    return {
        "vocabulary_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "stock_projected_positions": sequence_tokens,
        "efficient_projected_positions": PROJECTED_HIDDEN_TOKENS,
        "position_reduction_fraction": 1 - PROJECTED_HIDDEN_TOKENS / sequence_tokens,
        "stock_shifted_loss_positions": stock_shift,
        "efficient_shifted_loss_positions": efficient_shift,
        "stock_logits_bytes": {
            "float16": bytes_for(sequence_tokens, 2),
            "float32": bytes_for(sequence_tokens, 4),
            "float32_contiguous_shift": bytes_for(stock_shift, 4),
        },
        "efficient_logits_bytes": {
            "float16": bytes_for(PROJECTED_HIDDEN_TOKENS, 2),
            "float32": bytes_for(PROJECTED_HIDDEN_TOKENS, 4),
            "float32_contiguous_shift": bytes_for(efficient_shift, 4),
        },
        "scope_note": "static tensor sizes, not a CUDA allocator or runtime claim",
    }


def audit_processor_sample(
    config: Mapping[str, Any], records: Sequence[Any]
) -> dict[str, Any]:
    processor = highres.load_processor(config)
    record = highres.select_smoke_record(records, config, processor.tokenizer)
    _, diagnostics = _collator(config, processor).encode_training(record)
    if int(diagnostics["projected_label_tokens"]) != PROJECTED_HIDDEN_TOKENS:
        raise SafetyGateError("real processor sample did not use the fixed tail labels")
    return {
        "record_id": record.identifier,
        "processor_class": processor.__class__.__name__,
        "image_processor_class": processor.image_processor.__class__.__name__,
        "canvas_width": int(diagnostics["canvas_width"]),
        "canvas_height": int(diagnostics["canvas_height"]),
        "planned_image_tokens": int(diagnostics["planned_image_tokens"]),
        "processor_image_tokens": int(diagnostics["image_tokens"]),
        "sequence_tokens": int(diagnostics["sequence_tokens"]),
        "prompt_tokens": int(diagnostics["prompt_tokens"]),
        "supervised_tokens": int(diagnostics["supervised_tokens"]),
        "projected_label_tokens": int(diagnostics["projected_label_tokens"]),
    }


def run_memory_smoke(
    model: Any,
    collator: EfficientHighResolutionQwenCollator,
    record: Any,
    *,
    max_allocated_gib: float,
    max_reserved_gib: float,
) -> dict[str, Any]:
    """Exercise the actual tail loss through forward/backward under hard limits."""

    import torch

    device = _model_device(model)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        cpu_batch, tokens = collator.encode_training(record)
        batch = _to_device(cpu_batch, device)
        result, projection = tail_projected_forward(model, batch)
        loss = result.loss
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise SafetyGateError("Model51 smoke returned a missing/non-finite loss")
        loss.backward()
        torch.cuda.synchronize(device)
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
        loss_value = float(loss.detach().cpu().item())
    except torch.OutOfMemoryError as error:
        raise SafetyGateError("Model51 representative smoke ran out of VRAM") from error
    finally:
        model.zero_grad(set_to_none=True)
    enforce_memory_limits(
        allocated,
        reserved,
        max_allocated_gib=max_allocated_gib,
        max_reserved_gib=max_reserved_gib,
    )
    return {
        "record_id": record.identifier,
        **tokens,
        "head_projection": asdict(projection),
        "loss": loss_value,
        "peak_allocated_bytes": allocated,
        "peak_allocated_gib": allocated / highres.GIB,
        "peak_reserved_bytes": reserved,
        "peak_reserved_gib": reserved / highres.GIB,
        "allocated_limit_gib": max_allocated_gib,
        "reserved_limit_gib": max_reserved_gib,
    }


def dry_run(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    highres.enable_offline_mode()
    config, resolved = load_config(config_path)
    runtime = highres.audit_runtime_versions()
    qwen_forward = audit_installed_qwen_forward()
    provenance = highres.verify_starter_provenance(config)
    checkpoint = highres.verify_local_checkpoint(config)
    targets = highres.audit_checkpoint_target_shapes(config)
    context = highres.audit_model_context(config)
    records = highres._training_records(config)
    suffix = audit_supervised_suffix(config, records)
    sequence = highres.audit_sequence_budget(config, records)
    geometry = highres.audit_geometry(str(highres._nested(config, "image_dir")), config)
    processor_sample = audit_processor_sample(config, records)
    if processor_sample["supervised_tokens"] > MAX_SUPERVISED_TOKENS:
        raise SafetyGateError("real processor exceeds the audited suffix cap")
    folds: dict[str, Any] = {}
    for fold in range(5):
        training, validation = highres.split_outer_fold(records, fold)
        schedule = highres.deterministic_optimizer_schedule(
            len(training),
            int(highres._nested(config, "training", "gradient_accumulation")),
            int(highres._nested(config, "training", "epochs")),
        )
        highres.verify_configured_schedule(config, schedule)
        folds[str(fold)] = {
            "training_rows": len(training),
            "validation_rows": len(validation),
            "schedule": asdict(schedule),
        }
    report = {
        "status": "cpu_audits_passed_gpu_smoke_not_run",
        "model_number": MODEL_NUMBER,
        "parent_model": PARENT_MODEL,
        "resolved_config": str(resolved),
        "config_sha256": highres._sha256(resolved),
        "model43_source_sha256": highres._sha256(Path(highres.__file__).resolve()),
        "controlled_change": "training loss projects only a proven 37-token hidden suffix to the vocabulary",
        "unchanged": {
            "visual_interval": [896, 1024],
            "lora_rank": 16,
            "lora_targets": 196,
            "gradient_checkpointing": True,
            "full_determinism": True,
            "optimizer": "paged_adamw_8bit",
            "validation_generation": "stock full head",
        },
        "runtime": runtime,
        "qwen_forward": qwen_forward,
        "provenance": provenance,
        "checkpoint": checkpoint,
        "targets": asdict(targets),
        "model_context": context,
        "sequence": asdict(sequence),
        "supervised_suffix": suffix,
        "processor_sample": processor_sample,
        "geometry": geometry,
        "logit_materialization_at_worst_sequence": estimate_logit_materialization(
            sequence.observed_max_training_tokens
        ),
        "folds": folds,
    }
    highres._json_write(highres.reject_model22_destination(output_path), report)
    return report


def train(config_path: str | Path, fold: int, *, smoke_only: bool = False) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    highres.enable_offline_mode()
    config, resolved = load_config(config_path)
    runtime = highres.audit_runtime_versions()
    qwen_forward = audit_installed_qwen_forward()
    provenance = highres.verify_starter_provenance(config)
    checkpoint = highres.verify_local_checkpoint(config)
    header_targets = highres.audit_checkpoint_target_shapes(config)
    context = highres.audit_model_context(config)
    records = highres._training_records(config)
    suffix = audit_supervised_suffix(config, records)
    sequence = highres.audit_sequence_budget(config, records)
    training_records, validation_records = highres.split_outer_fold(records, fold)
    schedule = highres.deterministic_optimizer_schedule(
        len(training_records),
        int(highres._nested(config, "training", "gradient_accumulation")),
        int(highres._nested(config, "training", "epochs")),
    )
    highres.verify_configured_schedule(config, schedule)
    output_dir = highres.require_new_output_dir(
        str(highres._nested(config, "output_dir")).format(fold=fold)
    )
    seed = int(highres._nested(config, "seed"))
    highres.set_seed(seed)
    processor = highres.load_processor(config)
    model, model_audit = highres.load_training_model(config)
    collator = _collator(config, processor)
    smoke_record = highres.select_smoke_record(
        training_records, config, processor.tokenizer
    )
    smoke = run_memory_smoke(
        model,
        collator,
        smoke_record,
        max_allocated_gib=float(highres._nested(config, "smoke", "max_allocated_gib")),
        max_reserved_gib=float(highres._nested(config, "smoke", "max_reserved_gib")),
    )
    audit = {
        "model_number": MODEL_NUMBER,
        "parent_model": PARENT_MODEL,
        "fold": fold,
        "rows": {
            "all": len(records),
            "training": len(training_records),
            "validation": len(validation_records),
        },
        "runtime": runtime,
        "qwen_forward": qwen_forward,
        "provenance": provenance,
        "checkpoint": checkpoint,
        "header_targets": asdict(header_targets),
        "model_context": context,
        "sequence": asdict(sequence),
        "supervised_suffix": suffix,
        "schedule": asdict(schedule),
        "model": model_audit,
        "smoke": smoke,
    }
    if smoke_only:
        directory = highres.reject_model22_destination(
            str(highres._nested(config, "smoke", "report_dir"))
        )
        highres._json_write(directory / f"fold{fold}.json", audit)
        return audit
    output_dir.mkdir(parents=True, exist_ok=True)
    highres._json_write(
        output_dir / "run_manifest.json",
        {
            **highres._runtime_manifest(config),
            **audit,
            "resolved_config": str(resolved),
            "config_sha256": highres._sha256(resolved),
            "model43_source_sha256": highres._sha256(Path(highres.__file__).resolve()),
            "train_csv_sha256": highres._sha256(
                highres._absolute(str(highres._nested(config, "train_csv")))
            ),
        },
    )

    import torch
    from transformers import Trainer, TrainingArguments

    class ExactTrainer(
        highres.ExactEpochScheduleMixin, TailProjectionTrainerMixin, Trainer
    ):
        pass

    generation = highres._nested(config, "generation")
    validation_callback = highres._make_validation_callback(
        processor,
        collator,
        validation_records,
        output_dir,
        int(highres._nested(generation, "max_new_tokens")),
        repetition_penalty=float(highres._nested(generation, "repetition_penalty")),
        no_repeat_ngram_size=int(highres._nested(generation, "no_repeat_ngram_size")),
    )
    schedule_callback = highres.make_schedule_callback(schedule)
    training = highres._nested(config, "training")
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
        weight_decay=float(highres._nested(training, "weight_decay")),
        warmup_ratio=float(highres._nested(training, "warmup_ratio")),
        lr_scheduler_type=str(highres._nested(training, "scheduler")),
        max_grad_norm=float(highres._nested(training, "max_grad_norm")),
        logging_steps=int(highres._nested(training, "logging_steps")),
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
    highres.set_seed(seed)
    trainer = ExactTrainer(
        model=model,
        args=arguments,
        train_dataset=highres.RecordDataset(training_records),
        data_collator=collator,
        callbacks=[schedule_callback, validation_callback],
    )
    trainer.exact_optimizer_schedule = schedule
    result = trainer.train()
    if int(result.global_step) != schedule.total_optimizer_updates:
        raise SafetyGateError("Trainer stopped off the exact schedule")
    if validation_callback.best_epoch is None:
        raise SafetyGateError("no complete held-fold validation occurred")
    if trainer.tail_projection_calls != schedule.total_row_presentations:
        raise SafetyGateError(
            f"tail loss ran {trainer.tail_projection_calls} times, expected "
            f"{schedule.total_row_presentations}"
        )
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)
    complete = {
        "model_number": MODEL_NUMBER,
        "parent_model": PARENT_MODEL,
        "fold": fold,
        "best_epoch": validation_callback.best_epoch,
        "best_multi_score": validation_callback.best_score,
        "schedule": asdict(schedule),
        "observed_epoch_boundaries": schedule_callback.boundaries,
        "observed_optimizer_updates": int(result.global_step),
        "tail_projection_calls": trainer.tail_projection_calls,
        "training_metrics": result.metrics,
        "best_adapter": str(output_dir / "best_adapter"),
        "final_adapter": str(final_dir),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / highres.GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / highres.GIB,
    }
    highres._json_write(output_dir / "complete.json", complete)
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
    """Use the unmodified full-head Model43 inference path for saved adapters."""

    config, _ = load_config(config_path)
    report = highres.infer(
        config_path,
        input_csv,
        adapter_path,
        output_csv,
        fold=fold,
        sample_path=sample_path,
        submission_path=submission_path,
    )
    report["model_number"] = MODEL_NUMBER
    report["parent_model"] = PARENT_MODEL
    report["training_only_tail_projection"] = True
    highres._json_write(Path(report["output"]).with_suffix(".metrics.json"), report)
    return report


def train_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model51 exact tail-logit high-resolution QLoRA")
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            train(args.config, args.fold, smoke_only=args.smoke_only),
            indent=2,
            ensure_ascii=False,
        )
    )


def infer_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model51 full-head inference")
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
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

    parser = argparse.ArgumentParser(description="Model51 offline tail-logit audit")
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=highres.PROJECT_ROOT / "outputs/model51/dry_run.json",
    )
    args = parser.parse_args()
    print(json.dumps(dry_run(args.config, args.output), indent=2, ensure_ascii=False))
