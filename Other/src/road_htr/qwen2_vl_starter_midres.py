"""Model48: memory-bounded, organizer-aligned Qwen2-VL fallback.

Model48 deliberately keeps Model43's rank-16 language-only adapter and exact
training protocol, but reduces the aligned visual interval from 896..1024 to
512..640 merged tokens.  The Model43 implementation is imported read-only:
this module owns configuration validation, manifests, and command entrypoints
so an active Model43 run retains byte-for-byte provenance.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import qwen2_vl_starter_highres as highres
from .qwen2_vl_challenger import SafetyGateError
from .schema import SchemaError


MODEL_NUMBER = 48
MODEL_LABEL = "Model48"
MIN_IMAGE_TOKENS = 512
TARGET_IMAGE_TOKENS = 640
MAX_IMAGE_TOKENS = 640
TRAINING_SEQUENCE_CAP = 768
GENERATION_CONTEXT_CAP = 816
ORGANIZER_PROMPT = (
    "Transcribe EXACTLY as seen. Preserve line breaks, punctuation, and spacing. "
    "Do not add explanations. Return verbatim transcription. No paraphrasing. No "
    "normalization."
)

# Re-export the audited immutable architecture and pure utilities.  Keeping the
# aliases explicit makes tests prove that Model48 has not forked their logic.
LANGUAGE_LAYERS = highres.LANGUAGE_LAYERS
LANGUAGE_PROJECTIONS = highres.LANGUAGE_PROJECTIONS
LANGUAGE_SHAPES = highres.LANGUAGE_SHAPES
LORA_RANK = highres.LORA_RANK
EXPECTED_TARGET_MODULES = highres.EXPECTED_TARGET_MODULES
EXPECTED_TRAINABLE_PARAMETERS = highres.EXPECTED_TRAINABLE_PARAMETERS
EXPECTED_TRAINABLE_TENSORS = highres.EXPECTED_TRAINABLE_TENSORS
MERGED_PATCH_SIZE = highres.MERGED_PATCH_SIZE
MODEL_CONTEXT_TOKENS = highres.MODEL_CONTEXT_TOKENS
OptimizerSchedule = highres.OptimizerSchedule
calculate_lora_parameters = highres.calculate_lora_parameters
deterministic_optimizer_schedule = highres.deterministic_optimizer_schedule
expected_target_shapes = highres.expected_target_shapes
plan_letterbox = highres.plan_letterbox
qwen_grid_tokens = highres.qwen_grid_tokens
load_letterboxed_rgb = highres.load_letterboxed_rgb
split_outer_fold = highres.split_outer_fold


def _require_exact(mapping: Mapping[str, Any], expected: Mapping[str, Any], scope: str) -> None:
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
    """Fail closed unless this is the one frozen Model48 experiment."""

    if not isinstance(config, Mapping):
        raise ValueError("configuration must be a mapping")
    _require_exact(
        config.get("experiment", {}),
        {
            "model_number": MODEL_NUMBER,
            "profile": "starter_rank16_language_midres640",
            "parent_model": 43,
        },
        "experiment",
    )
    _require_exact(
        config.get("image", {}),
        {
            "merged_patch_size": MERGED_PATCH_SIZE,
            "min_pixels": MIN_IMAGE_TOKENS * MERGED_PATCH_SIZE**2,
            "max_pixels": MAX_IMAGE_TOKENS * MERGED_PATCH_SIZE**2,
            "min_effective_image_tokens": MIN_IMAGE_TOKENS,
            "target_image_tokens": TARGET_IMAGE_TOKENS,
            "max_image_tokens": MAX_IMAGE_TOKENS,
            "max_sequence_tokens": TRAINING_SEQUENCE_CAP,
            "max_generation_context_tokens": GENERATION_CONTEXT_CAP,
            "resize_policy": "aspect_preserving_letterbox_lanczos",
        },
        "image",
    )
    _require_exact(
        config.get("training", {}),
        {
            "epochs": 3,
            "batch_size": 1,
            "eval_batch_size": 1,
            "gradient_accumulation": 16,
            "expected_train_rows_per_fold": 3276,
            "expected_updates_per_epoch": 205,
            "expected_total_updates": 615,
            "learning_rate": 0.00003,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "scheduler": "cosine",
            "max_grad_norm": 1.0,
            "logging_steps": 10,
            "gradient_checkpointing": True,
        },
        "training",
    )
    _require_exact(
        config.get("generation", {}),
        {
            "max_new_tokens": 48,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 4,
        },
        "generation",
    )
    _require_exact(
        config.get("lora", {}),
        {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "language_projections": list(LANGUAGE_PROJECTIONS),
            "include_merger": False,
            "include_visual_blocks": False,
        },
        "lora",
    )
    _require_exact(
        config.get("smoke", {}),
        {
            "max_allocated_gib": 5.3,
            "max_reserved_gib": 5.3,
            "expected_target_modules": EXPECTED_TARGET_MODULES,
            "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
            "minimum_4bit_modules": 100,
            "max_trainable_parameters": 19_000_000,
            "report_dir": "outputs/model48/smoke",
        },
        "smoke",
    )
    if config.get("prompt") != ORGANIZER_PROMPT:
        raise ValueError("prompt must exactly match the organizer starter prompt")
    if config.get("output_dir") != "outputs/model48/fold{fold}":
        raise ValueError("output_dir must be the isolated Model48 fold path")
    if config.get("seed") != 20260903:
        raise ValueError("seed must match the frozen comparison seed")

    # Let Model43's mature validator verify every common model, checkpoint,
    # quantization, architecture, fold, and schedule invariant.  Only a private
    # copy receives its high-resolution image profile; Model43 stays untouched.
    canonical = copy.deepcopy(dict(config))
    canonical["image"].update(
        {
            "min_pixels": 896 * MERGED_PATCH_SIZE**2,
            "max_pixels": 1024 * MERGED_PATCH_SIZE**2,
            "min_effective_image_tokens": 896,
            "target_image_tokens": 1024,
            "max_image_tokens": 1024,
            "max_sequence_tokens": 1152,
            "max_generation_context_tokens": 1200,
        }
    )
    highres.validate_config(canonical)


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {resolved}")
    validate_config(config)
    return config, resolved


def dry_run(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    highres.enable_offline_mode()
    config, resolved = load_config(config_path)
    runtime = highres.audit_runtime_versions()
    provenance = highres.verify_starter_provenance(config)
    checkpoint = highres.verify_local_checkpoint(config)
    targets = highres.audit_checkpoint_target_shapes(config)
    context = highres.audit_model_context(config)
    records = highres._training_records(config)
    sequence = highres.audit_sequence_budget(config, records)
    geometry = highres.audit_geometry(str(highres._nested(config, "image_dir")), config)
    processor_sample = highres.audit_processor_sample(config, records)
    folds: dict[str, Any] = {}
    for fold in range(5):
        training, validation = split_outer_fold(records, fold)
        schedule = deterministic_optimizer_schedule(
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
        "parent_model": 43,
        "resolved_config": str(resolved),
        "config_sha256": highres._sha256(resolved),
        "controlled_change": "visual interval 896..1024 -> 512..640 tokens only",
        "visual_cost_comparison": {
            "maximum_tokens": MAX_IMAGE_TOKENS,
            "versus_model39_maximum_tokens": 2.0,
            "versus_model43_maximum_tokens": 0.625,
            "visual_quadratic_ratio_versus_model43": 0.390625,
        },
        "organizer_alignment": {
            "rank": 16,
            "alpha": 32,
            "target_families": list(LANGUAGE_PROJECTIONS),
            "target_scope": "exact 196 language paths",
            "merger_adapter": False,
            "visual_adapter": False,
            "prompt": "verbatim organizer starter prompt",
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
    highres._json_write(highres.reject_model22_destination(output_path), report)
    return report


def train(config_path: str | Path, fold: int, *, smoke_only: bool = False) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    highres.enable_offline_mode()
    config, resolved = load_config(config_path)
    runtime = highres.audit_runtime_versions()
    provenance = highres.verify_starter_provenance(config)
    checkpoint = highres.verify_local_checkpoint(config)
    header_targets = highres.audit_checkpoint_target_shapes(config)
    context = highres.audit_model_context(config)
    records = highres._training_records(config)
    sequence = highres.audit_sequence_budget(config, records)
    training_records, validation_records = split_outer_fold(records, fold)
    schedule = deterministic_optimizer_schedule(
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
    collator = highres._collator(config, processor)
    smoke_record = highres.select_smoke_record(
        training_records, config, processor.tokenizer
    )
    smoke = highres.run_memory_smoke(
        model,
        collator,
        smoke_record,
        max_allocated_gib=float(highres._nested(config, "smoke", "max_allocated_gib")),
        max_reserved_gib=float(highres._nested(config, "smoke", "max_reserved_gib")),
    )
    audit = {
        "model_number": MODEL_NUMBER,
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
        "sequence": asdict(sequence),
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
            "model_number": MODEL_NUMBER,
            "parent_model": 43,
            "resolved_config": str(resolved),
            "config_sha256": highres._sha256(resolved),
            "train_csv_sha256": highres._sha256(
                highres._absolute(str(highres._nested(config, "train_csv")))
            ),
        },
    )

    import torch
    from transformers import Trainer, TrainingArguments

    class ExactTrainer(highres.ExactEpochScheduleMixin, Trainer):
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
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)
    complete = {
        "model_number": MODEL_NUMBER,
        "fold": fold,
        "best_epoch": validation_callback.best_epoch,
        "best_multi_score": validation_callback.best_score,
        "schedule": asdict(schedule),
        "observed_epoch_boundaries": schedule_callback.boundaries,
        "observed_optimizer_updates": int(result.global_step),
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
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    highres.enable_offline_mode()
    config, _ = load_config(config_path)
    runtime = highres.audit_runtime_versions()
    highres.verify_starter_provenance(config)
    checkpoint = highres.verify_local_checkpoint(config)
    highres.audit_model_context(config)
    records = highres.load_records(
        highres._absolute(input_csv),
        highres._absolute(str(highres._nested(config, "image_dir"))),
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
    adapter = highres._absolute(adapter_path)
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    processor = highres.load_processor(config)
    model, targets = highres._load_inference_model(config, adapter)
    collator = highres._collator(config, processor)
    generation = highres._nested(config, "generation")
    predictions = highres.greedy_predict(
        model,
        processor,
        collator,
        records,
        max_new_tokens=int(highres._nested(generation, "max_new_tokens")),
        repetition_penalty=float(highres._nested(generation, "repetition_penalty")),
        no_repeat_ngram_size=int(highres._nested(generation, "no_repeat_ngram_size")),
    )
    output = highres.reject_model22_destination(output_csv)
    highres.write_predictions(output, records, predictions)
    report: dict[str, Any] = {
        "model_number": MODEL_NUMBER,
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
        report["metrics"] = highres.exact_metrics(records, predictions)
    highres._json_write(output.with_suffix(".metrics.json"), report)
    if submission_path is not None:
        if sample_path is None:
            raise ValueError("--sample is required with --submission")
        by_id = {
            record.identifier: prediction
            for record, prediction in zip(records, predictions, strict=True)
        }
        highres.write_submission(
            highres._absolute(sample_path),
            highres.reject_model22_destination(submission_path),
            by_id,
            cleaner=highres.clean_prediction_preserving_internal_spaces,
        )
    return report


def train_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model48 mid-resolution starter QLoRA")
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

    parser = argparse.ArgumentParser(description="Model48 mid-resolution inference")
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

    parser = argparse.ArgumentParser(description="Model48 offline audit")
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=highres.PROJECT_ROOT / "outputs/model48/dry_run.json",
    )
    args = parser.parse_args()
    print(json.dumps(dry_run(args.config, args.output), indent=2, ensure_ascii=False))
