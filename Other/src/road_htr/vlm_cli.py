"""Training and inference entry points shared by legal open-weight VLM models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .preprocess import load_rgb
from .schema import ImageIndex, read_table
from .submission import clean_prediction, write_submission
from .vlm_data import DEFAULT_PROMPT, apply_variant, build_vlm_dataset, read_id_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _absolute(path: str | Path, relative_to: Path = PROJECT_ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else relative_to / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return {
        str(file.relative_to(path)): _sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _verify_local_checkpoint(config: dict[str, Any], checkpoint: Path) -> None:
    """Fail closed when a configured local model snapshot drifts."""

    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    expected = config.get("expected_files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("local_path requires a nonempty expected_files_sha256 ledger")
    expected_names = {str(name) for name in expected}
    actual_names = {path.name for path in checkpoint.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError(
            "local checkpoint file set mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    for name in sorted(expected_names):
        expected_hash = str(expected[name]).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"invalid expected SHA-256 for {name!r}")
        actual_hash = _sha256(checkpoint / name)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"local checkpoint SHA-256 mismatch for {name}: "
                f"{actual_hash} != {expected_hash}"
            )

    manifest_path = checkpoint / "upstream_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("sha")) != str(config.get("revision")):
            raise RuntimeError("local checkpoint upstream revision mismatch")
        upstream_license = str(manifest.get("cardData", {}).get("license", "")).lower()
        configured_license = str(config.get("license", "")).lower()
        if upstream_license != configured_license:
            raise RuntimeError(
                f"local checkpoint license mismatch: {upstream_license!r} != "
                f"{configured_license!r}"
            )


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return config, path


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def runtime_manifest(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    import transformers

    try:
        import trl

        trl_version = trl.__version__
    except ImportError:
        trl_version = None
    return {
        "config": config,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _quantization_config(config: dict[str, Any]):
    if not config.get("load_in_4bit", False):
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _dtype(name: str):
    import torch

    values = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in values:
        raise ValueError(f"unsupported dtype {name!r}")
    return values[name]


def train(config_path: str | Path, fold_override: int | None = None) -> None:
    import torch
    import transformers
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    config, resolved_config = load_config(config_path)
    seed = int(config.get("seed", 20260903))
    set_seed(seed)
    fold = fold_override if fold_override is not None else config.get("validation_fold")
    if fold is not None:
        fold = int(fold)

    model_name = config["model_name"]
    local_path = config.get("local_path")
    model_source = str(_absolute(local_path)) if local_path else model_name
    local_only = bool(config.get("local_files_only", bool(local_path)))
    if local_path:
        _verify_local_checkpoint(config, Path(model_source))
    revision = config.get("revision")
    dtype = _dtype(config.get("dtype", "bfloat16"))
    quantization = _quantization_config(config)
    model_kwargs: dict[str, Any] = {
        "revision": None if local_path else revision,
        "dtype": dtype,
        "trust_remote_code": False,
        "local_files_only": local_only,
    }
    if quantization is not None:
        model_kwargs["quantization_config"] = quantization
        model_kwargs["device_map"] = "auto"
    attention = config.get("attn_implementation")
    if attention:
        model_kwargs["attn_implementation"] = attention

    model_class_name = config.get("model_class", "AutoModelForImageTextToText")
    processor_class_name = config.get("processor_class", "AutoProcessor")
    try:
        model_class = getattr(transformers, model_class_name)
        processor_class = getattr(transformers, processor_class_name)
    except AttributeError as exc:
        raise RuntimeError("configured training model/processor class is unavailable") from exc
    processor_kwargs: dict[str, Any] = {
        "revision": None if local_path else revision,
        "trust_remote_code": False,
        "local_files_only": local_only,
    }
    if bool(config.get("fix_mistral_regex", False)):
        processor_kwargs["fix_mistral_regex"] = True
    processor = processor_class.from_pretrained(
        model_source, **processor_kwargs
    )
    model = model_class.from_pretrained(model_source, **model_kwargs)
    model.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    exclude_path = config.get("exclude_ids")
    excluded = read_id_file(_absolute(exclude_path)) if exclude_path else set()
    csv_path = _absolute(config["train_csv"])
    image_dir = _absolute(config["image_dir"])
    prompt = config.get("prompt", DEFAULT_PROMPT)
    image_variant = config.get("image_variant", "full")
    train_dataset = build_vlm_dataset(
        csv_path,
        image_dir,
        prompt=prompt,
        fold=fold,
        fold_mode="train" if fold is not None else "all",
        exclude_ids=excluded,
        image_variant=image_variant,
        augmentation=config.get("augmentation"),
        synthetic_manifest=(
            _absolute(str(config["synthetic_manifest"]).format(fold=fold))
            if config.get("synthetic_manifest") else None
        ),
    )
    validation_dataset = None
    if fold is not None:
        validation_dataset = build_vlm_dataset(
            csv_path,
            image_dir,
            prompt=prompt,
            fold=fold,
            fold_mode="validation",
            exclude_ids=excluded,
            image_variant=image_variant,
            augmentation=None,
        )

    output_template = str(config.get("output_dir", "outputs/vlm/fold{fold}"))
    output_dir = _absolute(output_template.format(fold=fold if fold is not None else "all"))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = runtime_manifest(config)
    manifest["resolved_config"] = str(resolved_config)
    manifest["fold"] = fold
    manifest["excluded_ids"] = sorted(excluded)
    manifest["train_rows"] = len(train_dataset)
    if config.get("synthetic_manifest"):
        synthetic_path = _absolute(str(config["synthetic_manifest"]).format(fold=fold))
        manifest["synthetic_manifest_sha256"] = _sha256(synthetic_path)
        manifest["synthetic_rows"] = len(json.loads(synthetic_path.read_text())["rows"])
        manifest["real_train_rows"] = len(train_dataset) - manifest["synthetic_rows"]
    manifest["validation_rows"] = len(validation_dataset) if validation_dataset else 0
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    lora = config.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora.get("rank", 32)),
        lora_alpha=int(lora.get("alpha", 64)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    training = config.get("training", {})
    has_validation = validation_dataset is not None
    epochs = float(training.get("epochs", 3))
    batch_size = int(training.get("batch_size", 1))
    gradient_accumulation = int(training.get("gradient_accumulation", 16))
    configured_max_steps = int(training.get("max_steps", -1))
    if configured_max_steps == 0 or configured_max_steps < -1:
        raise ValueError("training.max_steps must be -1 or a positive integer")
    epoch_optimizer_steps = math.ceil(
        len(train_dataset) / max(batch_size * gradient_accumulation, 1)
    ) * math.ceil(epochs)
    optimizer_steps = (
        configured_max_steps if configured_max_steps > 0 else epoch_optimizer_steps
    )
    warmup_steps = int(
        round(optimizer_steps * float(training.get("warmup_ratio", 0.05)))
    )
    manifest["derived_training"] = {
        "optimizer_steps": optimizer_steps,
        "warmup_steps": warmup_steps,
        "warmup_ratio": float(training.get("warmup_ratio", 0.05)),
        "configured_max_steps": configured_max_steps,
        "epoch_optimizer_steps": epoch_optimizer_steps,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    args = SFTConfig(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
        max_length=None,
        completion_only_loss=True,
        num_train_epochs=epochs,
        max_steps=configured_max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(training.get("eval_batch_size", 1)),
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=float(training.get("learning_rate", 2e-5)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        warmup_steps=warmup_steps,
        lr_scheduler_type=training.get("scheduler", "cosine"),
        logging_steps=int(training.get("logging_steps", 10)),
        eval_strategy="steps" if has_validation else "no",
        eval_steps=int(training.get("eval_steps", 100)),
        save_strategy="steps",
        save_steps=int(training.get("save_steps", 100)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
        greater_is_better=False if has_validation else None,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=int(training.get("workers", 2)),
        dataloader_pin_memory=True,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir / "best_adapter"))
    processor.save_pretrained(str(output_dir / "best_adapter"))


_PREFIX = re.compile(r"^(?:transcription|text|answer)\s*:\s*", re.IGNORECASE)


def extract_text(value: str) -> str:
    value = value.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
        if value.lower().startswith("text\n"):
            value = value[5:]
    value = _PREFIX.sub("", value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return clean_prediction(value)


def _load_inference_model(config: dict[str, Any], adapter_path: Path | None):
    import torch
    import transformers

    model_name = config["model_name"]
    local_path = config.get("local_path")
    model_source = str(_absolute(local_path)) if local_path else model_name
    local_only = bool(config.get("local_files_only", bool(local_path)))
    if local_path:
        _verify_local_checkpoint(config, Path(model_source))
    revision = config.get("revision")
    dtype = _dtype(config.get("dtype", "bfloat16"))
    quantization = _quantization_config(config)
    kwargs: dict[str, Any] = {
        "revision": None if local_path else revision,
        "dtype": dtype,
        "trust_remote_code": False,
        "local_files_only": local_only,
    }
    if quantization is not None:
        kwargs.update(quantization_config=quantization, device_map="auto")
    else:
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
    attention = config.get("attn_implementation")
    if attention:
        kwargs["attn_implementation"] = attention
    model_class_name = config.get("model_class", "AutoModelForImageTextToText")
    try:
        model_class = getattr(transformers, model_class_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"installed transformers has no requested model class {model_class_name!r}"
        ) from exc
    model = model_class.from_pretrained(model_source, **kwargs)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    processor_source = (
        str(adapter_path)
        if adapter_path and (adapter_path / "processor_config.json").exists()
        else model_source
    )
    processor_class_name = config.get("processor_class", "AutoProcessor")
    try:
        processor_class = getattr(transformers, processor_class_name)
    except AttributeError as exc:
        raise RuntimeError(
            "installed transformers has no requested processor class "
            f"{processor_class_name!r}"
        ) from exc
    processor_kwargs: dict[str, Any] = {
        "revision": None if processor_source != model_name else revision,
        "trust_remote_code": False,
        "local_files_only": local_only,
    }
    if bool(config.get("fix_mistral_regex", False)):
        processor_kwargs["fix_mistral_regex"] = True
    processor = processor_class.from_pretrained(
        processor_source,
        **processor_kwargs,
    )
    model.eval()
    return model, processor


def _generate_one(model, processor, image, prompt: str, generation: dict[str, Any]):
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=[rendered], images=[image], return_tensors="pt")
    device = next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    kwargs = {
        "max_new_tokens": int(generation.get("max_new_tokens", 192)),
        "num_beams": int(generation.get("num_beams", 1)),
        "num_return_sequences": int(generation.get("num_return_sequences", 1)),
        "do_sample": False,
        "repetition_penalty": float(generation.get("repetition_penalty", 1.05)),
        "length_penalty": float(generation.get("length_penalty", 1.0)),
        "early_stopping": bool(generation.get("num_beams", 1) > 1),
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    with torch.inference_mode():
        result = model.generate(**inputs, **kwargs)
    input_ids = inputs.get("input_ids")
    candidates: list[str] = []
    for sequence in result.sequences:
        generated = sequence
        if input_ids is not None and len(sequence) >= input_ids.shape[1]:
            prefix = sequence[: input_ids.shape[1]]
            if torch.equal(prefix, input_ids[0]):
                generated = sequence[input_ids.shape[1] :]
        candidates.append(extract_text(processor.decode(generated, skip_special_tokens=True)))
    scores = getattr(result, "sequences_scores", None)
    score_values = scores.detach().float().cpu().tolist() if scores is not None else [None] * len(candidates)
    return candidates, score_values


def infer(
    config_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
    adapter_path: str | Path | None = None,
    fold: int | None = None,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
    image_variant: str | None = None,
    limit: int | None = None,
) -> None:
    started = time.monotonic()
    config, resolved_config = load_config(config_path)
    set_seed(int(config.get("seed", 20260903)))
    adapter = _absolute(adapter_path) if adapter_path else None
    model, processor = _load_inference_model(config, adapter)
    table = read_table(_absolute(input_csv), require_target=False)
    if fold is not None and "fold" not in table.fieldnames:
        raise ValueError("--fold was supplied but input has no fold column")
    image_index = ImageIndex(_absolute(config["image_dir"]))
    variant = image_variant or config.get("image_variant", "full")
    prompt = config.get("prompt", DEFAULT_PROMPT)
    generation = config.get("generation", {})

    output_path = _absolute(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path = output_path.with_suffix(".jsonl")
    predictions: dict[str, str] = {}
    with output_path.open("w", encoding="utf-8", newline="") as csv_handle, detail_path.open(
        "w", encoding="utf-8"
    ) as detail_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=[table.id_column, "prediction"])
        writer.writeheader()
        selected_rows = [
            row for row in table.rows if fold is None or int(row["fold"]) == fold
        ]
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            if submission_path:
                raise ValueError("limit cannot be used when creating a submission")
            selected_rows = selected_rows[:limit]
        for position, row in enumerate(selected_rows, start=1):
            identifier = row[table.id_column].strip()
            image = apply_variant(load_rgb(image_index.resolve(identifier)), variant)
            candidates, scores = _generate_one(model, processor, image, prompt, generation)
            nonempty = [candidate for candidate in candidates if candidate]
            chosen = nonempty[0] if nonempty else ""
            predictions[identifier] = chosen
            writer.writerow({table.id_column: identifier, "prediction": chosen})
            detail_handle.write(
                json.dumps(
                    {"id": identifier, "candidates": candidates, "scores": scores},
                    ensure_ascii=False,
                )
                + "\n"
            )
            if position % 25 == 0 or position == len(selected_rows):
                print(f"generated {position}/{len(selected_rows)}", flush=True)

    if submission_path:
        if not sample_path:
            raise ValueError("--sample is required with --submission")
        write_submission(_absolute(sample_path), _absolute(submission_path), predictions)

    input_path = _absolute(input_csv)
    manifest = {
        "schema_version": 1,
        "resolved_config": str(resolved_config),
        "config_sha256": _sha256(resolved_config),
        "input_csv": str(input_path),
        "input_sha256": _sha256(input_path),
        "output_csv": str(output_path),
        "output_sha256": _sha256(output_path),
        "detail_jsonl": str(detail_path),
        "detail_sha256": _sha256(detail_path),
        "model_name": config["model_name"],
        "revision": config.get("revision"),
        "local_path": str(_absolute(config["local_path"])) if config.get("local_path") else None,
        "local_files_only": bool(config.get("local_files_only", False)),
        "base_files_sha256": config.get("expected_files_sha256", {}),
        "adapter": str(adapter) if adapter else None,
        "adapter_files_sha256": _tree_sha256(adapter) if adapter else {},
        "fold": fold,
        "rows": len(predictions),
        "empty_prediction_ids": sorted(
            identifier for identifier, value in predictions.items() if not value
        ),
        "image_variant": variant,
        "prompt": prompt,
        "generation": generation,
        "seed": int(config.get("seed", 20260903)),
        "seconds": time.monotonic() - started,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_main(default_config: str | Path | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    train(args.config, args.fold)


def infer_main(default_config: str | Path | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--image-variant",
        choices=["full", "center", "enhanced", "center_enhanced"],
    )
    args = parser.parse_args()
    infer(
        args.config,
        args.input,
        args.output,
        adapter_path=args.adapter,
        fold=args.fold,
        sample_path=args.sample,
        submission_path=args.submission,
        image_variant=args.image_variant,
        limit=args.limit,
    )
