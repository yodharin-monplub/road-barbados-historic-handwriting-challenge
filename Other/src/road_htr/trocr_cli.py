"""Rectangular, aspect-preserving TrOCR fine-tuning and inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageEnhance, ImageFilter

from .preprocess import center_band, letterbox, load_rgb
from .schema import ImageIndex, read_table
from .submission import clean_prediction, write_submission
from .vlm_data import read_id_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _select_rows(config: dict[str, Any], mode: str, fold: int | None):
    table = read_table(_absolute(config["train_csv"]), require_target=True)
    if mode != "all" and (fold is None or "fold" not in table.fieldnames):
        raise ValueError("cross-validation mode needs a fold column and fold number")
    excluded = read_id_file(_absolute(config["exclude_ids"])) if config.get("exclude_ids") else set()
    rows: list[tuple[str, str]] = []
    assert table.target_column is not None
    for row in table.rows:
        identifier = row[table.id_column].strip()
        if identifier in excluded:
            continue
        if mode == "train" and int(row["fold"]) == fold:
            continue
        if mode == "validation" and int(row["fold"]) != fold:
            continue
        rows.append((identifier, row[table.target_column]))
    return rows, excluded


def _augment(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.7:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.35))
    if rng.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.82, 1.18))
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 0.8)))
    if rng.random() < 0.5:
        angle = rng.uniform(-1.2, 1.2)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    return image


class TrocrDataset:
    def __init__(
        self,
        rows: list[tuple[str, str]],
        image_index: ImageIndex,
        processor,
        image_height: int,
        image_width: int,
        max_label_length: int,
        center_fraction: float,
        augment: bool,
        seed: int,
    ):
        self.rows = rows
        self.image_index = image_index
        self.processor = processor
        self.image_height = image_height
        self.image_width = image_width
        self.max_label_length = max_label_length
        self.center_fraction = center_fraction
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        identifier, target = self.rows[index]
        image = load_rgb(self.image_index.resolve(identifier))
        if self.center_fraction < 1:
            image = center_band(image, self.center_fraction)
        if self.augment:
            # Deterministic per sample and epoch, independent of worker order.
            image = _augment(image, random.Random(self.seed + self.epoch * len(self) + index))
        image = letterbox(image, (self.image_height, self.image_width))
        pixel_values = self.processor.image_processor(
            images=image, do_resize=False, return_tensors="pt"
        ).pixel_values[0]
        label_ids = self.processor.tokenizer(
            target,
            max_length=self.max_label_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        pad_id = self.processor.tokenizer.pad_token_id
        label_ids[label_ids == pad_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": label_ids,
        }


def _collate_trocr(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors while keeping the interpolation switch a scalar bool."""

    import torch

    return {
        "pixel_values": torch.stack([feature["pixel_values"] for feature in features]),
        "labels": torch.stack([feature["labels"] for feature in features]),
        "interpolate_pos_encoding": True,
    }


def train(config_path: str | Path, fold_override: int | None = None) -> None:
    import torch
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    config = load_config(config_path)
    seed = int(config.get("seed", 20260903))
    set_seed(seed)
    fold_value = fold_override if fold_override is not None else config.get("validation_fold")
    fold = int(fold_value) if fold_value is not None else None
    train_rows, excluded = _select_rows(config, "train" if fold is not None else "all", fold)
    validation_rows = _select_rows(config, "validation", fold)[0] if fold is not None else []

    model_name = config["model_name"]
    revision = config.get("revision")
    processor = TrOCRProcessor.from_pretrained(model_name, revision=revision)
    model = VisionEncoderDecoderModel.from_pretrained(model_name, revision=revision)
    # transformers 5 keeps these only in generation_config, but the training forward
    # pass shifts labels using model.config's copies and fails without them (Model154).
    for name in ("decoder_start_token_id", "pad_token_id", "eos_token_id"):
        if getattr(model.config, name, None) is None:
            setattr(model.config, name, getattr(model.generation_config, name))
    model.config.use_cache = False
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.decoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    generation = config.get("generation", {})
    model.generation_config.max_new_tokens = int(generation.get("max_new_tokens", 192))
    model.generation_config.num_beams = int(generation.get("num_beams", 5))
    model.generation_config.length_penalty = float(generation.get("length_penalty", 1.0))
    # Beam-only flag: transformers 5 refuses to SAVE a generation config that sets it
    # with num_beams=1, which lost Model154's first fold at the very last step.
    model.generation_config.early_stopping = model.generation_config.num_beams > 1

    image_index = ImageIndex(_absolute(config["image_dir"]))
    image_height = int(config.get("image_height", 256))
    image_width = int(config.get("image_width", 1024))
    max_label_length = int(config.get("max_label_length", 192))
    center_fraction = float(config.get("center_fraction", 1.0))
    train_dataset = TrocrDataset(
        train_rows,
        image_index,
        processor,
        image_height,
        image_width,
        max_label_length,
        center_fraction,
        True,
        seed,
    )
    validation_dataset = (
        TrocrDataset(
            validation_rows,
            image_index,
            processor,
            image_height,
            image_width,
            max_label_length,
            center_fraction,
            False,
            seed,
        )
        if validation_rows
        else None
    )

    output_template = str(config.get("output_dir", "outputs/model3/fold{fold}"))
    output_dir = _absolute(output_template.format(fold=fold if fold is not None else "all"))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config,
        "fold": fold,
        "excluded_ids": sorted(excluded),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    training = config.get("training", {})
    has_validation = validation_dataset is not None
    # select_best: false trains a fixed number of epochs and keeps the final weights.
    # Picking the checkpoint by validation-fold loss would let the held-out fold
    # choose its own model and inflate the OOF score (Model154).
    select_best = has_validation and bool(training.get("select_best", True))
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
        num_train_epochs=float(training.get("epochs", 8)),
        per_device_train_batch_size=int(training.get("batch_size", 1)),
        per_device_eval_batch_size=int(training.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation", 8)),
        learning_rate=float(training.get("learning_rate", 2e-5)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        # transformers 5 removed warmup_ratio; a float in [0, 1) here is the same ratio.
        warmup_steps=float(training.get("warmup_ratio", 0.05)),
        lr_scheduler_type=training.get("scheduler", "cosine"),
        logging_steps=int(training.get("logging_steps", 10)),
        eval_strategy="steps" if select_best else "no",
        eval_steps=int(training.get("eval_steps", 100)),
        save_strategy="steps",
        save_steps=int(training.get("save_steps", 100)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        load_best_model_at_end=select_best,
        metric_for_best_model="eval_loss" if select_best else None,
        greater_is_better=False if select_best else None,
        predict_with_generate=False,
        bf16=bool(training.get("bf16", True)),
        fp16=bool(training.get("fp16", False)),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=int(training.get("workers", 2)),
        report_to="none",
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if select_best else []
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=_collate_trocr,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir / "best_model"))
    processor.save_pretrained(str(output_dir / "best_model"))


def _batched(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def infer(
    config_path: str | Path,
    input_csv: str | Path,
    model_path: str | Path,
    output_csv: str | Path,
    fold: int | None = None,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> None:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    config = load_config(config_path)
    set_seed(int(config.get("seed", 20260903)))
    trained_path = _absolute(model_path)
    processor = TrOCRProcessor.from_pretrained(trained_path)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = VisionEncoderDecoderModel.from_pretrained(trained_path, dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    table = read_table(_absolute(input_csv), require_target=False)
    if fold is not None and "fold" not in table.fieldnames:
        raise ValueError("--fold needs an input fold column")
    rows = [row for row in table.rows if fold is None or int(row["fold"]) == fold]
    image_index = ImageIndex(_absolute(config["image_dir"]))
    height = int(config.get("image_height", 256))
    width = int(config.get("image_width", 1024))
    center_fraction = float(config.get("center_fraction", 1.0))
    generation = config.get("generation", {})
    batch_size = int(generation.get("batch_size", 2))
    num_beams = int(generation.get("num_beams", 5))
    num_return = int(generation.get("num_return_sequences", num_beams))

    output = _absolute(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail = output.with_suffix(".jsonl")
    predictions: dict[str, str] = {}
    with output.open("w", encoding="utf-8", newline="") as csv_handle, detail.open(
        "w", encoding="utf-8"
    ) as detail_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=[table.id_column, "prediction"])
        writer.writeheader()
        for chunk_index, chunk in enumerate(_batched(rows, batch_size)):
            images: list[Image.Image] = []
            identifiers: list[str] = []
            for row in chunk:
                identifier = row[table.id_column].strip()
                image = load_rgb(image_index.resolve(identifier))
                if center_fraction < 1:
                    image = center_band(image, center_fraction)
                images.append(letterbox(image, (height, width)))
                identifiers.append(identifier)
            pixel_values = processor.image_processor(
                images=images, do_resize=False, return_tensors="pt"
            ).pixel_values.to(device=device, dtype=dtype)
            with torch.inference_mode():
                result = model.generate(
                    pixel_values,
                    interpolate_pos_encoding=True,
                    max_new_tokens=int(generation.get("max_new_tokens", 192)),
                    num_beams=num_beams,
                    num_return_sequences=num_return,
                    length_penalty=float(generation.get("length_penalty", 1.0)),
                    early_stopping=num_beams > 1,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            decoded = [
                clean_prediction(value)
                for value in processor.batch_decode(result.sequences, skip_special_tokens=True)
            ]
            scores = getattr(result, "sequences_scores", None)
            score_values = scores.detach().float().cpu().tolist() if scores is not None else [None] * len(decoded)
            for offset, identifier in enumerate(identifiers):
                start = offset * num_return
                candidates = decoded[start : start + num_return]
                candidate_scores = score_values[start : start + num_return]
                chosen = next((value for value in candidates if value), "")
                predictions[identifier] = chosen
                writer.writerow({table.id_column: identifier, "prediction": chosen})
                detail_handle.write(
                    json.dumps(
                        {"id": identifier, "candidates": candidates, "scores": candidate_scores},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            done = min((chunk_index + 1) * batch_size, len(rows))
            if done % 20 == 0 or done == len(rows):
                print(f"generated {done}/{len(rows)}", flush=True)

    if submission_path:
        if not sample_path:
            raise ValueError("--sample is required with --submission")
        write_submission(_absolute(sample_path), _absolute(submission_path), predictions)


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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    infer(
        args.config,
        args.input,
        args.model,
        args.output,
        fold=args.fold,
        sample_path=args.sample,
        submission_path=args.submission,
    )
