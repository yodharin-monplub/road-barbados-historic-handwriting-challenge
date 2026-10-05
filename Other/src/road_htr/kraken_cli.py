"""Reproducible Kraken 7 / PP-OCRv6 training and inference helpers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import yaml
from PIL import Image

from .preprocess import load_rgb
from .schema import ImageIndex, read_table
from .submission import (
    clean_prediction,
    clean_prediction_preserving_internal_spaces,
    write_submission,
)
from .vlm_data import apply_variant, read_id_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {resolved}")
    return config, resolved


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_rows(
    config: dict[str, Any], fold: int | None, mode: str
) -> tuple[Any, list[dict[str, str]], set[str]]:
    table = read_table(absolute(config["train_csv"]), require_target=True)
    if mode not in {"train", "validation", "all"}:
        raise ValueError("mode must be train, validation, or all")
    if mode != "all" and (fold is None or "fold" not in table.fieldnames):
        raise ValueError("fold selection requires a fold column and fold number")
    excluded = (
        read_id_file(absolute(config["exclude_ids"]))
        if config.get("exclude_ids")
        else set()
    )
    rows: list[dict[str, str]] = []
    for row in table.rows:
        identifier = row[table.id_column].strip()
        if identifier in excluded:
            continue
        if mode == "train" and int(row["fold"]) == fold:
            continue
        if mode == "validation" and int(row["fold"]) != fold:
            continue
        rows.append(row)
    return table, rows, excluded


def _safe_generated_name(identifier: str, suffix: str) -> str:
    """Keep generated line pairs flat and reject path-like competition IDs."""

    if Path(identifier).name != identifier:
        raise ValueError(f"ID must be a basename, got {identifier!r}")
    return f"{Path(identifier).stem}{suffix}"


def _all_data_evaluation_paths(
    config: dict[str, Any], train_paths: list[Path]
) -> list[Path]:
    """Choose a deterministic evaluation subset without removing train rows.

    Kraken only enables augmentation when an explicit evaluation set is
    supplied. Without one it also creates an internal 90/10 split, which is
    undesirable for a fixed-epoch fit on all labeled rows. The overlapping
    subset exists solely to select Kraken's explicit-evaluation code path; its
    metric is in-sample and must never be used for model selection.
    """

    training = config.get("training", {})
    count = training.get("all_data_evaluation_rows")
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(
            "training.all_data_evaluation_rows must be an integer for "
            "fixed-epoch --all-data training"
        )
    if count <= 0:
        raise ValueError("training.all_data_evaluation_rows must be positive")
    if count > len(train_paths):
        raise ValueError(
            "training.all_data_evaluation_rows cannot exceed the number of "
            f"eligible training rows ({len(train_paths)})"
        )

    # Rank by a stable digest of seed and generated basename, then retain the
    # original manifest order for deterministic, readable evaluation files.
    seed = int(config.get("seed", 20260903))
    ranked_indices = sorted(
        range(len(train_paths)),
        key=lambda index: hashlib.sha256(
            f"{seed}\0{train_paths[index].name}".encode("utf-8")
        ).digest(),
    )
    selected = set(ranked_indices[:count])
    return [path for index, path in enumerate(train_paths) if index in selected]


def _positive_training_integer(training: dict[str, Any], key: str) -> int:
    value = training.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"training.{key} must be a positive integer")
    return value


def _preflight_ctc_compatibility(
    config: dict[str, Any], samples: list[tuple[str, Path, str]]
) -> dict[str, Any]:
    """Reject samples Kraken would silently replace during PP-OCR training.

    Kraken scales an RGB bounding-box line proportionally to the configured
    height, adds horizontal padding, and then checks whether the encoded CTC
    target fits the network's subsampled width.  ``codec_resize: new`` creates
    a one-label-per-character codec for this corpus, so this reproduces the
    loader's check without importing the optional Kraken dependency.

    The preflight is enabled by explicitly configuring ``ctc_subsampling``.
    Other recognizers and older model configurations therefore retain their
    existing preparation behavior.
    """

    training = config.get("training", {})
    if "ctc_subsampling" not in training:
        return {"enabled": False, "checked_rows": 0}
    if str(training.get("codec_resize", "")).lower() != "new":
        raise ValueError(
            "training.ctc_subsampling preflight currently requires "
            "training.codec_resize: new"
        )

    height = _positive_training_integer(training, "height")
    padding = training.get("padding", 16)
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("training.padding must be a non-negative integer")
    max_width = _positive_training_integer(training, "max_width")
    subsampling = _positive_training_integer(training, "ctc_subsampling")
    normalization = str(training.get("normalization", "NFD"))
    if normalization not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(
            "training.normalization must be one of NFC, NFD, NFKC, or NFKD"
        )

    failures: list[str] = []
    for identifier, image_path, target in samples:
        normalized = unicodedata.normalize(normalization, target)
        with Image.open(image_path) as image:
            source_width, source_height = image.size
        if source_width <= 0 or source_height <= 0:
            failures.append(
                f"{identifier}: invalid source geometry "
                f"{source_width}x{source_height}"
            )
            continue

        # This mirrors kraken.lib.functional_im_transforms._fixed_resize and
        # ImageInputTransforms(..., pad=(padding, 0)).
        resized_width = int(source_width * height / source_height)
        transformed_width = resized_width + 2 * padding
        reasons: list[str] = []
        if transformed_width > max_width:
            reasons.append(
                f"transformed_width={transformed_width} exceeds max_width={max_width}"
            )

        adjacent_repeats = sum(
            left == right for left, right in zip(normalized, normalized[1:])
        )
        minimum_alignment = len(normalized) + adjacent_repeats
        output_steps = transformed_width // subsampling
        if minimum_alignment > output_steps:
            reasons.append(
                f"minimum_ctc_alignment={minimum_alignment} exceeds "
                f"output_steps={output_steps} (target_labels={len(normalized)}, "
                f"adjacent_repeats={adjacent_repeats})"
            )
        if reasons:
            failures.append(
                f"{identifier}: "
                + "; ".join(reasons)
                + f"; source={source_width}x{source_height}, height={height}, "
                f"padding={padding}, subsampling={subsampling}"
            )

    if failures:
        raise ValueError(
            "PP-OCR CTC preflight rejected sample(s) that Kraken would "
            "silently replace:\n- "
            + "\n- ".join(failures)
            + "\nFix their preprocessing/labels or add their IDs to exclude_ids."
        )
    return {
        "enabled": True,
        "checked_rows": len(samples),
        "height": height,
        "padding": padding,
        "max_width": max_width,
        "subsampling": subsampling,
        "normalization": normalization,
    }


def prepare_training_data(
    config: dict[str, Any], fold: int | None
) -> tuple[Path, Path | None, dict[str, Any]]:
    """Create Kraken path manifests and matching ``.gt.txt`` label files.

    These are generated artifacts under ``data/processed``. The supplied CSV
    and images are never edited.
    """

    table, train_rows, excluded = _selected_rows(
        config, fold, "train" if fold is not None else "all"
    )
    validation_rows = (
        _selected_rows(config, fold, "validation")[1] if fold is not None else []
    )
    assert table.target_column is not None
    image_index = ImageIndex(absolute(config["image_dir"]))
    prepared_dir = absolute(config.get("prepared_dir", "data/processed/model8"))
    variant = str(config.get("image_variant", "full"))
    training = config.get("training", {})
    strip_target_edges = training.get("strip_target_edges", False)
    if not isinstance(strip_target_edges, bool):
        raise ValueError("training.strip_target_edges must be a boolean")
    pair_dir = prepared_dir / f"lines_{variant}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    def materialize(
        rows: list[dict[str, str]],
    ) -> tuple[list[Path], list[tuple[str, Path, str]]]:
        paths: list[Path] = []
        samples: list[tuple[str, Path, str]] = []
        for row in rows:
            identifier = row[table.id_column].strip()
            source = image_index.resolve(identifier)
            image_name = _safe_generated_name(identifier, source.suffix.lower())
            image_path = pair_dir / image_name
            target_path = image_path.with_suffix(".gt.txt")
            if variant == "full":
                if image_path.is_symlink() and image_path.resolve() != source.resolve():
                    raise RuntimeError(f"stale generated symlink: {image_path}")
                if not image_path.exists():
                    image_path.symlink_to(source.resolve())
            else:
                image = apply_variant(load_rgb(source), variant)
                image.save(image_path, quality=95, subsampling=0)
            # Challenge scoring strips references and hypotheses. Preserve all
            # internal spacing but do not train boundary whitespace that can
            # never improve the competition metric.
            target = row[table.target_column]
            if strip_target_edges:
                target = target.strip()
            target_path.write_text(target, encoding="utf-8")
            # Keep the generated image path in the manifest. Resolving the
            # symlink would make Kraken search for ``.gt.txt`` beside the raw
            # image instead of beside this generated pair.
            paths.append(image_path.absolute())
            samples.append((identifier, image_path, target))
        return paths, samples

    train_paths, train_samples = materialize(train_rows)
    if fold is None:
        # Keep every eligible row in train_paths. This deliberately overlaps
        # evaluation for the fixed-epoch final fit; see helper documentation.
        validation_paths = _all_data_evaluation_paths(config, train_paths)
        validation_samples: list[tuple[str, Path, str]] = []
    else:
        validation_paths, validation_samples = materialize(validation_rows)
    # Train and validation are disjoint for folds. The all-data evaluation
    # paths overlap train and therefore need no second check.
    ctc_preflight = _preflight_ctc_compatibility(
        config, train_samples + validation_samples
    )
    split_name = f"fold{fold}" if fold is not None else "all"
    train_manifest = prepared_dir / f"train_{split_name}_{variant}.txt"
    validation_manifest = (
        prepared_dir
        / (
            f"validation_{split_name}_overlap_{variant}.txt"
            if fold is None
            else f"validation_{split_name}_{variant}.txt"
        )
        if validation_paths
        else None
    )
    train_manifest.write_text(
        "".join(f"{path}\n" for path in train_paths), encoding="utf-8"
    )
    if validation_manifest is not None:
        validation_manifest.write_text(
            "".join(f"{path}\n" for path in validation_paths), encoding="utf-8"
        )
    details = {
        "fold": fold,
        "image_variant": variant,
        "train_rows": len(train_paths),
        "validation_rows": len(validation_paths),
        "validation_overlaps_training": fold is None,
        "validation_metric_for_selection": fold is not None,
        "excluded_ids": sorted(excluded),
        "strip_target_edges": strip_target_edges,
        "ctc_preflight": ctc_preflight,
        "train_manifest": str(train_manifest),
        "validation_manifest": str(validation_manifest) if validation_manifest else None,
    }
    return train_manifest, validation_manifest, details


def _ketos_binary() -> str:
    installed = shutil.which("ketos")
    if installed:
        return installed
    sibling = Path(sys.executable).with_name("ketos")
    if sibling.is_file():
        return str(sibling)
    raise FileNotFoundError(
        "ketos is unavailable; activate the Kraken environment described in model8/readme.txt"
    )


def train(
    config_path: str | Path,
    fold_override: int | None = None,
    all_data: bool = False,
    dry_run: bool = False,
) -> None:
    config, resolved_config = load_config(config_path)
    if all_data and fold_override is not None:
        raise ValueError("--fold and --all-data are mutually exclusive")
    default_fold = config.get("validation_fold")
    fold = None if all_data else fold_override if fold_override is not None else default_fold
    if fold is not None:
        fold = int(fold)

    pretrained = absolute(config["pretrained_model"])
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    train_manifest, validation_manifest, preparation = prepare_training_data(config, fold)
    output_template = str(config.get("output_dir", "outputs/model8/fold{fold}"))
    output_dir = absolute(output_template.format(fold=fold if fold is not None else "all"))
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = config.get("runtime", {})
    training = config.get("training", {})
    epochs_key = "all_data_epochs" if fold is None else "epochs"
    epochs = int(training.get(epochs_key, 40 if fold is not None else 20))
    cosine_key = "all_data_cosine_epochs" if fold is None else "cosine_epochs"
    cosine_epochs = int(training.get(cosine_key, epochs))
    stop = "fixed" if fold is None else str(training.get("stop", "early"))
    command = [
        _ketos_binary(),
        "--device",
        str(runtime.get("device", "cuda:0")),
        "--precision",
        str(runtime.get("precision", "bf16-mixed")),
        "--workers",
        str(runtime.get("workers", 4)),
        "--threads",
        str(runtime.get("threads", 2)),
        "--seed",
        str(config.get("seed", 20260903)),
        "--deterministic",
        "train",
        "--load",
        str(pretrained),
        "--resize",
        str(training.get("codec_resize", "union")),
        "--batch-size",
        str(training.get("batch_size", 4)),
        "--pad",
        str(training.get("padding", 16)),
        "--height",
        str(training.get("height", 128)),
        "--max-width",
        str(training.get("max_width", 6144)),
        "--output",
        str(output_dir),
        "--freq",
        str(training.get("validation_frequency", 1)),
        "--quit",
        stop,
        "--epochs",
        str(epochs),
        "--min-epochs",
        str(training.get("min_epochs", 5)),
        "--lag",
        str(training.get("patience", 7)),
        "--optimizer",
        str(training.get("optimizer", "AdamW")),
        "--lrate",
        str(training.get("learning_rate", 0.0001)),
        "--weight-decay",
        str(training.get("weight_decay", 0.01)),
        "--gradient-clip-val",
        str(training.get("gradient_clip", 1.0)),
        "--accumulate-grad-batches",
        str(training.get("gradient_accumulation", 4)),
        "--warmup",
        str(training.get("warmup_steps", 200)),
        "--schedule",
        str(training.get("scheduler", "cosine")),
        "--cos-max",
        str(cosine_epochs),
        "--cos-min-lr",
        str(training.get("min_learning_rate", 0.000001)),
        "--normalization",
        str(training.get("normalization", "NFD")),
        "--no-normalize-whitespace",
        "--base-dir",
        "L",
        "--training-data",
        str(train_manifest),
        "--format-type",
        "path",
        "--linetype",
        "bbox",
    ]
    if validation_manifest is not None:
        command.extend(["--evaluation-data", str(validation_manifest)])
    command.append("--augment" if training.get("augment", True) else "--no-augment")

    manifest = {
        "config": config,
        "resolved_config": str(resolved_config),
        "pretrained_sha256": sha256(pretrained),
        "preparation": preparation,
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "started_utc_epoch": time.time(),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _make_batch_segmentation(items: list[tuple[str, Path, Any]]):
    from kraken.containers import BBoxLine, Segmentation
    from PIL import Image

    gap = 2
    width = max(image.width for _, _, image in items)
    height = sum(image.height for _, _, image in items) + gap * (len(items) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    lines = []
    top = 0
    for identifier, image_path, image in items:
        canvas.paste(image, (0, top))
        lines.append(
            BBoxLine(
                id=identifier,
                imagename=str(image_path),
                bbox=(0, top, image.width, top + image.height),
                text_direction="horizontal-lr",
            )
        )
        top += image.height + gap
    segmentation = Segmentation(
        type="bbox",
        imagename="in-memory-batch",
        text_direction="horizontal-lr",
        script_detection=False,
        lines=lines,
    )
    return canvas, segmentation


def _constrained_greedy_decoder(codec: Any, allowed_characters: set[str]):
    """Return a CTC greedy decoder restricted to the supplied label alphabet."""

    from kraken.lib.ctc_decoder import greedy_decoder

    allowed_labels = {0}
    missing: list[str] = []
    for character in sorted(allowed_characters):
        labels = codec.c2l.get(character)
        if labels is None:
            missing.append(character)
        else:
            allowed_labels.update(int(label) for label in labels)
    if missing:
        raise ValueError(f"model codec lacks allowed characters: {missing}")

    def decode(outputs, seq_lens=None):
        import torch

        values = torch.as_tensor(outputs)
        unconstrained = greedy_decoder(values, seq_lens)
        if all(
            int(label) in allowed_labels
            for sequence in unconstrained
            for label, *_ in sequence
        ):
            return unconstrained
        class_dimension = 0 if values.dim() == 2 else 1
        indices = torch.tensor(
            sorted(allowed_labels), dtype=torch.long, device=values.device
        )
        masked = torch.full_like(values, -1)
        masked.index_copy_(
            class_dimension,
            indices,
            values.index_select(class_dimension, indices),
        )
        return greedy_decoder(masked, seq_lens)

    return decode


def _training_character_inventory(config: dict[str, Any]) -> set[str]:
    table = read_table(absolute(config["train_csv"]), require_target=True)
    assert table.target_column is not None
    return set("".join(row[table.target_column] for row in table.rows))


def _prediction_cleaner(
    config: dict[str, Any],
) -> tuple[Callable[[object], str], bool]:
    inference = config.get("inference", {})
    preserve = inference.get("preserve_internal_spaces", False)
    if not isinstance(preserve, bool):
        raise ValueError("inference.preserve_internal_spaces must be a boolean")
    return (
        clean_prediction_preserving_internal_spaces if preserve else clean_prediction
    ), preserve


def _align_kraken_sequence_lengths_with_network(task_model: Any) -> bool:
    """Work around Kraken 7.1 leaving PP-OCR sequence lengths on CPU.

    Kraken's Fabric wrapper moves the image batch to the accelerator, but the
    companion width tensor is constructed inside the line iterator and remains
    on CPU. PP-OCRv6 consumes that tensor while building a CUDA mask. Keep this
    process-local and model-local instead of modifying the installed package.
    """

    network = task_model.net
    if getattr(network, "_road_htr_length_device_patch", False):
        return False
    original = network._rec_predict
    try:
        device = next(network.nn.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise RuntimeError("cannot determine Kraken recognition network device") from exc

    def device_safe_rec_predict(line: Any, lens: Any = None):
        if lens is not None:
            lens = lens.to(device=device)
        return original(line, lens)

    network._rec_predict = device_safe_rec_predict
    network._road_htr_length_device_patch = True
    return True


def infer(
    config_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
    checkpoint: str | Path | None = None,
    fold: int | None = None,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
    image_variant: str | None = None,
    limit: int | None = None,
    constrain_charset: bool | None = None,
    identifiers: set[str] | None = None,
    batch_size_override: int | None = None,
) -> None:
    from kraken.configs import RecognitionInferenceConfig
    from kraken.tasks import RecognitionTaskModel

    config, resolved_config = load_config(config_path)
    model_path = absolute(checkpoint or config["pretrained_model"])
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    table = read_table(absolute(input_csv))
    if fold is not None and "fold" not in table.fieldnames:
        raise ValueError("--fold needs an input CSV with a fold column")
    selected = [
        row for row in table.rows if fold is None or int(row["fold"]) == fold
    ]
    if identifiers:
        available = {row[table.id_column].strip() for row in selected}
        unknown = identifiers - available
        if unknown:
            raise ValueError(f"requested IDs are absent from the selected input: {sorted(unknown)}")
        selected = [
            row for row in selected if row[table.id_column].strip() in identifiers
        ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    variant = image_variant or str(config.get("image_variant", "full"))
    image_index = ImageIndex(absolute(config["image_dir"]))
    inference = config.get("inference", {})
    prediction_cleaner, preserve_internal_spaces = _prediction_cleaner(config)
    batch_size = (
        batch_size_override
        if batch_size_override is not None
        else int(inference.get("batch_size", 1))
    )
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    inference_config = RecognitionInferenceConfig(
        accelerator=str(inference.get("accelerator", "cpu")),
        device=str(inference.get("device", "auto")),
        precision=str(inference.get("precision", "32-true")),
        batch_size=batch_size,
        num_threads=int(inference.get("threads", max(1, (os.cpu_count() or 2) // 2))),
        num_line_workers=int(inference.get("line_workers", 0)),
        padding=int(inference.get("padding", 16)),
        bidi_reordering=True,
        raise_on_error=True,
    )
    model = RecognitionTaskModel.load_model(model_path)
    constrain = (
        bool(inference.get("constrain_charset", True))
        if constrain_charset is None
        else constrain_charset
    )
    if constrain:
        inference_config.decoder = _constrained_greedy_decoder(
            model.net.codec, _training_character_inventory(config)
        )
    # Preparing a PP-OCR model constructs its Fabric runtime and moves the
    # network to the selected device. Do this once, not once per line.
    model.net.prepare_for_inference(inference_config)
    _align_kraken_sequence_lengths_with_network(model)

    output = absolute(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    details = output.with_suffix(".jsonl")
    predictions: dict[str, str] = {}
    progress_every = int(inference.get("progress_every", 25))
    started = time.monotonic()
    with output.open("w", encoding="utf-8", newline="") as csv_handle, details.open(
        "w", encoding="utf-8"
    ) as detail_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=[table.id_column, "prediction"])
        writer.writeheader()
        completed = 0
        for start in range(0, len(selected), batch_size):
            batch_rows = selected[start : start + batch_size]
            items = []
            for row in batch_rows:
                identifier = row[table.id_column].strip()
                image_path = image_index.resolve(identifier)
                image = apply_variant(load_rgb(image_path), variant)
                items.append((identifier, image_path, image))
            canvas, segmentation = _make_batch_segmentation(items)
            records = list(model.net.predict(canvas, segmentation))
            if len(records) != len(items):
                raise RuntimeError(
                    f"expected {len(items)} recognized lines, got {len(records)}"
                )
            by_id = {record.id: record for record in records}
            if set(by_id) != {identifier for identifier, _, _ in items}:
                raise RuntimeError("Kraken returned unexpected line IDs")
            for identifier, _, _ in items:
                record = by_id[identifier]
                prediction = prediction_cleaner(record.prediction)
                confidence_values = [float(value) for value in record.confidences]
                confidence = fmean(confidence_values) if confidence_values else None
                predictions[identifier] = prediction
                writer.writerow({table.id_column: identifier, "prediction": prediction})
                detail_handle.write(
                    json.dumps(
                        {
                            "id": identifier,
                            "candidates": [prediction],
                            "scores": [confidence],
                            "character_confidences": confidence_values,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            completed += len(items)
            if completed % progress_every < len(items) or completed == len(selected):
                csv_handle.flush()
                detail_handle.flush()
                elapsed = time.monotonic() - started
                print(
                    f"recognized {completed}/{len(selected)} lines in {elapsed:.1f}s",
                    flush=True,
                )

    inference_manifest = {
        "config": config,
        "resolved_config": str(resolved_config),
        "config_sha256": sha256(resolved_config),
        "input_csv": str(absolute(input_csv)),
        "input_sha256": sha256(absolute(input_csv)),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "fold": fold,
        "image_variant": variant,
        "constrain_charset": constrain,
        "batch_size": batch_size,
        "preserve_internal_spaces": preserve_internal_spaces,
        "rows": len(selected),
        "seconds": time.monotonic() - started,
        "python": sys.version,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(inference_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if submission_path:
        if limit is not None:
            raise ValueError("cannot create a submission from a limited inference run")
        if sample_path is None:
            raise ValueError("--sample is required with --submission")
        write_submission(
            absolute(sample_path),
            absolute(submission_path),
            predictions,
            cleaner=prediction_cleaner,
        )


def train_main(default_config: str | Path | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--fold", type=int)
    selection.add_argument("--all-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args.config, args.fold, args.all_data, args.dry_run)


def infer_main(default_config: str | Path | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--variant", choices=["full", "center", "enhanced", "center_enhanced"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, help="batch independent lines together")
    parser.add_argument(
        "--id",
        dest="identifiers",
        action="append",
        help="recognize only this ID; repeat for a targeted diagnostic run",
    )
    parser.add_argument(
        "--constrain-charset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="restrict CTC decoding to characters observed in Train.csv",
    )
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    infer(
        args.config,
        args.input,
        args.output,
        checkpoint=args.checkpoint,
        fold=args.fold,
        sample_path=args.sample,
        submission_path=args.submission,
        image_variant=args.variant,
        limit=args.limit,
        constrain_charset=args.constrain_charset,
        identifiers=set(args.identifiers) if args.identifiers else None,
        batch_size_override=args.batch_size,
    )
