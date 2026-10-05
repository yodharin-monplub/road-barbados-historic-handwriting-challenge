#!/usr/bin/env python3
"""Prepare and optionally train the isolated Model30 weighted-manifest run."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from road_htr.kraken_cli import (  # noqa: E402
    _ketos_binary,
    absolute,
    load_config,
    prepare_training_data,
    sha256,
)

from model30.weighted_manifest import (  # noqa: E402
    atomic_json,
    build_weighting_plan,
    canonical_mode,
    load_training_examples,
    policy_from_config,
    stable_sha256,
    write_plan,
)


SCHEMA_VERSION = 1


def _positive_integer(mapping: dict[str, Any], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"training.{key} must be a positive integer")
    return value


def _scaled_epochs(base_epochs: int, unique_rows: int, manifest_rows: int) -> int:
    if min(base_epochs, unique_rows, manifest_rows) <= 0:
        raise ValueError("epoch scaling inputs must be positive")
    return max(1, round(base_epochs * unique_rows / manifest_rows))


def _estimated_optimizer_steps(
    rows: int, epochs: int, batch_size: int, accumulation: int
) -> int:
    batches = math.ceil(rows / batch_size)
    return epochs * math.ceil(batches / accumulation)


def _training_command(
    config: dict[str, Any],
    pretrained: Path,
    output_dir: Path,
    training_manifest: Path,
    validation_manifest: Path | None,
    epochs: int,
    cosine_epochs: int,
    all_data: bool,
) -> list[str]:
    runtime = config.get("runtime", {})
    training = config.get("training", {})
    if not isinstance(runtime, dict) or not isinstance(training, dict):
        raise ValueError("runtime and training configuration must be mappings")
    stop = "fixed" if all_data else str(training.get("stop", "early"))
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
        str(min(_positive_integer(training, "min_epochs", 5), epochs)),
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
        str(training_manifest),
        "--format-type",
        "path",
        "--linetype",
        "bbox",
    ]
    if validation_manifest is not None:
        command.extend(["--evaluation-data", str(validation_manifest)])
    command.append("--augment" if training.get("augment", True) else "--no-augment")
    return command


def _output_dir(config: dict[str, Any], mode: str, fold: int | None) -> Path:
    weighting = config.get("weighting", {})
    if not isinstance(weighting, dict):
        raise ValueError("weighting configuration must be a mapping")
    template = str(
        weighting.get("output_dir", "outputs/model30/{mode}/fold{fold}")
    )
    try:
        rendered = template.format(
            mode=mode, fold=fold if fold is not None else "all"
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"invalid weighting.output_dir template {template!r}") from exc
    output = absolute(rendered).resolve()
    model22_output = (ROOT / "outputs" / "model22").resolve()
    if output == model22_output or model22_output in output.parents:
        raise ValueError("Model30 output must never be under outputs/model22")
    return output


def _ensure_no_training_artifacts(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    protected = sorted(
        path.name
        for pattern in (
            "checkpoint_*.ckpt",
            "best_*.safetensors",
            "training_started.json",
            "training_complete.json",
        )
        for path in output_dir.glob(pattern)
    )
    if protected:
        raise RuntimeError(
            f"refusing to reuse populated training output {output_dir}: {protected[:5]}"
        )


def prepare_run(
    config_path: str | Path,
    fold_override: int | None = None,
    all_data: bool = False,
    mode_override: str | None = None,
) -> tuple[dict[str, Any], list[str], Path]:
    config, resolved_config = load_config(config_path)
    if all_data and fold_override is not None:
        raise ValueError("--fold and --all-data are mutually exclusive")
    default_fold = config.get("validation_fold")
    fold = None if all_data else fold_override if fold_override is not None else default_fold
    if fold is not None:
        fold = int(fold)
        if fold not in range(5):
            raise ValueError(f"competition fold must be 0, 1, 2, 3, or 4; got {fold}")

    source_config_sha256 = stable_sha256(resolved_config)
    train_csv = absolute(config["train_csv"])
    train_csv_sha256 = stable_sha256(train_csv)
    exclusion_path = absolute(config["exclude_ids"]) if config.get("exclude_ids") else None
    exclude_ids_sha256 = (
        stable_sha256(exclusion_path) if exclusion_path is not None else None
    )

    policy = policy_from_config(config, mode_override)
    mode = canonical_mode(policy.mode)
    output_dir = _output_dir(config, mode, fold)
    _ensure_no_training_artifacts(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained = absolute(config["pretrained_model"]).resolve()
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    base_manifest, validation_manifest, preparation = prepare_training_data(config, fold)
    base_manifest_sha256 = stable_sha256(base_manifest)
    validation_sha256 = (
        stable_sha256(validation_manifest) if validation_manifest is not None else None
    )
    examples = load_training_examples(config, fold, base_manifest)
    plan = build_weighting_plan(examples, policy, fold)

    split = f"fold{fold}" if fold is not None else "all"
    variant = str(config.get("image_variant", "full"))
    weighted_manifest = base_manifest.with_name(
        f"train_{split}_{variant}_{mode}.txt"
    )
    weighting_receipt_path = output_dir / "weighting_receipt.json"
    weighting_receipt = write_plan(
        plan, fold, weighted_manifest, weighting_receipt_path
    )
    if stable_sha256(resolved_config) != source_config_sha256:
        raise RuntimeError("configuration changed during Model30 preparation")
    if stable_sha256(train_csv) != train_csv_sha256:
        raise RuntimeError("training CSV changed during Model30 preparation")
    if (
        exclusion_path is not None
        and stable_sha256(exclusion_path) != exclude_ids_sha256
    ):
        raise RuntimeError("exclusion policy changed during Model30 preparation")
    if validation_manifest is not None and stable_sha256(validation_manifest) != validation_sha256:
        raise RuntimeError("validation manifest changed while weighting training rows")

    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("training configuration must be a mapping")
    epoch_key = "all_data_epochs" if all_data else "epochs"
    cosine_key = "all_data_cosine_epochs" if all_data else "cosine_epochs"
    base_epochs = _positive_integer(training, epoch_key, 20 if all_data else 40)
    base_cosine = _positive_integer(training, cosine_key, base_epochs)
    epochs = _scaled_epochs(base_epochs, len(examples), len(plan.expanded_paths))
    cosine_epochs = _scaled_epochs(
        base_cosine, len(examples), len(plan.expanded_paths)
    )
    batch_size = _positive_integer(training, "batch_size", 4)
    accumulation = _positive_integer(training, "gradient_accumulation", 4)
    base_steps = _estimated_optimizer_steps(
        len(examples), base_epochs, batch_size, accumulation
    )
    weighted_steps = _estimated_optimizer_steps(
        len(plan.expanded_paths), epochs, batch_size, accumulation
    )

    command = _training_command(
        config,
        pretrained,
        output_dir,
        weighted_manifest.resolve(),
        validation_manifest.resolve() if validation_manifest is not None else None,
        epochs,
        cosine_epochs,
        all_data,
    )
    effective_config = copy.deepcopy(config)
    effective_config["output_dir"] = str(output_dir)
    effective_training = effective_config.setdefault("training", {})
    effective_training[epoch_key] = epochs
    effective_training[cosine_key] = cosine_epochs
    preparation = {
        **preparation,
        "base_train_manifest": str(base_manifest.resolve()),
        "base_train_manifest_sha256": base_manifest_sha256,
        "train_manifest": str(weighted_manifest.resolve()),
        "train_manifest_sha256": weighting_receipt["manifest_sha256"],
        "train_rows": len(plan.expanded_paths),
        "unique_train_rows": len(examples),
        "validation_manifest_sha256": validation_sha256,
        "weighting_receipt": str(weighting_receipt_path.resolve()),
        "weighting_receipt_sha256": stable_sha256(weighting_receipt_path),
    }
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "model30_metric_aware_manifest",
        "config": effective_config,
        "source_config": str(resolved_config),
        "source_config_sha256": source_config_sha256,
        "train_csv_sha256": train_csv_sha256,
        "exclude_ids_sha256": exclude_ids_sha256,
        "pretrained_sha256": sha256(pretrained),
        "preparation": preparation,
        "weighting": {
            "mode": mode,
            "training_only_sha256": plan.training_only_sha256,
            "base_epochs": base_epochs,
            "weighted_epochs": epochs,
            "base_cosine_epochs": base_cosine,
            "weighted_cosine_epochs": cosine_epochs,
            "base_optimizer_steps_estimate": base_steps,
            "weighted_optimizer_steps_estimate": weighted_steps,
            "optimizer_step_ratio": weighted_steps / base_steps,
            "base_sample_presentations": len(examples) * base_epochs,
            "weighted_sample_presentations": len(plan.expanded_paths) * epochs,
        },
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("kraken", "torch", "lightning")
            if importlib.util.find_spec(name) is not None
        },
        "prepared_utc_epoch": time.time(),
    }
    atomic_json(output_dir / "run_manifest.json", run_manifest)
    return run_manifest, command, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or train Model30's deterministic weighted Kraken manifest"
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--fold", type=int)
    selection.add_argument("--all-data", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["metric_clipped", "uniform", "identity"],
        help="override weighting.mode; identity is an alias for uniform",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write deterministic manifests/receipts and print the command only",
    )
    args = parser.parse_args()

    manifest, command, output_dir = prepare_run(
        args.config, args.fold, args.all_data, args.mode
    )
    weighting = manifest["weighting"]
    print(shlex.join(command), flush=True)
    print(
        "Model30 prepared: "
        f"mode={manifest['weighting']['mode']} "
        f"unique_rows={manifest['preparation']['unique_train_rows']} "
        f"manifest_rows={manifest['preparation']['train_rows']} "
        f"epochs={weighting['weighted_epochs']} "
        f"optimizer_step_ratio={weighting['optimizer_step_ratio']:.6f}",
        flush=True,
    )
    print(f"run manifest: {output_dir / 'run_manifest.json'}", flush=True)
    if args.dry_run:
        return
    atomic_json(
        output_dir / "training_started.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_manifest_sha256": stable_sha256(output_dir / "run_manifest.json"),
            "started_utc_epoch": time.time(),
        },
    )
    subprocess.run(command, cwd=ROOT, check=True)
    checkpoints = sorted(output_dir.glob("checkpoint_*.ckpt"))
    best = sorted(output_dir.glob("best_*.safetensors"))
    atomic_json(
        output_dir / "training_complete.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_manifest_sha256": stable_sha256(output_dir / "run_manifest.json"),
            "finished_utc_epoch": time.time(),
            "checkpoints": [
                {"path": str(path), "sha256": stable_sha256(path)}
                for path in checkpoints
            ],
            "best_weights": [
                {"path": str(path), "sha256": stable_sha256(path)} for path in best
            ],
        },
    )


if __name__ == "__main__":
    main()
