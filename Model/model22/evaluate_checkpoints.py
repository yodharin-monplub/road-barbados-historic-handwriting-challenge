#!/usr/bin/env python3
"""Evaluate every Model22 epoch with the exact competition metric.

Kraken's training callback selects checkpoints by ordinary corpus-level CER.
The competition uses a different, weighted combination of CER and WER, so an
epoch must be decoded and scored to determine whether it is actually best.

This program intentionally converts one Lightning checkpoint per ``ketos``
invocation.  Conversion and inference run in subprocesses so GPU memory is
released between epochs, while scoring happens in-process with the same
``competition_error`` implementation used by ``scripts/score_csv.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.metrics import competition_error  # noqa: E402
from road_htr.schema import (  # noqa: E402
    SchemaError,
    predictions_by_id,
    read_table,
)


SCHEMA_VERSION = 1
CANONICAL_INPUT_SHA256 = (
    "1173b87f6415f657d89a13b62f4d3d84bc926e550338ad4499963c89a81a16aa"
)
CANONICAL_TOTAL_ROWS = 4095
CANONICAL_FOLD_ROWS = 819
CHECKPOINT_RE = re.compile(
    r"^checkpoint_(?P<epoch>\d+)-"
    r"(?P<metric>[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf)|nan)"
    r"(?:-v(?P<version>\d+))?\.ckpt$",
    re.IGNORECASE,
)
Runner = Callable[[list[str]], None]


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    epoch: int
    reported_val_accuracy: float
    version: int

    @property
    def artifact_stem(self) -> str:
        return self.path.stem


@dataclass(frozen=True)
class EvaluationOptions:
    checkpoint_dir: Path
    input_csv: Path
    output_dir: Path
    config: Path
    fold: int
    ketos: Path
    python: Path
    batch_size: int = 1
    expected_input_sha256: str = CANONICAL_INPUT_SHA256
    expected_total_rows: int = CANONICAL_TOTAL_ROWS
    expected_fold_rows: int = CANONICAL_FOLD_ROWS
    dry_run: bool = False
    force: bool = False
    allow_incomplete_training: bool = False


@dataclass(frozen=True)
class ArtifactPaths:
    weights: Path
    conversion_receipt: Path
    predictions: Path
    prediction_details: Path
    inference_manifest: Path
    inference_receipt: Path
    metrics: Path


@dataclass(frozen=True)
class TrainingLineage:
    manifest: Path
    manifest_sha256: str
    config_differences: tuple[str, ...]
    preserve_internal_spaces: bool
    expected_checkpoint_epochs: tuple[int, ...]


def _project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def _stable_sha256(path: Path) -> str:
    """Hash a file and fail if it changes while being read."""

    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing (is training still running?): {path}")
    return digest.hexdigest()


def _implementation_sha256() -> str:
    """Fingerprint code that affects decoding or challenge scoring."""

    files = (
        Path(__file__).resolve(),
        HERE / "infer.py",
        ROOT / "src" / "road_htr" / "kraken_cli.py",
        ROOT / "src" / "road_htr" / "metrics.py",
        ROOT / "src" / "road_htr" / "preprocess.py",
        ROOT / "src" / "road_htr" / "schema.py",
        ROOT / "src" / "road_htr" / "submission.py",
        ROOT / "src" / "road_htr" / "vlm_data.py",
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_stable_sha256(path)))
    return digest.hexdigest()


def discover_checkpoints(checkpoint_dir: Path) -> list[Checkpoint]:
    """Return scored checkpoints in epoch order, excluding abort snapshots."""

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    checkpoints: list[Checkpoint] = []
    malformed: list[str] = []
    for path in checkpoint_dir.glob("checkpoint_*.ckpt"):
        if path.name == "checkpoint_abort.ckpt":
            continue
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            malformed.append(path.name)
            continue
        checkpoints.append(
            Checkpoint(
                path=path.resolve(),
                epoch=int(match.group("epoch")),
                reported_val_accuracy=float(match.group("metric")),
                version=int(match.group("version") or 0),
            )
        )
    if malformed:
        raise ValueError(
            "unrecognized scored checkpoint name(s): " + ", ".join(sorted(malformed))
        )
    if not checkpoints:
        raise FileNotFoundError(
            f"no scored checkpoint_*.ckpt files found in {checkpoint_dir}"
        )
    checkpoints.sort(key=lambda item: (item.epoch, item.version, item.path.name))
    duplicate_epochs = sorted(
        {
            checkpoint.epoch
            for checkpoint in checkpoints
            if sum(item.epoch == checkpoint.epoch for item in checkpoints) > 1
        }
    )
    if duplicate_epochs:
        raise ValueError(
            "multiple checkpoint versions for epoch(s) "
            f"{duplicate_epochs}; the directory appears to mix training runs"
        )
    return checkpoints


def _checkpoint_snapshot(checkpoint_dir: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        sorted(
            (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in checkpoint_dir.glob("checkpoint_*.ckpt")
        )
    )


def _training_is_complete(checkpoint_dir: Path, checkpoints: list[Checkpoint]) -> bool:
    """A clean Kraken run writes best_*.safetensors after its final checkpoint."""

    best_weights = list(checkpoint_dir.glob("best_*.safetensors"))
    if not best_weights:
        return False
    required = [item.path for item in checkpoints]
    required.append(checkpoint_dir / "run_manifest.json")
    abort = checkpoint_dir / "checkpoint_abort.ckpt"
    if abort.is_file():
        required.append(abort)
    if not all(path.is_file() for path in required):
        return False
    newest_required = max(path.stat().st_mtime_ns for path in required)
    return max(path.stat().st_mtime_ns for path in best_weights) >= newest_required


def _artifact_paths(output_dir: Path, checkpoint: Checkpoint) -> ArtifactPaths:
    stem = checkpoint.artifact_stem
    weights = output_dir / "converted" / f"{stem}.safetensors"
    predictions = output_dir / "predictions" / f"{stem}.csv"
    return ArtifactPaths(
        weights=weights,
        conversion_receipt=output_dir / "converted" / f"{stem}.conversion.json",
        predictions=predictions,
        prediction_details=predictions.with_suffix(".jsonl"),
        inference_manifest=predictions.with_suffix(".manifest.json"),
        inference_receipt=output_dir / "predictions" / f"{stem}.evaluation.json",
        metrics=output_dir / "metrics" / f"{stem}.json",
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _validate_output_is_disjoint(options: EvaluationOptions) -> None:
    """Keep evaluator artifacts outside the immutable training tree."""

    checkpoint_dir = options.checkpoint_dir.resolve()
    output_dir = options.output_dir.resolve()
    if (
        checkpoint_dir == output_dir
        or checkpoint_dir in output_dir.parents
        or output_dir in checkpoint_dir.parents
    ):
        raise ValueError(
            "output_dir and checkpoint_dir must be disjoint sibling trees; "
            f"got output_dir={output_dir}, checkpoint_dir={checkpoint_dir}"
        )


def _flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key in sorted(value):
        child = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten_mapping(value[key], child))
    return flattened


def _effective_inference_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Return config values that can affect decoded text in this evaluator."""

    if "image_dir" not in config:
        raise ValueError("configuration is missing image_dir")
    inference = config.get("inference", {})
    if not isinstance(inference, dict):
        raise ValueError("configuration inference section must be a mapping")
    return {
        "image_dir": str(_project_path(config["image_dir"])),
        "image_variant": str(config.get("image_variant", "full")),
        "accelerator": str(inference.get("accelerator", "cpu")),
        "device": str(inference.get("device", "auto")),
        "precision": str(inference.get("precision", "32-true")),
        "threads": int(inference.get("threads", max(1, (os.cpu_count() or 2) // 2))),
        "line_workers": int(inference.get("line_workers", 0)),
        "padding": int(inference.get("padding", 16)),
        # batch_size and constrain_charset are deliberately absent because the
        # evaluator overrides both on the command line. Space cleanup is also
        # current evaluation policy rather than a recognizer-training input;
        # its value is validated separately and embedded in every signature.
    }


def _load_training_lineage(
    options: EvaluationOptions, input_sha256: str
) -> TrainingLineage:
    """Validate the live config against the immutable training-run snapshot.

    Model22's current config may gain training-only safeguards after a run has
    started. Those changes are safe for checkpoint decoding. Image selection
    and effective inference settings are not, so mismatches there fail before
    conversion or artifact creation.
    """

    manifest_path = options.checkpoint_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"training run manifest is missing: {manifest_path}")
    manifest_sha256 = _stable_sha256(manifest_path)
    run_manifest = _read_json(manifest_path)
    if run_manifest is None:
        raise ValueError(f"training run manifest is missing or invalid: {manifest_path}")
    run_config = run_manifest.get("config")
    preparation = run_manifest.get("preparation")
    if not isinstance(run_config, dict) or not isinstance(preparation, dict):
        raise ValueError(f"training run manifest lacks config/preparation: {manifest_path}")
    if preparation.get("fold") != options.fold:
        raise ValueError(
            f"training run fold {preparation.get('fold')!r} does not match "
            f"requested fold {options.fold}"
        )

    output_template = run_config.get("output_dir")
    if not isinstance(output_template, str):
        raise ValueError("training run config is missing output_dir")
    try:
        recorded_checkpoint_dir = _project_path(
            output_template.format(fold=options.fold)
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"invalid training output_dir template {output_template!r}"
        ) from exc
    if recorded_checkpoint_dir != options.checkpoint_dir.resolve():
        raise ValueError(
            "checkpoint_dir does not match the training run output_dir: "
            f"{options.checkpoint_dir.resolve()} != {recorded_checkpoint_dir}"
        )

    try:
        current_config = yaml.safe_load(options.config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read current config {options.config}: {exc}") from exc
    if not isinstance(current_config, dict):
        raise ValueError(f"current config must be a mapping: {options.config}")
    current_inference = current_config.get("inference", {})
    preserve_internal_spaces = current_inference.get(
        "preserve_internal_spaces", False
    )
    if not isinstance(preserve_internal_spaces, bool):
        raise ValueError("inference.preserve_internal_spaces must be a boolean")

    for label, config in (("training run", run_config), ("current", current_config)):
        train_csv = config.get("train_csv")
        if not isinstance(train_csv, str):
            raise ValueError(f"{label} config is missing train_csv")
        configured_hash = _stable_sha256(_project_path(train_csv))
        if configured_hash != input_sha256:
            raise ValueError(
                f"{label} train_csv does not match --input by SHA-256"
            )

    run_contract = _effective_inference_contract(run_config)
    current_contract = _effective_inference_contract(current_config)
    contract_mismatches = {
        key: (run_contract[key], current_contract[key])
        for key in run_contract
        if run_contract[key] != current_contract[key]
    }
    if contract_mismatches:
        raise ValueError(
            "current config changes inference-critical training-run values: "
            f"{contract_mismatches}"
        )

    old_flat = _flatten_mapping(run_config)
    new_flat = _flatten_mapping(current_config)
    differences = tuple(
        key
        for key in sorted(set(old_flat) | set(new_flat))
        if old_flat.get(key) != new_flat.get(key)
    )
    if _stable_sha256(manifest_path) != manifest_sha256:
        raise RuntimeError(f"training run manifest changed while reading: {manifest_path}")
    run_training = run_config.get("training", {})
    expected_checkpoint_epochs: tuple[int, ...] = ()
    if isinstance(run_training, dict):
        epochs = run_training.get("epochs")
        frequency = run_training.get("validation_frequency")
        stop = run_training.get("stop")
        # Kraken retains its top 10 checkpoints. A fixed Model22 run of at
        # most 10 epochs with per-epoch validation must therefore contain
        # every epoch; accepting less would silently rank an incomplete run.
        if (
            not isinstance(epochs, bool)
            and isinstance(epochs, int)
            and 0 < epochs <= 10
            and frequency == 1
            and stop == "fixed"
        ):
            expected_checkpoint_epochs = tuple(range(epochs))
    return TrainingLineage(
        manifest=manifest_path.resolve(),
        manifest_sha256=manifest_sha256,
        config_differences=differences,
        preserve_internal_spaces=preserve_internal_spaces,
        expected_checkpoint_epochs=expected_checkpoint_epochs,
    )


def _run_subprocess(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _show_command(command: list[str]) -> None:
    print("+ " + shlex.join(command), flush=True)


def _conversion_command(options: EvaluationOptions, checkpoint: Checkpoint, output: Path) -> list[str]:
    # Keep exactly one checkpoint argument. Passing a glob or multiple epochs
    # would combine models into a single weights file instead of converting one.
    return [
        str(options.ketos),
        "convert",
        "--output",
        str(output),
        str(checkpoint.path),
    ]


def _inference_command(
    options: EvaluationOptions, weights: Path, predictions: Path
) -> list[str]:
    return [
        str(options.python),
        str(HERE / "infer.py"),
        "--config",
        str(options.config),
        "--checkpoint",
        str(weights),
        "--input",
        str(options.input_csv),
        "--fold",
        str(options.fold),
        "--output",
        str(predictions),
        "--no-constrain-charset",
        "--batch-size",
        str(options.batch_size),
    ]


def _conversion_is_reusable(
    artifacts: ArtifactPaths, checkpoint_sha256: str
) -> tuple[bool, str | None]:
    receipt = _read_json(artifacts.conversion_receipt)
    if receipt is None or receipt.get("schema_version") != SCHEMA_VERSION:
        return False, None
    if receipt.get("checkpoint_sha256") != checkpoint_sha256:
        return False, None
    expected_weights_sha256 = receipt.get("weights_sha256")
    if not isinstance(expected_weights_sha256, str) or not artifacts.weights.is_file():
        return False, None
    actual_weights_sha256 = _stable_sha256(artifacts.weights)
    return actual_weights_sha256 == expected_weights_sha256, actual_weights_sha256


def _ensure_conversion(
    options: EvaluationOptions,
    checkpoint: Checkpoint,
    artifacts: ArtifactPaths,
    checkpoint_sha256: str,
    runner: Runner,
) -> tuple[str, bool]:
    if not options.force:
        reusable, weights_sha256 = _conversion_is_reusable(
            artifacts, checkpoint_sha256
        )
        if reusable and weights_sha256 is not None:
            print(f"reuse conversion: {artifacts.weights}", flush=True)
            return weights_sha256, True

    artifacts.weights.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifacts.weights.with_name(
        f".{artifacts.weights.stem}.{uuid.uuid4().hex}.safetensors"
    )
    try:
        command = _conversion_command(options, checkpoint, temporary)
        _show_command(command)
        runner(command)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"ketos did not produce a non-empty weights file: {temporary}")
        weights_sha256 = _stable_sha256(temporary)
        os.replace(temporary, artifacts.weights)
        _atomic_json(
            artifacts.conversion_receipt,
            {
                "schema_version": SCHEMA_VERSION,
                "checkpoint": str(checkpoint.path),
                "checkpoint_sha256": checkpoint_sha256,
                "weights": str(artifacts.weights),
                "weights_sha256": weights_sha256,
            },
        )
        return weights_sha256, False
    finally:
        temporary.unlink(missing_ok=True)


def _inference_is_reusable(
    artifacts: ArtifactPaths,
    signature: dict[str, Any],
    weights_sha256: str,
) -> tuple[bool, str | None]:
    receipt = _read_json(artifacts.inference_receipt)
    if receipt is None or receipt.get("schema_version") != SCHEMA_VERSION:
        return False, None
    if receipt.get("signature") != signature:
        return False, None
    if receipt.get("weights_sha256") != weights_sha256:
        return False, None
    expected_hashes = {
        artifacts.predictions: receipt.get("predictions_sha256"),
        artifacts.prediction_details: receipt.get("prediction_details_sha256"),
        artifacts.inference_manifest: receipt.get("inference_manifest_sha256"),
    }
    if not all(isinstance(value, str) for value in expected_hashes.values()):
        return False, None
    if not all(path.is_file() for path in expected_hashes):
        return False, None
    actual_hashes = {path: _stable_sha256(path) for path in expected_hashes}
    if any(actual_hashes[path] != expected for path, expected in expected_hashes.items()):
        return False, None
    return True, actual_hashes[artifacts.predictions]


def _ensure_inference(
    options: EvaluationOptions,
    artifacts: ArtifactPaths,
    signature: dict[str, Any],
    weights_sha256: str,
    runner: Runner,
) -> tuple[str, bool]:
    if not options.force:
        reusable, predictions_sha256 = _inference_is_reusable(
            artifacts, signature, weights_sha256
        )
        if reusable and predictions_sha256 is not None:
            print(f"reuse predictions: {artifacts.predictions}", flush=True)
            return predictions_sha256, True

    artifacts.predictions.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifacts.predictions.with_name(
        f".{artifacts.predictions.stem}.{uuid.uuid4().hex}.csv"
    )
    generated = (
        temporary,
        temporary.with_suffix(".jsonl"),
        temporary.with_suffix(".manifest.json"),
    )
    try:
        command = _inference_command(options, artifacts.weights, temporary)
        _show_command(command)
        runner(command)
        if not all(path.is_file() for path in generated):
            missing = [str(path) for path in generated if not path.is_file()]
            raise RuntimeError(f"inference did not create expected artifact(s): {missing}")
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"inference produced an empty CSV: {temporary}")

        inference_manifest = _read_json(temporary.with_suffix(".manifest.json"))
        if inference_manifest is None:
            raise RuntimeError("inference manifest is missing or invalid JSON")
        expected_manifest = {
            "input_sha256": signature["input_sha256"],
            "config_sha256": signature["config_sha256"],
            "model_sha256": weights_sha256,
            "fold": options.fold,
            "constrain_charset": False,
            "batch_size": options.batch_size,
            "preserve_internal_spaces": signature["preserve_internal_spaces"],
            # Training exclusions only control Kraken manifests. External OOF
            # decoding must still cover every row in the canonical fold CSV.
            "rows": signature["expected_rows"],
        }
        mismatches = {
            key: (inference_manifest.get(key), expected)
            for key, expected in expected_manifest.items()
            if inference_manifest.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"inference manifest mismatch: {mismatches}")

        # Validate the full canonical ID set and exact metric before making
        # this inference reusable. Otherwise a subprocess that reports the
        # right row count but writes wrong/duplicate IDs would poison the
        # cache and require --force to recover.
        score_predictions(options.input_csv, temporary, options.fold)
        predictions_sha256 = _stable_sha256(temporary)
        prediction_details_sha256 = _stable_sha256(temporary.with_suffix(".jsonl"))
        inference_manifest_sha256 = _stable_sha256(
            temporary.with_suffix(".manifest.json")
        )
        os.replace(temporary.with_suffix(".jsonl"), artifacts.prediction_details)
        os.replace(temporary.with_suffix(".manifest.json"), artifacts.inference_manifest)
        os.replace(temporary, artifacts.predictions)
        _atomic_json(
            artifacts.inference_receipt,
            {
                "schema_version": SCHEMA_VERSION,
                "signature": signature,
                "weights_sha256": weights_sha256,
                "predictions_sha256": predictions_sha256,
                "prediction_details_sha256": prediction_details_sha256,
                "inference_manifest_sha256": inference_manifest_sha256,
            },
        )
        return predictions_sha256, False
    finally:
        for path in generated:
            path.unlink(missing_ok=True)


def score_predictions(input_csv: Path, predictions_csv: Path, fold: int) -> dict[str, float]:
    """Mirror ``scripts/score_csv.py --fold`` exactly."""

    truth = read_table(input_csv, require_target=True)
    if "fold" not in truth.fieldnames:
        raise SchemaError("--fold requires an input CSV with a fold column")
    predictions = read_table(predictions_csv)
    if predictions.target_column is None:
        non_id = [name for name in predictions.fieldnames if name != predictions.id_column]
        if len(non_id) != 1:
            raise SchemaError(
                "prediction CSV is ambiguous; expected exactly one non-ID column"
            )
        prediction_column = non_id[0]
    else:
        prediction_column = predictions.target_column
    by_id = predictions_by_id(predictions, prediction_column)

    assert truth.target_column is not None
    rows = [row for row in truth.rows if int(row["fold"]) == fold]
    if not rows:
        raise SchemaError(f"input CSV has no rows for fold {fold}")
    identifiers = [row[truth.id_column].strip() for row in rows]
    missing = sorted(set(identifiers) - set(by_id))
    extra = sorted(set(by_id) - set(identifiers))
    if missing or extra:
        raise SchemaError(
            f"ID mismatch: {len(missing)} missing, {len(extra)} extra; "
            f"examples missing={missing[:5]}, extra={extra[:5]}"
        )
    return competition_error(
        [row[truth.target_column] for row in rows],
        [by_id[identifier] for identifier in identifiers],
    )


def _completed_result(
    artifacts: ArtifactPaths, signature: dict[str, Any]
) -> dict[str, Any] | None:
    payload = _read_json(artifacts.metrics)
    if payload is None or payload.get("schema_version") != SCHEMA_VERSION:
        return None
    if payload.get("signature") != signature:
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    predictions_sha256 = result.get("predictions_sha256")
    weights_sha256 = result.get("weights_sha256")
    if not isinstance(predictions_sha256, str) or not isinstance(weights_sha256, str):
        return None
    conversion_reusable, actual_weights_sha256 = _conversion_is_reusable(
        artifacts, signature["checkpoint_sha256"]
    )
    if not conversion_reusable or actual_weights_sha256 != weights_sha256:
        return None
    inference_reusable, actual_predictions_sha256 = _inference_is_reusable(
        artifacts, signature, weights_sha256
    )
    if not inference_reusable or actual_predictions_sha256 != predictions_sha256:
        return None
    required_metrics = {
        "weighted_wer",
        "weighted_cer",
        "normalized_wer_error",
        "normalized_cer_error",
        "mean_error",
        "multi_score",
    }
    if not required_metrics.issubset(result):
        return None
    return result


def _signature(
    checkpoint: Checkpoint,
    checkpoint_sha256: str,
    input_sha256: str,
    config_sha256: str,
    implementation_sha256: str,
    options: EvaluationOptions,
    expected_rows: int,
    training_manifest_sha256: str,
    preserve_internal_spaces: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": checkpoint.path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "input_sha256": input_sha256,
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "fold": options.fold,
        "batch_size": options.batch_size,
        "constrain_charset": False,
        "expected_rows": expected_rows,
        "training_manifest_sha256": training_manifest_sha256,
        "preserve_internal_spaces": preserve_internal_spaces,
    }


def _make_result(
    checkpoint: Checkpoint,
    artifacts: ArtifactPaths,
    checkpoint_sha256: str,
    weights_sha256: str,
    predictions_sha256: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "epoch": checkpoint.epoch,
        "checkpoint_version": checkpoint.version,
        "checkpoint_val_accuracy": checkpoint.reported_val_accuracy,
        "checkpoint": str(checkpoint.path),
        "checkpoint_sha256": checkpoint_sha256,
        "weights": str(artifacts.weights),
        "weights_sha256": weights_sha256,
        "predictions": str(artifacts.predictions),
        "predictions_sha256": predictions_sha256,
        **metrics,
    }


def _dry_run_plan(
    options: EvaluationOptions,
    checkpoint: Checkpoint,
    artifacts: ArtifactPaths,
    signature: dict[str, Any],
) -> None:
    if not options.force and _completed_result(artifacts, signature) is not None:
        print(f"[epoch {checkpoint.epoch:02d}] complete: skip", flush=True)
        return
    print(f"[epoch {checkpoint.epoch:02d}] evaluate {checkpoint.path.name}", flush=True)
    reusable_conversion, _ = _conversion_is_reusable(
        artifacts, signature["checkpoint_sha256"]
    )
    if options.force or not reusable_conversion:
        _show_command(_conversion_command(options, checkpoint, artifacts.weights))
    else:
        print(f"reuse conversion: {artifacts.weights}", flush=True)
    # A dry run cannot know the weight hash of a conversion that has not run.
    # Still show the exact inference command and subsequent scoring action.
    _show_command(_inference_command(options, artifacts.weights, artifacts.predictions))
    print(
        f"score exact competition metric: {artifacts.predictions} (fold {options.fold})",
        flush=True,
    )


def _write_leaderboards(
    options: EvaluationOptions,
    results: list[dict[str, Any]],
    input_sha256: str,
    lineage: TrainingLineage,
) -> dict[str, Any]:
    ranked = sorted(
        results,
        key=lambda row: (
            -float(row["multi_score"]),
            int(row["epoch"]),
            int(row["checkpoint_version"]),
            str(row["checkpoint"]),
        ),
    )
    rank_by_checkpoint = {
        str(row["checkpoint"]): rank for rank, row in enumerate(ranked, start=1)
    }
    epoch_order = [
        {"challenge_rank": rank_by_checkpoint[str(row["checkpoint"])], **row}
        for row in results
    ]
    best = epoch_order[
        next(
            index
            for index, row in enumerate(epoch_order)
            if str(row["checkpoint"]) == str(ranked[0]["checkpoint"])
        )
    ]
    leaderboard = {
        "schema_version": SCHEMA_VERSION,
        "selection_metric": "multi_score",
        "fold": options.fold,
        "input": str(options.input_csv),
        "input_sha256": input_sha256,
        "checkpoint_dir": str(options.checkpoint_dir),
        "training_manifest": str(lineage.manifest),
        "training_manifest_sha256": lineage.manifest_sha256,
        "training_current_config_differences": list(lineage.config_differences),
        "best": best,
        "results": epoch_order,
    }
    _atomic_json(options.output_dir / "leaderboard.json", leaderboard)

    fields = [
        "challenge_rank",
        "epoch",
        "checkpoint_version",
        "checkpoint_val_accuracy",
        "multi_score",
        "mean_error",
        "weighted_wer",
        "weighted_cer",
        "normalized_wer_error",
        "normalized_cer_error",
        "checkpoint",
        "checkpoint_sha256",
        "weights",
        "weights_sha256",
        "predictions",
        "predictions_sha256",
    ]
    _atomic_csv(
        options.output_dir / "leaderboard.csv",
        fields,
        [{name: row[name] for name in fields} for row in epoch_order],
    )
    return leaderboard


def evaluate_checkpoints(
    options: EvaluationOptions, runner: Runner = _run_subprocess
) -> dict[str, Any]:
    if options.fold < 0:
        raise ValueError("fold must be non-negative")
    if options.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not options.input_csv.is_file():
        raise FileNotFoundError(options.input_csv)
    if not options.config.is_file():
        raise FileNotFoundError(options.config)
    _validate_output_is_disjoint(options)

    checkpoints = discover_checkpoints(options.checkpoint_dir)
    initial_snapshot = _checkpoint_snapshot(options.checkpoint_dir)
    complete = _training_is_complete(options.checkpoint_dir, checkpoints)
    if not complete:
        message = (
            "no up-to-date best_*.safetensors completion marker found; "
            "training may still be active"
        )
        if options.dry_run:
            print(f"warning: {message}", file=sys.stderr, flush=True)
        elif not options.allow_incomplete_training:
            raise RuntimeError(message + " (use --allow-incomplete-training to override)")

    # Validate the truth table before starting expensive conversion/inference.
    truth = read_table(options.input_csv, require_target=True)
    if "fold" not in truth.fieldnames:
        raise SchemaError("input CSV has no fold column")
    if len(truth.rows) != options.expected_total_rows:
        raise SchemaError(
            f"canonical input must contain {options.expected_total_rows} rows, "
            f"found {len(truth.rows)}"
        )
    expected_rows = sum(int(row["fold"]) == options.fold for row in truth.rows)
    if expected_rows != options.expected_fold_rows:
        raise SchemaError(
            f"canonical fold {options.fold} must contain "
            f"{options.expected_fold_rows} rows, found {expected_rows}"
        )

    input_sha256 = _stable_sha256(options.input_csv)
    if input_sha256 != options.expected_input_sha256:
        raise SchemaError(
            "canonical input SHA-256 mismatch: "
            f"{input_sha256} != {options.expected_input_sha256}"
        )
    config_sha256 = _stable_sha256(options.config)
    implementation_sha256 = _implementation_sha256()
    lineage = _load_training_lineage(options, input_sha256)
    if lineage.config_differences:
        print(
            "training/current config differ only in evaluator-safe or "
            "CLI-overridden fields: "
            + ", ".join(lineage.config_differences),
            flush=True,
        )
    if complete and lineage.expected_checkpoint_epochs:
        actual_epochs = tuple(checkpoint.epoch for checkpoint in checkpoints)
        if actual_epochs != lineage.expected_checkpoint_epochs:
            raise RuntimeError(
                "completed fixed training run has an incomplete checkpoint set: "
                f"expected {list(lineage.expected_checkpoint_epochs)}, "
                f"found {list(actual_epochs)}"
            )
    checkpoint_hashes = {
        checkpoint.path: _stable_sha256(checkpoint.path) for checkpoint in checkpoints
    }

    if options.dry_run:
        for checkpoint in checkpoints:
            artifacts = _artifact_paths(options.output_dir, checkpoint)
            signature = _signature(
                checkpoint,
                checkpoint_hashes[checkpoint.path],
                input_sha256,
                config_sha256,
                implementation_sha256,
                options,
                expected_rows,
                lineage.manifest_sha256,
                lineage.preserve_internal_spaces,
            )
            _dry_run_plan(options, checkpoint, artifacts, signature)
        return {
            "dry_run": True,
            "fold": options.fold,
            "checkpoints": [str(item.path) for item in checkpoints],
        }

    for name in ("converted", "predictions", "metrics"):
        (options.output_dir / name).mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_sha256 = checkpoint_hashes[checkpoint.path]
        artifacts = _artifact_paths(options.output_dir, checkpoint)
        signature = _signature(
            checkpoint,
            checkpoint_sha256,
            input_sha256,
            config_sha256,
            implementation_sha256,
            options,
            expected_rows,
            lineage.manifest_sha256,
            lineage.preserve_internal_spaces,
        )
        if not options.force:
            completed = _completed_result(artifacts, signature)
            if completed is not None:
                print(
                    f"[epoch {checkpoint.epoch:02d}] reuse complete score "
                    f"{float(completed['multi_score']):.9f}",
                    flush=True,
                )
                results.append(completed)
                continue

        print(f"[epoch {checkpoint.epoch:02d}] {checkpoint.path.name}", flush=True)
        weights_sha256, _ = _ensure_conversion(
            options, checkpoint, artifacts, checkpoint_sha256, runner
        )
        predictions_sha256, _ = _ensure_inference(
            options, artifacts, signature, weights_sha256, runner
        )
        metrics = score_predictions(options.input_csv, artifacts.predictions, options.fold)
        result = _make_result(
            checkpoint,
            artifacts,
            checkpoint_sha256,
            weights_sha256,
            predictions_sha256,
            metrics,
        )
        _atomic_json(
            artifacts.metrics,
            {
                "schema_version": SCHEMA_VERSION,
                "signature": signature,
                "result": result,
            },
        )
        print(
            f"[epoch {checkpoint.epoch:02d}] multi_score={metrics['multi_score']:.9f} "
            f"weighted_wer={metrics['weighted_wer']:.6f} "
            f"weighted_cer={metrics['weighted_cer']:.6f}",
            flush=True,
        )
        results.append(result)

    mutable_inputs = {
        "checkpoint directory": _checkpoint_snapshot(options.checkpoint_dir)
        != initial_snapshot,
        "truth CSV": _stable_sha256(options.input_csv) != input_sha256,
        "current config": _stable_sha256(options.config) != config_sha256,
        "training run manifest": _stable_sha256(lineage.manifest)
        != lineage.manifest_sha256,
        "implementation": _implementation_sha256() != implementation_sha256,
    }
    changed = [name for name, did_change in mutable_inputs.items() if did_change]
    if changed:
        raise RuntimeError(
            "evaluation input changed while running: "
            + ", ".join(changed)
            + "; per-checkpoint results remain resumable, but rerun from stable inputs"
        )

    leaderboard = _write_leaderboards(options, results, input_sha256, lineage)
    best = leaderboard["best"]
    print(
        f"best challenge checkpoint: epoch {best['epoch']} "
        f"({Path(best['checkpoint']).name}), multi_score={best['multi_score']:.9f}",
        flush=True,
    )
    print(f"leaderboard: {options.output_dir / 'leaderboard.json'}", flush=True)
    return leaderboard


def _find_ketos(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = _project_path(explicit)
    else:
        local = ROOT / ".venv-kraken" / "bin" / "ketos"
        installed = shutil.which("ketos")
        candidate = local if local.is_file() else Path(installed) if installed else local
    if not candidate.is_file():
        raise FileNotFoundError(
            f"ketos executable not found: {candidate}; pass --ketos explicitly"
        )
    return candidate.resolve()


def _find_python(explicit: Path | None, ketos: Path) -> Path:
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT / candidate
    else:
        sibling = ketos.with_name("python")
        candidate = sibling if sibling.is_file() else Path(sys.executable)
    if not candidate.is_file():
        raise FileNotFoundError(f"Python executable not found: {candidate}")
    # Do not resolve a virtual-environment Python symlink. CPython discovers
    # pyvenv.cfg from the invoked launcher path; calling its resolved base
    # interpreter bypasses the venv and loses Kraken/site-packages.
    return Path(os.path.abspath(candidate))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert, decode, and competition-score every Kraken checkpoint "
            "from one completed Model22 fold"
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input", dest="input_csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--ketos", type=Path)
    parser.add_argument("--python", type=Path, help="Kraken environment's Python")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resumable plan without creating or changing artifacts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="reconvert and reinfer even when matching completed artifacts exist",
    )
    parser.add_argument(
        "--allow-incomplete-training",
        action="store_true",
        help=(
            "evaluate without Kraken's final best_*.safetensors marker; only use "
            "after confirming the training process has stopped"
        ),
    )
    args = parser.parse_args()

    ketos = _find_ketos(args.ketos)
    python = _find_python(args.python, ketos)
    options = EvaluationOptions(
        checkpoint_dir=_project_path(args.checkpoint_dir),
        input_csv=_project_path(args.input_csv),
        output_dir=_project_path(args.output_dir),
        config=_project_path(args.config),
        fold=args.fold,
        ketos=ketos,
        python=python,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        force=args.force,
        allow_incomplete_training=args.allow_incomplete_training,
    )
    evaluate_checkpoints(options)


if __name__ == "__main__":
    main()
