"""Opt-in, resumable PP-OCR CTC probability export for Model29.

Kraken's bbox ``ocr_record.logits`` currently contains decoded tuples instead
of the full emission matrix.  This utility captures the tensor at the supported
decoder callback seam and then delegates to Kraken's stock greedy decoder.  It
does not modify ``road_htr.kraken_cli.infer`` or any installed Kraken file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from .kraken_cli import _make_batch_segmentation
from .preprocess import load_rgb
from .schema import ImageIndex, SchemaError, read_table
from .vlm_data import apply_variant


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_MODEL22_OUTPUT = (PROJECT_ROOT / "outputs" / "model22").resolve()
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExportOptions:
    config: Path
    input_csv: Path
    checkpoint: Path
    output_dir: Path
    fold: int | None
    training_manifest: Path | None = None
    conversion_receipt: Path | None = None
    training_excluded_folds: tuple[int, ...] | None = None
    dry_run: bool = False


def project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def stable_sha256(path: str | Path) -> str:
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


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


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "model29" / "export_probabilities.py",
        PROJECT_ROOT / "src" / "road_htr" / "kraken_cli.py",
        PROJECT_ROOT / "src" / "road_htr" / "preprocess.py",
        PROJECT_ROOT / "src" / "road_htr" / "schema.py",
        PROJECT_ROOT / "src" / "road_htr" / "vlm_data.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(stable_sha256(path)))
    return digest.hexdigest()


class CapturingGreedyDecoder:
    """Capture a single line's full probabilities and delegate unchanged."""

    def __init__(self, delegate: Callable[..., Any]):
        self.delegate = delegate
        self._capture: Any | None = None

    @staticmethod
    def _dimensions(value: Any) -> int:
        if hasattr(value, "dim"):
            return int(value.dim())
        dimensions = 0
        current = value
        while isinstance(current, (list, tuple)):
            dimensions += 1
            current = current[0] if current else None
        return dimensions

    def __call__(self, outputs: Any, seq_lens: Any = None) -> Any:
        dimensions = self._dimensions(outputs)
        if dimensions == 2:
            batch_outputs = outputs.unsqueeze(0) if hasattr(outputs, "unsqueeze") else [outputs]
        elif dimensions == 3:
            batch_outputs = outputs
        else:
            raise ValueError("Kraken CTC outputs must have shape [C,T] or [N,C,T]")
        batch_size = int(batch_outputs.shape[0]) if hasattr(batch_outputs, "shape") else len(batch_outputs)
        if batch_size != 1:
            raise ValueError(
                "Model29 probability capture requires batch_size=1 for exact ID mapping"
            )
        class_time = batch_outputs[0]
        time_steps = (
            int(class_time.shape[-1])
            if hasattr(class_time, "shape")
            else len(class_time[0])
        )
        if seq_lens is not None:
            length = int(seq_lens[0])
        else:
            length = time_steps
        if length <= 0 or length > time_steps:
            raise ValueError(
                f"invalid Kraken output length {length} for {time_steps} timesteps"
            )

        if hasattr(class_time, "detach"):
            # Capture [time, classes] float32 on CPU.  ``clone`` makes the
            # artifact independent of Kraken's mutable ``net.outputs`` field.
            self._capture = (
                class_time[..., :length]
                .detach()
                .float()
                .cpu()
                .transpose(0, 1)
                .contiguous()
                .clone()
            )
        else:
            self._capture = [
                [float(class_time[class_index][time_index]) for class_index in range(len(class_time))]
                for time_index in range(length)
            ]
        # Passing the original objects, including padded lengths, is important:
        # the wrapper must not change Kraken's ordinary decoded prediction.
        return self.delegate(outputs, seq_lens)

    def take(self) -> Any:
        if self._capture is None:
            raise RuntimeError("Kraken did not invoke the probability-capture decoder")
        capture = self._capture
        self._capture = None
        return capture


def _codec_labels(codec: Any, output_classes: int) -> list[str | None]:
    """Return exact class-index mapping, rejecting unsupported compound codes."""

    labels: list[str | None] = [None] * output_classes
    if not isinstance(getattr(codec, "c2l", None), dict):
        raise ValueError("recognition model has no usable CTC codec")
    for text, raw_indices in codec.c2l.items():
        indices = [int(value) for value in raw_indices]
        if len(indices) != 1:
            raise ValueError(
                "Model29 requires one CTC class per codec text; "
                f"{text!r} maps to {indices}"
            )
        index = indices[0]
        if index <= 0 or index >= output_classes:
            raise ValueError(f"codec class {index} is outside [1,{output_classes - 1}]")
        if labels[index] is not None:
            raise ValueError(f"multiple codec texts map to CTC class {index}")
        labels[index] = str(text)
    missing = [index for index, value in enumerate(labels[1:], start=1) if value is None]
    if missing:
        raise ValueError(f"codec does not define contiguous CTC classes: {missing[:10]}")
    return labels


def _validate_output_location(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved == PROTECTED_MODEL22_OUTPUT or PROTECTED_MODEL22_OUTPUT in resolved.parents:
        raise ValueError("Model29 probability artifacts must not be written under outputs/model22")


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read Model29 config {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("Model29 configuration must be a mapping")
    return config


def _export_runtime(config: dict[str, Any]) -> dict[str, Any]:
    runtime = config.get("probability_export", {})
    if not isinstance(runtime, dict):
        raise ValueError("probability_export must be a mapping")
    values = {
        "accelerator": str(runtime.get("accelerator", "cpu")),
        "device": str(runtime.get("device", "auto")),
        "precision": str(runtime.get("precision", "32-true")),
        "threads": int(runtime.get("threads", 4)),
        "line_workers": int(runtime.get("line_workers", 0)),
        "padding": int(runtime.get("padding", 16)),
        "temperature": float(runtime.get("temperature", 1.0)),
        "progress_every": int(runtime.get("progress_every", 25)),
        "batch_size": 1,
    }
    if values["accelerator"] != "cpu":
        raise ValueError("Model29 probability export is intentionally CPU-only")
    if values["precision"] != "32-true":
        raise ValueError("Model29 probability export requires precision 32-true")
    if values["line_workers"] != 0:
        raise ValueError("Model29 requires line_workers=0 for deterministic ID capture")
    if values["threads"] <= 0 or values["padding"] < 0:
        raise ValueError("invalid probability-export threads or padding")
    if not math.isfinite(values["temperature"]) or values["temperature"] <= 0:
        raise ValueError("probability-export temperature must be positive")
    if values["progress_every"] <= 0:
        raise ValueError("probability-export progress_every must be positive")
    return values


def _select_rows(
    input_csv: Path, fold: int | None
) -> tuple[Any, list[dict[str, str]], list[str]]:
    table = read_table(input_csv)
    if fold is not None and "fold" not in table.fieldnames:
        raise SchemaError("--fold requires an input CSV with a fold column")
    rows = [
        row for row in table.rows if fold is None or int(row["fold"]) == fold
    ]
    if not rows:
        raise SchemaError(f"input contains no rows for fold {fold}")
    identifiers = [row[table.id_column].strip() for row in rows]
    return table, rows, identifiers


def _enforce_oof_contract(
    config: dict[str, Any], input_csv: Path, fold: int | None, rows: int
) -> None:
    if fold is None:
        return
    contract = config.get("oof_contract", {})
    if not isinstance(contract, dict):
        raise ValueError("oof_contract must be a mapping")
    expected_hash = contract.get("input_sha256")
    expected_total = contract.get("total_rows")
    expected_fold = contract.get("fold_rows")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("oof_contract.input_sha256 must be a SHA-256 digest")
    table = read_table(input_csv)
    if stable_sha256(input_csv) != expected_hash:
        raise SchemaError("OOF input SHA-256 does not match Model29's pinned contract")
    if len(table.rows) != expected_total or rows != expected_fold:
        raise SchemaError(
            "OOF row contract mismatch: "
            f"total={len(table.rows)} (expected {expected_total}), "
            f"fold={rows} (expected {expected_fold})"
        )


def _checkpoint_provenance(
    options: ExportOptions,
    input_csv: Path,
    input_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Fail closed on fold-model lineage for any artifact claimed as OOF."""

    if options.fold is None and options.training_manifest is None and options.conversion_receipt is None:
        return {"kind": "unverified_non_oof", "training_excluded_folds": []}
    if options.training_manifest is None or options.conversion_receipt is None:
        raise ValueError(
            "--training-manifest and --conversion-receipt are both required "
            "for fold probability export"
        )
    training_manifest_path = project_path(options.training_manifest)
    conversion_receipt_path = project_path(options.conversion_receipt)
    training_manifest = _read_json(training_manifest_path)
    conversion_receipt = _read_json(conversion_receipt_path)
    if training_manifest is None or conversion_receipt is None:
        raise ValueError("checkpoint provenance JSON is missing or invalid")
    preparation = training_manifest.get("preparation")
    run_config = training_manifest.get("config")
    if not isinstance(preparation, dict) or not isinstance(run_config, dict):
        raise ValueError("training run manifest lacks config/preparation")
    raw_excluded_folds = preparation.get("excluded_folds")
    if raw_excluded_folds is None:
        raw_fold = preparation.get("fold")
        actual_excluded_folds = () if raw_fold is None else (int(raw_fold),)
    elif isinstance(raw_excluded_folds, list):
        actual_excluded_folds = tuple(sorted(int(value) for value in raw_excluded_folds))
    else:
        raise ValueError("training preparation.excluded_folds must be a list")
    expected_excluded_folds = options.training_excluded_folds
    if expected_excluded_folds is None:
        expected_excluded_folds = () if options.fold is None else (options.fold,)
    expected_excluded_folds = tuple(sorted(set(expected_excluded_folds)))
    if actual_excluded_folds != expected_excluded_folds:
        raise ValueError(
            "checkpoint training-fold exclusion mismatch: "
            f"manifest={actual_excluded_folds}, expected={expected_excluded_folds}"
        )
    if options.fold is not None and options.fold not in actual_excluded_folds:
        raise ValueError(
            f"checkpoint trained on exported fold {options.fold}; artifact is not OOF"
        )

    configured_train_csv = run_config.get("train_csv")
    if not isinstance(configured_train_csv, str):
        raise ValueError("training run config is missing train_csv")
    if stable_sha256(project_path(configured_train_csv)) != input_sha256:
        raise ValueError("training run used a different canonical input CSV")
    output_template = run_config.get("output_dir")
    if not isinstance(output_template, str):
        raise ValueError("training run config is missing output_dir")
    manifest_parent = training_manifest_path.parent.resolve()
    if "{fold}" in output_template and preparation.get("fold") is not None:
        expected_run_dir = project_path(
            output_template.format(fold=int(preparation["fold"]))
        )
        if expected_run_dir != manifest_parent:
            raise ValueError("training manifest is outside its configured output directory")

    if conversion_receipt.get("schema_version") != 1:
        raise ValueError("unsupported or missing conversion receipt schema")
    recorded_weights = conversion_receipt.get("weights")
    if not isinstance(recorded_weights, str) or project_path(recorded_weights) != checkpoint:
        raise ValueError("conversion receipt does not name the exported weights")
    if conversion_receipt.get("weights_sha256") != checkpoint_sha256:
        raise ValueError("converted checkpoint SHA-256 differs from its receipt")
    source_checkpoint_value = conversion_receipt.get("checkpoint")
    source_checkpoint_sha256 = conversion_receipt.get("checkpoint_sha256")
    if not isinstance(source_checkpoint_value, str) or not isinstance(
        source_checkpoint_sha256, str
    ):
        raise ValueError("conversion receipt lacks source checkpoint provenance")
    source_checkpoint = project_path(source_checkpoint_value)
    if source_checkpoint.parent != manifest_parent:
        raise ValueError("converted checkpoint came from a different training run")
    if stable_sha256(source_checkpoint) != source_checkpoint_sha256:
        raise ValueError("source training checkpoint SHA-256 differs from its receipt")
    return {
        "kind": "model22_converted_checkpoint",
        "training_manifest": str(training_manifest_path),
        "training_manifest_sha256": stable_sha256(training_manifest_path),
        "conversion_receipt": str(conversion_receipt_path),
        "conversion_receipt_sha256": stable_sha256(conversion_receipt_path),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "training_excluded_folds": list(actual_excluded_folds),
        "exported_fold": options.fold,
        "input_csv": str(input_csv),
    }


def _receipt_valid(
    output_dir: Path,
    index: int,
    identifier: str,
    class_count: int,
    image_path: Path,
    image_sha256: str,
) -> dict[str, Any] | None:
    import numpy as np

    receipt_path = output_dir / "receipts" / f"{index:06d}.json"
    receipt = _read_json(receipt_path)
    expected_file = f"probabilities/{index:06d}.npz"
    if receipt is None:
        return None
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("index") != index
        or receipt.get("id") != identifier
        or receipt.get("file") != expected_file
        or receipt.get("dtype") != "float32"
        or receipt.get("standalone_greedy_verified") is not True
        or receipt.get("source_image") != str(image_path)
        or receipt.get("source_image_sha256") != image_sha256
    ):
        return None
    shape = receipt.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or shape[1] != class_count
        or not isinstance(shape[0], int)
        or shape[0] <= 0
    ):
        return None
    shard = output_dir / expected_file
    expected_sha = receipt.get("sha256")
    if not shard.is_file() or not isinstance(expected_sha, str):
        return None
    if stable_sha256(shard) != expected_sha:
        return None
    try:
        with np.load(shard, allow_pickle=False) as archive:
            if archive.files != ["probabilities"]:
                return None
            probabilities = archive["probabilities"]
    except (OSError, ValueError):
        return None
    if list(probabilities.shape) != shape or str(probabilities.dtype) != "float32":
        return None
    if not np.isfinite(probabilities).all():
        return None
    if probabilities.min() < 0 or probabilities.max() > 1:
        return None
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=2e-4):
        return None
    return receipt


def _write_probability_shard(
    output_dir: Path,
    index: int,
    identifier: str,
    raw_probabilities: Any,
    greedy_prediction: str,
    class_count: int,
    codec_labels: Sequence[str | None],
    image_path: Path,
    image_sha256: str,
) -> dict[str, Any]:
    import numpy as np

    probabilities = np.asarray(raw_probabilities, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != class_count:
        raise RuntimeError(
            f"captured CTC shape {probabilities.shape} does not match "
            f"[time,{class_count}]"
        )
    if probabilities.shape[0] <= 0 or not np.isfinite(probabilities).all():
        raise RuntimeError("captured CTC probabilities are empty or non-finite")
    if probabilities.min() < 0 or probabilities.max() > 1:
        raise RuntimeError("captured CTC values fall outside [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=2e-4):
        raise RuntimeError("captured CTC class probabilities do not sum to one")
    standalone_greedy = _ctc_greedy_text(probabilities, codec_labels)
    if standalone_greedy != greedy_prediction:
        raise RuntimeError(
            "standalone CTC greedy text differs from Kraken's record; "
            "codec or BiDi ordering is not safe for Model29: "
            f"{standalone_greedy!r} != {greedy_prediction!r}"
        )

    shard = output_dir / "probabilities" / f"{index:06d}.npz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_name(f".{shard.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, probabilities=probabilities)
        shard_sha256 = stable_sha256(temporary)
        os.replace(temporary, shard)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "index": index,
        "id": identifier,
        "file": str(shard.relative_to(output_dir)),
        "sha256": shard_sha256,
        "shape": list(probabilities.shape),
        "dtype": "float32",
        "greedy_prediction": str(greedy_prediction),
        "standalone_greedy_verified": True,
        "source_image": str(image_path),
        "source_image_sha256": image_sha256,
    }
    _atomic_json(output_dir / "receipts" / f"{index:06d}.json", receipt)
    return receipt


def _ctc_greedy_text(
    probabilities: Sequence[Sequence[float]],
    codec_labels: Sequence[str | None],
) -> str:
    """Decode class probabilities independently for exporter parity checks."""

    if not codec_labels or codec_labels[0] is not None:
        raise ValueError("codec class zero must be the CTC blank")
    previous = 0
    output: list[str] = []
    for raw_row in probabilities:
        values = raw_row.tolist() if hasattr(raw_row, "tolist") else list(raw_row)
        if len(values) != len(codec_labels):
            raise ValueError("probability row and codec class counts differ")
        token = max(range(len(values)), key=lambda index: float(values[index]))
        if token != 0 and token != previous:
            text = codec_labels[token]
            if not isinstance(text, str) or not text:
                raise ValueError(f"CTC class {token} has no codec text")
            output.append(text)
        previous = token
    return "".join(output)


def _completed_export(
    output_dir: Path,
    signature: dict[str, Any],
    identifiers: Sequence[str],
    class_count: int,
    image_paths: Sequence[Path],
    image_hashes: Sequence[str],
) -> bool:
    manifest = _read_json(output_dir / "manifest.json")
    index_payload = _read_json(output_dir / "index.json")
    if (
        manifest is None
        or index_payload is None
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("signature") != signature
        or manifest.get("rows") != len(identifiers)
        or index_payload.get("schema_version") != SCHEMA_VERSION
    ):
        return False
    entries = index_payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(identifiers):
        return False
    if manifest.get("index_sha256") != stable_sha256(output_dir / "index.json"):
        return False
    for index, (identifier, image_path, image_sha256) in enumerate(
        zip(identifiers, image_paths, image_hashes, strict=True)
    ):
        receipt = _receipt_valid(
            output_dir,
            index,
            identifier,
            class_count,
            image_path,
            image_sha256,
        )
        if receipt is None or entries[index] != {
            **receipt,
            "receipt_sha256": stable_sha256(
                output_dir / "receipts" / f"{index:06d}.json"
            ),
        }:
            return False
    return True


def export_probabilities(options: ExportOptions) -> dict[str, Any]:
    """Export one full softmax matrix per selected line, resumably and on CPU."""

    from kraken.configs import RecognitionInferenceConfig
    from kraken.lib.ctc_decoder import greedy_decoder
    from kraken.tasks import RecognitionTaskModel

    config_path = project_path(options.config)
    input_csv = project_path(options.input_csv)
    checkpoint = project_path(options.checkpoint)
    output_dir = project_path(options.output_dir)
    _validate_output_location(output_dir)
    for path in (config_path, input_csv, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if options.fold is not None and options.fold < 0:
        raise ValueError("fold must be non-negative")

    config = _load_config(config_path)
    runtime = _export_runtime(config)
    table, selected, identifiers = _select_rows(input_csv, options.fold)
    _enforce_oof_contract(config, input_csv, options.fold, len(selected))
    image_dir = project_path(config["image_dir"])
    image_variant = str(config.get("image_variant", "full"))
    image_index = ImageIndex(image_dir)
    image_paths = [image_index.resolve(identifier) for identifier in identifiers]
    image_hashes = [stable_sha256(path) for path in image_paths]
    input_sha256 = stable_sha256(input_csv)
    config_sha256 = stable_sha256(config_path)
    checkpoint_sha256 = stable_sha256(checkpoint)
    provenance = _checkpoint_provenance(
        options,
        input_csv,
        input_sha256,
        checkpoint,
        checkpoint_sha256,
    )

    model = RecognitionTaskModel.load_model(checkpoint)
    if type(model.net).__name__ != "PPOCRv6Model":
        raise ValueError(
            f"Model29 requires a PP-OCRv6 checkpoint, got {type(model.net).__name__}"
        )
    output_classes = int(model.net.num_classes)
    codec_labels = _codec_labels(model.net.codec, output_classes)
    signature = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "implementation_sha256": _implementation_sha256(),
        "fold": options.fold,
        "ids_sha256": _json_sha256(identifiers),
        "rows": len(identifiers),
        "image_dir": str(image_dir),
        "images_sha256": _json_sha256(
            [
                {"id": identifier, "path": str(path), "sha256": digest}
                for identifier, path, digest in zip(
                    identifiers, image_paths, image_hashes, strict=True
                )
            ]
        ),
        "image_variant": image_variant,
        "runtime": runtime,
        "codec_labels": codec_labels,
        "checkpoint_provenance": provenance,
    }
    if options.dry_run:
        return {
            "dry_run": True,
            "output_dir": str(output_dir),
            "rows": len(identifiers),
            "fold": options.fold,
            "classes": output_classes,
            "signature": signature,
        }

    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"probability output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        state = _read_json(output_dir / "export_state.json")
        if state is None or state.get("signature") != signature:
            raise FileExistsError(
                f"refusing nonempty or incompatible probability directory: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            output_dir / "export_state.json",
            {
                "schema_version": SCHEMA_VERSION,
                "signature": signature,
                "status": "in_progress",
            },
        )
    if _completed_export(
        output_dir,
        signature,
        identifiers,
        output_classes,
        image_paths,
        image_hashes,
    ):
        print(f"reuse complete probability export: {output_dir}", flush=True)
        return _read_json(output_dir / "manifest.json") or {}

    capture_decoder = CapturingGreedyDecoder(greedy_decoder)
    inference_config = RecognitionInferenceConfig(
        accelerator=runtime["accelerator"],
        device=runtime["device"],
        precision=runtime["precision"],
        batch_size=1,
        num_threads=runtime["threads"],
        num_line_workers=0,
        padding=runtime["padding"],
        temperature=runtime["temperature"],
        bidi_reordering=True,
        raise_on_error=True,
        decoder=capture_decoder,
    )
    model.net.prepare_for_inference(inference_config)
    started = time.monotonic()
    receipts: list[dict[str, Any]] = []
    for index, (row, identifier, image_path, image_sha256) in enumerate(
        zip(selected, identifiers, image_paths, image_hashes, strict=True)
    ):
        reusable = _receipt_valid(
            output_dir,
            index,
            identifier,
            output_classes,
            image_path,
            image_sha256,
        )
        if reusable is not None:
            receipts.append(reusable)
            continue
        image = apply_variant(load_rgb(image_path), image_variant)
        canvas, segmentation = _make_batch_segmentation(
            [(identifier, image_path, image)]
        )
        records = list(model.net.predict(canvas, segmentation))
        if len(records) != 1 or records[0].id != identifier:
            raise RuntimeError(
                f"Kraken returned unexpected record IDs for {identifier!r}"
            )
        probabilities = capture_decoder.take()
        receipt = _write_probability_shard(
            output_dir,
            index,
            identifier,
            probabilities,
            records[0].prediction,
            output_classes,
            codec_labels,
            image_path,
            image_sha256,
        )
        receipts.append(receipt)
        completed = index + 1
        if completed % runtime["progress_every"] == 0 or completed == len(selected):
            print(
                f"exported {completed}/{len(selected)} probability matrices "
                f"in {time.monotonic() - started:.1f}s",
                flush=True,
            )

    entries = []
    for index, receipt in enumerate(receipts):
        receipt_path = output_dir / "receipts" / f"{index:06d}.json"
        entries.append({**receipt, "receipt_sha256": stable_sha256(receipt_path)})
    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "id_column": table.id_column,
        "entries": entries,
    }
    _atomic_json(output_dir / "index.json", index_payload)
    changed = {
        "input CSV": stable_sha256(input_csv) != input_sha256,
        "config": stable_sha256(config_path) != config_sha256,
        "checkpoint": stable_sha256(checkpoint) != checkpoint_sha256,
        "implementation": _implementation_sha256() != signature["implementation_sha256"],
        "source images": any(
            stable_sha256(path) != digest
            for path, digest in zip(image_paths, image_hashes, strict=True)
        ),
        "checkpoint provenance": (
            provenance.get("kind") == "model22_converted_checkpoint"
            and any(
                stable_sha256(Path(str(provenance[path_key])))
                != provenance[hash_key]
                for path_key, hash_key in (
                    ("training_manifest", "training_manifest_sha256"),
                    ("conversion_receipt", "conversion_receipt_sha256"),
                    ("source_checkpoint", "source_checkpoint_sha256"),
                )
            )
        ),
    }
    if any(changed.values()):
        names = [name for name, did_change in changed.items() if did_change]
        raise RuntimeError(
            "probability-export input changed while running: " + ", ".join(names)
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "signature": signature,
        "input_csv": str(input_csv),
        "checkpoint": str(checkpoint),
        "checkpoint_provenance": provenance,
        "fold": options.fold,
        "rows": len(identifiers),
        "blank_class": 0,
        "codec_labels": codec_labels,
        "matrix_layout": "time,classes",
        "matrix_values": "softmax probabilities after configured temperature",
        "storage": "one compressed float32 NPZ per ID",
        "index_sha256": stable_sha256(output_dir / "index.json"),
        "seconds": time.monotonic() - started,
        "python": sys.version,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    _atomic_json(
        output_dir / "export_state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "signature": signature,
            "status": "complete",
            "manifest_sha256": stable_sha256(output_dir / "manifest.json"),
        },
    )

    if not _completed_export(
        output_dir,
        signature,
        identifiers,
        output_classes,
        image_paths,
        image_hashes,
    ):
        raise RuntimeError("completed probability export failed its own integrity check")
    return manifest
