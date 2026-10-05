"""Fold-local, provenance-bound Models 11--15 postprocessing.

Unlike :mod:`crossfit_postprocess`, this workflow receives predictions from one
specific OCR fold model on *all* training rows.  For held-out fold ``f``, every
recipe is fitted from that same model's predictions and labels on folds other
than ``f`` and is then applied once to ``f``.  Scores outside ``f`` are never
reported because those OCR predictions are in-sample rather than OOF.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .crossfit_postprocess import STAGES, StageDefinition
from .metrics import competition_error
from .schema import SchemaError, predictions_by_id, read_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PROTOCOL = "same-fold-model train-side fit then single heldout apply"
FIT_SIDE_BLANK_FILL = "?"
SUPPORTED_MODEL_FAMILIES = ("model22", "model36", "model71")
RELOCATABLE_PROJECT_ROOTS = (Path("/workspace/road"),)
PROTECTED_OUTPUT_ROOTS = tuple(
    (PROJECT_ROOT / "outputs" / name).resolve() for name in SUPPORTED_MODEL_FAMILIES
)
RESERVED_ACTIVE_OUTPUT = (
    PROJECT_ROOT / "outputs" / "model35" / "fold0_epoch07_train_side_fit"
).resolve()
_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)-.+\.ckpt$")


@dataclass(frozen=True)
class FoldModelData:
    id_column: str
    ids: tuple[str, ...]
    references: tuple[str, ...]
    folds: tuple[int, ...]
    predictions: tuple[str, ...]


@dataclass(frozen=True)
class FoldPipelineResult:
    heldout_ids: tuple[str, ...]
    stage_predictions: Mapping[str, tuple[str, ...]]
    stage_changes: Mapping[str, tuple[dict[str, Any], ...]]
    stage_fits: Mapping[str, Mapping[str, Any]]
    supervised_mapping: Mapping[str, Mapping[str, Any]]


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for identifier in sorted(ids):
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return value


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


def _atomic_prediction_csv(
    path: Path, id_column: str, ids: Sequence[str], values: Sequence[str]
) -> None:
    if len(ids) != len(values):
        raise ValueError("prediction row count mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[id_column, "prediction"])
            writer.writeheader()
            for identifier, value in zip(ids, values, strict=True):
                writer.writerow({id_column: identifier, "prediction": value})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_output_dir(path: str | Path) -> Path:
    output = _absolute(path)
    if output == RESERVED_ACTIVE_OUTPUT or RESERVED_ACTIVE_OUTPUT in output.parents:
        raise ValueError(f"reserved active Model35 output may not be modified: {output}")
    if any(output == root or root in output.parents for root in PROTECTED_OUTPUT_ROOTS):
        raise ValueError(f"Model35 may not write into OCR/GPU artifacts: {output}")
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _prediction_column(table: Any, path: Path) -> str:
    if table.target_column is not None:
        return table.target_column
    columns = [name for name in table.fieldnames if name != table.id_column]
    if len(columns) != 1:
        raise SchemaError(f"prediction CSV must have one value column: {path}")
    return columns[0]


def load_fold_model_data(
    truth_path: str | Path,
    predictions_path: str | Path,
    *,
    allow_blank_predictions: bool = False,
) -> FoldModelData:
    truth_file = _absolute(truth_path)
    predictions_file = _absolute(predictions_path)
    truth = read_table(truth_file, require_target=True)
    predictions = read_table(predictions_file)
    if "fold" not in truth.fieldnames:
        raise SchemaError("truth CSV must contain a fold column")
    assert truth.target_column is not None
    ids = tuple(row[truth.id_column].strip() for row in truth.rows)
    references = tuple(row[truth.target_column] for row in truth.rows)
    try:
        folds = tuple(int(row["fold"]) for row in truth.rows)
    except (TypeError, ValueError) as error:
        raise SchemaError("truth fold values must be integers") from error
    if len(set(folds)) < 2:
        raise SchemaError("fold-model postprocessing requires at least two folds")
    prediction_column = _prediction_column(predictions, predictions_file)
    by_id = dict(predictions_by_id(predictions, prediction_column))
    expected = set(ids)
    if set(by_id) != expected:
        raise SchemaError(
            f"all-row prediction ID mismatch: missing={sorted(expected-set(by_id))[:5]}, "
            f"extra={sorted(set(by_id)-expected)[:5]}"
        )
    if not allow_blank_predictions and any(not by_id[identifier] for identifier in ids):
        raise SchemaError("all-row prediction CSV contains a blank prediction")
    return FoldModelData(
        truth.id_column,
        ids,
        references,
        folds,
        tuple(by_id[identifier] for identifier in ids),
    )


def _prediction_map(path: Path) -> dict[str, str]:
    table = read_table(path)
    column = _prediction_column(table, path)
    values = dict(predictions_by_id(table, column))
    if any(not value for value in values.values()):
        raise SchemaError(f"blank prediction in {path}")
    return values


def _manifest_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest is missing {label}")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() or candidate.exists():
        return _absolute(candidate)
    for source_root in RELOCATABLE_PROJECT_ROOTS:
        try:
            relative = candidate.relative_to(source_root)
        except ValueError:
            continue
        relocated = (PROJECT_ROOT / relative).resolve()
        if (
            relocated == PROJECT_ROOT
            or PROJECT_ROOT in relocated.parents
        ) and relocated.exists():
            return relocated
    return candidate.resolve()


def _require_manifest_value(
    manifest: Mapping[str, Any], key: str, expected: Any, label: str
) -> None:
    if manifest.get(key) != expected:
        raise ValueError(
            f"{label} {key} mismatch: {manifest.get(key)!r} != {expected!r}"
        )


def _manifest_ids(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                raise SchemaError(f"blank line at {path}:{line_number}")
            identifier = Path(value).stem
            if not identifier or identifier in ids:
                raise SchemaError(f"blank/duplicate manifest ID at {path}:{line_number}")
            ids.append(identifier)
    return tuple(ids)


def verify_fold_model_lineage(
    truth_path: str | Path,
    predictions_path: str | Path,
    inference_manifest_path: str | Path,
    training_manifest_path: str | Path,
    heldout_oof_predictions_path: str | Path,
    oof_evaluation_receipt_path: str | Path,
    heldout_fold: int,
    checkpoint_epoch: int,
    model_family: str,
) -> dict[str, Any]:
    """Fail closed unless the all-row run is bound to a genuine OOF fold model."""

    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            "model_family must be one of " + ", ".join(SUPPORTED_MODEL_FAMILIES)
        )
    truth = _absolute(truth_path)
    predictions = _absolute(predictions_path)
    inference_manifest_file = _absolute(inference_manifest_path)
    training_manifest_file = _absolute(training_manifest_path)
    heldout_predictions = _absolute(heldout_oof_predictions_path)
    evaluation_receipt_file = _absolute(oof_evaluation_receipt_path)
    if inference_manifest_file != predictions.with_suffix(".manifest.json"):
        raise ValueError("all-row inference manifest must be the prediction CSV sidecar")
    if evaluation_receipt_file != heldout_predictions.with_suffix(".evaluation.json"):
        raise ValueError("OOF evaluation receipt must be the heldout CSV sidecar")
    heldout_manifest_file = heldout_predictions.with_suffix(".manifest.json")

    # A failed decode can be blank on a fit-side row.  It must never invalidate
    # the signed heldout slice or silently enter a fitted recipe.  We therefore
    # locate blanks here, after binding IDs/folds, reject heldout blanks, and
    # expose fit-side blanks for a fixed label-independent fill in run_fold().
    data = load_fold_model_data(truth, predictions, allow_blank_predictions=True)
    if heldout_fold not in set(data.folds):
        raise ValueError(f"truth has no fold {heldout_fold}")
    truth_sha256 = _sha256(truth)
    expected_heldout_ids = {
        identifier
        for identifier, fold in zip(data.ids, data.folds, strict=True)
        if fold == heldout_fold
    }
    expected_training_ids = set(data.ids) - expected_heldout_ids
    blank_ids = {
        identifier
        for identifier, value in zip(data.ids, data.predictions, strict=True)
        if not value
    }
    heldout_blank_ids = sorted(blank_ids & expected_heldout_ids)
    if heldout_blank_ids:
        raise SchemaError(
            "all-row prediction CSV contains a blank prediction on heldout fold: "
            f"{heldout_blank_ids[:5]}"
        )
    fit_side_blank_ids = sorted(blank_ids & expected_training_ids)

    inference = _read_json(inference_manifest_file)
    heldout_inference = _read_json(heldout_manifest_file)
    training = _read_json(training_manifest_file)
    evaluation = _read_json(evaluation_receipt_file)
    signature = evaluation.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("OOF evaluation receipt has no signature")

    _require_manifest_value(inference, "input_sha256", truth_sha256, "all-row inference")
    _require_manifest_value(inference, "rows", len(data.ids), "all-row inference")
    _require_manifest_value(inference, "fold", None, "all-row inference")
    _require_manifest_value(inference, "preserve_internal_spaces", True, "all-row inference")
    _require_manifest_value(heldout_inference, "input_sha256", truth_sha256, "OOF inference")
    _require_manifest_value(
        heldout_inference, "rows", len(expected_heldout_ids), "OOF inference"
    )
    _require_manifest_value(heldout_inference, "fold", heldout_fold, "OOF inference")
    _require_manifest_value(
        heldout_inference, "preserve_internal_spaces", True, "OOF inference"
    )

    comparable_settings = (
        "model_sha256",
        "config_sha256",
        "image_variant",
        "constrain_charset",
        "preserve_internal_spaces",
    )
    for key in comparable_settings:
        if inference.get(key) != heldout_inference.get(key):
            raise ValueError(f"all-row and OOF inference differ in {key}")
    model_sha256 = inference.get("model_sha256")
    if not isinstance(model_sha256, str) or len(model_sha256) != 64:
        raise ValueError("inference manifest has no valid model SHA-256")
    model_path = _manifest_path(inference.get("model"), "model")
    if _sha256(model_path) != model_sha256:
        raise ValueError("inference model bytes do not match model_sha256")
    resolved_config = _manifest_path(inference.get("resolved_config"), "resolved_config")
    if resolved_config.parent.name != model_family:
        raise ValueError("model_family does not match inference resolved_config")
    if _sha256(resolved_config) != inference.get("config_sha256"):
        raise ValueError("resolved inference config bytes do not match config_sha256")

    training_sha256 = _sha256(training_manifest_file)
    preparation = training.get("preparation")
    training_config = training.get("config")
    if not isinstance(preparation, dict) or not isinstance(training_config, dict):
        raise ValueError("training manifest lacks config/preparation")
    _require_manifest_value(preparation, "fold", heldout_fold, "training preparation")
    raw_excluded = preparation.get("excluded_ids") or []
    if not isinstance(raw_excluded, list):
        raise ValueError("training excluded_ids must be a list")
    excluded = {str(value) for value in raw_excluded}
    expected_effective_validation = expected_heldout_ids - excluded
    _require_manifest_value(
        preparation,
        "validation_rows",
        len(expected_effective_validation),
        "training preparation",
    )
    training_resolved_config = _manifest_path(
        training.get("resolved_config"), "training resolved_config"
    )
    if training_resolved_config.parent.name != model_family:
        raise ValueError("model_family does not match training manifest")
    configured_truth = _manifest_path(training_config.get("train_csv"), "train_csv")
    if _sha256(configured_truth) != truth_sha256:
        raise ValueError("training manifest points to a different truth CSV")
    training_options = training_config.get("training")
    if not isinstance(training_options, dict):
        raise ValueError("training manifest has no training options")
    epochs = training_options.get("epochs")
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < checkpoint_epoch + 1
        or training_options.get("stop") != "fixed"
    ):
        raise ValueError("checkpoint epoch is incompatible with the fixed training run")

    train_manifest = _manifest_path(preparation.get("train_manifest"), "train_manifest")
    validation_manifest = _manifest_path(
        preparation.get("validation_manifest"), "validation_manifest"
    )
    train_ids = set(_manifest_ids(train_manifest))
    validation_ids = set(_manifest_ids(validation_manifest))
    if validation_ids != expected_effective_validation:
        raise ValueError(
            "training validation manifest is not exactly the eligible heldout fold"
        )
    expected_effective_training = expected_training_ids - excluded
    if train_ids != expected_effective_training:
        raise ValueError("training manifest is not exactly the non-heldout eligible rows")
    if preparation.get("train_rows") != len(train_ids):
        raise ValueError("training manifest row count is inconsistent")

    _require_manifest_value(signature, "fold", heldout_fold, "OOF receipt")
    _require_manifest_value(signature, "input_sha256", truth_sha256, "OOF receipt")
    _require_manifest_value(
        signature, "expected_rows", len(expected_heldout_ids), "OOF receipt"
    )
    _require_manifest_value(
        signature, "training_manifest_sha256", training_sha256, "OOF receipt"
    )
    _require_manifest_value(
        signature, "config_sha256", inference.get("config_sha256"), "OOF receipt"
    )
    _require_manifest_value(
        signature,
        "preserve_internal_spaces",
        inference.get("preserve_internal_spaces"),
        "OOF receipt",
    )
    _require_manifest_value(
        signature,
        "constrain_charset",
        inference.get("constrain_charset"),
        "OOF receipt",
    )
    _require_manifest_value(evaluation, "weights_sha256", model_sha256, "OOF receipt")
    _require_manifest_value(
        evaluation,
        "inference_manifest_sha256",
        _sha256(heldout_manifest_file),
        "OOF receipt",
    )
    _require_manifest_value(
        evaluation,
        "predictions_sha256",
        _sha256(heldout_predictions),
        "OOF receipt",
    )
    checkpoint_name = signature.get("checkpoint")
    match = _CHECKPOINT_RE.match(str(checkpoint_name))
    if match is None or int(match.group(1)) != checkpoint_epoch:
        raise ValueError("OOF receipt checkpoint does not match fixed checkpoint_epoch")

    heldout_map = _prediction_map(heldout_predictions)
    if set(heldout_map) != expected_heldout_ids:
        raise SchemaError("heldout OOF prediction IDs do not match the truth fold")
    full_map = dict(zip(data.ids, data.predictions, strict=True))
    differing = [
        identifier
        for identifier in sorted(expected_heldout_ids)
        if full_map[identifier] != heldout_map[identifier]
    ]
    all_row_batch_size = inference.get("batch_size")
    heldout_batch_size = heldout_inference.get("batch_size")
    if differing and all_row_batch_size == heldout_batch_size:
        raise ValueError(
            "all-row inference does not reproduce the signed OOF predictions "
            f"on heldout fold {heldout_fold}: {differing[:5]}"
        )

    return {
        "model_family": model_family,
        "heldout_fold": heldout_fold,
        "checkpoint_epoch_zero_based": checkpoint_epoch,
        "truth": str(truth),
        "truth_sha256": truth_sha256,
        "all_row_predictions": str(predictions),
        "all_row_predictions_sha256": _sha256(predictions),
        "all_row_inference_manifest": str(inference_manifest_file),
        "all_row_inference_manifest_sha256": _sha256(inference_manifest_file),
        "heldout_oof_predictions": str(heldout_predictions),
        "heldout_oof_predictions_sha256": _sha256(heldout_predictions),
        "heldout_oof_manifest": str(heldout_manifest_file),
        "heldout_oof_manifest_sha256": _sha256(heldout_manifest_file),
        "oof_evaluation_receipt": str(evaluation_receipt_file),
        "oof_evaluation_receipt_sha256": _sha256(evaluation_receipt_file),
        "training_manifest": str(training_manifest_file),
        "training_manifest_sha256": training_sha256,
        "model": str(model_path),
        "model_sha256": model_sha256,
        "config": str(resolved_config),
        "config_sha256": inference.get("config_sha256"),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": _sha256(train_manifest),
        "validation_manifest": str(validation_manifest),
        "validation_manifest_sha256": _sha256(validation_manifest),
        "train_ids_sha256": _ids_sha256(tuple(train_ids)),
        "heldout_ids_sha256": _ids_sha256(tuple(expected_heldout_ids)),
        "train_rows": len(train_ids),
        "heldout_rows": len(expected_heldout_ids),
        "excluded_truth_rows": sorted(expected_training_ids - train_ids),
        "excluded_validation_truth_rows": sorted(
            expected_heldout_ids - validation_ids
        ),
        "fit_side_blank_prediction_ids": fit_side_blank_ids,
        "fit_side_blank_fill": FIT_SIDE_BLANK_FILL,
        "fit_side_blank_policy": (
            "fixed label-independent '?' fill; heldout blanks are rejected"
        ),
        "all_row_inference_batch_size": all_row_batch_size,
        "heldout_oof_inference_batch_size": heldout_batch_size,
        "heldout_prediction_source": "signed OOF artifact",
        "all_row_and_oof_heldout_predictions_identical": not differing,
        "batch_sensitive_heldout_difference_count": len(differing),
        "batch_sensitive_heldout_difference_ids_sha256": _ids_sha256(differing),
        "batch_sensitive_heldout_difference_id_sample": differing[:20],
    }


def _changed_counts(
    before: Sequence[str],
    after: Sequence[str],
    operations: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[int, int]:
    return (
        sum(left != right for left, right in zip(before, after, strict=True)),
        sum(len(row) for row in operations),
    )


def fit_apply_fold_pipeline(
    data: FoldModelData,
    heldout_fold: int,
    stages: Sequence[StageDefinition] = STAGES,
) -> FoldPipelineResult:
    """Fit sequential stages on non-f rows and transform f without its labels."""

    training_indices = [
        index for index, fold in enumerate(data.folds) if fold != heldout_fold
    ]
    heldout_indices = [
        index for index, fold in enumerate(data.folds) if fold == heldout_fold
    ]
    if not training_indices or not heldout_indices:
        raise ValueError(f"cannot split heldout fold {heldout_fold}")
    training_references = [data.references[index] for index in training_indices]
    training_values = [data.predictions[index] for index in training_indices]
    heldout_values = [data.predictions[index] for index in heldout_indices]
    heldout_ids = tuple(data.ids[index] for index in heldout_indices)
    predictions: dict[str, tuple[str, ...]] = {"raw": tuple(heldout_values)}
    changes: dict[str, tuple[dict[str, Any], ...]] = {}
    fits: dict[str, Mapping[str, Any]] = {}
    supervised_mapping: Mapping[str, Mapping[str, Any]] = {}

    for stage in stages:
        fit = stage.fit(training_values, training_references)
        before = heldout_values
        heldout_values, operations = stage.transform(before, fit.state, fit.config)
        if len(heldout_values) != len(heldout_indices) or len(operations) != len(
            heldout_indices
        ):
            raise RuntimeError(f"{stage.name} returned the wrong heldout row count")
        changed_rows, changed_operations = _changed_counts(
            before, heldout_values, operations
        )
        stage_changes = tuple(
            {
                "id": identifier,
                "fold": heldout_fold,
                "before": old,
                "after": new,
                "operations": operation,
            }
            for identifier, old, new, operation in zip(
                heldout_ids, before, heldout_values, operations, strict=True
            )
            if old != new
        )
        training_report = dict(fit.report)
        fits[stage.name] = {
            "selected_config": dict(fit.config),
            "fit_rows": len(training_indices),
            "fit_folds": sorted(
                {data.folds[index] for index in training_indices}
            ),
            "heldout_fold": heldout_fold,
            "heldout_changed_rows": changed_rows,
            "heldout_changed_operations": changed_operations,
            "recipe_search_diagnostic_not_oof": training_report,
            "diagnostic_status": "TRAIN_SIDE_IN_SAMPLE_NOT_OOF_DO_NOT_CLAIM",
        }
        predictions[stage.name] = tuple(heldout_values)
        changes[stage.name] = stage_changes
        training_values = list(fit.training_predictions)
        if stage.name == "supervised_errors":
            supervised_mapping = dict(fit.state)
    return FoldPipelineResult(
        heldout_ids,
        predictions,
        changes,
        fits,
        supervised_mapping,
    )


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), Path(STAGES[0].fit.__code__.co_filename).resolve()):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def run_fold(
    truth_path: str | Path,
    predictions_path: str | Path,
    inference_manifest_path: str | Path,
    training_manifest_path: str | Path,
    heldout_oof_predictions_path: str | Path,
    oof_evaluation_receipt_path: str | Path,
    output_dir: str | Path,
    heldout_fold: int,
    checkpoint_epoch: int,
    checkpoint_policy: str,
    policy_selection_fold: int,
    model_family: str,
) -> dict[str, Any]:
    if not checkpoint_policy.strip():
        raise ValueError("checkpoint_policy cannot be blank")
    lineage = verify_fold_model_lineage(
        truth_path,
        predictions_path,
        inference_manifest_path,
        training_manifest_path,
        heldout_oof_predictions_path,
        oof_evaluation_receipt_path,
        heldout_fold,
        checkpoint_epoch,
        model_family,
    )
    raw_data = load_fold_model_data(
        truth_path, predictions_path, allow_blank_predictions=True
    )
    signed_heldout = _prediction_map(_absolute(heldout_oof_predictions_path))
    data = FoldModelData(
        raw_data.id_column,
        raw_data.ids,
        raw_data.references,
        raw_data.folds,
        tuple(
            signed_heldout[identifier]
            if fold == heldout_fold
            else (FIT_SIDE_BLANK_FILL if not value else value)
            for identifier, value, fold in zip(
                raw_data.ids, raw_data.predictions, raw_data.folds, strict=True
            )
        ),
    )
    pipeline = fit_apply_fold_pipeline(data, heldout_fold)
    output = _prepare_output_dir(output_dir)
    heldout_reference_map = dict(zip(data.ids, data.references, strict=True))
    references = [heldout_reference_map[identifier] for identifier in pipeline.heldout_ids]
    stage_order = ["raw", *(stage.name for stage in STAGES)]
    scores: dict[str, Any] = {}
    for position, stage_name in enumerate(stage_order):
        artifact = output / "predictions" / f"{position:02d}_{stage_name}_fold{heldout_fold}.csv"
        values = pipeline.stage_predictions[stage_name]
        _atomic_prediction_csv(artifact, data.id_column, pipeline.heldout_ids, values)
        scores[stage_name] = {
            "scope": f"HELDOUT_FOLD_{heldout_fold}_ONLY",
            "score": competition_error(references, values),
            "prediction_artifact": str(artifact),
            "prediction_sha256": _sha256(artifact),
        }
        if stage_name != "raw":
            changes_path = output / "changes" / f"{position:02d}_{stage_name}_fold{heldout_fold}.jsonl"
            _atomic_jsonl(changes_path, pipeline.stage_changes[stage_name])
            scores[stage_name]["change_artifact"] = str(changes_path)
            scores[stage_name]["change_artifact_sha256"] = _sha256(changes_path)
    mapping_path = output / "fit" / f"fold{heldout_fold}_supervised_mappings.json"
    _atomic_json(mapping_path, pipeline.supervised_mapping)
    fit_payload = {
        "heldout_fold": heldout_fold,
        "fit_ids_sha256": lineage["train_ids_sha256"],
        "fit_uses_same_fold_model_predictions": True,
        "stages": pipeline.stage_fits,
        "supervised_mapping_artifact": str(mapping_path),
        "supervised_mapping_sha256": _sha256(mapping_path),
        "nonheldout_score_warning": (
            "All recipe-search metrics are diagnostics on predictions from rows used "
            "to train this OCR fold model. They are not OOF and must not be claimed."
        ),
        "fit_side_blank_prediction_ids": lineage["fit_side_blank_prediction_ids"],
        "fit_side_blank_fill": lineage["fit_side_blank_fill"],
    }
    fit_path = output / "fit" / f"fold{heldout_fold}_recipe.json"
    _atomic_json(fit_path, fit_payload)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "implementation_sha256": _implementation_sha256(),
        "model_family": model_family,
        "heldout_fold": heldout_fold,
        "checkpoint_epoch_zero_based": checkpoint_epoch,
        "checkpoint_policy": checkpoint_policy,
        "policy_selection_fold": policy_selection_fold,
        "rows": len(data.ids),
        "fit_rows": len(data.ids) - len(pipeline.heldout_ids),
        "heldout_rows": len(pipeline.heldout_ids),
        "stage_order": stage_order,
        "scores": scores,
        "fit_recipe_artifact": str(fit_path),
        "fit_recipe_sha256": _sha256(fit_path),
        "lineage": lineage,
        "claim_scope": (
            f"Only scores under scores.* for heldout fold {heldout_fold} are OOF. "
            "Predictions and recipe-search diagnostics on folds != heldout_fold "
            "are not OOF; no such score is emitted or claimable."
        ),
        "leakage_guarantees": [
            "Every recipe and setting is fitted only on labels from folds != heldout_fold.",
            "Fit-side and heldout inputs come from the exact same OCR weights and settings.",
            "The frozen sequential recipe is applied once to heldout_fold; heldout labels only score the result.",
            "Raw heldout predictions come directly from the signed OOF artifact; the all-row run is used only on fit-side rows.",
            "If inference batch sizes differ, batch-sensitive heldout differences are measured and recorded rather than used.",
            "Blank OCR outputs are rejected on heldout_fold; fit-side blanks use one fixed label-independent '?' fill.",
        ],
    }
    report_path = output / "report.json"
    _atomic_json(report_path, report)
    return report


def _score_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "weighted_wer",
        "weighted_cer",
        "normalized_wer_error",
        "normalized_cer_error",
        "mean_error",
        "multi_score",
    )
    return all(
        math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1e-12)
        for key in keys
    )


def combine_fold_reports(
    truth_path: str | Path,
    report_paths: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Combine only the signed heldout slices and compute complete OOF scores."""

    truth = _absolute(truth_path)
    table = read_table(truth, require_target=True)
    if "fold" not in table.fieldnames:
        raise SchemaError("truth CSV must contain a fold column")
    assert table.target_column is not None
    ids = [row[table.id_column].strip() for row in table.rows]
    references = {row[table.id_column].strip(): row[table.target_column] for row in table.rows}
    folds = {row[table.id_column].strip(): int(row["fold"]) for row in table.rows}
    expected_folds = sorted(set(folds.values()))
    truth_sha256 = _sha256(truth)
    reports: dict[int, tuple[Path, dict[str, Any]]] = {}
    for raw_path in report_paths:
        path = _absolute(raw_path)
        report = _read_json(path)
        if report.get("schema_version") != SCHEMA_VERSION or report.get("protocol") != PROTOCOL:
            raise ValueError(f"not a compatible Model35 fold report: {path}")
        fold = report.get("heldout_fold")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold in reports:
            raise ValueError("fold reports must have unique integer heldout folds")
        if report.get("lineage", {}).get("truth_sha256") != truth_sha256:
            raise ValueError(f"fold {fold} report uses a different truth file")
        reports[fold] = (path, report)
    if sorted(reports) != expected_folds:
        raise ValueError(
            f"report folds {sorted(reports)} do not match truth folds {expected_folds}"
        )
    policies = {
        (
            report["checkpoint_epoch_zero_based"],
            report["checkpoint_policy"],
            report["policy_selection_fold"],
        )
        for _, report in reports.values()
    }
    if len(policies) != 1:
        raise ValueError("fold reports do not share one pre-fixed checkpoint policy")
    stage_order = ["raw", *(stage.name for stage in STAGES)]
    if any(report.get("stage_order") != stage_order for _, report in reports.values()):
        raise ValueError("fold reports have inconsistent stage order")

    combined_by_stage: dict[str, dict[str, str]] = {
        stage: {} for stage in stage_order
    }
    source_receipts: dict[str, Any] = {}
    for fold, (report_path, report) in sorted(reports.items()):
        expected_ids = {identifier for identifier in ids if folds[identifier] == fold}
        source_receipts[str(fold)] = {
            "report": str(report_path),
            "report_sha256": _sha256(report_path),
            "model_family": report["model_family"],
            "model_sha256": report["lineage"]["model_sha256"],
        }
        for stage in stage_order:
            record = report["scores"][stage]
            prediction_path = _manifest_path(
                record["prediction_artifact"], "prediction_artifact"
            )
            if _sha256(prediction_path) != record["prediction_sha256"]:
                raise ValueError(f"fold {fold} {stage} prediction checksum mismatch")
            values = _prediction_map(prediction_path)
            if set(values) != expected_ids:
                raise SchemaError(f"fold {fold} {stage} IDs are not its heldout slice")
            recomputed = competition_error(
                [references[identifier] for identifier in ids if identifier in expected_ids],
                [values[identifier] for identifier in ids if identifier in expected_ids],
            )
            if not _score_equal(recomputed, record["score"]):
                raise ValueError(f"fold {fold} {stage} reported score is not reproducible")
            combined_by_stage[stage].update(values)

    output = _prepare_output_dir(output_dir)
    score_records: dict[str, Any] = {}
    for position, stage in enumerate(stage_order):
        if set(combined_by_stage[stage]) != set(ids):
            raise RuntimeError(f"combined {stage} OOF predictions are incomplete")
        values = [combined_by_stage[stage][identifier] for identifier in ids]
        artifact = output / "predictions" / f"{position:02d}_{stage}_oof.csv"
        _atomic_prediction_csv(artifact, table.id_column, ids, values)
        per_fold: dict[str, Any] = {}
        for fold in expected_folds:
            fold_ids = [identifier for identifier in ids if folds[identifier] == fold]
            per_fold[str(fold)] = competition_error(
                [references[identifier] for identifier in fold_ids],
                [combined_by_stage[stage][identifier] for identifier in fold_ids],
            )
        score_records[stage] = {
            "global": competition_error(
                [references[identifier] for identifier in ids], values
            ),
            "per_fold": per_fold,
            "prediction_artifact": str(artifact),
            "prediction_sha256": _sha256(artifact),
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "combine provenance-verified Model35 heldout slices",
        "truth": str(truth),
        "truth_sha256": truth_sha256,
        "rows": len(ids),
        "folds": expected_folds,
        "checkpoint_policy": list(policies)[0][1],
        "checkpoint_epoch_zero_based": list(policies)[0][0],
        "policy_selection_fold": list(policies)[0][2],
        "stage_order": stage_order,
        "scores": score_records,
        "fold_reports": source_receipts,
        "claim_scope": (
            "These global scores combine one provenance-verified heldout slice per fold. "
            "Train-side diagnostics from individual fold models remain non-OOF and excluded."
        ),
    }
    _atomic_json(output / "report.json", report)
    return report


def fold_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--heldout-oof-predictions", type=Path, required=True)
    parser.add_argument("--oof-evaluation-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heldout-fold", type=int, required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--checkpoint-policy", required=True)
    parser.add_argument("--policy-selection-fold", type=int, required=True)
    parser.add_argument("--model-family", choices=SUPPORTED_MODEL_FAMILIES, required=True)
    args = parser.parse_args()
    report = run_fold(
        args.truth,
        args.predictions,
        args.inference_manifest,
        args.training_manifest,
        args.heldout_oof_predictions,
        args.oof_evaluation_receipt,
        args.output_dir,
        args.heldout_fold,
        args.checkpoint_epoch,
        args.checkpoint_policy,
        args.policy_selection_fold,
        args.model_family,
    )
    print(
        json.dumps(
            {
                "report": str(_absolute(args.output_dir) / "report.json"),
                "heldout_fold": report["heldout_fold"],
                "raw_heldout_score": report["scores"]["raw"]["score"]["multi_score"],
                "final_heldout_score": report["scores"]["supervised_errors"]["score"]["multi_score"],
            },
            indent=2,
        )
    )


def combine_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--fold-report", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = combine_fold_reports(args.truth, args.fold_report, args.output_dir)
    print(
        json.dumps(
            {
                "report": str(_absolute(args.output_dir) / "report.json"),
                "raw_oof_score": report["scores"]["raw"]["global"]["multi_score"],
                "final_oof_score": report["scores"]["supervised_errors"]["global"]["multi_score"],
            },
            indent=2,
        )
    )
