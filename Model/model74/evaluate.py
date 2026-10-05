#!/usr/bin/env python3
"""Verify and score one Model73 heldout prediction artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from road_htr.metrics import competition_error
from road_htr.schema import SchemaError, predictions_by_id, read_table
from road_htr.vlm_cli import _sha256, _tree_sha256


def evaluate(
    truth_path: Path,
    prediction_path: Path,
    manifest_path: Path,
    receipt_path: Path,
    fold: int,
) -> dict:
    truth_path = truth_path.resolve()
    prediction_path = prediction_path.resolve()
    manifest_path = manifest_path.resolve()
    receipt_path = receipt_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_path != prediction_path.with_suffix(".manifest.json"):
        raise ValueError("manifest must be the prediction CSV sidecar")
    if manifest.get("output_sha256") != _sha256(prediction_path):
        raise ValueError("prediction checksum does not match inference manifest")
    if manifest.get("input_sha256") != _sha256(truth_path):
        raise ValueError("truth checksum does not match inference manifest")
    if manifest.get("fold") != fold:
        raise ValueError("inference manifest fold mismatch")
    adapter = Path(str(manifest.get("adapter", "")))
    if not adapter.is_dir() or manifest.get("adapter_files_sha256") != _tree_sha256(adapter):
        raise ValueError("adapter tree does not match inference manifest")

    truth = read_table(truth_path, require_target=True)
    predictions = read_table(prediction_path)
    if "fold" not in truth.fieldnames:
        raise SchemaError("truth CSV has no fold column")
    assert truth.target_column is not None
    columns = [name for name in predictions.fieldnames if name != predictions.id_column]
    prediction_column = predictions.target_column or (columns[0] if len(columns) == 1 else None)
    if prediction_column is None:
        raise SchemaError("prediction CSV must contain exactly one value column")
    by_id = dict(predictions_by_id(predictions, prediction_column))
    heldout = [row for row in truth.rows if int(row["fold"]) == fold]
    expected_ids = [row[truth.id_column].strip() for row in heldout]
    if set(by_id) != set(expected_ids) or len(by_id) != len(expected_ids):
        raise SchemaError("prediction IDs do not exactly match the heldout fold")
    values = [by_id[identifier] for identifier in expected_ids]
    references = [row[truth.target_column] for row in heldout]
    score = competition_error(references, values)
    receipt = {
        "schema_version": 1,
        "fold": fold,
        "rows": len(expected_ids),
        "truth": str(truth_path),
        "truth_sha256": _sha256(truth_path),
        "predictions": str(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
        "inference_manifest": str(manifest_path),
        "inference_manifest_sha256": _sha256(manifest_path),
        "adapter": str(adapter.resolve()),
        "adapter_files_sha256": manifest["adapter_files_sha256"],
        "empty_prediction_ids": sorted(identifier for identifier in expected_ids if not by_id[identifier]),
        "score": score,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    result = evaluate(args.truth, args.predictions, args.manifest, args.receipt, args.fold)
    print(json.dumps(result["score"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
