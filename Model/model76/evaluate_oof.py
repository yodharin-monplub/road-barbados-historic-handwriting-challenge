#!/usr/bin/env python3
"""Combine and independently verify all five signed Model73 OOF folds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from road_htr.metrics import competition_error
from road_htr.schema import SchemaError, predictions_by_id, read_table
from road_htr.vlm_cli import _sha256


def evaluate(truth_path: Path, artifact_root: Path, output_path: Path, receipt_path: Path) -> dict:
    truth_path = truth_path.resolve()
    artifact_root = artifact_root.resolve()
    output_path = output_path.resolve()
    receipt_path = receipt_path.resolve()
    truth = read_table(truth_path, require_target=True)
    if "fold" not in truth.fieldnames:
        raise SchemaError("truth CSV has no fold column")
    assert truth.target_column is not None

    combined: dict[str, str] = {}
    fold_artifacts: list[dict] = []
    for fold in range(5):
        prediction_path = artifact_root / "model73" / f"fold{fold}_oof_step400.csv"
        manifest_path = prediction_path.with_suffix(".manifest.json")
        signed_path = artifact_root / "model74" / f"fold{fold}_step400.evaluation.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        signed = json.loads(signed_path.read_text(encoding="utf-8"))
        if manifest.get("fold") != fold or signed.get("fold") != fold:
            raise ValueError(f"fold {fold} provenance mismatch")
        if manifest.get("input_sha256") != _sha256(truth_path):
            raise ValueError(f"fold {fold} truth checksum mismatch")
        prediction_sha = _sha256(prediction_path)
        if manifest.get("output_sha256") != prediction_sha:
            raise ValueError(f"fold {fold} inference output checksum mismatch")
        if signed.get("predictions_sha256") != prediction_sha:
            raise ValueError(f"fold {fold} signed output checksum mismatch")
        if signed.get("inference_manifest_sha256") != _sha256(manifest_path):
            raise ValueError(f"fold {fold} signed manifest checksum mismatch")

        table = read_table(prediction_path)
        value_columns = [name for name in table.fieldnames if name != table.id_column]
        value_column = table.target_column or (
            value_columns[0] if len(value_columns) == 1 else None
        )
        if value_column is None:
            raise SchemaError(f"fold {fold} predictions need exactly one value column")
        by_id = dict(predictions_by_id(table, value_column))
        expected = {
            row[truth.id_column].strip()
            for row in truth.rows
            if int(row["fold"]) == fold
        }
        if set(by_id) != expected or len(by_id) != len(expected):
            raise SchemaError(f"fold {fold} prediction IDs do not match heldout IDs")
        overlap = set(combined).intersection(by_id)
        if overlap:
            raise SchemaError(f"duplicate OOF IDs across folds: {sorted(overlap)[:3]}")
        combined.update(by_id)
        fold_artifacts.append(
            {
                "fold": fold,
                "rows": len(by_id),
                "score": signed["score"],
                "predictions": str(prediction_path),
                "predictions_sha256": prediction_sha,
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "signed_receipt": str(signed_path),
                "signed_receipt_sha256": _sha256(signed_path),
            }
        )

    ordered_ids = [row[truth.id_column].strip() for row in truth.rows]
    if set(combined) != set(ordered_ids) or len(combined) != len(ordered_ids):
        raise SchemaError("combined OOF IDs do not exactly cover the truth table")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[truth.id_column, "prediction"])
        writer.writeheader()
        writer.writerows(
            {truth.id_column: identifier, "prediction": combined[identifier]}
            for identifier in ordered_ids
        )
    predictions = [combined[identifier] for identifier in ordered_ids]
    references = [row[truth.target_column] for row in truth.rows]
    score = competition_error(references, predictions)
    receipt = {
        "schema_version": 1,
        "rows": len(ordered_ids),
        "truth": str(truth_path),
        "truth_sha256": _sha256(truth_path),
        "combined_predictions": str(output_path),
        "combined_predictions_sha256": _sha256(output_path),
        "empty_prediction_ids": sorted(
            identifier for identifier in ordered_ids if not combined[identifier]
        ),
        "fold_artifacts": fold_artifacts,
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
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.truth, args.artifact_root, args.output, args.receipt)
    print(json.dumps(result["score"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
