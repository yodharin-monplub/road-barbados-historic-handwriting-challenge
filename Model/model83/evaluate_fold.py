#!/usr/bin/env python3
"""Strict held-fold scorer for Model83's fixed five-view medoid screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from road_htr.deployment_medoid import (
    EXPECTED_SOURCE_COUNT,
    parse_five_sources,
    select_medoid,
    validate_complete_sources,
)
from road_htr.ensemble import write_prediction_csv
from road_htr.metrics import competition_error
from road_htr.schema import read_table
from road_htr.vlm_cli import _sha256, _tree_sha256


VLM_VARIANTS = {
    "full": "full",
    "enhanced": "enhanced",
    "center": "center",
    "center_enhanced": "center_enhanced",
}
EXPECTED_NAMES = (*VLM_VARIANTS, "kraken")
INFERENCE_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def evaluate(
    truth_path: Path,
    fold: int,
    source_specs: list[str],
    output_path: Path,
    report_path: Path,
    adapter_dir: Path | None = None,
) -> dict:
    truth_path = truth_path.resolve()
    truth = read_table(truth_path, require_target=True)
    if "fold" not in truth.fieldnames:
        raise ValueError("truth table has no fold column")
    assert truth.target_column is not None
    heldout = [row for row in truth.rows if int(row["fold"]) == fold]
    ids = [row[truth.id_column].strip() for row in heldout]
    references = [row[truth.target_column] for row in heldout]

    sources = parse_five_sources(source_specs)
    if tuple(sources) != EXPECTED_NAMES:
        raise ValueError(f"sources must be ordered exactly as {EXPECTED_NAMES}")
    validate_complete_sources(sources, ids)

    specifications = {
        item.split("=", 1)[0].strip(): Path(item.split("=", 1)[1].strip()).resolve()
        for item in source_specs
    }
    adapter_hashes: dict[str, dict[str, str]] = {}
    relocated_tree = None
    if adapter_dir is not None:
        adapter_dir = adapter_dir.resolve()
        if not adapter_dir.is_dir():
            raise ValueError(f"relocated adapter directory is unavailable: {adapter_dir}")
        relocated_tree = _tree_sha256(adapter_dir)
    manifests: dict[str, dict] = {}
    for name, variant in VLM_VARIANTS.items():
        prediction_path = specifications[name]
        manifest_path = prediction_path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("output_sha256") != _sha256(prediction_path):
            raise ValueError(f"{name} output hash does not match its manifest")
        if manifest.get("input_sha256") != _sha256(truth_path):
            raise ValueError(f"{name} truth hash does not match its manifest")
        if manifest.get("fold") != fold:
            raise ValueError(f"{name} manifest fold mismatch")
        if manifest.get("image_variant") != variant:
            raise ValueError(f"{name} manifest image variant mismatch")
        declared_tree = manifest.get("adapter_files_sha256")
        if not isinstance(declared_tree, dict):
            raise ValueError(f"{name} manifest has no adapter hash ledger")
        if relocated_tree is None:
            adapter = Path(str(manifest.get("adapter", "")))
            if not adapter.is_dir():
                raise ValueError(f"{name} adapter directory is unavailable: {adapter}")
            tree = _tree_sha256(adapter)
            if declared_tree != tree:
                raise ValueError(f"{name} adapter tree hash mismatch")
        else:
            missing_or_changed = {
                relative: digest
                for relative, digest in declared_tree.items()
                if relocated_tree.get(relative) != digest
            }
            if missing_or_changed:
                raise ValueError(
                    f"{name} adapter ledger does not match relocated checkpoint: "
                    f"{sorted(missing_or_changed)}"
                )
        adapter_hashes[name] = {
            relative: declared_tree.get(relative, "")
            for relative in INFERENCE_ADAPTER_FILES
        }
        if any(not digest for digest in adapter_hashes[name].values()):
            raise ValueError(f"{name} manifest lacks an inference adapter file")
        manifests[name] = {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        }
    if len({json.dumps(value, sort_keys=True) for value in adapter_hashes.values()}) != 1:
        raise ValueError("the four LightOnOCR views do not use the same adapter tree")

    individual = {}
    for name, records in sources.items():
        predictions = [records[name_id].candidates[0] for name_id in ids]
        individual[name] = competition_error(references, predictions)

    selections = [select_medoid(identifier, sources) for identifier in ids]
    medoid_predictions = [selection.prediction for selection in selections]
    medoid_score = competition_error(references, medoid_predictions)
    output_path = output_path.resolve()
    write_prediction_csv(output_path, truth.id_column, ids, medoid_predictions)

    result = {
        "schema_version": 1,
        "claim_scope": f"heldout fold {fold} only",
        "fold": fold,
        "rows": len(ids),
        "truth": str(truth_path),
        "truth_sha256": _sha256(truth_path),
        "source_order": list(sources),
        "source_count": EXPECTED_SOURCE_COUNT,
        "source_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in specifications.items()
        },
        "vlm_manifests": manifests,
        "adapter_verification": {
            "mode": "relocated_superset" if relocated_tree is not None else "exact_tree",
            "directory": str(adapter_dir) if adapter_dir is not None else None,
            "tree_sha256": relocated_tree,
            "required_files": list(INFERENCE_ADAPTER_FILES),
        },
        "individual_scores": individual,
        "medoid_score": medoid_score,
        "delta_vs_full": medoid_score["multi_score"]
        - individual["full"]["multi_score"],
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "selection_reasons": {
            reason: sum(
                selection.diagnostic["selection_reason"] == reason
                for selection in selections
            )
            for reason in ("exact_majority", "medoid", "confidence_tiebreak", "lexical_tiebreak")
        },
        "empty_prediction_ids": [],
    }
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        help=(
            "local complete checkpoint used to verify every manifest's declared "
            "inference-file hashes when artifacts have moved between machines"
        ),
    )
    args = parser.parse_args()
    evaluate(
        args.truth,
        args.fold,
        args.source,
        args.output,
        args.report,
        adapter_dir=args.adapter_dir,
    )


if __name__ == "__main__":
    main()
