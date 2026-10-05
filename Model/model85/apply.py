#!/usr/bin/env python3
"""Build Model85's verified Model77 TTA consensus submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from road_htr.deployment_medoid import apply as apply_medoid
from road_htr.schema import read_table
from road_htr.vlm_cli import _sha256, _tree_sha256


VLM_VARIANTS = {
    "full": "full",
    "enhanced": "enhanced",
    "center": "center",
    "center_enhanced": "center_enhanced",
}
EXPECTED_SOURCE_ORDER = (*VLM_VARIANTS, "kraken")
EXPECTED_VALIDATION_FOLDS = (0, 1)


def _parse_vlm_sources(values: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"source must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in sources:
            raise ValueError(f"blank or duplicate source name {name!r}")
        sources[name] = Path(raw_path.strip()).resolve()
    if tuple(sources) != tuple(VLM_VARIANTS):
        raise ValueError(f"VLM sources must be ordered exactly as {tuple(VLM_VARIANTS)}")
    return sources


def _verify_validation_reports(paths: list[Path]) -> list[dict]:
    if len(paths) != len(EXPECTED_VALIDATION_FOLDS):
        raise ValueError("exactly two validation reports are required")
    evidence: list[dict] = []
    for expected_fold, path in zip(EXPECTED_VALIDATION_FOLDS, paths, strict=True):
        resolved = path.resolve()
        report = json.loads(resolved.read_text(encoding="utf-8"))
        if report.get("fold") != expected_fold:
            raise ValueError(f"validation fold mismatch in {resolved}")
        if tuple(report.get("source_order", ())) != EXPECTED_SOURCE_ORDER:
            raise ValueError(f"validation source order mismatch in {resolved}")
        delta = float(report.get("delta_vs_full", float("nan")))
        if not delta > 0.0:
            raise ValueError(f"validation uplift is not positive in {resolved}")
        evidence.append(
            {
                "path": str(resolved),
                "sha256": _sha256(resolved),
                "fold": expected_fold,
                "rows": int(report["rows"]),
                "full_score": float(report["individual_scores"]["full"]["multi_score"]),
                "medoid_score": float(report["medoid_score"]["multi_score"]),
                "delta_vs_full": delta,
            }
        )
    return evidence


def build(
    test_path: Path,
    sample_path: Path,
    adapter_path: Path,
    vlm_source_specs: list[str],
    kraken_path: Path,
    validation_reports: list[Path],
    output_path: Path,
    agreement_path: Path,
    submission_path: Path,
    report_path: Path,
) -> dict:
    test_path = test_path.resolve()
    sample_path = sample_path.resolve()
    adapter_path = adapter_path.resolve()
    kraken_path = kraken_path.resolve()
    sources = _parse_vlm_sources(vlm_source_specs)
    adapter_tree = _tree_sha256(adapter_path)
    if not adapter_tree:
        raise ValueError("adapter tree is empty")

    test = read_table(test_path)
    test_ids = [row[test.id_column].strip() for row in test.rows]
    manifests: dict[str, dict] = {}
    for name, variant in VLM_VARIANTS.items():
        prediction_path = sources[name]
        manifest_path = prediction_path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("output_sha256") != _sha256(prediction_path):
            raise ValueError(f"{name} output hash does not match its manifest")
        if manifest.get("input_sha256") != _sha256(test_path):
            raise ValueError(f"{name} test hash does not match its manifest")
        if manifest.get("fold") is not None:
            raise ValueError(f"{name} unexpectedly used a validation fold")
        if manifest.get("image_variant") != variant:
            raise ValueError(f"{name} image variant mismatch")
        if int(manifest.get("rows", -1)) != len(test_ids):
            raise ValueError(f"{name} row count mismatch")
        if manifest.get("empty_prediction_ids"):
            raise ValueError(f"{name} contains empty predictions")
        if manifest.get("adapter_files_sha256") != adapter_tree:
            raise ValueError(f"{name} adapter tree hash mismatch")
        manifests[name] = {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        }

    evidence = _verify_validation_reports(validation_reports)
    source_specs = [f"{name}={path}" for name, path in sources.items()]
    source_specs.append(f"kraken={kraken_path}")
    medoid_result = apply_medoid(
        test_path=test_path,
        source_specs=source_specs,
        output_path=output_path,
        agreement_output=agreement_path,
        sample_path=sample_path,
        submission_path=submission_path,
    )

    result = {
        "schema_version": 1,
        "claim_scope": "deployment candidate; no test-label claim",
        "policy": "fixed five-source mixed character/word medoid",
        "validation_evidence": evidence,
        "test": {"path": str(test_path), "sha256": _sha256(test_path)},
        "sample": {"path": str(sample_path), "sha256": _sha256(sample_path)},
        "adapter": {
            "path": str(adapter_path),
            "files_sha256": adapter_tree,
        },
        "source_order": list(EXPECTED_SOURCE_ORDER),
        "source_files": {
            **{
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in sources.items()
            },
            "kraken": {"path": str(kraken_path), "sha256": _sha256(kraken_path)},
        },
        "vlm_manifests": manifests,
        "medoid_result": medoid_result,
        "output_sha256": _sha256(output_path.resolve()),
        "agreement_sha256": _sha256(agreement_path.resolve()),
        "submission_sha256": _sha256(submission_path.resolve()),
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
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--kraken", type=Path, required=True)
    parser.add_argument("--validation-report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agreement-output", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    build(
        test_path=args.test,
        sample_path=args.sample,
        adapter_path=args.adapter,
        vlm_source_specs=args.source,
        kraken_path=args.kraken,
        validation_reports=args.validation_report,
        output_path=args.output,
        agreement_path=args.agreement_output,
        submission_path=args.submission,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
