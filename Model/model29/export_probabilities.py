#!/usr/bin/env python3
"""Export full PP-OCR CTC probabilities without changing normal inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.kraken_probability_export import (  # noqa: E402
    ExportOptions,
    export_probabilities,
    project_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only, resumable PP-OCR CTC probability export"
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        help="Model22 fold run_manifest.json; mandatory with --fold",
    )
    parser.add_argument(
        "--conversion-receipt",
        type=Path,
        help="selected evaluator *.conversion.json; mandatory with --fold",
    )
    parser.add_argument(
        "--training-excluded-fold",
        type=int,
        action="append",
        help="exact fold(s) excluded from checkpoint training; repeat for inner models",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    result = export_probabilities(
        ExportOptions(
            config=project_path(arguments.config),
            input_csv=project_path(arguments.input),
            checkpoint=project_path(arguments.checkpoint),
            output_dir=project_path(arguments.output_dir),
            fold=arguments.fold,
            training_manifest=(
                project_path(arguments.training_manifest)
                if arguments.training_manifest is not None
                else None
            ),
            conversion_receipt=(
                project_path(arguments.conversion_receipt)
                if arguments.conversion_receipt is not None
                else None
            ),
            training_excluded_folds=(
                tuple(arguments.training_excluded_fold)
                if arguments.training_excluded_fold
                else None
            ),
            dry_run=arguments.dry_run,
        )
    )
    signature = result.get("signature", {})
    print(
        json.dumps(
            {
                "dry_run": bool(result.get("dry_run", False)),
                "complete": bool(result.get("complete", False)),
                "output_dir": str(arguments.output_dir),
                "fold": result.get("fold"),
                "rows": result.get("rows"),
                "classes": result.get("classes", len(result.get("codec_labels", []))),
                "checkpoint_sha256": signature.get("checkpoint_sha256"),
                "input_sha256": signature.get("input_sha256"),
                "implementation_sha256": signature.get("implementation_sha256"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
