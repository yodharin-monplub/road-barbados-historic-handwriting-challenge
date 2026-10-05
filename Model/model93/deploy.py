#!/usr/bin/env python3
"""Create gated and aggressive Model93 submissions after confirmation passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.crossfit_candidate_selector import (  # noqa: E402
    apply_deployment,
    fit_deployment,
    read_predictions,
)
from road_htr.submission import (  # noqa: E402
    clean_prediction_preserving_internal_spaces,
    validate_submission,
    write_submission,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-report", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--primary-oof", type=Path, required=True)
    parser.add_argument("--llm-oof", type=Path, required=True)
    parser.add_argument("--primary-test", type=Path, required=True)
    parser.add_argument("--llm-test", type=Path, required=True)
    parser.add_argument("--selector-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    confirmation_path = args.confirmation_report.resolve()
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if confirmation.get("promoted") is not True:
        raise RuntimeError("Model93 confirmation gate did not pass; deployment is blocked")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"deployment output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    gate_path = output / "gate.json"
    gated_predictions = output / "gated_test.csv"
    gated_submission = output / "submission_gated.csv"
    aggressive_submission = output / "submission_aggressive.csv"

    state = fit_deployment(
        args.truth,
        args.primary_oof,
        args.llm_oof,
        gate_path,
        args.selector_config,
    )
    applied = apply_deployment(
        args.test,
        args.primary_test,
        args.llm_test,
        gate_path,
        gated_predictions,
        sample_path=args.sample,
        submission_path=gated_submission,
    )
    write_submission(
        args.sample,
        aggressive_submission,
        read_predictions(args.llm_test),
        cleaner=clean_prediction_preserving_internal_spaces,
    )
    gated_validation = validate_submission(gated_submission, args.sample)
    aggressive_validation = validate_submission(aggressive_submission, args.sample)
    report = {
        "schema_version": 1,
        "kind": "model93_deployment",
        "confirmation_report": str(confirmation_path),
        "confirmation_report_sha256": digest(confirmation_path),
        "training_rows": state["training_rows"],
        "test_rows": applied["rows"],
        "selected_llm_rows": applied["selected_b_rows"],
        "gated_submission": str(gated_submission),
        "gated_submission_sha256": digest(gated_submission),
        "gated_validation": gated_validation,
        "aggressive_submission": str(aggressive_submission),
        "aggressive_submission_sha256": digest(aggressive_submission),
        "aggressive_validation": aggressive_validation,
        "automatic_zindi_submission": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
