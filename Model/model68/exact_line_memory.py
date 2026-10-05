#!/usr/bin/env python3
"""Model68: exact case-sensitive OOF prediction-to-reference memory."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from model54.restore_final_tokens import resolve  # noqa: E402
from road_htr.crossfold_pair_stage import apply_pair_stage, evaluate_pair_stage  # noqa: E402

OUTPUT_ROOT = (ROOT / "outputs/model68").resolve()


class Model68GateError(RuntimeError):
    pass


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve(path)
    config = json.loads(resolved.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "model_number": 68,
        "source_model": "model66",
        "fit_policy": "outer_fold_excluded_exact_full_line_memory",
        "minimum_support": 1,
        "minimum_precision": 1.0,
        "normalization": "exact_case_sensitive_prediction_string",
        "output_root": "outputs/model68",
    }
    if config != expected:
        raise Model68GateError("Model68 config is immutable")
    return config, resolved


def fit_mapping(
    predictions: Sequence[str], references: Sequence[str], config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if not predictions or len(predictions) != len(references):
        raise Model68GateError("mapping fit requires aligned nonempty OOF pairs")
    outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    for prediction, reference in zip(predictions, references, strict=True):
        outcomes[prediction][reference] += 1
    mapping: dict[str, dict[str, Any]] = {}
    for source, counts in outcomes.items():
        target, support = max(counts.items(), key=lambda item: (item[1], item[0]))
        observations = sum(counts.values())
        precision = support / observations
        if (
            target != source
            and support >= int(config["minimum_support"])
            and precision >= float(config["minimum_precision"])
        ):
            mapping[source] = {
                "target": target,
                "support": support,
                "observations": observations,
                "precision": precision,
            }
    return mapping


def build_outer_fold_maps(
    predictions: Sequence[str], references: Sequence[str], folds: Sequence[int], config: Mapping[str, Any]
) -> dict[int, dict[str, dict[str, Any]]]:
    return {
        heldout: fit_mapping(
            [value for value, fold in zip(predictions, folds, strict=True) if fold != heldout],
            [value for value, fold in zip(references, folds, strict=True) if fold != heldout],
            config,
        )
        for heldout in sorted(set(folds))
    }


def repair_exact_line(
    value: str, mapping: Mapping[str, Mapping[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    rule = mapping.get(value)
    if rule is None:
        return value, []
    target = str(rule["target"])
    return target, [{
        "before": value,
        "after": target,
        "support": int(rule["support"]),
        "observations": int(rule["observations"]),
        "precision": float(rule["precision"]),
    }]


def main() -> None:
    parser = argparse.ArgumentParser(description="Model68 exact full-line translation memory")
    subs = parser.add_subparsers(dest="command", required=True)
    oof = subs.add_parser("oof")
    oof.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    oof.add_argument("--truth", type=Path, default=ROOT / "data/processed/train_folds.csv")
    oof.add_argument("--predictions", type=Path, required=True)
    oof.add_argument("--output-dir", type=Path, required=True)
    app = subs.add_parser("apply")
    app.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    app.add_argument("--truth", type=Path, default=ROOT / "data/processed/train_folds.csv")
    app.add_argument("--training-predictions", type=Path, required=True)
    app.add_argument("--test", type=Path, default=ROOT / "data/raw/Test.csv")
    app.add_argument("--predictions", type=Path, required=True)
    app.add_argument("--output-dir", type=Path, required=True)
    app.add_argument("--sample", type=Path)
    app.add_argument("--submission", type=Path)
    args = parser.parse_args()
    config, path = load_config(args.config)
    common = {
        "model_number": 68,
        "config": config,
        "config_path": path,
        "implementation_path": Path(__file__).resolve(),
        "output_root": OUTPUT_ROOT,
        "truth_path": args.truth,
    }
    if args.command == "oof":
        report = evaluate_pair_stage(
            **common,
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            build_maps=build_outer_fold_maps,
            repair=repair_exact_line,
        )
    else:
        report = apply_pair_stage(
            **common,
            training_predictions_path=args.training_predictions,
            test_path=args.test,
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            fit_mapping=fit_mapping,
            repair=repair_exact_line,
            sample_path=args.sample,
            submission_path=args.submission,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
