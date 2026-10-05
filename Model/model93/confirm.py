#!/usr/bin/env python3
"""Train the frozen Model93 gate on fold 0 and confirm it on folds 1--4."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.crossfit_candidate_selector import (  # noqa: E402
    TextLanguageModel,
    _nested_training_features,
    extract_features,
    fit_logistic,
    load_config,
    load_labeled_rows,
    read_predictions,
    row_costs,
)
from road_htr.metrics import competition_error  # noqa: E402
from road_htr.model34_token_fusion import _row_cost, paired_bootstrap  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--selector-config", type=Path, required=True)
    parser.add_argument("--development-fold", type=int, default=0)
    parser.add_argument("--minimum-gain", type=float, default=0.0005)
    parser.add_argument("--minimum-nonnegative-folds", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--minimum-positive-fraction", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    id_column, rows = load_labeled_rows(args.truth)
    primary = read_predictions(args.primary)
    llm = read_predictions(args.llm)
    ids = {row.identifier for row in rows}
    for name, values in (("primary", primary), ("llm", llm)):
        missing = ids - set(values)
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} IDs: {sorted(missing)[:5]}")
    fit_rows = [row for row in rows if row.fold == args.development_fold]
    confirmation_rows = [row for row in rows if row.fold != args.development_fold]
    if len(fit_rows) < 2 or not confirmation_rows:
        raise ValueError("development/confirmation partition is empty")

    config = load_config(args.selector_config)
    matrix = _nested_training_features(
        fit_rows, primary, llm, {}, {}, config, "model93-fold0-gate"
    )
    costs = row_costs(fit_rows, primary, llm)
    selector = fit_logistic(
        matrix,
        [costs[row.identifier].choose_b for row in fit_rows],
        [costs[row.identifier].importance for row in fit_rows],
        config,
    )
    language_model = TextLanguageModel.fit(
        (row.target for row in fit_rows), config.char_order, config.lm_alpha
    )
    selected = {}
    probabilities = {}
    for row in confirmation_rows:
        features = extract_features(
            primary[row.identifier], llm[row.identifier], None, None, language_model
        )
        probability = selector.probability_b(features)
        probabilities[row.identifier] = probability
        selected[row.identifier] = (
            llm[row.identifier] if probability > 0.5 else primary[row.identifier]
        )

    references = [row.target for row in confirmation_rows]
    baseline_values = [primary[row.identifier] for row in confirmation_rows]
    aggressive_values = [llm[row.identifier] for row in confirmation_rows]
    selected_values = [selected[row.identifier] for row in confirmation_rows]
    baseline_score = competition_error(references, baseline_values)
    aggressive_score = competition_error(references, aggressive_values)
    gated_score = competition_error(references, selected_values)
    fold_scores = {}
    nonnegative = 0
    for fold in sorted({row.fold for row in confirmation_rows}):
        fold_rows = [row for row in confirmation_rows if row.fold == fold]
        fold_references = [row.target for row in fold_rows]
        fold_baseline = competition_error(
            fold_references, [primary[row.identifier] for row in fold_rows]
        )
        fold_gated = competition_error(
            fold_references, [selected[row.identifier] for row in fold_rows]
        )
        delta = fold_gated["multi_score"] - fold_baseline["multi_score"]
        nonnegative += int(delta >= 0)
        fold_scores[str(fold)] = {
            "baseline": fold_baseline,
            "gated": fold_gated,
            "delta": delta,
        }
    bootstrap = paired_bootstrap(
        [_row_cost(reference, hypothesis) for reference, hypothesis in zip(references, baseline_values, strict=True)],
        [_row_cost(reference, hypothesis) for reference, hypothesis in zip(references, selected_values, strict=True)],
        samples=args.bootstrap_samples,
        seed=20260906,
    )
    delta = gated_score["multi_score"] - baseline_score["multi_score"]
    promoted = (
        delta >= args.minimum_gain
        and nonnegative >= args.minimum_nonnegative_folds
        and float(bootstrap["positive_fraction"]) >= args.minimum_positive_fraction
    )

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"confirmation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "confirmation_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_column, "prediction"])
        writer.writeheader()
        for row in confirmation_rows:
            writer.writerow({id_column: row.identifier, "prediction": selected[row.identifier]})
    with (output / "selection_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[id_column, "fold", "selected_candidate", "probability_llm"]
        )
        writer.writeheader()
        for row in confirmation_rows:
            writer.writerow(
                {
                    id_column: row.identifier,
                    "fold": row.fold,
                    "selected_candidate": "llm" if probabilities[row.identifier] > 0.5 else "primary",
                    "probability_llm": f"{probabilities[row.identifier]:.12g}",
                }
            )
    gate = {
        "selector": selector.to_dict(),
        "language_model": language_model.to_dict(),
        "development_fold": args.development_fold,
        "training_rows": len(fit_rows),
        "config": config.to_dict(),
    }
    (output / "fold0_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "development_fold": args.development_fold,
        "development_rows": len(fit_rows),
        "confirmation_rows": len(confirmation_rows),
        "baseline_score": baseline_score,
        "aggressive_score": aggressive_score,
        "gated_score": gated_score,
        "gated_delta": delta,
        "selected_llm_rows": sum(
            probabilities[row.identifier] > 0.5 for row in confirmation_rows
        ),
        "nonnegative_confirmation_folds": nonnegative,
        "fold_scores": fold_scores,
        "bootstrap": bootstrap,
        "thresholds": {
            "minimum_gain": args.minimum_gain,
            "minimum_nonnegative_folds": args.minimum_nonnegative_folds,
            "minimum_positive_fraction": args.minimum_positive_fraction,
        },
        "promoted": promoted,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
