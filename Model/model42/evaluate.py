#!/usr/bin/env python3
"""CPU-only, leakage-safe audit of Model35-confusions versus Model25.

This is deliberately an *assessment*, not a deployable blender. Both input
candidates are frozen before it starts. For every salt and every inner fold,
the held partition is excluded from selector labels and importance weights.
The fixed pair-geometry features do not consume targets at all.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr import crossfit_candidate_selector as selector  # noqa: E402
from road_htr.metrics import competition_error  # noqa: E402
from road_htr.schema import SchemaError  # noqa: E402


OUTPUT_ROOT = (ROOT / "outputs" / "model42").resolve()


@dataclass(frozen=True)
class Variant:
    name: str
    feature_set: str
    logistic_l2: float


@dataclass(frozen=True)
class CompactLogistic:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]

    def probabilities(self, rows: Sequence[Sequence[float]]) -> list[float]:
        result = []
        for row in rows:
            score = self.coefficients[0]
            for value, mean, scale, coefficient in zip(
                row, self.means, self.scales, self.coefficients[1:], strict=True
            ):
                score += coefficient * ((value - mean) / scale)
            result.append(_sigmoid(score))
        return result


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for identifier in sorted(ids):
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


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


def _atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_artifacts(config: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in config["artifacts"].items():
        if set(spec) != {"path", "sha256"}:
            raise SchemaError(f"{name} must specify only path and sha256")
        path = _absolute(str(spec["path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = str(spec["sha256"])
        if len(expected) != 64 or _sha256(path) != expected:
            raise SchemaError(f"{name} SHA-256 mismatch")
        paths[name] = path
    return paths


def _feature_indices(feature_set: str) -> tuple[int, ...]:
    names = selector.feature_names()
    compact_absolute = {
        f"{prefix}_{suffix}"
        for prefix in ("a", "b")
        for suffix in (
            "log_chars", "log_words", "lm_char_logprob", "lm_word_logprob",
            "lm_known_word_fraction", "confidence_mean", "confidence_q10",
            "confidence_source_distance",
        )
    }
    if feature_set == "compact":
        included = lambda name: name.startswith("pair_") or name in compact_absolute
    elif feature_set == "pair_only":
        included = lambda name: name.startswith("pair_")
    elif feature_set == "pair_geometry":
        included = lambda name: (
            name.startswith("pair_") and "_lm_" not in name and "confidence" not in name
        )
    else:
        raise ValueError(f"unsupported Model42 feature set: {feature_set}")
    result = tuple(index for index, name in enumerate(names) if included(name))
    if not result:
        raise ValueError("empty feature set")
    return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exponential = math.exp(max(value, -60.0))
    return exponential / (1.0 + exponential)


def _fit_logistic(
    matrix: Sequence[Sequence[float]], labels: Sequence[bool], importance: Sequence[float],
    config: selector.SelectorConfig,
) -> CompactLogistic:
    if not matrix or len(matrix) != len(labels) or len(matrix) != len(importance):
        raise ValueError("selector inputs must be nonempty and aligned")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("selector feature width mismatch")
    means = tuple(statistics.fmean(row[column] for row in matrix) for column in range(width))
    scales = tuple(
        max(math.sqrt(statistics.fmean((row[column] - means[column]) ** 2 for row in matrix)), config.minimum_scale)
        for column in range(width)
    )
    design = [[1.0, *((value - mean) / scale for value, mean, scale in zip(row, means, scales, strict=True))] for row in matrix]
    weights = [max(float(value), 0.0) for value in importance]
    total = sum(weights)
    if total <= 0:
        return CompactLogistic(means, scales, (0.0,) * (width + 1))
    weights = [weight * len(matrix) / total for weight in weights]
    positive = sum(weight for weight, label in zip(weights, labels, strict=True) if label)
    negative = sum(weight for weight, label in zip(weights, labels, strict=True) if not label)
    coefficients = [math.log((positive + 0.5) / (negative + 0.5)), *([0.0] * width)]
    first, second = [0.0] * (width + 1), [0.0] * (width + 1)
    beta1, beta2 = 0.9, 0.999
    for epoch in range(1, config.logistic_epochs + 1):
        gradient = [0.0] * (width + 1)
        for features, label, weight in zip(design, labels, weights, strict=True):
            residual = weight * (_sigmoid(sum(a * b for a, b in zip(coefficients, features, strict=True))) - float(label)) / sum(weights)
            for column, value in enumerate(features):
                gradient[column] += residual * value
        for column in range(1, width + 1):
            gradient[column] += config.logistic_l2 * coefficients[column]
        for column, value in enumerate(gradient):
            first[column] = beta1 * first[column] + (1 - beta1) * value
            second[column] = beta2 * second[column] + (1 - beta2) * value * value
            coefficients[column] -= config.logistic_learning_rate * (first[column] / (1 - beta1**epoch)) / (math.sqrt(second[column] / (1 - beta2**epoch)) + 1e-8)
    return CompactLogistic(means, scales, tuple(coefficients))


def _score(rows: Sequence[selector.LabeledRow], predictions: Mapping[str, str]) -> dict[str, float]:
    return competition_error(
        [row.target for row in rows], [predictions[row.identifier] for row in rows]
    )


def _validate_model35_report(report: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    if report.get("heldout_fold") != 0 or report.get("heldout_rows") != 819:
        raise SchemaError("Model35 report is not the expected held-out fold 0 run")
    score = report.get("scores", {}).get("confusions", {})
    if score.get("prediction_sha256") != _sha256(paths["candidate_model35_confusions"]):
        raise SchemaError("Model35 report does not bind the confusions CSV")
    if score.get("scope") != "HELDOUT_FOLD_0_ONLY":
        raise SchemaError("Model35 confusions artifact is not held-out fold 0 only")
    guarantees = report.get("leakage_guarantees")
    if not isinstance(guarantees, list) or not guarantees:
        raise SchemaError("Model35 report lacks its fit-side leakage declaration")


def _fixed_geometry_matrix(
    rows: Sequence[selector.LabeledRow],
    candidate_a: Mapping[str, str],
    candidate_b: Mapping[str, str],
    details_a: Mapping[str, selector.OcrDetails],
    details_b: Mapping[str, selector.OcrDetails],
    configuration: selector.SelectorConfig,
    split_name: str,
) -> list[list[float]]:
    # The predeclared geometry-only specification does not use a target LM at
    # all. A blank fixed model satisfies Model33's feature API while making it
    # impossible for any partition target to affect a feature value.
    del split_name
    blank_lm = selector.TextLanguageModel.fit((), configuration.char_order, configuration.lm_alpha)
    return selector._feature_matrix(
        rows, candidate_a, candidate_b, details_a, details_b, blank_lm
    )


def _evaluate_seed(
    rows: Sequence[selector.LabeledRow],
    candidate_a: Mapping[str, str],
    candidate_b: Mapping[str, str],
    details_a: Mapping[str, selector.OcrDetails],
    details_b: Mapping[str, selector.OcrDetails],
    config: Mapping[str, Any],
    seed: str,
    variants: Sequence[Variant],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ids = [row.identifier for row in rows]
    split_count = int(config["splits"])
    assignment = selector.stable_groups(ids, split_count, f"{seed}:fold0")
    base = selector.SelectorConfig(
        single_fold_inner_splits=split_count,
        nested_lm_splits=2,
        seed=seed,
        char_order=int(config["char_order"]),
        lm_alpha=float(config["lm_alpha"]),
        logistic_epochs=int(config["logistic_epochs"]),
        logistic_learning_rate=float(config["logistic_learning_rate"]),
        logistic_l2=0.0,
        minimum_scale=float(config["minimum_scale"]),
    )
    probabilities: dict[str, dict[str, float]] = {variant.name: {} for variant in variants}
    audits: list[dict[str, Any]] = []
    # Geometry features are entirely label-independent and do not vary by
    # partition, so compute them once. This is also materially cheaper than
    # repeating character-level edit distances for every inner fit.
    fixed_by_id = dict(
        zip(
            ids,
            _fixed_geometry_matrix(
                rows, candidate_a, candidate_b, details_a, details_b, base, "fold0:all"
            ),
            strict=True,
        )
    )
    for group in range(split_count):
        fit_rows = [row for row in rows if assignment[row.identifier] != group]
        held_rows = [row for row in rows if assignment[row.identifier] == group]
        if not fit_rows or not held_rows:
            raise RuntimeError("empty inner partition")
        train_matrix = [fixed_by_id[row.identifier] for row in fit_rows]
        held_matrix = [fixed_by_id[row.identifier] for row in held_rows]
        costs = selector.row_costs(fit_rows, candidate_a, candidate_b)
        labels = [costs[row.identifier].choose_b for row in fit_rows]
        importance = [costs[row.identifier].importance for row in fit_rows]
        audits.append({
            "partition": group,
            "fit_rows": len(fit_rows),
            "evaluation_rows": len(held_rows),
            "fit_ids_sha256": _ids_sha256(row.identifier for row in fit_rows),
            "evaluation_ids_sha256": _ids_sha256(row.identifier for row in held_rows),
            "evaluation_labels_excluded_from_selector": True,
            "target_language_model_used": False,
            "selector_features_are_label_independent": True,
        })
        for variant in variants:
            indices = _feature_indices(variant.feature_set)
            fitted = _fit_logistic(
                [[features[index] for index in indices] for features in train_matrix], labels, importance,
                replace(base, logistic_l2=variant.logistic_l2),
            )
            values = fitted.probabilities(
                [[features[index] for index in indices] for features in held_matrix]
            )
            probabilities[variant.name].update(
                (row.identifier, float(value))
                for row, value in zip(held_rows, values, strict=True)
            )

    baseline = _score(rows, candidate_a)
    all_costs = selector.row_costs(rows, candidate_a, candidate_b)
    runs: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for variant in variants:
        values = probabilities[variant.name]
        if set(values) != set(ids):
            raise RuntimeError("incomplete held-out probability map")
        picks_b = {identifier: values[identifier] > 0.5 for identifier in ids}
        prediction = {
            identifier: candidate_b[identifier] if picks_b[identifier] else candidate_a[identifier]
            for identifier in ids
        }
        exact = _score(rows, prediction)
        partition_gains = []
        for group in range(split_count):
            subset = [row for row in rows if assignment[row.identifier] == group]
            partition_gains.append(
                _score(subset, prediction)["multi_score"] - _score(subset, candidate_a)["multi_score"]
            )
        runs.append({
            "seed": seed,
            "variant": variant.name,
            "feature_set": variant.feature_set,
            "feature_count": len(_feature_indices(variant.feature_set)),
            "logistic_l2": variant.logistic_l2,
            "multi_score": exact["multi_score"],
            "mean_error": exact["mean_error"],
            "gain_vs_model35": exact["multi_score"] - baseline["multi_score"],
            "selected_model25_rows": sum(picks_b.values()),
            "minimum_partition_gain_vs_model35": min(partition_gains),
            "positive_partition_count": sum(gain > 0 for gain in partition_gains),
        })
        for row in rows:
            cost = all_costs[row.identifier]
            selections.append({
                "ID": row.identifier,
                "seed": seed,
                "variant": variant.name,
                "partition": assignment[row.identifier],
                "probability_model25": f"{values[row.identifier]:.12g}",
                "selected_candidate": "model25" if picks_b[row.identifier] else "model35_confusions",
                "oracle_candidate": "model25" if cost.choose_b else "model35_confusions",
                "oracle_importance": f"{cost.importance:.12g}",
            })
    return runs, selections, {
        "seed": seed,
        "partition_ids_sha256": {
            str(group): _ids_sha256(identifier for identifier in ids if assignment[identifier] == group)
            for group in range(split_count)
        },
        "split_audits": audits,
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "minimum": ordered[0], "mean": statistics.fmean(ordered),
        "maximum": ordered[-1], "population_std": statistics.pstdev(ordered),
    }


def evaluate(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_file = _absolute(config_path)
    implementation = Path(__file__).resolve()
    initial_implementation_sha256 = _sha256(implementation)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise SchemaError("unsupported Model42 schema")
    paths = _verify_artifacts(config)
    report35 = json.loads(paths["model35_report"].read_text(encoding="utf-8"))
    _validate_model35_report(report35, paths)
    id_column, all_rows = selector.load_labeled_rows(paths["truth"])
    rows = [row for row in all_rows if row.fold == int(config["outer_fold"])]
    if len(all_rows) != int(config["expected_truth_rows"]) or len(rows) != int(config["expected_fold_rows"]):
        raise SchemaError("unexpected canonical truth coverage")
    ids = [row.identifier for row in rows]
    candidate_a = selector.read_predictions(paths["candidate_model35_confusions"])
    candidate_b = selector.read_predictions(paths["candidate_model25"])
    # The fixed pair-geometry specification deliberately consumes no OCR
    # confidence or target-language feature, so raw Model22/Model33 artifacts
    # cannot introduce an additional within-fold lineage dependency.
    details_a: dict[str, selector.OcrDetails] = {}
    details_b: dict[str, selector.OcrDetails] = {}
    for name, values in (("Model35", candidate_a), ("Model25", candidate_b)):
        missing = set(ids) - set(values)
        if missing:
            raise SchemaError(f"{name} lacks fold-0 IDs: {sorted(missing)[:5]}")
    if set(candidate_a) != set(ids):
        raise SchemaError("Model35 confusions must contain exactly the held-out fold IDs")
    variants = tuple(Variant(**value) for value in config["variants"])
    if len({variant.name for variant in variants}) != len(variants):
        raise SchemaError("duplicate Model42 variant")
    seeds = tuple(str(value) for value in config["partition_seeds"])
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise SchemaError("need at least two fixed, distinct salts")
    output = _absolute(output_dir)
    if not _below(output, OUTPUT_ROOT):
        raise SchemaError(f"Model42 output must stay under {OUTPUT_ROOT}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is nonempty: {output}")

    runs: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    seed_audits: list[dict[str, Any]] = []
    for seed in seeds:
        seed_runs, seed_selections, audit = _evaluate_seed(
            rows, candidate_a, candidate_b, details_a, details_b, config, seed, variants
        )
        runs.extend(seed_runs)
        selections.extend(seed_selections)
        seed_audits.append(audit)

    baseline = _score(rows, candidate_a)
    candidate25 = _score(rows, candidate_b)
    summaries = {}
    for variant in variants:
        selected = [run for run in runs if run["variant"] == variant.name]
        summaries[variant.name] = {
            "feature_set": variant.feature_set,
            "feature_count": len(_feature_indices(variant.feature_set)),
            "score": _distribution([run["multi_score"] for run in selected]),
            "gain_vs_model35": _distribution([run["gain_vs_model35"] for run in selected]),
            "minimum_partition_gain": min(run["minimum_partition_gain_vs_model35"] for run in selected),
            "salts_with_all_partitions_positive": sum(run["positive_partition_count"] == int(config["splits"]) for run in selected),
            "selected_model25_rows": _distribution([run["selected_model25_rows"] for run in selected]),
        }
    rule = config["promotion_rule"]
    primary = summaries[str(rule["primary_variant"])]
    robust_gain = primary["gain_vs_model35"]["minimum"]
    accepted = (
        robust_gain >= float(rule["minimum_score_gain"])
        and primary["salts_with_all_partitions_positive"] == len(seeds)
        and primary["minimum_partition_gain"] > 0
    )
    decision = {
        "accepted": accepted,
        "primary_variant": rule["primary_variant"],
        "minimum_required_gain": rule["minimum_score_gain"],
        "observed_minimum_salt_gain": robust_gain,
        "observed_minimum_partition_gain": primary["minimum_partition_gain"],
        "reason": (
            "passes the predeclared conditional fold-0 robustness rule"
            if accepted else "rejected: the predeclared robust-gain rule was not met"
        ),
        "scope": rule["scope"],
    }
    if _sha256(implementation) != initial_implementation_sha256:
        raise RuntimeError("Model42 implementation changed during evaluation")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output / "runs.csv", tuple(runs[0]), runs)
    _atomic_csv(output / "selections.csv", tuple(selections[0]), selections)
    payload = {
        "schema_version": 1,
        "method": "frozen Model35-confusions vs Model25, nested inner-fold selector",
        "claim_scope": "conditional fold-0 inner-crossfit assessment only; not deployable or complete OOF",
        "id_column": id_column,
        "rows": len(rows),
        "baseline_model35_confusions": baseline,
        "candidate_model25": candidate25,
        "lineage": {name + "_sha256": _sha256(path) for name, path in paths.items()},
        "config": config,
        "config_sha256": _sha256(config_file),
        "implementation": str(implementation),
        "implementation_sha256": initial_implementation_sha256,
        "seed_audits": seed_audits,
        "summaries": summaries,
        "decision": decision,
        "artifacts": {
            "runs": str(output / "runs.csv"), "runs_sha256": _sha256(output / "runs.csv"),
            "selections": str(output / "selections.csv"), "selections_sha256": _sha256(output / "selections.csv"),
        },
    }
    _atomic_json(output / "report.json", payload)
    return payload


def _below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def main() -> None:
    parser = argparse.ArgumentParser(description="Model42 nested CPU selector audit")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
