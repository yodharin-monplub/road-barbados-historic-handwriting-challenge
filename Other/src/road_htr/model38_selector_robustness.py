"""Robustness audit for the Model33 fold-0 candidate selector.

This module deliberately does not search for a winning salt or feature set.
It evaluates the complete Cartesian product declared in ``model38/config.json``
and reports every result.  Model33's feature extraction and language model are
reused read-only; a NumPy implementation of the same weighted Adam logistic
fit makes the stress panel practical on CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import crossfit_candidate_selector as model33
from .metrics import competition_error
from .schema import SchemaError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AuditVariant:
    name: str
    feature_set: str
    logistic_l2: float


@dataclass(frozen=True)
class VectorLogisticModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]

    def probabilities(self, matrix: Sequence[Sequence[float]]) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("selector feature width mismatch")
        standardized = (values - np.asarray(self.means)) / np.asarray(self.scales)
        scores = self.coefficients[0] + standardized @ np.asarray(
            self.coefficients[1:]
        )
        return _sigmoid_array(scores)


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


def _verify_artifact(name: str, spec: Mapping[str, Any]) -> Path:
    if set(spec) != {"path", "sha256"}:
        raise SchemaError(f"{name} provenance must contain only path and sha256")
    path = _absolute(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(spec["sha256"]).lower()
    actual = _sha256(path)
    if actual != expected:
        raise SchemaError(
            f"{name} SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return path


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


def _atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sigmoid_array(scores: np.ndarray) -> np.ndarray:
    result = np.empty_like(scores, dtype=np.float64)
    positive = scores >= 0
    inverse = np.exp(-np.minimum(scores[positive], 60.0))
    result[positive] = 1.0 / (1.0 + inverse)
    exponential = np.exp(np.maximum(scores[~positive], -60.0))
    result[~positive] = exponential / (1.0 + exponential)
    return result


def fit_logistic_numpy(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[bool],
    importance: Sequence[float],
    config: model33.SelectorConfig,
    active_indices: Sequence[int],
) -> VectorLogisticModel:
    """Vectorized equivalent of Model33's weighted Adam logistic fit."""

    if not matrix or len(matrix) != len(labels) or len(matrix) != len(importance):
        raise ValueError("logistic inputs must be equally sized and nonempty")
    all_names = model33.feature_names()
    indices = tuple(int(index) for index in active_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("active feature indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= len(all_names):
        raise ValueError("active feature index is out of range")
    raw = np.asarray(matrix, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(all_names):
        raise ValueError("logistic matrix has inconsistent feature width")
    values = raw[:, indices]
    means = values.mean(axis=0)
    scales = np.maximum(values.std(axis=0), config.minimum_scale)
    standardized = (values - means) / scales
    design = np.concatenate(
        (np.ones((len(values), 1), dtype=np.float64), standardized), axis=1
    )
    weights = np.maximum(np.asarray(importance, dtype=np.float64), 0.0)
    total_importance = float(weights.sum())
    names = tuple(all_names[index] for index in indices)
    if total_importance <= 0:
        return VectorLogisticModel(
            names,
            tuple(float(value) for value in means),
            tuple(float(value) for value in scales),
            (0.0,) * (len(indices) + 1),
        )
    weights *= len(values) / total_importance
    target = np.asarray(labels, dtype=np.float64)
    positive = float(weights[target == 1.0].sum())
    negative = float(weights[target == 0.0].sum())
    coefficients = np.zeros(len(indices) + 1, dtype=np.float64)
    coefficients[0] = math.log((positive + 0.5) / (negative + 0.5))
    first_moment = np.zeros_like(coefficients)
    second_moment = np.zeros_like(coefficients)
    beta1, beta2 = 0.9, 0.999
    weight_sum = float(weights.sum())
    for epoch in range(1, config.logistic_epochs + 1):
        probabilities = _sigmoid_array(design @ coefficients)
        residual = weights * (probabilities - target) / weight_sum
        gradient = design.T @ residual
        gradient[1:] += config.logistic_l2 * coefficients[1:]
        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient * gradient
        corrected_first = first_moment / (1 - beta1**epoch)
        corrected_second = second_moment / (1 - beta2**epoch)
        coefficients -= (
            config.logistic_learning_rate
            * corrected_first
            / (np.sqrt(corrected_second) + 1e-8)
        )
    return VectorLogisticModel(
        names,
        tuple(float(value) for value in means),
        tuple(float(value) for value in scales),
        tuple(float(value) for value in coefficients),
    )


def feature_indices(feature_set: str) -> tuple[int, ...]:
    names = model33.feature_names()
    compact_absolute = {
        f"{prefix}_{suffix}"
        for prefix in ("a", "b")
        for suffix in (
            "log_chars",
            "log_words",
            "lm_char_logprob",
            "lm_word_logprob",
            "lm_known_word_fraction",
            "confidence_mean",
            "confidence_q10",
            "confidence_source_distance",
        )
    }

    def included(name: str) -> bool:
        if feature_set == "all":
            return True
        if feature_set == "compact":
            return name.startswith("pair_") or name in compact_absolute
        if feature_set == "no_lm":
            return "_lm_" not in name
        if feature_set == "no_confidence":
            return "confidence" not in name
        if feature_set == "pair_only":
            return name.startswith("pair_")
        if feature_set == "pair_geometry":
            return (
                name.startswith("pair_")
                and "_lm_" not in name
                and "confidence" not in name
            )
        raise ValueError(f"unknown feature set {feature_set!r}")

    result = tuple(index for index, name in enumerate(names) if included(name))
    if not result:
        raise ValueError(f"feature set {feature_set!r} is empty")
    return result


def _variants(config: Mapping[str, Any]) -> tuple[AuditVariant, ...]:
    values = tuple(
        AuditVariant(
            name=str(value["name"]),
            feature_set=str(value["feature_set"]),
            logistic_l2=float(value["logistic_l2"]),
        )
        for value in config["variants"]
    )
    if len({value.name for value in values}) != len(values):
        raise ValueError("audit variant names must be unique")
    for value in values:
        if value.logistic_l2 < 0:
            raise ValueError("audit L2 values must be non-negative")
        feature_indices(value.feature_set)
    return values


def load_verified_inputs(
    config: Mapping[str, Any]
) -> tuple[
    str,
    list[model33.LabeledRow],
    dict[str, str],
    dict[str, str],
    dict[str, model33.OcrDetails],
    dict[str, model33.OcrDetails],
    model33.SelectorConfig,
    dict[str, str],
]:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise SchemaError("unsupported Model38 config schema")
    artifacts = config["artifacts"]
    paths = {
        name: _verify_artifact(name, spec) for name, spec in artifacts.items()
    }
    id_column, all_rows = model33.load_labeled_rows(paths["truth"])
    outer_fold = int(config["outer_fold"])
    rows = [row for row in all_rows if row.fold == outer_fold]
    if len(all_rows) != int(config["expected_truth_rows"]):
        raise SchemaError("unexpected truth row count")
    if len(rows) != int(config["expected_fold_rows"]):
        raise SchemaError("unexpected audit-fold row count")
    all_ids = {row.identifier for row in all_rows}
    ids = {row.identifier for row in rows}
    candidate_a = model33.read_predictions(paths["candidate_a"])
    candidate_b = model33.read_predictions(paths["candidate_b"])
    details_a = model33.read_details(paths["details_a"])
    details_b = model33.read_details(paths["details_b"])
    if set(candidate_a) != all_ids or set(details_a) != all_ids:
        raise SchemaError("candidate A/details must exactly cover the full truth table")
    if set(candidate_b) != ids or set(details_b) != ids:
        raise SchemaError("candidate B/details must exactly cover the audit fold")
    if any(details_b[i].source_text != candidate_b[i] for i in ids):
        raise SchemaError("candidate B text does not match its confidence details")
    selector_config = model33.load_config(paths["model33_config"])
    expected_selector = config["expected_model33_config"]
    if selector_config.to_dict() != expected_selector:
        raise SchemaError("Model33 selector configuration drifted")
    lineage = {
        name + "_sha256": str(artifacts[name]["sha256"])
        for name in sorted(artifacts)
    }
    return (
        id_column,
        rows,
        candidate_a,
        candidate_b,
        details_a,
        details_b,
        selector_config,
        lineage,
    )


def _weighted_calibration(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    importance: Sequence[float],
    bins: int,
) -> dict[str, Any]:
    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    weight = np.maximum(np.asarray(importance, dtype=np.float64), 0.0)
    if len(probability) == 0 or len(probability) != len(target):
        raise ValueError("calibration arrays must be equally sized and non-empty")
    if bins < 2:
        raise ValueError("calibration needs at least two bins")
    total_weight = float(weight.sum())
    if total_weight <= 0:
        weight = np.ones_like(weight)
        total_weight = float(len(weight))
    brier = float(np.sum(weight * (probability - target) ** 2) / total_weight)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    log_loss = float(
        -np.sum(
            weight
            * (target * np.log(clipped) + (1 - target) * np.log(1 - clipped))
        )
        / total_weight
    )
    ece = 0.0
    bin_rows: list[dict[str, float | int]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        if not np.any(mask):
            continue
        bin_weight = float(weight[mask].sum())
        if bin_weight > 0:
            mean_probability = float(
                np.average(probability[mask], weights=weight[mask])
            )
            observed = float(np.average(target[mask], weights=weight[mask]))
        else:
            # Zero-importance ties do not affect the weighted audit, but keep
            # their bin diagnostics defined.
            mean_probability = float(probability[mask].mean())
            observed = float(target[mask].mean())
        ece += bin_weight / total_weight * abs(mean_probability - observed)
        bin_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "rows": int(mask.sum()),
                "importance_weight": bin_weight,
                "mean_probability_b": mean_probability,
                "observed_b_fraction": observed,
            }
        )
    choices = probability > 0.5
    weighted_accuracy = float(np.sum(weight * (choices == target)) / total_weight)
    return {
        "weighted_brier": brier,
        "weighted_log_loss": log_loss,
        "weighted_ece": ece,
        "weighted_choice_accuracy": weighted_accuracy,
        "near_threshold_fraction": float(np.mean((probability >= 0.45) & (probability <= 0.55))),
        "probability_b_mean": float(probability.mean()),
        "probability_b_q10": float(np.quantile(probability, 0.10)),
        "probability_b_median": float(np.quantile(probability, 0.50)),
        "probability_b_q90": float(np.quantile(probability, 0.90)),
        "bins": bin_rows,
    }


def _score_partition(
    rows: Sequence[model33.LabeledRow], predictions: Mapping[str, str]
) -> dict[str, float]:
    return competition_error(
        [row.target for row in rows],
        [predictions[row.identifier] for row in rows],
    )


def evaluate_seed(
    rows: Sequence[model33.LabeledRow],
    candidate_a: Mapping[str, str],
    candidate_b: Mapping[str, str],
    details_a: Mapping[str, model33.OcrDetails],
    details_b: Mapping[str, model33.OcrDetails],
    base_config: model33.SelectorConfig,
    variants: Sequence[AuditVariant],
    seed: str,
    *,
    outer_fold: int,
    calibration_bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate every fixed variant under one deterministic partition seed."""

    ids = [row.identifier for row in rows]
    split_count = min(base_config.single_fold_inner_splits, len(rows))
    selector_config = replace(base_config, seed=seed)
    partitions = model33.stable_groups(
        ids, split_count, f"{seed}:single-fold:{outer_fold}"
    )
    probabilities: dict[str, dict[str, float]] = {
        variant.name: {} for variant in variants
    }
    coefficients: dict[str, list[float]] = {variant.name: [] for variant in variants}
    split_audits: list[dict[str, Any]] = []

    for partition in range(split_count):
        fit_rows = [row for row in rows if partitions[row.identifier] != partition]
        evaluation_rows = [
            row for row in rows if partitions[row.identifier] == partition
        ]
        if not fit_rows or not evaluation_rows:
            raise ValueError("empty robustness-audit split")
        if {row.identifier for row in fit_rows} & {
            row.identifier for row in evaluation_rows
        }:
            raise RuntimeError("fit/evaluation ID overlap")
        split_name = f"single_fold_inner_crossfit:{partition}"
        train_matrix = model33._nested_training_features(
            fit_rows,
            candidate_a,
            candidate_b,
            details_a,
            details_b,
            selector_config,
            split_name,
        )
        evaluation_lm = model33.TextLanguageModel.fit(
            (row.target for row in fit_rows),
            selector_config.char_order,
            selector_config.lm_alpha,
        )
        evaluation_matrix = model33._feature_matrix(
            evaluation_rows,
            candidate_a,
            candidate_b,
            details_a,
            details_b,
            evaluation_lm,
        )
        costs = model33.row_costs(fit_rows, candidate_a, candidate_b)
        labels = [costs[row.identifier].choose_b for row in fit_rows]
        importance = [costs[row.identifier].importance for row in fit_rows]
        nested_count = min(selector_config.nested_lm_splits, len(fit_rows))
        nested_groups = model33.stable_groups(
            [row.identifier for row in fit_rows],
            nested_count,
            f"{seed}:{split_name}:lm",
        )
        split_audits.append(
            {
                "partition": partition,
                "fit_rows": len(fit_rows),
                "evaluation_rows": len(evaluation_rows),
                "fit_ids_sha256": _ids_sha256(row.identifier for row in fit_rows),
                "evaluation_ids_sha256": _ids_sha256(
                    row.identifier for row in evaluation_rows
                ),
                "evaluation_labels_excluded_from_selector": True,
                "evaluation_labels_excluded_from_evaluation_lm": True,
                "selector_training_lm_is_nested": True,
                "nested_lm_group_count": nested_count,
                "nested_lm_group_sizes": [
                    sum(nested_groups[row.identifier] == group for row in fit_rows)
                    for group in range(nested_count)
                ],
            }
        )
        for variant in variants:
            variant_config = replace(
                selector_config, logistic_l2=variant.logistic_l2
            )
            indices = feature_indices(variant.feature_set)
            model = fit_logistic_numpy(
                train_matrix, labels, importance, variant_config, indices
            )
            values = model.probabilities(
                [[row[index] for index in indices] for row in evaluation_matrix]
            )
            probabilities[variant.name].update(
                (row.identifier, float(value))
                for row, value in zip(evaluation_rows, values, strict=True)
            )
            coefficients[variant.name].append(
                math.sqrt(sum(value * value for value in model.coefficients[1:]))
            )

    full_costs = model33.row_costs(rows, candidate_a, candidate_b)
    oracle_labels = {
        row.identifier: full_costs[row.identifier].choose_b for row in rows
    }
    oracle_importance = {
        row.identifier: full_costs[row.identifier].importance for row in rows
    }
    oracle_predictions = {
        row.identifier: (
            candidate_b[row.identifier]
            if oracle_labels[row.identifier]
            else candidate_a[row.identifier]
        )
        for row in rows
    }
    oracle_score = _score_partition(rows, oracle_predictions)
    baseline_score = _score_partition(rows, candidate_a)
    run_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_probabilities = probabilities[variant.name]
        if set(variant_probabilities) != set(ids):
            raise RuntimeError("incomplete cross-fitted probability map")
        choices = {
            identifier: variant_probabilities[identifier] > 0.5
            for identifier in ids
        }
        predictions = {
            identifier: (
                candidate_b[identifier] if choices[identifier] else candidate_a[identifier]
            )
            for identifier in ids
        }
        score = _score_partition(rows, predictions)
        partition_gains: list[float] = []
        for partition in range(split_count):
            partition_rows = [
                row for row in rows if partitions[row.identifier] == partition
            ]
            partition_gains.append(
                float(_score_partition(partition_rows, predictions)["multi_score"])
                - float(_score_partition(partition_rows, candidate_a)["multi_score"])
            )
        calibration = _weighted_calibration(
            [variant_probabilities[identifier] for identifier in ids],
            [oracle_labels[identifier] for identifier in ids],
            [oracle_importance[identifier] for identifier in ids],
            calibration_bins,
        )
        run_rows.append(
            {
                "seed": seed,
                "variant": variant.name,
                "feature_set": variant.feature_set,
                "feature_count": len(feature_indices(variant.feature_set)),
                "logistic_l2": variant.logistic_l2,
                "multi_score": float(score["multi_score"]),
                "mean_error": float(score["mean_error"]),
                "weighted_wer": float(score["weighted_wer"]),
                "weighted_cer": float(score["weighted_cer"]),
                "gain_vs_model25": float(score["multi_score"])
                - float(baseline_score["multi_score"]),
                "oracle_gap": float(oracle_score["multi_score"])
                - float(score["multi_score"]),
                "selected_b_rows": sum(choices.values()),
                "minimum_partition_gain_vs_model25": min(partition_gains),
                "positive_partition_count": sum(gain > 0 for gain in partition_gains),
                "coefficient_l2_mean": statistics.fmean(coefficients[variant.name]),
                "coefficient_l2_max": max(coefficients[variant.name]),
                **{
                    key: value
                    for key, value in calibration.items()
                    if key != "bins"
                },
            }
        )
        for row in rows:
            identifier = row.identifier
            selection_rows.append(
                {
                    "ID": identifier,
                    "seed": seed,
                    "variant": variant.name,
                    "partition": partitions[identifier],
                    "probability_b": f"{variant_probabilities[identifier]:.12g}",
                    "selected_candidate": "b" if choices[identifier] else "a",
                    "oracle_candidate": "b" if oracle_labels[identifier] else "a",
                    "oracle_importance": f"{oracle_importance[identifier]:.12g}",
                }
            )
    seed_audit = {
        "seed": seed,
        "partition_ids_sha256": {
            str(partition): _ids_sha256(
                identifier
                for identifier in ids
                if partitions[identifier] == partition
            )
            for partition in range(split_count)
        },
        "split_audits": split_audits,
    }
    return run_rows, selection_rows, seed_audit


def _distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "population_std": statistics.pstdev(ordered),
    }


def _variant_summary(
    name: str,
    runs: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    ids: Sequence[str],
    seeds: Sequence[str],
) -> dict[str, Any]:
    selected_runs = [row for row in runs if row["variant"] == name]
    selected_rows = [row for row in selections if row["variant"] == name]
    by_key = {
        (str(row["seed"]), str(row["ID"])): row["selected_candidate"] == "b"
        for row in selected_rows
    }
    choice_rates = {
        identifier: sum(by_key[(seed, identifier)] for seed in seeds) / len(seeds)
        for identifier in ids
    }
    pairwise: list[float] = []
    for left_index, left in enumerate(seeds):
        for right in seeds[left_index + 1 :]:
            pairwise.append(
                sum(
                    by_key[(left, identifier)] == by_key[(right, identifier)]
                    for identifier in ids
                )
                / len(ids)
            )
    return {
        "variant": name,
        "feature_set": selected_runs[0]["feature_set"],
        "feature_count": selected_runs[0]["feature_count"],
        "logistic_l2": selected_runs[0]["logistic_l2"],
        "score": _distribution([float(row["multi_score"]) for row in selected_runs]),
        "gain_vs_model25": _distribution(
            [float(row["gain_vs_model25"]) for row in selected_runs]
        ),
        "weighted_brier": _distribution(
            [float(row["weighted_brier"]) for row in selected_runs]
        ),
        "weighted_ece": _distribution(
            [float(row["weighted_ece"]) for row in selected_runs]
        ),
        "selected_b_rows": _distribution(
            [float(row["selected_b_rows"]) for row in selected_runs]
        ),
        "minimum_partition_gain": min(
            float(row["minimum_partition_gain_vs_model25"])
            for row in selected_runs
        ),
        "runs_beating_model25": sum(
            float(row["gain_vs_model25"]) > 0 for row in selected_runs
        ),
        "runs_all_five_partitions_positive": sum(
            int(row["positive_partition_count"]) == 5 for row in selected_runs
        ),
        "mean_pairwise_choice_agreement": statistics.fmean(pairwise),
        "minimum_pairwise_choice_agreement": min(pairwise),
        "unanimous_choice_row_fraction": sum(
            rate in {0.0, 1.0} for rate in choice_rates.values()
        )
        / len(ids),
        "unstable_choice_row_fraction": sum(
            0.25 <= rate <= 0.75 for rate in choice_rates.values()
        )
        / len(ids),
    }


def run_audit(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_file = _absolute(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    (
        id_column,
        rows,
        candidate_a,
        candidate_b,
        details_a,
        details_b,
        base_config,
        lineage,
    ) = load_verified_inputs(config)
    variants = _variants(config)
    seeds = tuple(str(value) for value in config["partition_seeds"])
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("partition seed panel must contain unique alternatives")
    ids = [row.identifier for row in rows]
    all_runs: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    seed_audits: list[dict[str, Any]] = []
    for seed in seeds:
        runs, selections, seed_audit = evaluate_seed(
            rows,
            candidate_a,
            candidate_b,
            details_a,
            details_b,
            base_config,
            variants,
            seed,
            outer_fold=int(config["outer_fold"]),
            calibration_bins=int(config["calibration_bins"]),
        )
        all_runs.extend(runs)
        all_selections.extend(selections)
        seed_audits.append(seed_audit)

    output = _absolute(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_fields = tuple(all_runs[0])
    selection_fields = tuple(all_selections[0])
    _atomic_csv(output / "runs.csv", run_fields, all_runs)
    _atomic_csv(output / "selections.csv", selection_fields, all_selections)
    summaries = [
        _variant_summary(variant.name, all_runs, all_selections, ids, seeds)
        for variant in variants
    ]
    summary_rows = [
        {
            "variant": value["variant"],
            "feature_set": value["feature_set"],
            "feature_count": value["feature_count"],
            "logistic_l2": value["logistic_l2"],
            "score_min": value["score"]["minimum"],
            "score_mean": value["score"]["mean"],
            "score_max": value["score"]["maximum"],
            "score_std": value["score"]["population_std"],
            "gain_min": value["gain_vs_model25"]["minimum"],
            "gain_mean": value["gain_vs_model25"]["mean"],
            "gain_max": value["gain_vs_model25"]["maximum"],
            "runs_beating_model25": value["runs_beating_model25"],
            "runs_all_partitions_positive": value[
                "runs_all_five_partitions_positive"
            ],
            "choice_agreement_mean": value["mean_pairwise_choice_agreement"],
            "unanimous_row_fraction": value["unanimous_choice_row_fraction"],
            "unstable_row_fraction": value["unstable_choice_row_fraction"],
            "weighted_brier_mean": value["weighted_brier"]["mean"],
            "weighted_ece_mean": value["weighted_ece"]["mean"],
        }
        for value in summaries
    ]
    _atomic_csv(output / "summary.csv", tuple(summary_rows[0]), summary_rows)
    baseline_score = _score_partition(rows, candidate_a)
    challenger_score = _score_partition(rows, candidate_b)
    costs = model33.row_costs(rows, candidate_a, candidate_b)
    oracle = {
        row.identifier: (
            candidate_b[row.identifier]
            if costs[row.identifier].choose_b
            else candidate_a[row.identifier]
        )
        for row in rows
    }
    oracle_score = _score_partition(rows, oracle)
    reference_seed = str(config["reference_seed"])
    reference_variant = str(config["reference_variant"])
    reference_run = next(
        row
        for row in all_runs
        if row["seed"] == reference_seed and row["variant"] == reference_variant
    )
    expected_reference_score = float(config["expected_reference_score"])
    reference_delta = float(reference_run["multi_score"]) - expected_reference_score
    if abs(reference_delta) > float(config["reference_score_tolerance"]):
        raise RuntimeError(
            f"NumPy audit failed Model33 reproduction: delta={reference_delta}"
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "method": "fixed salt-by-feature/regularization Cartesian robustness panel",
        "outer_fold": int(config["outer_fold"]),
        "rows": len(rows),
        "id_column": id_column,
        "partition_seed_count": len(seeds),
        "variant_count": len(variants),
        "run_count": len(all_runs),
        "partition_seeds": list(seeds),
        "variants": [asdict(value) for value in variants],
        "baseline_model25_score": baseline_score,
        "challenger_model22_score": challenger_score,
        "row_oracle_score_ceiling_not_a_model": oracle_score,
        "reference_reproduction": {
            "seed": reference_seed,
            "variant": reference_variant,
            "expected_score": expected_reference_score,
            "audit_score": reference_run["multi_score"],
            "difference": reference_delta,
        },
        "variant_summaries": summaries,
        "seed_audits": seed_audits,
        "lineage": lineage,
        "config_sha256": _sha256(config_file),
        "artifacts": {
            "runs": str(output / "runs.csv"),
            "runs_sha256": _sha256(output / "runs.csv"),
            "selections": str(output / "selections.csv"),
            "selections_sha256": _sha256(output / "selections.csv"),
            "summary": str(output / "summary.csv"),
            "summary_sha256": _sha256(output / "summary.csv"),
        },
        "leakage_assertion": (
            "For every seed and partition, evaluation IDs are absent from "
            "selector labels, selector importance, evaluation-LM targets, and "
            "all nested selector-training LM targets. No run is selected by "
            "its held-out result; the full declared panel is reported."
        ),
        "checkpoint_selection_limitation": (
            "The audit conditions on the already selected Model22 epoch-7 "
            "artifact, so it does not remove the earlier fold-0 checkpoint-screen bias."
        ),
    }
    _atomic_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.config, args.output_dir)
    compact = {
        "rows": report["rows"],
        "run_count": report["run_count"],
        "baseline_model25_score": report["baseline_model25_score"]["multi_score"],
        "reference_reproduction": report["reference_reproduction"],
        "variant_summaries": report["variant_summaries"],
        "artifacts": report["artifacts"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
