"""Outer-fold cross-fitted postprocessing for complete OCR OOF predictions.

The reusable correctors behind Models 11--15 expose useful transformation
primitives, but their standalone tuners choose settings on the rows they score.
This module instead treats each supplied fold as an untouched outer validation
partition.  Every model, setting, and supervised mapping applied to that fold
is fitted using rows from the other folds only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .approximate_boundary_corrector import (
    CONTEXT_WEIGHTS as JOIN_CONTEXT_WEIGHTS,
    DISTANCE_PENALTIES as JOIN_DISTANCE_PENALTIES,
    MAXIMUM_DISTANCES as JOIN_MAXIMUM_DISTANCES,
    MINIMUM_BIGRAM_COUNTS,
    MINIMUM_TOKEN_LENGTHS,
    TrainingJoinedWordModel,
    repair_approximate_joins,
)
from .boundary_corrector import (
    MERGE_THRESHOLDS,
    SPLIT_THRESHOLDS,
    TrainingBoundaryModel,
    repair_boundaries,
)
from .confusion_corrector import (
    CANDIDATE_MINIMUM_FREQUENCY,
    CONTEXT_WEIGHTS as CONFUSION_CONTEXT_WEIGHTS,
    SCORE_MARGINS,
    TrainingConfusionModel,
    repair_confusions,
)
from .contextual_corrector import (
    CONTEXT_WEIGHTS,
    DISTANCE_PENALTIES,
    FALLBACK_FREQUENCIES,
    MINIMUM_FREQUENCIES,
    SEED_AMBIGUITY_RATIO,
    SEED_MINIMUM_FREQUENCY,
    TWO_EDIT_MIN_LENGTH,
    TrainingLanguageModel,
    correct_contextually,
)
from .ensemble import write_prediction_csv
from .metrics import competition_error
from .schema import SchemaError, predictions_by_id, read_table
from .supervised_error_corrector import (
    MAXIMUM_DISTANCES as ERROR_MAXIMUM_DISTANCES,
    MINIMUM_LENGTHS,
    MINIMUM_PRECISIONS,
    MINIMUM_SUPPORTS,
    SupervisedErrorModel,
    repair_supervised_errors,
)


SCHEMA_VERSION = 1
TOP_TRIALS = 5
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_MODEL22_OUTPUT = (PROJECT_ROOT / "outputs" / "model22").resolve()


@dataclass(frozen=True)
class OofData:
    id_column: str
    prediction_column: str
    ids: tuple[str, ...]
    references: tuple[str, ...]
    folds: tuple[int, ...]
    predictions: tuple[str, ...]


@dataclass
class StageFit:
    config: dict[str, Any]
    state: Any
    training_predictions: list[str]
    training_operations: list[list[dict[str, Any]]]
    report: dict[str, Any]


Transform = Callable[
    [Sequence[str], Any, Mapping[str, Any]],
    tuple[list[str], list[list[dict[str, Any]]]],
]
FitStage = Callable[[Sequence[str], Sequence[str]], StageFit]


@dataclass(frozen=True)
class StageDefinition:
    name: str
    artifact_stem: str
    fit: FitStage
    transform: Transform


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_column(table: Any, path: Path) -> str:
    if table.target_column is not None:
        return table.target_column
    columns = [column for column in table.fieldnames if column != table.id_column]
    if len(columns) != 1:
        raise SchemaError(f"OOF CSV must have exactly one prediction column: {path}")
    return columns[0]


def load_complete_oof(
    truth_path: str | Path, predictions_path: str | Path
) -> OofData:
    """Load a complete OOF CSV in truth order and validate its fold schema."""

    truth_file = Path(truth_path)
    prediction_file = Path(predictions_path)
    truth = read_table(truth_file, require_target=True)
    predictions = read_table(prediction_file)
    if "fold" not in truth.fieldnames:
        raise SchemaError("truth CSV must contain a fold column")
    if not truth.rows:
        raise SchemaError("truth CSV contains no rows")
    assert truth.target_column is not None

    ids = tuple(row[truth.id_column].strip() for row in truth.rows)
    references = tuple(row[truth.target_column] for row in truth.rows)
    try:
        folds = tuple(int(row["fold"]) for row in truth.rows)
    except (TypeError, ValueError) as error:
        raise SchemaError("every truth fold value must be an integer") from error
    unique_folds = sorted(set(folds))
    if len(unique_folds) < 2:
        raise SchemaError("cross-fitted evaluation requires at least two folds")

    prediction_column = _prediction_column(predictions, prediction_file)
    prediction_map = predictions_by_id(predictions, prediction_column)
    expected = set(ids)
    missing = expected - set(prediction_map)
    extra = set(prediction_map) - expected
    if missing or extra:
        raise SchemaError(
            f"OOF prediction ID mismatch: {len(missing)} missing, {len(extra)} extra; "
            f"examples missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )
    ordered_predictions = tuple(prediction_map[identifier] for identifier in ids)
    return OofData(
        id_column=truth.id_column,
        prediction_column=prediction_column,
        ids=ids,
        references=references,
        folds=folds,
        predictions=ordered_predictions,
    )


def _identity(
    values: Sequence[str], _state: Any, _config: Mapping[str, Any]
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    return list(values), [[] for _ in values]


def _changed_counts(
    before: Sequence[str],
    after: Sequence[str],
    operations: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[int, int]:
    return (
        sum(left != right for left, right in zip(before, after, strict=True)),
        sum(len(items) for items in operations),
    )


def _select_configuration(
    source: Sequence[str],
    references: Sequence[str],
    configurations: Iterable[dict[str, Any]],
    evaluate: Callable[
        [Mapping[str, Any]],
        tuple[list[str], list[list[dict[str, Any]]], Any],
    ],
) -> StageFit:
    if not source or len(source) != len(references):
        raise ValueError("stage fitting needs equally sized nonempty inputs and references")
    baseline = competition_error(references, source)
    trials: list[dict[str, Any]] = []
    best_key: tuple[float, int, int, int] | None = None
    best: tuple[
        dict[str, Any], Any, list[str], list[list[dict[str, Any]]]
    ] | None = None

    for order, raw_config in enumerate(configurations):
        config = dict(raw_config)
        transformed, operations, state = evaluate(config)
        if len(transformed) != len(source) or len(operations) != len(source):
            raise RuntimeError("stage returned the wrong number of training rows")
        changed_rows, changed_operations = _changed_counts(
            source, transformed, operations
        )
        score = competition_error(references, transformed)
        record = {
            "config": config,
            "changed_rows": changed_rows,
            "changed_operations": changed_operations,
            "score": score,
        }
        trials.append(record)
        key = (
            float(score["mean_error"]),
            changed_operations,
            changed_rows,
            order,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (config, state, transformed, operations)

    if best is None:
        raise ValueError("stage configuration grid is empty")
    config, state, transformed, operations = best
    changed_rows, changed_operations = _changed_counts(source, transformed, operations)
    selected_score = competition_error(references, transformed)
    ranked_trials = sorted(
        trials,
        key=lambda row: (
            float(row["score"]["mean_error"]),
            int(row["changed_operations"]),
            int(row["changed_rows"]),
        ),
    )
    report = {
        "selected_config": config,
        "training_rows": len(source),
        "training_baseline_score": baseline,
        "training_selected_score": selected_score,
        "training_multi_score_delta": (
            float(selected_score["multi_score"]) - float(baseline["multi_score"])
        ),
        "training_changed_rows": changed_rows,
        "training_changed_operations": changed_operations,
        "trial_count": len(trials),
        "top_trials": ranked_trials[:TOP_TRIALS],
    }
    return StageFit(config, state, transformed, operations, report)


def _context_configurations() -> Iterable[dict[str, Any]]:
    yield {"enabled": False}
    combinations = itertools.product(
        MINIMUM_FREQUENCIES,
        CONTEXT_WEIGHTS,
        DISTANCE_PENALTIES,
        FALLBACK_FREQUENCIES,
    )
    for (
        minimum_frequency,
        context_weight,
        distance_penalty,
        fallback_frequency,
    ) in combinations:
        yield {
            "enabled": True,
            "minimum_frequency": minimum_frequency,
            "context_weight": context_weight,
            "distance_penalty": distance_penalty,
            "fallback_frequency": fallback_frequency,
            "two_edit_min_length": TWO_EDIT_MIN_LENGTH,
            "seed_minimum_frequency": SEED_MINIMUM_FREQUENCY,
            "seed_ambiguity_ratio": SEED_AMBIGUITY_RATIO,
        }


def _transform_context(
    values: Sequence[str], state: TrainingLanguageModel, config: Mapping[str, Any]
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    if not config.get("enabled", True):
        return _identity(values, state, config)
    transformed: list[str] = []
    operations: list[list[dict[str, Any]]] = []
    for value in values:
        result, changes = correct_contextually(
            value,
            state,
            int(config["minimum_frequency"]),
            float(config["context_weight"]),
            float(config["distance_penalty"]),
            int(config["fallback_frequency"]),
            int(config["two_edit_min_length"]),
            int(config["seed_minimum_frequency"]),
            float(config["seed_ambiguity_ratio"]),
        )
        transformed.append(result)
        operations.append(changes)
    return transformed, operations


def _fit_context(source: Sequence[str], references: Sequence[str]) -> StageFit:
    model = TrainingLanguageModel(references)

    def evaluate(config: Mapping[str, Any]):
        transformed, operations = _transform_context(source, model, config)
        return transformed, operations, model

    return _select_configuration(
        source, references, _context_configurations(), evaluate
    )


def _boundary_configurations() -> Iterable[dict[str, Any]]:
    yield {"enabled": False}
    for split_threshold, merge_threshold in itertools.product(
        SPLIT_THRESHOLDS, MERGE_THRESHOLDS
    ):
        yield {
            "enabled": True,
            "split_minimum_count": split_threshold,
            "merge_minimum_frequency": merge_threshold,
        }


def _transform_boundaries(
    values: Sequence[str], state: TrainingBoundaryModel, config: Mapping[str, Any]
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    if not config.get("enabled", True):
        return _identity(values, state, config)
    transformed: list[str] = []
    operations: list[list[dict[str, Any]]] = []
    for value in values:
        result, changes = repair_boundaries(
            value,
            state,
            int(config["split_minimum_count"]),
            int(config["merge_minimum_frequency"]),
        )
        transformed.append(result)
        operations.append(changes)
    return transformed, operations


def _fit_boundaries(source: Sequence[str], references: Sequence[str]) -> StageFit:
    model = TrainingBoundaryModel(references)

    def evaluate(config: Mapping[str, Any]):
        transformed, operations = _transform_boundaries(source, model, config)
        return transformed, operations, model

    return _select_configuration(
        source, references, _boundary_configurations(), evaluate
    )


def _confusion_configurations() -> Iterable[dict[str, Any]]:
    yield {"enabled": False}
    for context_weight, score_margin in itertools.product(
        CONFUSION_CONTEXT_WEIGHTS, SCORE_MARGINS
    ):
        yield {
            "enabled": True,
            "context_weight": context_weight,
            "score_margin": score_margin,
            "candidate_minimum_frequency": CANDIDATE_MINIMUM_FREQUENCY,
        }


def _transform_confusions(
    values: Sequence[str], state: TrainingConfusionModel, config: Mapping[str, Any]
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    if not config.get("enabled", True):
        return _identity(values, state, config)
    transformed: list[str] = []
    operations: list[list[dict[str, Any]]] = []
    for value in values:
        result, changes = repair_confusions(
            value,
            state,
            float(config["context_weight"]),
            float(config["score_margin"]),
            int(config["candidate_minimum_frequency"]),
        )
        transformed.append(result)
        operations.append(changes)
    return transformed, operations


def _fit_confusions(source: Sequence[str], references: Sequence[str]) -> StageFit:
    model = TrainingConfusionModel(references)

    def evaluate(config: Mapping[str, Any]):
        transformed, operations = _transform_confusions(source, model, config)
        return transformed, operations, model

    return _select_configuration(
        source, references, _confusion_configurations(), evaluate
    )


def _join_configurations() -> Iterable[dict[str, Any]]:
    yield {"enabled": False}
    combinations = itertools.product(
        JOIN_MAXIMUM_DISTANCES,
        MINIMUM_TOKEN_LENGTHS,
        MINIMUM_BIGRAM_COUNTS,
        JOIN_CONTEXT_WEIGHTS,
        JOIN_DISTANCE_PENALTIES,
    )
    for (
        maximum_distance,
        minimum_token_length,
        minimum_bigram_count,
        context_weight,
        distance_penalty,
    ) in combinations:
        yield {
            "enabled": True,
            "maximum_distance": maximum_distance,
            "minimum_token_length": minimum_token_length,
            "minimum_bigram_count": minimum_bigram_count,
            "context_weight": context_weight,
            "distance_penalty": distance_penalty,
        }


def _transform_joins(
    values: Sequence[str], state: TrainingJoinedWordModel, config: Mapping[str, Any]
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    if not config.get("enabled", True):
        return _identity(values, state, config)
    transformed: list[str] = []
    operations: list[list[dict[str, Any]]] = []
    for value in values:
        result, changes = repair_approximate_joins(
            value,
            state,
            int(config["maximum_distance"]),
            int(config["minimum_token_length"]),
            int(config["minimum_bigram_count"]),
            float(config["context_weight"]),
            float(config["distance_penalty"]),
        )
        transformed.append(result)
        operations.append(changes)
    return transformed, operations


def _fit_joins(source: Sequence[str], references: Sequence[str]) -> StageFit:
    model = TrainingJoinedWordModel(references)

    def evaluate(config: Mapping[str, Any]):
        transformed, operations = _transform_joins(source, model, config)
        return transformed, operations, model

    return _select_configuration(source, references, _join_configurations(), evaluate)


def _supervised_configurations() -> Iterable[dict[str, Any]]:
    yield {"enabled": False}
    for maximum_distance, minimum_support, minimum_precision, minimum_length in itertools.product(
        ERROR_MAXIMUM_DISTANCES,
        MINIMUM_SUPPORTS,
        MINIMUM_PRECISIONS,
        MINIMUM_LENGTHS,
    ):
        yield {
            "enabled": True,
            "maximum_distance": maximum_distance,
            "minimum_support": minimum_support,
            "minimum_precision": minimum_precision,
            "minimum_length": minimum_length,
        }


def _transform_supervised(
    values: Sequence[str], state: Mapping[str, dict[str, Any]], config: Mapping[str, Any]
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    if not config.get("enabled", True):
        return _identity(values, state, config)
    transformed: list[str] = []
    operations: list[list[dict[str, Any]]] = []
    for value in values:
        result, changes = repair_supervised_errors(value, dict(state))
        transformed.append(result)
        operations.append(changes)
    return transformed, operations


def _fit_supervised(source: Sequence[str], references: Sequence[str]) -> StageFit:
    model = SupervisedErrorModel(zip(source, references, strict=True))

    def evaluate(config: Mapping[str, Any]):
        mappings = (
            {}
            if not config.get("enabled", True)
            else model.mappings(
                int(config["maximum_distance"]),
                int(config["minimum_support"]),
                float(config["minimum_precision"]),
                int(config["minimum_length"]),
            )
        )
        transformed, operations = _transform_supervised(source, mappings, config)
        return transformed, operations, mappings

    fit = _select_configuration(
        source, references, _supervised_configurations(), evaluate
    )
    fit.report["selected_mapping_count"] = len(fit.state)
    return fit


STAGES = (
    StageDefinition("context", "11_context", _fit_context, _transform_context),
    StageDefinition("boundaries", "12_boundaries", _fit_boundaries, _transform_boundaries),
    StageDefinition("confusions", "13_confusions", _fit_confusions, _transform_confusions),
    StageDefinition(
        "approximate_joins", "14_approximate_joins", _fit_joins, _transform_joins
    ),
    StageDefinition(
        "supervised_errors",
        "15_supervised_errors",
        _fit_supervised,
        _transform_supervised,
    ),
)


def _scores(
    data: OofData, predictions: Mapping[str, str]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    values = [predictions[identifier] for identifier in data.ids]
    global_score = competition_error(data.references, values)
    fold_scores: dict[str, dict[str, float]] = {}
    for fold in sorted(set(data.folds)):
        indices = [index for index, value in enumerate(data.folds) if value == fold]
        fold_scores[str(fold)] = competition_error(
            [data.references[index] for index in indices],
            [values[index] for index in indices],
        )
    return global_score, fold_scores


def _prepare_output_dir(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output == PROTECTED_MODEL22_OUTPUT or PROTECTED_MODEL22_OUTPUT in output.parents:
        raise ValueError(
            f"Model28 artifacts may not be written under active Model22 outputs: {output}"
        )
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_paths(output: Path) -> dict[str, Path]:
    result = {"raw": output / "predictions" / "00_raw_oof.csv"}
    for stage in STAGES:
        result[stage.name] = output / "predictions" / f"{stage.artifact_stem}_oof.csv"
    return result


def run_crossfit(
    truth_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Fit the sequential pipeline outside each fold and write OOF artifacts."""

    truth_file = Path(truth_path).resolve()
    predictions_file = Path(predictions_path).resolve()
    data = load_complete_oof(truth_file, predictions_file)
    output = _prepare_output_dir(output_dir)
    stage_predictions: dict[str, dict[str, str]] = {
        "raw": dict(zip(data.ids, data.predictions, strict=True)),
        **{stage.name: {} for stage in STAGES},
    }
    stage_changes: dict[str, list[dict[str, Any]]] = {
        stage.name: [] for stage in STAGES
    }
    fold_records: dict[str, dict[str, Any]] = {}
    supervised_mappings: dict[int, Mapping[str, dict[str, Any]]] = {}

    unique_folds = sorted(set(data.folds))
    for heldout_fold in unique_folds:
        training_indices = [
            index for index, fold in enumerate(data.folds) if fold != heldout_fold
        ]
        heldout_indices = [
            index for index, fold in enumerate(data.folds) if fold == heldout_fold
        ]
        training_folds = sorted({data.folds[index] for index in training_indices})
        training_references = [data.references[index] for index in training_indices]
        training_values = [data.predictions[index] for index in training_indices]
        heldout_values = [data.predictions[index] for index in heldout_indices]
        fold_record: dict[str, Any] = {
            "heldout_fold": heldout_fold,
            "heldout_rows": len(heldout_indices),
            "fit_folds": training_folds,
            "fit_rows": len(training_indices),
            "stages": {},
        }

        for stage in STAGES:
            fit = stage.fit(training_values, training_references)
            heldout_before = heldout_values
            heldout_values, operations = stage.transform(
                heldout_before, fit.state, fit.config
            )
            if len(heldout_values) != len(heldout_indices):
                raise RuntimeError(f"{stage.name} returned the wrong held-out row count")
            for position, index in enumerate(heldout_indices):
                identifier = data.ids[index]
                stage_predictions[stage.name][identifier] = heldout_values[position]
                if heldout_before[position] != heldout_values[position]:
                    stage_changes[stage.name].append(
                        {
                            "id": identifier,
                            "fold": heldout_fold,
                            "before": heldout_before[position],
                            "after": heldout_values[position],
                            "operations": operations[position],
                        }
                    )
            heldout_changed_rows, heldout_changed_operations = _changed_counts(
                heldout_before, heldout_values, operations
            )
            fit.report["heldout_changed_rows"] = heldout_changed_rows
            fit.report["heldout_changed_operations"] = heldout_changed_operations
            fold_record["stages"][stage.name] = fit.report
            training_values = fit.training_predictions
            if stage.name == "supervised_errors":
                supervised_mappings[heldout_fold] = dict(fit.state)

        fold_records[str(heldout_fold)] = fold_record

    for stage in STAGES:
        if set(stage_predictions[stage.name]) != set(data.ids):
            raise RuntimeError(f"{stage.name} did not produce every OOF row")

    paths = _artifact_paths(output)
    for stage_name, path in paths.items():
        values = [stage_predictions[stage_name][identifier] for identifier in data.ids]
        write_prediction_csv(path, data.id_column, list(data.ids), values)

    change_paths: dict[str, str] = {}
    for stage in STAGES:
        path = output / "changes" / f"{stage.artifact_stem}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in stage_changes[stage.name]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        change_paths[stage.name] = str(path)

    fold_config_paths: dict[str, str] = {}
    for fold in unique_folds:
        mapping_path = output / "folds" / f"fold_{fold}_supervised_mappings.json"
        _write_json(mapping_path, supervised_mappings[fold])
        fold_records[str(fold)]["stages"]["supervised_errors"][
            "mapping_artifact"
        ] = str(mapping_path)
        config_path = output / "folds" / f"fold_{fold}_fit.json"
        _write_json(config_path, fold_records[str(fold)])
        fold_config_paths[str(fold)] = str(config_path)

    scores: dict[str, dict[str, Any]] = {}
    previous_name: str | None = None
    for name in ("raw", *(stage.name for stage in STAGES)):
        global_score, fold_scores = _scores(data, stage_predictions[name])
        record: dict[str, Any] = {
            "global": global_score,
            "per_fold": fold_scores,
            "prediction_artifact": str(paths[name]),
        }
        if previous_name is not None:
            previous_global = scores[previous_name]["global"]
            record["global_multi_score_delta"] = (
                float(global_score["multi_score"])
                - float(previous_global["multi_score"])
            )
            record["changed_rows_from_previous"] = len(stage_changes[name])
        scores[name] = record
        previous_name = name

    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "outer-fold sequential cross-fit",
        "selection_metric": "exact challenge multi_score",
        "truth": str(truth_file),
        "truth_sha256": _sha256(truth_file),
        "input_predictions": str(predictions_file),
        "input_predictions_sha256": _sha256(predictions_file),
        "input_prediction_column": data.prediction_column,
        "rows": len(data.ids),
        "folds": unique_folds,
        "fold_rows": {
            str(fold): sum(value == fold for value in data.folds)
            for fold in unique_folds
        },
        "stage_order": ["raw", *(stage.name for stage in STAGES)],
        "scores": scores,
        "fold_fits": fold_records,
        "fold_fit_artifacts": fold_config_paths,
        "change_artifacts": change_paths,
        "leakage_guarantees": [
            (
                "A held-out fold's references never enter its models, setting "
                "search, or supervised mappings."
            ),
            (
                "Later-stage fitting uses only earlier-stage outputs from the "
                "same other-fold training partition."
            ),
            (
                "Every stage includes a no-op candidate and held-out scores "
                "never control stage selection."
            ),
        ],
        "provenance_limit": (
            "The program validates OOF completeness and fold isolation during postprocessing, "
            "but cannot prove that the supplied OCR predictions came from checkpoints that "
            "excluded their corresponding folds."
        ),
    }
    report_path = output / "report.json"
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe outer-fold refit of Models11-15-style correctors"
    )
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_crossfit(args.truth, args.predictions, args.output_dir)
    final = report["scores"]["supervised_errors"]["global"]
    print(
        json.dumps(
            {
                "report": str(Path(args.output_dir).resolve() / "report.json"),
                "rows": report["rows"],
                "folds": report["folds"],
                "raw_multi_score": report["scores"]["raw"]["global"][
                    "multi_score"
                ],
                "final_multi_score": final["multi_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
