"""CPU-only Model35 deployment core; artifact verification belongs to the caller.

Fit on the final recognizer's predictions of labeled training rows, then apply
to that SAME recognizer's test predictions. This intentionally accepts no test
labels and emits no validation score or submission-ready claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .crossfit_postprocess import STAGES, StageDefinition


FIXED_STAGE_NAMES = ("context", "boundaries", "confusions")


@dataclass(frozen=True)
class DeploymentResult:
    predictions: tuple[str, ...]
    stage_predictions: Mapping[str, tuple[str, ...]]
    fits: Mapping[str, Mapping[str, Any]]


def fit_apply_deployment(
    training_predictions: Sequence[str],
    training_references: Sequence[str],
    test_predictions: Sequence[str],
    *,
    stages: Sequence[StageDefinition] = STAGES[:3],
) -> DeploymentResult:
    """Reuse the fixed fold recipe, carrying corrected train values forward.

    Caller must verify model/input provenance and complete five-fold promotion
    evidence before invoking this for deployment. An all-fold OOF CSV is not
    a substitute for the final recognizer's in-sample training predictions.
    No text normalization is introduced beyond the existing stage functions.
    """
    if tuple(stage.name for stage in stages) != FIXED_STAGE_NAMES:
        raise ValueError("deployment must use context -> boundaries -> confusions")
    if not training_predictions or len(training_predictions) != len(training_references):
        raise ValueError("training predictions and references must be nonempty and aligned")
    if not test_predictions:
        raise ValueError("test predictions must be nonempty")
    for values in (training_predictions, training_references, test_predictions):
        if any(not isinstance(value, str) for value in values):
            raise ValueError("all predictions and references must be strings")
    train_values = list(training_predictions)
    references = list(training_references)
    test_values = list(test_predictions)
    predictions = {"raw": tuple(test_values)}
    reports: dict[str, Mapping[str, Any]] = {}
    for stage in stages:
        fitted = stage.fit(train_values, references)
        if len(fitted.training_predictions) != len(references):
            raise RuntimeError(f"{stage.name} returned the wrong training row count")
        before = test_values
        test_values, operations = stage.transform(before, fitted.state, fitted.config)
        if len(test_values) != len(test_predictions) or len(operations) != len(test_predictions):
            raise RuntimeError(f"{stage.name} returned the wrong test row count")
        if any(not isinstance(value, str) for value in test_values + list(fitted.training_predictions)):
            raise RuntimeError(f"{stage.name} returned non-string predictions")
        predictions[stage.name] = tuple(test_values)
        reports[stage.name] = {
            "selected_config": dict(fitted.config),
            "fit_rows": len(references),
            "test_rows": len(test_values),
            "test_changed_rows": sum(a != b for a, b in zip(before, test_values, strict=True)),
            "diagnostic_status": "TRAIN_SIDE_IN_SAMPLE_NOT_OOF_DO_NOT_CLAIM",
            "recipe_search_diagnostic_not_oof": dict(fitted.report),
        }
        train_values = list(fitted.training_predictions)
    return DeploymentResult(tuple(test_values), predictions, reports)
