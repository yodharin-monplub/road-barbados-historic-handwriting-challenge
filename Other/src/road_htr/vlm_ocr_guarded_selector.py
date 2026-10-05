"""Model52: conservative, fold-safe VLM-versus-OCR whole-line selector.

The selector is deliberately target-blind at decision time.  A character
language model is fitted on labels outside the held fold; each decision then
receives only the fallback and VLM strings.  Fixed structural gates keep an
unreliable generative model from replacing a proven OCR line merely because it
looks fluent.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .ctc_decode import CharNGramLM
from .metrics import competition_error, levenshtein
from .schema import SchemaError, read_table
from .submission import write_submission


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (PROJECT_ROOT / "outputs" / "model52").resolve()
ROLE_MARKER = re.compile(
    r"<\|(?:assistant|user|system|im_start|im_end)[^>]*\|>|^\s*(?:assistant|user)\s*:",
    re.IGNORECASE,
)


class Model52GateError(RuntimeError):
    """Raised when an artifact or frozen-policy contract is violated."""


@dataclass(frozen=True)
class GuardedPolicy:
    char_lm_order: int
    char_lm_smoothing: float
    minimum_language_advantage: float
    edit_distance_penalty: float
    maximum_normalized_char_distance: float
    minimum_character_length_ratio: float
    maximum_character_length_ratio: float
    minimum_token_length_ratio: float
    maximum_token_length_ratio: float
    repeated_word_ngram_size: int
    maximum_repeated_word_ngram_count: int


@dataclass(frozen=True)
class SelectionDecision:
    selected: str
    selected_source: str
    reason: str
    fallback_language_quality: float
    vlm_language_quality: float
    language_advantage: float
    required_language_advantage: float
    normalized_char_distance: float
    character_length_ratio: float
    token_length_ratio: float


EXPECTED_POLICY = GuardedPolicy(
    char_lm_order=5,
    char_lm_smoothing=0.2,
    minimum_language_advantage=0.04,
    edit_distance_penalty=0.08,
    maximum_normalized_char_distance=0.60,
    minimum_character_length_ratio=0.65,
    maximum_character_length_ratio=1.40,
    minimum_token_length_ratio=0.60,
    maximum_token_length_ratio=1.50,
    repeated_word_ngram_size=4,
    maximum_repeated_word_ngram_count=1,
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_predictions(
    path: Path, id_column: str, ids: Sequence[str], predictions: Sequence[str]
) -> None:
    if len(ids) != len(predictions):
        raise Model52GateError("prediction vector length changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[id_column, "prediction"])
            writer.writeheader()
            for identifier, prediction in zip(ids, predictions, strict=True):
                writer.writerow({id_column: identifier, "prediction": prediction})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _inside_output_root(path: Path) -> bool:
    return path != OUTPUT_ROOT and OUTPUT_ROOT in path.parents


def require_new_output_dir(path: str | Path) -> Path:
    output = resolve(path)
    if not _inside_output_root(output):
        raise Model52GateError("output directory must be below outputs/model52")
    if output.exists():
        raise Model52GateError("output directory must not already exist")
    return output


def load_config(path: str | Path) -> tuple[dict[str, Any], Path, GuardedPolicy]:
    resolved = resolve(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Model52GateError("config must be a JSON object")
    if payload.get("model_number") != 52:
        raise Model52GateError("config is not Model52")
    if payload.get("policy_name") != "frozen_guarded_char_lm_v1":
        raise Model52GateError("unexpected selector policy")
    if payload.get("selection_inputs") != [
        "fallback_prediction",
        "vlm_prediction",
        "fold_excluded_label_char_lm",
    ]:
        raise Model52GateError("selection-input ledger changed")
    forbidden = set(payload.get("forbidden_selection_inputs", []))
    if forbidden != {"heldout_target", "identifier", "image_geometry", "test_membership"}:
        raise Model52GateError("forbidden-input ledger changed")
    try:
        policy = GuardedPolicy(**payload["policy"])
    except (KeyError, TypeError) as error:
        raise Model52GateError("policy fields are incomplete") from error
    if policy != EXPECTED_POLICY:
        raise Model52GateError("Model52 policy is immutable")
    if payload.get("fallback_model") != "model25" or payload.get("vlm_stage") != "model46_confusions":
        raise Model52GateError("candidate roles changed")
    if payload.get("output_root") != "outputs/model52":
        raise Model52GateError("output root changed")
    return payload, resolved, policy


def language_quality(text: str, model: CharNGramLM) -> float:
    """Mean character log probability under a train-label language model."""

    if not text:
        return -100.0
    total = 0.0
    prefix = ""
    for character in text:
        total += model.log_probability(prefix, character)
        prefix += character
    return total / len(text)


def _repeats_word_ngram(text: str, size: int, maximum_count: int) -> bool:
    words = text.casefold().split()
    if len(words) < size:
        return False
    counts = Counter(tuple(words[index : index + size]) for index in range(len(words) - size + 1))
    return bool(counts and max(counts.values()) > maximum_count)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def select_line(
    fallback: str,
    vlm: str,
    language_model: CharNGramLM,
    policy: GuardedPolicy = EXPECTED_POLICY,
) -> SelectionDecision:
    """Choose one line without receiving an ID, fold, image, or target."""

    fallback = str(fallback).strip()
    vlm = str(vlm).strip()
    if not fallback:
        raise Model52GateError("fallback OCR prediction is blank")

    fallback_quality = language_quality(fallback, language_model)
    vlm_quality = language_quality(vlm, language_model)
    advantage = vlm_quality - fallback_quality
    char_ratio = _ratio(len(vlm), len(fallback))
    token_ratio = _ratio(len(vlm.split()), len(fallback.split()))
    distance = _ratio(levenshtein(tuple(fallback), tuple(vlm)), max(len(fallback), len(vlm)))
    required = policy.minimum_language_advantage + policy.edit_distance_penalty * distance

    def decision(source: str, reason: str) -> SelectionDecision:
        return SelectionDecision(
            selected=vlm if source == "vlm" else fallback,
            selected_source=source,
            reason=reason,
            fallback_language_quality=fallback_quality,
            vlm_language_quality=vlm_quality,
            language_advantage=advantage,
            required_language_advantage=required,
            normalized_char_distance=distance,
            character_length_ratio=char_ratio,
            token_length_ratio=token_ratio,
        )

    if not vlm:
        return decision("fallback", "blank_vlm")
    if ROLE_MARKER.search(vlm):
        return decision("fallback", "role_marker")
    if vlm == fallback:
        return decision("fallback", "identical")
    if _repeats_word_ngram(
        vlm,
        policy.repeated_word_ngram_size,
        policy.maximum_repeated_word_ngram_count,
    ):
        return decision("fallback", "repeated_word_ngram")
    if not policy.minimum_character_length_ratio <= char_ratio <= policy.maximum_character_length_ratio:
        return decision("fallback", "character_length_ratio")
    if not policy.minimum_token_length_ratio <= token_ratio <= policy.maximum_token_length_ratio:
        return decision("fallback", "token_length_ratio")
    if distance > policy.maximum_normalized_char_distance:
        return decision("fallback", "char_distance")
    if not math.isfinite(advantage) or advantage < required:
        return decision("fallback", "insufficient_language_advantage")
    return decision("vlm", "guarded_language_advantage")


def _prediction_map(
    path: str | Path,
    expected_ids: Sequence[str],
    *,
    allow_superset: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    resolved = resolve(path)
    table = read_table(resolved)
    if "prediction" not in table.fieldnames:
        raise SchemaError(f"{resolved} must contain an explicit prediction column")
    mapping = {
        row[table.id_column].strip(): (row.get("prediction") or "").strip()
        for row in table.rows
    }
    expected = set(expected_ids)
    observed = set(mapping)
    valid = expected <= observed if allow_superset else expected == observed
    if not valid:
        raise Model52GateError(
            f"prediction IDs differ: missing={len(expected-observed)}, "
            f"extra={len(observed-expected)}"
        )
    selected = {identifier: mapping[identifier] for identifier in expected_ids}
    return selected, {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "artifact_rows": len(mapping),
        "used_rows": len(selected),
        "accepted_superset": allow_superset and observed != expected,
    }


def _fit_lm(references: Iterable[str], policy: GuardedPolicy) -> CharNGramLM:
    values = tuple(references)
    if not values:
        raise Model52GateError("language-model fit partition is empty")
    return CharNGramLM(policy.char_lm_order, policy.char_lm_smoothing).fit(values)


def _select_vector(
    ids: Sequence[str],
    fallback: Mapping[str, str],
    vlm: Mapping[str, str],
    model: CharNGramLM,
    policy: GuardedPolicy,
) -> tuple[list[str], list[SelectionDecision]]:
    decisions = [select_line(fallback[identifier], vlm[identifier], model, policy) for identifier in ids]
    return [decision.selected for decision in decisions], decisions


def _reason_counts(decisions: Sequence[SelectionDecision]) -> dict[str, int]:
    return dict(sorted(Counter(decision.reason for decision in decisions).items()))


def _write_decisions(path: Path, ids: Sequence[str], decisions: Sequence[SelectionDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for identifier, decision in zip(ids, decisions, strict=True):
            handle.write(json.dumps({"ID": identifier, **asdict(decision)}, ensure_ascii=False) + "\n")


def evaluate(
    config_path: str | Path,
    truth_path: str | Path,
    fallback_path: str | Path,
    vlm_path: str | Path,
    heldout_fold: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    config, resolved_config, policy = load_config(config_path)
    truth = read_table(resolve(truth_path), require_target=True)
    if truth.target_column is None or "fold" not in truth.fieldnames:
        raise Model52GateError("truth must contain target and fold columns")
    try:
        folds = [int(row["fold"]) for row in truth.rows]
    except ValueError as error:
        raise Model52GateError("fold values must be integers") from error
    if set(folds) != set(range(5)):
        raise Model52GateError("expected canonical folds 0..4")
    held_rows = [row for row, fold in zip(truth.rows, folds, strict=True) if fold == heldout_fold]
    fit_rows = [row for row, fold in zip(truth.rows, folds, strict=True) if fold != heldout_fold]
    if len(held_rows) != 819 or len(fit_rows) != 3276:
        raise Model52GateError("expected 819 held and 3,276 fit rows")
    ids = [row[truth.id_column].strip() for row in held_rows]
    references = [row[truth.target_column] for row in held_rows]
    model = _fit_lm((row[truth.target_column] for row in fit_rows), policy)
    fallback, fallback_receipt = _prediction_map(fallback_path, ids, allow_superset=True)
    vlm, vlm_receipt = _prediction_map(vlm_path, ids)
    selected, decisions = _select_vector(ids, fallback, vlm, model, policy)

    destination = require_new_output_dir(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    prediction_path = destination / "predictions.csv"
    decision_path = destination / "decisions.jsonl"
    _atomic_predictions(prediction_path, truth.id_column, ids, selected)
    _write_decisions(decision_path, ids, decisions)
    fallback_values = [fallback[identifier] for identifier in ids]
    vlm_values = [vlm[identifier] for identifier in ids]
    report = {
        "status": "complete_conditional_heldout_evaluation_not_full_oof",
        "model_number": 52,
        "scope": f"conditional_fold_{heldout_fold}_only",
        "heldout_fold": heldout_fold,
        "rows": len(ids),
        "fit_rows": len(fit_rows),
        "fit_folds": sorted(set(folds) - {heldout_fold}),
        "selection_uses_heldout_targets": False,
        "selection_inputs": config["selection_inputs"],
        "policy_name": config["policy_name"],
        "policy": asdict(policy),
        "config": {"path": str(resolved_config), "sha256": sha256(resolved_config)},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "inputs": {"fallback": fallback_receipt, "vlm": vlm_receipt},
        "selected_counts": dict(Counter(decision.selected_source for decision in decisions)),
        "reason_counts": _reason_counts(decisions),
        "scores": {
            "fallback": competition_error(references, fallback_values),
            "vlm": competition_error(references, vlm_values),
            "selected": competition_error(references, selected),
        },
        "artifacts": {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "decisions": str(decision_path),
            "decisions_sha256": sha256(decision_path),
        },
    }
    _atomic_json(destination / "report.json", report)
    return report


def apply(
    config_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    fallback_path: str | Path,
    vlm_path: str | Path,
    output_dir: str | Path,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> dict[str, Any]:
    config, resolved_config, policy = load_config(config_path)
    train = read_table(resolve(train_path), require_target=True)
    test = read_table(resolve(test_path))
    assert train.target_column is not None
    ids = [row[test.id_column].strip() for row in test.rows]
    model = _fit_lm((row[train.target_column] for row in train.rows), policy)
    fallback, fallback_receipt = _prediction_map(fallback_path, ids)
    vlm, vlm_receipt = _prediction_map(vlm_path, ids)
    selected, decisions = _select_vector(ids, fallback, vlm, model, policy)
    if any(not value for value in selected):
        raise Model52GateError("selector produced a blank test prediction")

    destination = require_new_output_dir(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    prediction_path = destination / "predictions.csv"
    decision_path = destination / "decisions.jsonl"
    _atomic_predictions(prediction_path, test.id_column, ids, selected)
    _write_decisions(decision_path, ids, decisions)
    if submission_path is not None:
        if sample_path is None:
            raise Model52GateError("sample path is required with submission path")
        write_submission(
            resolve(sample_path),
            resolve(submission_path),
            dict(zip(ids, selected, strict=True)),
        )
    report = {
        "status": "complete_test_selection_no_score_claimed",
        "model_number": 52,
        "rows": len(ids),
        "fit_rows": len(train.rows),
        "selection_inputs": config["selection_inputs"],
        "policy_name": config["policy_name"],
        "policy": asdict(policy),
        "config": {"path": str(resolved_config), "sha256": sha256(resolved_config)},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "inputs": {"fallback": fallback_receipt, "vlm": vlm_receipt},
        "selected_counts": dict(Counter(decision.selected_source for decision in decisions)),
        "reason_counts": _reason_counts(decisions),
        "artifacts": {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "decisions": str(decision_path),
            "decisions_sha256": sha256(decision_path),
            "submission": str(resolve(submission_path)) if submission_path else None,
        },
    }
    _atomic_json(destination / "report.json", report)
    return report


def audit(config_path: str | Path, truth_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    config, resolved_config, policy = load_config(config_path)
    truth = read_table(resolve(truth_path), require_target=True)
    if truth.target_column is None or "fold" not in truth.fieldnames:
        raise Model52GateError("truth must contain target and fold columns")
    folds = [int(row["fold"]) for row in truth.rows]
    fold_counts = Counter(folds)
    if fold_counts != Counter({fold: 819 for fold in range(5)}):
        raise Model52GateError("truth is not the canonical balanced five-fold table")
    # Exercise the selector using only labels from folds 1..4 as LM fit text.
    model = _fit_lm(
        (row[truth.target_column] for row, fold in zip(truth.rows, folds, strict=True) if fold != 0),
        policy,
    )
    probe = select_line("sealed and delivered", "Sealed and delivered", model, policy)
    report = {
        "status": "cpu_dry_audit_passed_waiting_for_model50_and_model46_predictions",
        "model_number": 52,
        "rows": len(truth.rows),
        "fold_counts": dict(sorted(fold_counts.items())),
        "selection_uses_heldout_targets": False,
        "selection_inputs": config["selection_inputs"],
        "forbidden_selection_inputs": config["forbidden_selection_inputs"],
        "policy": asdict(policy),
        "config": {"path": str(resolved_config), "sha256": sha256(resolved_config)},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "probe": asdict(probe),
    }
    output = resolve(output_path)
    if output != OUTPUT_ROOT / "dry_run.json":
        raise Model52GateError("audit output must be outputs/model52/dry_run.json")
    _atomic_json(output, report)
    return report
