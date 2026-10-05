"""Model53: conservative token-level fusion of a VLM into an OCR fallback."""

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
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .lexicon_corrector import TOKEN_PATTERN
from .metrics import competition_error, levenshtein
from .schema import SchemaError, read_table
from .submission import write_submission


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (PROJECT_ROOT / "outputs" / "model53").resolve()
WORD_SPAN = re.compile(r"\S+")


class Model53GateError(RuntimeError):
    """Raised when Model53 cannot prove its fixed data contract."""


@dataclass(frozen=True)
class FusionPolicy:
    minimum_candidate_frequency: int
    minimum_candidate_source_ratio: float
    context_weight: float
    minimum_context_score_advantage: float
    maximum_short_word_edit_distance: int
    maximum_long_word_edit_distance: int
    long_word_minimum_length: int
    require_agreed_neighbor: bool
    require_same_edge_punctuation: bool


@dataclass(frozen=True)
class TokenChange:
    fallback_index: int
    vlm_index: int
    before: str
    after: str
    source_core: str
    candidate_core: str
    edit_distance: int
    source_frequency: int
    candidate_frequency: int
    left_support: int
    right_support: int
    score_advantage: float


EXPECTED_POLICY = FusionPolicy(
    minimum_candidate_frequency=5,
    minimum_candidate_source_ratio=4.0,
    context_weight=4.0,
    minimum_context_score_advantage=4.0,
    maximum_short_word_edit_distance=1,
    maximum_long_word_edit_distance=2,
    long_word_minimum_length=6,
    require_agreed_neighbor=True,
    require_same_edge_punctuation=True,
)


class TokenLanguageModel:
    def __init__(self, texts: Iterable[str]):
        self.unigrams: Counter[str] = Counter()
        self.bigrams: Counter[tuple[str, str]] = Counter()
        rows = 0
        for text in texts:
            rows += 1
            words = [_core(token) for token in str(text).split()]
            words = [word for word in words if word]
            self.unigrams.update(words)
            sequence = ["<s>", *words, "</s>"]
            self.bigrams.update(zip(sequence, sequence[1:], strict=False))
        if rows == 0 or not self.unigrams:
            raise Model53GateError("token language-model fit partition is empty")


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


def _atomic_predictions(path: Path, id_column: str, ids: Sequence[str], values: Sequence[str]) -> None:
    if len(ids) != len(values):
        raise Model53GateError("prediction vector length changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[id_column, "prediction"])
            writer.writeheader()
            for identifier, value in zip(ids, values, strict=True):
                writer.writerow({id_column: identifier, "prediction": value})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _core(token: str) -> str:
    match = TOKEN_PATTERN.fullmatch(token)
    return match.group(2).casefold() if match else ""


def _token_parts(token: str) -> tuple[str, str, str] | None:
    match = TOKEN_PATTERN.fullmatch(token)
    return match.groups() if match else None


def load_config(path: str | Path) -> tuple[dict[str, Any], Path, FusionPolicy]:
    resolved = resolve(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("model_number") != 53:
        raise Model53GateError("config is not Model53")
    if payload.get("policy_name") != "frozen_aligned_token_fusion_v1":
        raise Model53GateError("unexpected fusion policy")
    if payload.get("selection_inputs") != [
        "fallback_prediction",
        "vlm_prediction",
        "fold_excluded_label_token_counts",
    ]:
        raise Model53GateError("selection-input ledger changed")
    if set(payload.get("forbidden_selection_inputs", [])) != {
        "heldout_target",
        "identifier",
        "image_geometry",
        "test_membership",
    }:
        raise Model53GateError("forbidden-input ledger changed")
    try:
        policy = FusionPolicy(**payload["policy"])
    except (KeyError, TypeError) as error:
        raise Model53GateError("policy fields are incomplete") from error
    if policy != EXPECTED_POLICY:
        raise Model53GateError("Model53 policy is immutable")
    if payload.get("fallback_model") != "model25" or payload.get("vlm_stage") != "model46_confusions":
        raise Model53GateError("candidate roles changed")
    if payload.get("output_root") != "outputs/model53":
        raise Model53GateError("output root changed")
    return payload, resolved, policy


def _context_score(model: TokenLanguageModel, word: str, left: str, right: str, weight: float) -> float:
    return math.log1p(model.unigrams[word]) + weight * (
        math.log1p(model.bigrams[(left, word)])
        + math.log1p(model.bigrams[(word, right)])
    )


def _agreed_neighbor(
    fallback_tokens: Sequence[str],
    vlm_tokens: Sequence[str],
    fallback_index: int,
    vlm_index: int,
) -> tuple[bool, bool]:
    left = (
        fallback_index > 0
        and vlm_index > 0
        and fallback_tokens[fallback_index - 1] == vlm_tokens[vlm_index - 1]
    )
    right = (
        fallback_index + 1 < len(fallback_tokens)
        and vlm_index + 1 < len(vlm_tokens)
        and fallback_tokens[fallback_index + 1] == vlm_tokens[vlm_index + 1]
    )
    return left, right


def _candidate_change(
    fallback_tokens: Sequence[str],
    vlm_tokens: Sequence[str],
    fallback_index: int,
    vlm_index: int,
    model: TokenLanguageModel,
    policy: FusionPolicy,
) -> TokenChange | None:
    before = fallback_tokens[fallback_index]
    after = vlm_tokens[vlm_index]
    source_parts = _token_parts(before)
    candidate_parts = _token_parts(after)
    if source_parts is None or candidate_parts is None:
        return None
    source_prefix, source_word, source_suffix = source_parts
    candidate_prefix, candidate_word, candidate_suffix = candidate_parts
    if policy.require_same_edge_punctuation and (
        source_prefix != candidate_prefix or source_suffix != candidate_suffix
    ):
        return None
    source = source_word.casefold()
    candidate = candidate_word.casefold()
    if source == candidate:
        return None
    maximum_distance = (
        policy.maximum_long_word_edit_distance
        if max(len(source), len(candidate)) >= policy.long_word_minimum_length
        else policy.maximum_short_word_edit_distance
    )
    distance = levenshtein(source, candidate)
    if distance > maximum_distance:
        return None
    candidate_frequency = model.unigrams[candidate]
    source_frequency = model.unigrams[source]
    if candidate_frequency < policy.minimum_candidate_frequency:
        return None
    if source_frequency and candidate_frequency < policy.minimum_candidate_source_ratio * source_frequency:
        return None
    left_agrees, right_agrees = _agreed_neighbor(
        fallback_tokens, vlm_tokens, fallback_index, vlm_index
    )
    if policy.require_agreed_neighbor and not (left_agrees or right_agrees):
        return None
    fallback_cores = [_core(token) for token in fallback_tokens]
    left = fallback_cores[fallback_index - 1] if fallback_index else "<s>"
    right = fallback_cores[fallback_index + 1] if fallback_index + 1 < len(fallback_cores) else "</s>"
    if not left:
        left = "<s>" if fallback_index == 0 else fallback_cores[fallback_index - 1]
    if not right:
        right = "</s>" if fallback_index + 1 == len(fallback_cores) else fallback_cores[fallback_index + 1]
    source_score = _context_score(model, source, left, right, policy.context_weight)
    candidate_score = _context_score(model, candidate, left, right, policy.context_weight)
    advantage = candidate_score - source_score
    if advantage < policy.minimum_context_score_advantage:
        return None
    return TokenChange(
        fallback_index=fallback_index,
        vlm_index=vlm_index,
        before=before,
        after=after,
        source_core=source,
        candidate_core=candidate,
        edit_distance=distance,
        source_frequency=source_frequency,
        candidate_frequency=candidate_frequency,
        left_support=model.bigrams[(left, candidate)],
        right_support=model.bigrams[(candidate, right)],
        score_advantage=advantage,
    )


def fuse_line(
    fallback: str,
    vlm: str,
    model: TokenLanguageModel,
    policy: FusionPolicy = EXPECTED_POLICY,
) -> tuple[str, list[TokenChange]]:
    """Copy only isolated one-for-one, locally anchored VLM token proposals."""

    fallback = str(fallback)
    vlm = str(vlm)
    if not fallback.strip():
        raise Model53GateError("fallback OCR prediction is blank")
    if not vlm.strip() or fallback == vlm:
        return fallback, []
    fallback_matches = list(WORD_SPAN.finditer(fallback))
    vlm_matches = list(WORD_SPAN.finditer(vlm))
    fallback_tokens = [match.group() for match in fallback_matches]
    vlm_tokens = [match.group() for match in vlm_matches]
    matcher = SequenceMatcher(a=fallback_tokens, b=vlm_tokens, autojunk=False)
    changes: list[TokenChange] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace" or i2 - i1 != 1 or j2 - j1 != 1:
            continue
        change = _candidate_change(
            fallback_tokens, vlm_tokens, i1, j1, model, policy
        )
        if change is not None:
            changes.append(change)
    result = fallback
    for change in sorted(changes, key=lambda item: item.fallback_index, reverse=True):
        match = fallback_matches[change.fallback_index]
        result = result[: match.start()] + change.after + result[match.end() :]
    return result, changes


def _prediction_map(
    path: str | Path,
    expected_ids: Sequence[str],
    *,
    allow_superset: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    resolved = resolve(path)
    table = read_table(resolved)
    if "prediction" not in table.fieldnames:
        raise SchemaError("candidate must have an explicit prediction column")
    observed = {
        row[table.id_column].strip(): (row.get("prediction") or "")
        for row in table.rows
    }
    expected = set(expected_ids)
    observed_ids = set(observed)
    if not (expected <= observed_ids if allow_superset else expected == observed_ids):
        raise Model53GateError(
            f"prediction IDs differ: missing={len(expected-observed_ids)}, "
            f"extra={len(observed_ids-expected)}"
        )
    selected = {identifier: observed[identifier] for identifier in expected_ids}
    return selected, {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "artifact_rows": len(observed),
        "used_rows": len(selected),
        "accepted_superset": allow_superset and expected != observed_ids,
    }


def _require_new_output(path: str | Path) -> Path:
    output = resolve(path)
    if output == OUTPUT_ROOT or OUTPUT_ROOT not in output.parents:
        raise Model53GateError("output directory must be below outputs/model53")
    if output.exists():
        raise Model53GateError("output directory must not already exist")
    return output


def _run_vector(
    ids: Sequence[str],
    fallback: Mapping[str, str],
    vlm: Mapping[str, str],
    model: TokenLanguageModel,
    policy: FusionPolicy,
) -> tuple[list[str], list[dict[str, Any]]]:
    predictions: list[str] = []
    rows: list[dict[str, Any]] = []
    for identifier in ids:
        result, changes = fuse_line(fallback[identifier], vlm[identifier], model, policy)
        predictions.append(result)
        if changes:
            rows.append(
                {
                    "ID": identifier,
                    "before": fallback[identifier],
                    "vlm": vlm[identifier],
                    "after": result,
                    "changes": [asdict(change) for change in changes],
                }
            )
    return predictions, rows


def _write_changes(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        raise Model53GateError("truth must contain target and fold")
    folds = [int(row["fold"]) for row in truth.rows]
    if Counter(folds) != Counter({fold: 819 for fold in range(5)}):
        raise Model53GateError("truth is not the canonical balanced five-fold table")
    held_rows = [row for row, fold in zip(truth.rows, folds, strict=True) if fold == heldout_fold]
    fit_rows = [row for row, fold in zip(truth.rows, folds, strict=True) if fold != heldout_fold]
    ids = [row[truth.id_column].strip() for row in held_rows]
    references = [row[truth.target_column] for row in held_rows]
    model = TokenLanguageModel(row[truth.target_column] for row in fit_rows)
    fallback, fallback_receipt = _prediction_map(fallback_path, ids, allow_superset=True)
    vlm, vlm_receipt = _prediction_map(vlm_path, ids)
    selected, changed_rows = _run_vector(ids, fallback, vlm, model, policy)
    destination = _require_new_output(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    predictions_path = destination / "predictions.csv"
    changes_path = destination / "changes.jsonl"
    _atomic_predictions(predictions_path, truth.id_column, ids, selected)
    _write_changes(changes_path, changed_rows)
    fallback_values = [fallback[identifier] for identifier in ids]
    vlm_values = [vlm[identifier] for identifier in ids]
    report = {
        "status": "complete_conditional_heldout_evaluation_not_full_oof",
        "model_number": 53,
        "scope": f"conditional_fold_{heldout_fold}_only",
        "heldout_fold": heldout_fold,
        "rows": len(ids),
        "fit_rows": len(fit_rows),
        "fit_folds": sorted(set(folds) - {heldout_fold}),
        "selection_uses_heldout_targets": False,
        "policy_name": config["policy_name"],
        "policy": asdict(policy),
        "config": {"path": str(resolved_config), "sha256": sha256(resolved_config)},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "inputs": {"fallback": fallback_receipt, "vlm": vlm_receipt},
        "changed_rows": len(changed_rows),
        "changed_tokens": sum(len(row["changes"]) for row in changed_rows),
        "scores": {
            "fallback": competition_error(references, fallback_values),
            "vlm": competition_error(references, vlm_values),
            "selected": competition_error(references, selected),
        },
        "artifacts": {
            "predictions": str(predictions_path),
            "predictions_sha256": sha256(predictions_path),
            "changes": str(changes_path),
            "changes_sha256": sha256(changes_path),
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
    model = TokenLanguageModel(row[train.target_column] for row in train.rows)
    fallback, fallback_receipt = _prediction_map(fallback_path, ids)
    vlm, vlm_receipt = _prediction_map(vlm_path, ids)
    selected, changed_rows = _run_vector(ids, fallback, vlm, model, policy)
    if any(not value.strip() for value in selected):
        raise Model53GateError("fusion produced a blank test prediction")
    destination = _require_new_output(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    predictions_path = destination / "predictions.csv"
    changes_path = destination / "changes.jsonl"
    _atomic_predictions(predictions_path, test.id_column, ids, selected)
    _write_changes(changes_path, changed_rows)
    if submission_path is not None:
        if sample_path is None:
            raise Model53GateError("sample path is required with submission path")
        write_submission(
            resolve(sample_path),
            resolve(submission_path),
            dict(zip(ids, selected, strict=True)),
        )
    report = {
        "status": "complete_test_fusion_no_score_claimed",
        "model_number": 53,
        "rows": len(ids),
        "fit_rows": len(train.rows),
        "policy_name": config["policy_name"],
        "policy": asdict(policy),
        "config": {"path": str(resolved_config), "sha256": sha256(resolved_config)},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "inputs": {"fallback": fallback_receipt, "vlm": vlm_receipt},
        "changed_rows": len(changed_rows),
        "changed_tokens": sum(len(row["changes"]) for row in changed_rows),
        "artifacts": {
            "predictions": str(predictions_path),
            "predictions_sha256": sha256(predictions_path),
            "changes": str(changes_path),
            "changes_sha256": sha256(changes_path),
            "submission": str(resolve(submission_path)) if submission_path else None,
        },
    }
    _atomic_json(destination / "report.json", report)
    return report


def audit(config_path: str | Path, truth_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    config, resolved_config, policy = load_config(config_path)
    truth = read_table(resolve(truth_path), require_target=True)
    if truth.target_column is None or "fold" not in truth.fieldnames:
        raise Model53GateError("truth must contain target and fold")
    folds = [int(row["fold"]) for row in truth.rows]
    fold_counts = Counter(folds)
    if fold_counts != Counter({fold: 819 for fold in range(5)}):
        raise Model53GateError("truth is not the canonical balanced five-fold table")
    model = TokenLanguageModel(
        row[truth.target_column]
        for row, fold in zip(truth.rows, folds, strict=True)
        if fold != 0
    )
    probe, changes = fuse_line(
        "sealed and delivored in the presence of",
        "sealed and delivered in the presence of",
        model,
        policy,
    )
    report = {
        "status": "cpu_dry_audit_passed_waiting_for_model50_and_model46_predictions",
        "model_number": 53,
        "rows": len(truth.rows),
        "fold_counts": dict(sorted(fold_counts.items())),
        "selection_uses_heldout_targets": False,
        "selection_inputs": config["selection_inputs"],
        "forbidden_selection_inputs": config["forbidden_selection_inputs"],
        "policy": asdict(policy),
        "config": {"path": str(resolved_config), "sha256": sha256(resolved_config)},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "probe": {"output": probe, "changes": [asdict(change) for change in changes]},
    }
    output = resolve(output_path)
    if output != OUTPUT_ROOT / "dry_run.json":
        raise Model53GateError("audit output must be outputs/model53/dry_run.json")
    _atomic_json(output, report)
    return report

