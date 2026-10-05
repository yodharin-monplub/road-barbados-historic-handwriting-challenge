#!/usr/bin/env python3
"""Model57: fold-safe, context-conditioned medial case restoration."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from model54.restore_final_tokens import (  # noqa: E402
    _atomic_json,
    _prediction_map,
    _write_changes,
    _write_predictions,
    resolve,
    sha256,
)
from road_htr.lexicon_corrector import TOKEN_PATTERN  # noqa: E402
from road_htr.metrics import competition_error  # noqa: E402
from road_htr.schema import read_table  # noqa: E402
from road_htr.submission import write_submission  # noqa: E402


OUTPUT_ROOT = (ROOT / "outputs" / "model57").resolve()
TOKEN_SPAN = re.compile(r"\S+")


class Model57GateError(RuntimeError):
    pass


def token_parts(token: str) -> tuple[str, str, str] | None:
    match = TOKEN_PATTERN.fullmatch(str(token))
    return match.groups() if match else None


def token_core(token: str) -> str:
    parts = token_parts(token)
    return parts[1].casefold() if parts else ""


def change_kind(before: str, after: str) -> str:
    source = token_parts(before)
    target = token_parts(after)
    if source is None or target is None:
        return "unsupported"
    core_changed = source[1] != target[1]
    edge_changed = (source[0], source[2]) != (target[0], target[2])
    if core_changed and edge_changed:
        return "case_and_edge_punctuation"
    if core_changed:
        return "case"
    if edge_changed:
        return "edge_punctuation_only"
    return "same"


def build_medial_map(
    labels: Iterable[str], minimum_count: int, minimum_ratio: float
) -> dict[tuple[str, str, str], dict[str, Any]]:
    variants: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for label in labels:
        tokens = str(label).split()
        cores = [token_core(token) for token in tokens]
        for index in range(1, len(tokens) - 1):
            key = (cores[index - 1], cores[index], cores[index + 1])
            if all(key):
                variants[key][tokens[index]] += 1
    mapping: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, counts in variants.items():
        candidate, support = counts.most_common(1)[0]
        observations = sum(counts.values())
        if support >= minimum_count and support / observations >= minimum_ratio:
            mapping[key] = {
                "target": candidate,
                "support": support,
                "observations": observations,
                "ratio": support / observations,
                "left_core": key[0],
                "current_core": key[1],
                "right_core": key[2],
            }
    return mapping


def restore_medial_case(
    text: str,
    mapping: Mapping[tuple[str, str, str], Mapping[str, Any]],
    allowed_change_kinds: frozenset[str] = frozenset({"case", "case_and_edge_punctuation"}),
) -> tuple[str, list[dict[str, Any]]]:
    value = str(text)
    spans = list(TOKEN_SPAN.finditer(value))
    tokens = [span.group() for span in spans]
    cores = [token_core(token) for token in tokens]
    changes: list[dict[str, Any]] = []
    for index in range(1, len(tokens) - 1):
        rule = mapping.get((cores[index - 1], cores[index], cores[index + 1]))
        if rule is None:
            continue
        before = tokens[index]
        after = str(rule["target"])
        kind = change_kind(before, after)
        if before == after or kind not in allowed_change_kinds:
            continue
        changes.append(
            {"token_index": index, "before": before, "after": after, "change_kind": kind, **rule}
        )
    result = value
    for change in reversed(changes):
        match = spans[int(change["token_index"])]
        result = result[: match.start()] + str(change["after"]) + result[match.end() :]
    return result, changes


def build_fold_maps(
    rows: Iterable[Mapping[str, str]],
    target_column: str,
    fold_column: str,
    minimum_count: int,
    minimum_ratio: float,
) -> dict[int, dict[tuple[str, str, str], dict[str, Any]]]:
    materialized = tuple(rows)
    folds = sorted({int(row[fold_column]) for row in materialized})
    return {
        fold: build_medial_map(
            (
                row[target_column]
                for row in materialized
                if int(row[fold_column]) != fold
            ),
            minimum_count,
            minimum_ratio,
        )
        for fold in folds
    }


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve(path)
    config = json.loads(resolved.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "model_number": 57,
        "position": "medial_token_conditioned_on_both_neighbor_cores",
        "minimum_count": 10,
        "minimum_ratio": 1.0,
        "allowed_change_kinds": ["case", "case_and_edge_punctuation"],
        "key": "left_current_right_ascii_alpha_core_casefold",
        "source_model": "model56",
        "output_root": "outputs/model57",
    }
    if config != expected:
        raise Model57GateError("Model57 config is immutable")
    return config, resolved


def _require_new_output(path: str | Path) -> Path:
    output = resolve(path)
    if output == OUTPUT_ROOT or OUTPUT_ROOT not in output.parents:
        raise Model57GateError("output directory must be below outputs/model57")
    if output.exists():
        raise Model57GateError("output directory must be new")
    return output


def evaluate_oof(
    config_path: str | Path,
    truth_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config, config_resolved = load_config(config_path)
    truth = read_table(resolve(truth_path), require_target=True)
    if truth.target_column is None or "fold" not in truth.fieldnames:
        raise Model57GateError("OOF truth must contain target and fold")
    folds = [int(row["fold"]) for row in truth.rows]
    if Counter(folds) != Counter({fold: 819 for fold in range(5)}):
        raise Model57GateError("truth is not the canonical balanced five-fold table")
    ids = [row[truth.id_column].strip() for row in truth.rows]
    references = [row[truth.target_column] for row in truth.rows]
    source, source_receipt = _prediction_map(predictions_path, ids)
    maps = build_fold_maps(
        truth.rows,
        truth.target_column,
        "fold",
        int(config["minimum_count"]),
        float(config["minimum_ratio"]),
    )
    allowed = frozenset(config["allowed_change_kinds"])
    values: list[str] = []
    changes: list[dict[str, Any]] = []
    for row in truth.rows:
        identifier = row[truth.id_column].strip()
        fold = int(row["fold"])
        result, operations = restore_medial_case(source[identifier], maps[fold], allowed)
        values.append(result)
        if operations:
            changes.append({"ID": identifier, "fold": fold, "operations": operations})
    destination = _require_new_output(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    predictions_output = destination / "allfold_oof.csv"
    changes_output = destination / "changes.jsonl"
    _write_predictions(predictions_output, truth.id_column, ids, values)
    _write_changes(changes_output, changes)
    baseline = [source[identifier] for identifier in ids]
    fold_scores: dict[str, Any] = {}
    for fold in range(5):
        positions = [index for index, value in enumerate(folds) if value == fold]
        before = competition_error(
            [references[index] for index in positions], [baseline[index] for index in positions]
        )
        after = competition_error(
            [references[index] for index in positions], [values[index] for index in positions]
        )
        fold_scores[str(fold)] = {
            "baseline": before,
            "model57": after,
            "multi_score_gain": after["multi_score"] - before["multi_score"],
        }
    baseline_score = competition_error(references, baseline)
    score = competition_error(references, values)
    report = {
        "status": "complete_five_fold_oof",
        "model_number": 57,
        "rows": len(ids),
        "changed_rows": len(changes),
        "changed_tokens": sum(len(row["operations"]) for row in changes),
        "config": {"path": str(config_resolved), "sha256": sha256(config_resolved)},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "source": source_receipt,
        "map_sizes": {str(fold): len(mapping) for fold, mapping in maps.items()},
        "baseline_score": baseline_score,
        "score": score,
        "multi_score_gain": score["multi_score"] - baseline_score["multi_score"],
        "fold_scores": fold_scores,
        "all_folds_improve": all(
            fold_scores[str(fold)]["multi_score_gain"] > 0 for fold in range(5)
        ),
        "artifacts": {
            "predictions": str(predictions_output),
            "predictions_sha256": sha256(predictions_output),
            "changes": str(changes_output),
            "changes_sha256": sha256(changes_output),
        },
    }
    _atomic_json(destination / "report.json", report)
    return report


def apply_test(
    config_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> dict[str, Any]:
    config, config_resolved = load_config(config_path)
    train = read_table(resolve(train_path), require_target=True)
    test = read_table(resolve(test_path))
    assert train.target_column is not None
    ids = [row[test.id_column].strip() for row in test.rows]
    source, source_receipt = _prediction_map(predictions_path, ids)
    mapping = build_medial_map(
        (row[train.target_column] for row in train.rows),
        int(config["minimum_count"]),
        float(config["minimum_ratio"]),
    )
    allowed = frozenset(config["allowed_change_kinds"])
    values: list[str] = []
    changes: list[dict[str, Any]] = []
    for identifier in ids:
        result, operations = restore_medial_case(source[identifier], mapping, allowed)
        values.append(result)
        if operations:
            changes.append({"ID": identifier, "operations": operations})
    destination = _require_new_output(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    predictions_output = destination / "test.csv"
    changes_output = destination / "changes.jsonl"
    _write_predictions(predictions_output, test.id_column, ids, values)
    _write_changes(changes_output, changes)
    if submission_path is not None:
        if sample_path is None:
            raise Model57GateError("sample is required with submission")
        write_submission(
            resolve(sample_path),
            resolve(submission_path),
            dict(zip(ids, values, strict=True)),
        )
    report = {
        "status": "complete_test_candidate_no_leaderboard_score_claimed",
        "model_number": 57,
        "rows": len(ids),
        "changed_rows": len(changes),
        "changed_tokens": sum(len(row["operations"]) for row in changes),
        "config": {"path": str(config_resolved), "sha256": sha256(config_resolved)},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "source": source_receipt,
        "map_size": len(mapping),
        "artifacts": {
            "predictions": str(predictions_output),
            "predictions_sha256": sha256(predictions_output),
            "changes": str(changes_output),
            "changes_sha256": sha256(changes_output),
            "submission": str(resolve(submission_path)) if submission_path else None,
            "submission_sha256": sha256(resolve(submission_path)) if submission_path else None,
        },
    }
    _atomic_json(destination / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Model57 contextual medial case restorer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    oof = subparsers.add_parser("oof")
    oof.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    oof.add_argument("--truth", type=Path, default=ROOT / "data/processed/train_folds.csv")
    oof.add_argument("--predictions", type=Path, required=True)
    oof.add_argument("--output-dir", type=Path, required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    apply_parser.add_argument("--train", type=Path, default=ROOT / "data/processed/train_folds.csv")
    apply_parser.add_argument("--test", type=Path, default=ROOT / "data/raw/Test.csv")
    apply_parser.add_argument("--predictions", type=Path, required=True)
    apply_parser.add_argument("--output-dir", type=Path, required=True)
    apply_parser.add_argument("--sample", type=Path)
    apply_parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if args.command == "oof":
        report = evaluate_oof(args.config, args.truth, args.predictions, args.output_dir)
    else:
        report = apply_test(
            args.config,
            args.train,
            args.test,
            args.predictions,
            args.output_dir,
            args.sample,
            args.submission,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
