#!/usr/bin/env python3
"""Model56: fold-safe next-token-conditioned initial-token restoration."""

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
    token_key,
)
from road_htr.metrics import competition_error  # noqa: E402
from road_htr.schema import read_table  # noqa: E402
from road_htr.submission import write_submission  # noqa: E402


OUTPUT_ROOT = (ROOT / "outputs" / "model56").resolve()


class Model56GateError(RuntimeError):
    pass


def build_contextual_initial_map(
    labels: Iterable[str], minimum_count: int, minimum_ratio: float
) -> dict[tuple[str, str], dict[str, Any]]:
    variants: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for label in labels:
        tokens = str(label).split()
        if len(tokens) < 2:
            continue
        key = (token_key(tokens[0]), token_key(tokens[1]))
        if all(key):
            variants[key][tokens[0]] += 1
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for key, counts in variants.items():
        candidate, support = counts.most_common(1)[0]
        observations = sum(counts.values())
        if support >= minimum_count and support / observations >= minimum_ratio:
            mapping[key] = {
                "target": candidate,
                "support": support,
                "observations": observations,
                "ratio": support / observations,
                "initial_core": key[0],
                "next_core": key[1],
            }
    return mapping


def restore_contextual_initial_token(
    text: str, mapping: Mapping[tuple[str, str], Mapping[str, Any]]
) -> tuple[str, dict[str, Any] | None]:
    value = str(text)
    spans = list(re.finditer(r"\S+", value))
    if len(spans) < 2:
        return value, None
    before = spans[0].group()
    key = (token_key(before), token_key(spans[1].group()))
    rule = mapping.get(key)
    if rule is None:
        return value, None
    after = str(rule["target"])
    if before == after:
        return value, None
    match = spans[0]
    result = value[: match.start()] + after + value[match.end() :]
    return result, {"before": before, "after": after, **rule}


def build_fold_maps(
    rows: Iterable[Mapping[str, str]],
    target_column: str,
    fold_column: str,
    minimum_count: int,
    minimum_ratio: float,
) -> dict[int, dict[tuple[str, str], dict[str, Any]]]:
    materialized = tuple(rows)
    folds = sorted({int(row[fold_column]) for row in materialized})
    return {
        fold: build_contextual_initial_map(
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
        "model_number": 56,
        "position": "initial_token_conditioned_on_next_core",
        "minimum_count": 20,
        "minimum_ratio": 0.9,
        "key": "initial_and_next_ascii_alpha_core_casefold_strip_edge_nonalpha",
        "source_model": "model55",
        "output_root": "outputs/model56",
    }
    if config != expected:
        raise Model56GateError("Model56 config is immutable")
    return config, resolved


def _require_new_output(path: str | Path) -> Path:
    output = resolve(path)
    if output == OUTPUT_ROOT or OUTPUT_ROOT not in output.parents:
        raise Model56GateError("output directory must be below outputs/model56")
    if output.exists():
        raise Model56GateError("output directory must be new")
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
        raise Model56GateError("OOF truth must contain target and fold")
    folds = [int(row["fold"]) for row in truth.rows]
    if Counter(folds) != Counter({fold: 819 for fold in range(5)}):
        raise Model56GateError("truth is not the canonical balanced five-fold table")
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
    values: list[str] = []
    changes: list[dict[str, Any]] = []
    for row in truth.rows:
        identifier = row[truth.id_column].strip()
        fold = int(row["fold"])
        result, change = restore_contextual_initial_token(source[identifier], maps[fold])
        values.append(result)
        if change is not None:
            changes.append({"ID": identifier, "fold": fold, **change})
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
            [references[index] for index in positions],
            [baseline[index] for index in positions],
        )
        after = competition_error(
            [references[index] for index in positions],
            [values[index] for index in positions],
        )
        fold_scores[str(fold)] = {
            "baseline": before,
            "model56": after,
            "multi_score_gain": after["multi_score"] - before["multi_score"],
        }
    baseline_score = competition_error(references, baseline)
    score = competition_error(references, values)
    report = {
        "status": "complete_five_fold_oof",
        "model_number": 56,
        "rows": len(ids),
        "changed_rows": len(changes),
        "config": {"path": str(config_resolved), "sha256": sha256(config_resolved)},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "source": source_receipt,
        "map_sizes": {str(fold): len(mapping) for fold, mapping in maps.items()},
        "baseline_score": baseline_score,
        "score": score,
        "multi_score_gain": score["multi_score"] - baseline_score["multi_score"],
        "fold_scores": fold_scores,
        "no_fold_regresses": all(
            fold_scores[str(fold)]["multi_score_gain"] >= 0 for fold in range(5)
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
    mapping = build_contextual_initial_map(
        (row[train.target_column] for row in train.rows),
        int(config["minimum_count"]),
        float(config["minimum_ratio"]),
    )
    values: list[str] = []
    changes: list[dict[str, Any]] = []
    for identifier in ids:
        result, change = restore_contextual_initial_token(source[identifier], mapping)
        values.append(result)
        if change is not None:
            changes.append({"ID": identifier, **change})
    destination = _require_new_output(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    predictions_output = destination / "test.csv"
    changes_output = destination / "changes.jsonl"
    _write_predictions(predictions_output, test.id_column, ids, values)
    _write_changes(changes_output, changes)
    if submission_path is not None:
        if sample_path is None:
            raise Model56GateError("sample is required with submission")
        write_submission(
            resolve(sample_path),
            resolve(submission_path),
            dict(zip(ids, values, strict=True)),
        )
    report = {
        "status": "complete_test_candidate_no_leaderboard_score_claimed",
        "model_number": 56,
        "rows": len(ids),
        "changed_rows": len(changes),
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
    parser = argparse.ArgumentParser(description="Model56 contextual initial-token restorer")
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
