#!/usr/bin/env python3
"""Model54: fold-safe unanimous canonicalization of the final token only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (ROOT / "outputs" / "model54").resolve()
EDGE_NONALPHA = re.compile(r"^[^A-Za-z]+|[^A-Za-z]+$")
FINAL_TOKEN = re.compile(r"\S+(?=\s*$)")


class Model54GateError(RuntimeError):
    pass


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_key(token: str) -> str:
    return EDGE_NONALPHA.sub("", str(token)).casefold()


def build_final_map(
    labels: Iterable[str], minimum_count: int, minimum_ratio: float
) -> dict[str, dict[str, Any]]:
    variants: dict[str, Counter[str]] = defaultdict(Counter)
    for label in labels:
        tokens = str(label).split()
        if not tokens:
            continue
        key = token_key(tokens[-1])
        if key:
            variants[key][tokens[-1]] += 1
    mapping: dict[str, dict[str, Any]] = {}
    for key, counts in variants.items():
        candidate, support = counts.most_common(1)[0]
        observations = sum(counts.values())
        if support >= minimum_count and support / observations >= minimum_ratio:
            mapping[key] = {
                "target": candidate,
                "support": support,
                "observations": observations,
                "ratio": support / observations,
            }
    return mapping


def restore_final_token(
    text: str, mapping: Mapping[str, Mapping[str, Any]]
) -> tuple[str, dict[str, Any] | None]:
    """Replace only the last non-whitespace span; preserve all other bytes."""

    value = str(text)
    match = FINAL_TOKEN.search(value)
    if match is None:
        return value, None
    before = match.group()
    rule = mapping.get(token_key(before))
    if rule is None:
        return value, None
    after = str(rule["target"])
    if before == after:
        return value, None
    result = value[: match.start()] + after + value[match.end() :]
    return result, {"before": before, "after": after, **rule}


def build_fold_maps(
    rows: Iterable[Mapping[str, str]],
    target_column: str,
    fold_column: str,
    minimum_count: int,
    minimum_ratio: float,
) -> dict[int, dict[str, dict[str, Any]]]:
    materialized = tuple(rows)
    folds = sorted({int(row[fold_column]) for row in materialized})
    return {
        fold: build_final_map(
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
        "model_number": 54,
        "position": "final_token_only",
        "minimum_count": 15,
        "minimum_ratio": 1.0,
        "key": "ascii_alpha_core_casefold_strip_edge_nonalpha",
        "source_model": "model25",
        "output_root": "outputs/model54",
    }
    if config != expected:
        raise Model54GateError("Model54 config is immutable")
    return config, resolved


def _prediction_map(path: str | Path, ids: Sequence[str]) -> tuple[dict[str, str], dict[str, Any]]:
    from road_htr.schema import SchemaError, read_table

    resolved = resolve(path)
    table = read_table(resolved)
    if "prediction" not in table.fieldnames:
        raise SchemaError("source must contain an explicit prediction column")
    values = {
        row[table.id_column].strip(): row.get("prediction", "")
        for row in table.rows
    }
    expected = set(ids)
    if set(values) != expected:
        raise Model54GateError(
            f"source ID mismatch: missing={len(expected-set(values))}, "
            f"extra={len(set(values)-expected)}"
        )
    if any(not values[identifier].strip() for identifier in ids):
        raise Model54GateError("source contains a blank prediction")
    return values, {"path": str(resolved), "sha256": sha256(resolved), "rows": len(values)}


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


def _write_predictions(path: Path, id_column: str, ids: Sequence[str], values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_column, "prediction"])
        writer.writeheader()
        for identifier, value in zip(ids, values, strict=True):
            writer.writerow({id_column: identifier, "prediction": value})


def _write_changes(path: Path, changes: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for change in changes:
            handle.write(json.dumps(change, ensure_ascii=False) + "\n")


def _require_output(path: str | Path, *, directory: bool) -> Path:
    resolved = resolve(path)
    if resolved == OUTPUT_ROOT or OUTPUT_ROOT not in resolved.parents:
        raise Model54GateError("output must be below outputs/model54")
    if directory and resolved.exists():
        raise Model54GateError("output directory must be new")
    return resolved


def evaluate_oof(
    config_path: str | Path,
    truth_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    from road_htr.metrics import competition_error
    from road_htr.schema import read_table

    config, config_resolved = load_config(config_path)
    truth = read_table(resolve(truth_path), require_target=True)
    if truth.target_column is None or "fold" not in truth.fieldnames:
        raise Model54GateError("OOF truth must contain target and fold")
    folds = [int(row["fold"]) for row in truth.rows]
    if Counter(folds) != Counter({fold: 819 for fold in range(5)}):
        raise Model54GateError("truth is not the canonical balanced five-fold table")
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
        result, change = restore_final_token(source[identifier], maps[fold])
        values.append(result)
        if change is not None:
            changes.append({"ID": identifier, "fold": fold, **change})
    destination = _require_output(output_dir, directory=True)
    destination.mkdir(parents=True, exist_ok=False)
    prediction_output = destination / "allfold_oof.csv"
    changes_output = destination / "changes.jsonl"
    _write_predictions(prediction_output, truth.id_column, ids, values)
    _write_changes(changes_output, changes)
    baseline = [source[identifier] for identifier in ids]
    fold_scores: dict[str, Any] = {}
    for fold in range(5):
        positions = [index for index, value in enumerate(folds) if value == fold]
        fold_scores[str(fold)] = {
            "baseline": competition_error(
                [references[index] for index in positions],
                [baseline[index] for index in positions],
            ),
            "model54": competition_error(
                [references[index] for index in positions],
                [values[index] for index in positions],
            ),
        }
        fold_scores[str(fold)]["multi_score_gain"] = (
            fold_scores[str(fold)]["model54"]["multi_score"]
            - fold_scores[str(fold)]["baseline"]["multi_score"]
        )
    baseline_score = competition_error(references, baseline)
    score = competition_error(references, values)
    report = {
        "status": "complete_five_fold_oof",
        "model_number": 54,
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
        "all_folds_improve": all(
            fold_scores[str(fold)]["multi_score_gain"] > 0 for fold in range(5)
        ),
        "artifacts": {
            "predictions": str(prediction_output),
            "predictions_sha256": sha256(prediction_output),
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
    from road_htr.schema import read_table
    from road_htr.submission import write_submission

    config, config_resolved = load_config(config_path)
    train = read_table(resolve(train_path), require_target=True)
    test = read_table(resolve(test_path))
    assert train.target_column is not None
    ids = [row[test.id_column].strip() for row in test.rows]
    source, source_receipt = _prediction_map(predictions_path, ids)
    mapping = build_final_map(
        (row[train.target_column] for row in train.rows),
        int(config["minimum_count"]),
        float(config["minimum_ratio"]),
    )
    values: list[str] = []
    changes: list[dict[str, Any]] = []
    for identifier in ids:
        result, change = restore_final_token(source[identifier], mapping)
        values.append(result)
        if change is not None:
            changes.append({"ID": identifier, **change})
    destination = _require_output(output_dir, directory=True)
    destination.mkdir(parents=True, exist_ok=False)
    prediction_output = destination / "test.csv"
    changes_output = destination / "changes.jsonl"
    _write_predictions(prediction_output, test.id_column, ids, values)
    _write_changes(changes_output, changes)
    if submission_path is not None:
        if sample_path is None:
            raise Model54GateError("sample is required with submission")
        write_submission(
            resolve(sample_path),
            resolve(submission_path),
            dict(zip(ids, values, strict=True)),
        )
    report = {
        "status": "complete_test_candidate_no_leaderboard_score_claimed",
        "model_number": 54,
        "rows": len(ids),
        "changed_rows": len(changes),
        "config": {"path": str(config_resolved), "sha256": sha256(config_resolved)},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "source": source_receipt,
        "map_size": len(mapping),
        "artifacts": {
            "predictions": str(prediction_output),
            "predictions_sha256": sha256(prediction_output),
            "changes": str(changes_output),
            "changes_sha256": sha256(changes_output),
            "submission": str(resolve(submission_path)) if submission_path else None,
            "submission_sha256": sha256(resolve(submission_path)) if submission_path else None,
        },
    }
    _atomic_json(destination / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Model54 final-token canonicalizer")
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
