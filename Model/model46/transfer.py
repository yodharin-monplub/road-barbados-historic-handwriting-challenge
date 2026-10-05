#!/usr/bin/env python3
"""Model46: frozen Model35 language-repair transfer to one genuine fold-0 vector.

There is no search in this module.  The candidate vector is never used to fit
state or choose a configuration: every state sees only truth labels for folds
1--4, then the frozen Model35 context/boundary/confusion recipe is applied once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


class Model46GateError(RuntimeError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, id_column: str, ids: Sequence[str], values: Sequence[str]) -> None:
    if len(ids) != len(values):
        raise Model46GateError("stage prediction count changed")
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


def write_changes(path: Path, ids: Sequence[str], before: Sequence[str], after: Sequence[str], operations: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    if not (len(ids) == len(before) == len(after) == len(operations)):
        raise Model46GateError("change vector lengths differ")
    changed_rows = 0
    changed_operations = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for identifier, old, new, detail in zip(ids, before, after, operations, strict=True):
            if old != new:
                changed_rows += 1
                changed_operations += len(detail)
                handle.write(json.dumps({"id": identifier, "before": old, "after": new, "operations": detail}, ensure_ascii=False) + "\n")
    return {"changed_rows": changed_rows, "changed_operations": changed_operations}


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Model46GateError("config must be a JSON object")
    validate_config(value)
    return value, resolved


def validate_config(config: Mapping[str, Any]) -> None:
    expected_context = {"context_weight": 16.0, "distance_penalty": 32.0, "enabled": True, "fallback_frequency": 8, "minimum_frequency": 1, "seed_ambiguity_ratio": 1.0, "seed_minimum_frequency": 3, "two_edit_min_length": 6}
    expected_boundaries = {"enabled": True, "merge_minimum_frequency": 1, "split_minimum_count": 1}
    expected_confusions = {"candidate_minimum_frequency": 2, "context_weight": 2.0, "enabled": True, "score_margin": 12.0}
    if config.get("model_number") != 46 or config.get("heldout_fold") != 0:
        raise Model46GateError("Model46 is frozen to heldout fold 0")
    if config.get("scope") != "conditional_fold0_transfer_only_not_full_oof":
        raise Model46GateError("claim scope must remain conditional fold 0")
    if config.get("terminal_stage") != "confusions" or config.get("stage_order") != ["raw", "context", "boundaries", "confusions"]:
        raise Model46GateError("terminal policy must stop after confusions")
    if config.get("frozen_recipe") != {"context": expected_context, "boundaries": expected_boundaries, "confusions": expected_confusions}:
        raise Model46GateError("the Model35 fold-0 recipe is immutable")
    if config.get("output_root") != "outputs/model46":
        raise Model46GateError("outputs must stay isolated under outputs/model46")
    if not isinstance(config.get("implementations"), Mapping) or not isinstance(config.get("model35"), Mapping):
        raise Model46GateError("complete Model35/source pin ledger required")


def verify_pins(config: Mapping[str, Any]) -> dict[str, Any]:
    receipts: dict[str, str] = {}
    ledger: dict[str, Mapping[str, str]] = {"truth": config["truth"], **config["model35"], **config["implementations"]}
    for name, spec in ledger.items():
        path = resolve(spec["path"])
        if not path.is_file() or path.is_symlink():
            raise Model46GateError(f"pinned artifact missing or symlinked: {path}")
        digest = sha256(path)
        if digest != spec["sha256"]:
            raise Model46GateError(f"pinned artifact hash mismatch for {name}")
        receipts[name] = digest
    policy = json.loads(resolve(config["model35"]["terminal_policy"]["path"]).read_text(encoding="utf-8"))
    recipe = json.loads(resolve(config["model35"]["recipe"]["path"]).read_text(encoding="utf-8"))
    report = json.loads(resolve(config["model35"]["report"]["path"]).read_text(encoding="utf-8"))
    if policy.get("terminal_stage") != "confusions" or policy.get("terminal_stage_index") != 3 or policy.get("selection_fold") != 0:
        raise Model46GateError("Model35 terminal policy does not freeze fold-0 confusions")
    if {name: recipe.get("stages", {}).get(name, {}).get("selected_config") for name in ("context", "boundaries", "confusions")} != config["frozen_recipe"]:
        raise Model46GateError("Model35 recipe differs from frozen transfer config")
    if report.get("scores", {}).get("confusions", {}).get("score", {}).get("multi_score") != policy.get("selection_score"):
        raise Model46GateError("Model35 report/policy terminal score mismatch")
    return {"hashes": receipts, "model35_terminal_selection_score": policy["selection_score"]}


def load_truth(config: Mapping[str, Any]) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    from road_htr.schema import SchemaError, read_table
    table = read_table(resolve(config["truth"]["path"]), require_target=True)
    if table.target_column is None or "fold" not in table.fieldnames:
        raise Model46GateError("truth must carry target and fold")
    ids = tuple(row[table.id_column].strip() for row in table.rows)
    references = tuple(row[table.target_column] for row in table.rows)
    try:
        folds = tuple(int(row["fold"]) for row in table.rows)
    except ValueError as error:
        raise SchemaError("truth folds must be integers") from error
    if set(folds) != set(range(5)) or len(ids) != 4095:
        raise Model46GateError("expected the canonical five-fold 4,095-row truth table")
    return table.id_column, ids, references, folds


def stage_states(references: Sequence[str], folds: Sequence[int], heldout: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit every Model11--13 state from labels whose fold is not heldout."""
    from road_htr.boundary_corrector import TrainingBoundaryModel
    from road_htr.confusion_corrector import TrainingConfusionModel
    from road_htr.contextual_corrector import TrainingLanguageModel

    fit_rows = [(reference, fold) for reference, fold in zip(references, folds, strict=True) if fold != heldout]
    if not fit_rows or any(fold == heldout for _, fold in fit_rows):
        raise Model46GateError("heldout labels reached a state fitter")
    fit_references = tuple(reference for reference, _ in fit_rows)
    audit = {"heldout_fold": heldout, "fit_folds": sorted(set(fold for _, fold in fit_rows)), "fit_rows": len(fit_references), "fit_reference_sha256": hashlib.sha256("".join(f"{value}\n" for value in fit_references).encode("utf-8")).hexdigest(), "fit_uses_candidate_predictions": False, "exclusion_rule": "truth.fold != heldout_fold before all three constructors"}
    return {"context": TrainingLanguageModel(fit_references), "boundaries": TrainingBoundaryModel(fit_references), "confusions": TrainingConfusionModel(fit_references)}, audit


def load_candidate(config: Mapping[str, Any], prediction_csv: str | Path, ids: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
    from road_htr.schema import SchemaError, read_table
    path = resolve(prediction_csv)
    if not path.is_file() or path.is_symlink():
        raise Model46GateError(f"candidate CSV missing or symlinked: {path}")
    table = read_table(path)
    if "prediction" not in table.fieldnames:
        raise SchemaError("candidate CSV must have an explicit prediction column")
    if "fold" in table.fieldnames and any((row.get("fold") or "").strip() not in {"", "0"} for row in table.rows):
        raise Model46GateError("candidate CSV is not exclusively a fold-0 vector")
    by_id = {row[table.id_column].strip(): row.get("prediction", "") for row in table.rows}
    expected = set(ids)
    if set(by_id) != expected:
        raise Model46GateError(f"candidate IDs must exactly equal truth fold 0: missing={len(expected-set(by_id))}, extra={len(set(by_id)-expected)}")
    values = [by_id[identifier] for identifier in ids]
    if any(not value for value in values):
        raise Model46GateError("candidate has a blank prediction")
    return values, {"path": str(path), "sha256": sha256(path), "rows": len(values), "id_column": table.id_column}


def _receipt_strings(value: Any) -> list[str]:
    if isinstance(value, str): return [value]
    if isinstance(value, Mapping): return [item for child in value.values() for item in _receipt_strings(child)]
    if isinstance(value, list): return [item for child in value for item in _receipt_strings(child)]
    return []


def verify_optional_lineage(receipt_path: str | Path | None, candidate: Mapping[str, Any]) -> dict[str, Any]:
    if receipt_path is None:
        return {"status": "not_supplied_id_set_only_model_lineage_not_asserted"}
    path = resolve(receipt_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    strings = _receipt_strings(payload)
    candidate_path = str(Path(candidate["path"]).resolve())
    bound_path = any((resolve(value) == Path(candidate_path) if value else False) for value in strings if "/" in value or value.endswith(".csv"))
    bound_hash = candidate["sha256"] in strings
    if not (bound_path or bound_hash):
        raise Model46GateError("optional lineage receipt does not bind candidate path or SHA256")
    return {"status": "receipt_binds_candidate", "path": str(path), "sha256": sha256(path), "bound_by": "path" if bound_path else "sha256"}


def transform(values: Sequence[str], states: Mapping[str, Any], recipe: Mapping[str, Any]) -> tuple[dict[str, list[str]], dict[str, list[list[dict[str, Any]]]]]:
    from road_htr.boundary_corrector import repair_boundaries
    from road_htr.confusion_corrector import repair_confusions
    from road_htr.contextual_corrector import correct_contextually
    context = recipe["context"]
    context_values, context_ops = zip(*(correct_contextually(value, states["context"], int(context["minimum_frequency"]), float(context["context_weight"]), float(context["distance_penalty"]), int(context["fallback_frequency"]), int(context["two_edit_min_length"]), int(context["seed_minimum_frequency"]), float(context["seed_ambiguity_ratio"])) for value in values), strict=True)
    boundary = recipe["boundaries"]
    boundary_values, boundary_ops = zip(*(repair_boundaries(value, states["boundaries"], int(boundary["split_minimum_count"]), int(boundary["merge_minimum_frequency"])) for value in context_values), strict=True)
    confusion = recipe["confusions"]
    confusion_values, confusion_ops = zip(*(repair_confusions(value, states["confusions"], float(confusion["context_weight"]), float(confusion["score_margin"]), int(confusion["candidate_minimum_frequency"])) for value in boundary_values), strict=True)
    return {"context": list(context_values), "boundaries": list(boundary_values), "confusions": list(confusion_values)}, {"context": list(context_ops), "boundaries": list(boundary_ops), "confusions": list(confusion_ops)}


def output_dir(config: Mapping[str, Any], value: str | Path) -> Path:
    root = resolve(config["output_root"])
    path = resolve(value)
    if path == root or root not in path.parents or path.exists():
        raise Model46GateError("output must be a new directory below outputs/model46")
    return path


def dry_audit(config_path: str | Path) -> dict[str, Any]:
    config, path = load_config(config_path)
    pins = verify_pins(config)
    _, _, references, folds = load_truth(config)
    _, state_audit = stage_states(references, folds, 0)
    return {"status": "cpu_dry_audit_passed_no_candidate_score_claimed", "config_path": str(path), "config_sha256": sha256(path), "model46_transfer_sha256": sha256(Path(__file__).resolve()), "pins": pins, "state_fit_audit": state_audit, "scope": config["scope"], "terminal_stage": "confusions"}


def run(config_path: str | Path, prediction_csv: str | Path, output: str | Path, lineage_receipt: str | Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    config, config_path = load_config(config_path)
    config_digest = sha256(config_path)
    source_digest = sha256(Path(__file__).resolve())
    pins_before = verify_pins(config)
    id_column, all_ids, all_references, folds = load_truth(config)
    heldout = config["heldout_fold"]
    heldout_ids = tuple(identifier for identifier, fold in zip(all_ids, folds, strict=True) if fold == heldout)
    heldout_references = tuple(reference for reference, fold in zip(all_references, folds, strict=True) if fold == heldout)
    raw, candidate = load_candidate(config, prediction_csv, heldout_ids)
    lineage = verify_optional_lineage(lineage_receipt, candidate)
    states, fit_audit = stage_states(all_references, folds, heldout)
    staged, operations = transform(raw, states, config["frozen_recipe"])
    destination = output_dir(config, output)
    destination.mkdir(parents=True, exist_ok=False)
    from road_htr.metrics import competition_error
    values = {"raw": raw, **staged}
    scores: dict[str, Any] = {}
    previous = raw
    for index, stage in enumerate(config["stage_order"]):
        current = values[stage]
        csv_path = destination / "predictions" / f"{index:02d}_{stage}_fold0.csv"
        write_csv(csv_path, id_column, heldout_ids, current)
        entry: dict[str, Any] = {"scope": "CONDITIONAL_HELDOUT_FOLD_0_ONLY_NOT_FULL_OOF", "score": competition_error(heldout_references, current), "prediction_artifact": str(csv_path), "prediction_sha256": sha256(csv_path)}
        if stage != "raw":
            changes_path = destination / "changes" / f"{index:02d}_{stage}_fold0.jsonl"
            entry.update(write_changes(changes_path, heldout_ids, previous, current, operations[stage]))
            entry["change_artifact"] = str(changes_path)
            entry["change_sha256"] = sha256(changes_path)
            entry["multi_score_delta_from_previous"] = entry["score"]["multi_score"] - scores[config["stage_order"][index - 1]]["score"]["multi_score"]
        scores[stage] = entry
        previous = current
    pins_after = verify_pins(config)
    if pins_before != pins_after or candidate["sha256"] != sha256(candidate["path"]) or config_digest != sha256(config_path) or source_digest != sha256(Path(__file__).resolve()):
        raise Model46GateError("TOCTOU: config, source, pinned artifact, or candidate CSV changed during transfer")
    report = {"status": "complete_conditional_fold0_transfer_not_full_oof", "model_number": 46, "scope": config["scope"], "runtime_seconds": time.monotonic() - started, "config_sha256": config_digest, "model46_transfer_sha256": source_digest, "candidate": candidate, "lineage": lineage, "pins": pins_before, "state_fit_audit": fit_audit, "recipe": config["frozen_recipe"], "terminal_stage": "confusions", "selection_performed": False, "scores": scores}
    write_json(destination / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Model46 frozen Model35 fold-0 language-repair transfer")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lineage-receipt", type=Path)
    parser.add_argument("--dry-audit", action="store_true")
    args = parser.parse_args()
    if args.dry_audit:
        print(json.dumps(dry_audit(args.config), ensure_ascii=False, indent=2))
    elif args.predictions is None or args.output_dir is None:
        parser.error("--predictions and --output-dir are required unless --dry-audit")
    else:
        print(json.dumps(run(args.config, args.predictions, args.output_dir, args.lineage_receipt), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
