"""Guarded final Model35 submission assembly. No training or upload occurs here."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import yaml

from .fold_model_postprocess import (
    PROJECT_ROOT, _absolute, _atomic_json, _atomic_prediction_csv,
    _manifest_ids, _prediction_map, _read_json, _score_equal, _sha256,
    verify_fold_model_lineage,
)
from .metrics import competition_error
from .model35_deployment import fit_apply_deployment
from .schema import read_table
from .submission import (
    clean_prediction_preserving_internal_spaces, validate_submission, write_submission,
)


def check_oof_scores(truth_path: Path, report: dict[str, Any], fallback: Path) -> dict[str, Any]:
    """Recompute the fixed candidate and require improvement on every fold.

    This numeric gate is conservative and is not a confidence interval or a
    claim of fully nested validation. Source lineage is checked separately.
    """
    policy = _read_json(PROJECT_ROOT / "model35/terminal_policy.json")
    truth = read_table(truth_path, require_target=True)
    if report.get("truth_sha256") != _sha256(truth_path):
        raise ValueError("OOF truth checksum mismatch")
    for key, value in {
        "folds": list(range(5)), "rows": len(truth.rows),
        "checkpoint_epoch_zero_based": policy["checkpoint_epoch_zero_based"],
        "checkpoint_policy": policy["checkpoint_policy"],
        "policy_selection_fold": policy["selection_fold"],
    }.items():
        if report.get(key) != value:
            raise ValueError(f"incomplete or incompatible OOF report: {key}")
    if {int(row["fold"]) for row in truth.rows} != set(range(5)):
        raise ValueError("truth must contain all five canonical folds")
    ids = [row[truth.id_column].strip() for row in truth.rows]
    maps = {"fallback": _prediction_map(fallback)}
    for stage in ("raw", policy["terminal_stage"]):
        record = report["scores"][stage]
        path = _absolute(record["prediction_artifact"])
        if _sha256(path) != record["prediction_sha256"]:
            raise ValueError(f"OOF prediction checksum mismatch: {stage}")
        maps[stage] = _prediction_map(path)
    if any(set(values) != set(ids) for values in maps.values()):
        raise ValueError("OOF prediction ID mismatch")
    scores: dict[str, Any] = {}
    for name, values in maps.items():
        subsets = {"global": truth.rows, "confirmation_folds_1_4": tuple(
            row for row in truth.rows if int(row["fold"]) != 0
        )}
        subsets.update({str(fold): tuple(row for row in truth.rows if int(row["fold"]) == fold)
                        for fold in range(5)})
        scores[name] = {label: competition_error(
            [row[truth.target_column] for row in rows],
            [values[row[truth.id_column].strip()] for row in rows],
        ) for label, rows in subsets.items()}
        if name != "fallback":
            record = report["scores"][name]
            if not _score_equal(scores[name]["global"], record["global"]):
                raise ValueError(f"reported global score differs: {name}")
            for fold in range(5):
                if not _score_equal(scores[name][str(fold)], record["per_fold"][str(fold)]):
                    raise ValueError(f"reported fold score differs: {name}, {fold}")
    selected = scores[policy["terminal_stage"]]
    for baseline in ("raw", "fallback"):
        for subset in ("global", "confirmation_folds_1_4", *(str(fold) for fold in range(5))):
            if selected[subset]["multi_score"] <= scores[baseline][subset]["multi_score"]:
                raise ValueError(f"candidate does not improve {baseline} on {subset}")
    return {"scores": scores, "numeric_gate": "strict_improvement_over_raw_and_fallback_on_all_folds",
            "claim_scope": "Conditional OOF evidence; fold 0 selected the policy. Not a public score or prize guarantee."}


def verify_recorded_fold_lineage(truth: Path, original: dict[str, Any]) -> None:
    """Recheck provenance AND require it to match the snapshot used for fitting."""
    lineage = original["lineage"]
    fresh = verify_fold_model_lineage(
        truth, lineage["all_row_predictions"], lineage["all_row_inference_manifest"],
        lineage["training_manifest"], lineage["heldout_oof_predictions"],
        lineage["oof_evaluation_receipt"], original["heldout_fold"],
        original["checkpoint_epoch_zero_based"], original["model_family"],
    )
    # Older valid reports predate the excluded-validation-rows diagnostic.
    # They must match every field they recorded, without requiring new fields.
    changed = [key for key, value in lineage.items() if key not in fresh or fresh[key] != value]
    if changed:
        raise ValueError(f"source lineage changed since recipe fitting: {changed}")
    recipe = _absolute(original["fit_recipe_artifact"])
    if _sha256(recipe) != original["fit_recipe_sha256"]:
        raise ValueError("source fit recipe checksum mismatch")


def verify_oof_sources(truth: Path, report: dict[str, Any]) -> None:
    sources = report.get("fold_reports", {})
    if set(sources) != {str(fold) for fold in range(5)}:
        raise ValueError("missing five-fold source reports")
    for fold in range(5):
        source = sources[str(fold)]
        path = _absolute(source["report"])
        if _sha256(path) != source["report_sha256"]:
            raise ValueError("source fold report checksum mismatch")
        original = _read_json(path)
        if original["heldout_fold"] != fold:
            raise ValueError("source fold mismatch")
        for key in ("checkpoint_epoch_zero_based", "checkpoint_policy", "policy_selection_fold"):
            if original.get(key) != report.get(key):
                raise ValueError("source fold policy mismatch")
        verify_recorded_fold_lineage(truth, original)
        for stage in ("raw", "confusions"):
            record = original["scores"][stage]
            artifact = _absolute(record["prediction_artifact"])
            if _sha256(artifact) != record["prediction_sha256"]:
                raise ValueError("source stage prediction checksum mismatch")
            combined = _prediction_map(_absolute(report["scores"][stage]["prediction_artifact"]))
            table = read_table(truth)
            expected_ids = {row[table.id_column].strip() for row in table.rows if int(row["fold"]) == fold}
            source_values = _prediction_map(artifact)
            if set(source_values) != expected_ids:
                raise ValueError("source stage IDs do not match the fold")
            if any(combined.get(key) != value for key, value in source_values.items()):
                raise ValueError("combined OOF differs from its source predictions")


def verify_final_inference(truth: Path, test: Path, train_predictions: Path,
                           test_predictions: Path, training_path: Path,
                           conversion_path: Path) -> dict[str, str]:
    """Bind both inference inputs to one converted all-data epoch-7 model."""
    training = _read_json(training_path)
    prep, config = training["preparation"], training["config"]
    canonical_config = PROJECT_ROOT / "model36/config.yaml"
    if config != yaml.safe_load(canonical_config.read_text()):
        raise ValueError("deployment configuration differs from Model36")
    if prep.get("fold", "missing") is not None:
        raise ValueError("deployment requires an all-data training run")
    if config["training"].get("all_data_epochs") != 8 or config["training"].get("all_data_cosine_epochs") != 10:
        raise ValueError("deployment training schedule differs from the fixed policy")
    if _sha256(_absolute(config["train_csv"])) != _sha256(truth):
        raise ValueError("all-data model used a different truth file")
    truth_table = read_table(truth, require_target=True)
    truth_ids = {row[truth_table.id_column].strip() for row in truth_table.rows}
    excluded = {line.strip() for line in (PROJECT_ROOT / "model22/exclude_ids.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")}
    if set(prep["excluded_ids"]) != excluded:
        raise ValueError("unexpected all-data exclusions")
    train_ids = set(_manifest_ids(_absolute(prep["train_manifest"])))
    if train_ids != truth_ids - excluded or prep["train_rows"] != len(train_ids):
        raise ValueError("all-data training manifest does not cover eligible truth")
    conversion = _read_json(conversion_path)
    if (conversion.get("training_manifest_sha256") != _sha256(training_path)
            or _absolute(conversion.get("training_manifest", "missing")) != training_path):
        raise ValueError("conversion training-manifest provenance mismatch")
    checkpoint = _absolute(conversion["checkpoint"])
    weights = _absolute(conversion["weights"])
    if checkpoint.parent != training_path.parent or not re.fullmatch(r"checkpoint_0?7-.+\.ckpt", checkpoint.name):
        raise ValueError("conversion is not the final run's epoch-7 checkpoint")
    checkpoint_paths = list(training_path.parent.glob("checkpoint_*.ckpt"))
    epochs = [re.fullmatch(r"checkpoint_(\d+)-.+\.ckpt", path.name) for path in checkpoint_paths]
    if (len(epochs) != 8 or any(match is None for match in epochs)
            or {int(match.group(1)) for match in epochs if match} != set(range(8))
            or not list(training_path.parent.glob("best_*.safetensors"))):
        raise ValueError("all-data training completion artifacts are missing")
    for path, key in ((checkpoint, "checkpoint_sha256"), (weights, "weights_sha256")):
        if _sha256(path) != conversion[key]:
            raise ValueError(f"conversion checksum mismatch: {key}")
    expected_config_hash = _sha256(PROJECT_ROOT / "model36/config.yaml")
    for source, prediction in ((truth, train_predictions), (test, test_predictions)):
        manifest = _read_json(prediction.with_suffix(".manifest.json"))
        if manifest.get("config") != config:
            raise ValueError("training and inference configurations differ")
        for key, value in {"fold": None, "image_variant": "full", "constrain_charset": False,
                           "preserve_internal_spaces": True, "input_sha256": _sha256(source),
                           "predictions_sha256": _sha256(prediction),
                           "config_sha256": expected_config_hash,
                           "model_sha256": conversion["weights_sha256"]}.items():
            if key not in manifest or manifest[key] != value:
                raise ValueError(f"inference provenance mismatch: {key}")
        if _absolute(manifest["model"]) != weights:
            raise ValueError("inference model path differs from converted weights")
        table = read_table(source)
        ids = {row[table.id_column].strip() for row in table.rows}
        if set(_prediction_map(prediction)) != ids or manifest["rows"] != len(ids):
            raise ValueError("inference does not cover the complete input")
    return {"model_sha256": conversion["weights_sha256"],
            "checkpoint_sha256": conversion["checkpoint_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("truth", "data/processed/train_folds.csv"), ("test", "data/raw/Test.csv"),
                          ("sample", "data/raw/SampleSubmission.csv"),
                          ("fallback", "outputs/model66/oof/allfold_oof.csv")):
        parser.add_argument("--" + name, type=Path, default=PROJECT_ROOT / default)
    for name in ("oof-report", "training-manifest", "training-predictions", "test-predictions",
                 "conversion-receipt", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        setattr(args, name, _absolute(value))
    output = args.output_dir
    if not output.is_relative_to(PROJECT_ROOT / "outputs/model35") or output == PROJECT_ROOT / "outputs/model35":
        raise ValueError("output must be a new subdirectory of outputs/model35")
    if output.exists():
        raise FileExistsError(output)
    report = _read_json(args.oof_report)
    evidence = check_oof_scores(args.truth, report, args.fallback)
    verify_oof_sources(args.truth, report)
    lineage = verify_final_inference(args.truth, args.test, args.training_predictions,
                                    args.test_predictions, args.training_manifest, args.conversion_receipt)
    truth, test, sample = read_table(args.truth, True), read_table(args.test), read_table(args.sample)
    test_ids = [row[test.id_column].strip() for row in test.rows]
    train_ids = [row[truth.id_column].strip() for row in truth.rows]
    if set(test_ids) & set(train_ids):
        raise ValueError("train and test IDs overlap")
    if set(test_ids) != {row[sample.id_column].strip() for row in sample.rows}:
        raise ValueError("test and sample IDs differ")
    paths = {name: value for name, value in vars(args).items() if name != "output_dir"}
    paths.update({"training_inference": args.training_predictions.with_suffix(".manifest.json"),
                  "test_inference": args.test_predictions.with_suffix(".manifest.json"),
                  "terminal_policy": PROJECT_ROOT / "model35/terminal_policy.json"})
    hashes = {name: _sha256(path) for name, path in paths.items()}
    training_values, test_values = _prediction_map(args.training_predictions), _prediction_map(args.test_predictions)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "status.json", {"status": "fitting_not_submission_ready", "input_sha256": hashes})
    result = fit_apply_deployment([training_values[key] for key in train_ids],
                                 [row[truth.target_column] for row in truth.rows],
                                 [test_values[key] for key in test_ids])
    if any(_sha256(path) != hashes[name] for name, path in paths.items()):
        raise RuntimeError("an input changed during CPU fitting")
    for stage, values in result.stage_predictions.items():
        _atomic_prediction_csv(output / f"{stage}_test.csv", test.id_column, test_ids, values)
    submission = output / "submission.csv"
    write_submission(args.sample, submission, dict(zip(test_ids, result.predictions, strict=True)),
                     cleaner=clean_prediction_preserving_internal_spaces)
    validation = validate_submission(submission, args.sample)
    _atomic_json(output / "report.json", {"status": "validated_candidate_not_uploaded",
                 "input_sha256": hashes, "lineage": lineage, "oof_evidence": evidence,
                 "fit_diagnostics_not_oof": result.fits, "submission": str(submission),
                 "submission_sha256": _sha256(submission), "validation": validation})
    _atomic_json(output / "status.json", {"status": "validated_candidate_not_uploaded"})
    print(f"Candidate: {submission}\nSHA256: {_sha256(submission)}", flush=True)


if __name__ == "__main__":
    main()
