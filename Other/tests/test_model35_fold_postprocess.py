from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from road_htr.crossfit_postprocess import STAGES, StageDefinition, StageFit
from road_htr.fold_model_postprocess import (
    PROTOCOL,
    RESERVED_ACTIVE_OUTPUT,
    SUPPORTED_MODEL_FAMILIES,
    FoldModelData,
    _prepare_output_dir,
    _manifest_path,
    _sha256,
    combine_fold_reports,
    fit_apply_fold_pipeline,
    run_fold,
    verify_fold_model_lineage,
)
from road_htr.metrics import competition_error
from road_htr.schema import SchemaError, read_table


class Model35FoldPostprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _rows() -> list[tuple[str, str, int]]:
        return [
            ("a0", "alpha zero", 0),
            ("b0", "beta zero", 0),
            ("a1", "alpha one", 1),
            ("b1", "beta one", 1),
            ("a2", "alpha two", 2),
            ("b2", "beta two", 2),
        ]

    def _write_truth(self, name: str = "truth.csv") -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Target", "fold"])
            writer.writeheader()
            for identifier, target, fold in self._rows():
                writer.writerow({"ID": identifier, "Target": target, "fold": fold})
        return path

    @staticmethod
    def _write_predictions(path: Path, rows: list[tuple[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
            writer.writeheader()
            for identifier, prediction in rows:
                writer.writerow({"ID": identifier, "prediction": prediction})

    def _lineage_fixture(self, family: str = "model22") -> dict[str, Path]:
        truth = self._write_truth()
        rows = self._rows()
        all_predictions = self.root / "all.csv"
        values = [(identifier, f"raw {identifier}") for identifier, _, _ in rows]
        self._write_predictions(all_predictions, values)
        heldout = self.root / "oof.csv"
        self._write_predictions(
            heldout, [(identifier, value) for identifier, value in values if identifier.endswith("0")]
        )

        family_dir = self.root / family
        family_dir.mkdir()
        config = family_dir / "config.yaml"
        config.write_text("fixed: true\n", encoding="utf-8")
        weights = self.root / "epoch07.safetensors"
        weights.write_bytes(b"fixed weights")
        truth_hash = _sha256(truth)
        config_hash = _sha256(config)
        weights_hash = _sha256(weights)

        common = {
            "resolved_config": str(config),
            "config_sha256": config_hash,
            "input_csv": str(truth),
            "input_sha256": truth_hash,
            "model": str(weights),
            "model_sha256": weights_hash,
            "image_variant": "full",
            "constrain_charset": False,
            "preserve_internal_spaces": True,
        }
        all_manifest = all_predictions.with_suffix(".manifest.json")
        all_manifest.write_text(
            json.dumps({**common, "fold": None, "rows": 6}), encoding="utf-8"
        )
        heldout_manifest = heldout.with_suffix(".manifest.json")
        heldout_manifest.write_text(
            json.dumps({**common, "fold": 0, "rows": 2}), encoding="utf-8"
        )

        train_lines = self.root / "train.txt"
        train_lines.write_text(
            "\n".join(str(self.root / f"{identifier}.jpg") for identifier, _, fold in rows if fold != 0)
            + "\n",
            encoding="utf-8",
        )
        validation_lines = self.root / "validation.txt"
        validation_lines.write_text(
            "\n".join(str(self.root / f"{identifier}.jpg") for identifier, _, fold in rows if fold == 0)
            + "\n",
            encoding="utf-8",
        )
        training_manifest = self.root / "training.json"
        training_manifest.write_text(
            json.dumps(
                {
                    "resolved_config": str(config),
                    "config": {
                        "train_csv": str(truth),
                        "training": {"epochs": 8, "stop": "fixed"},
                    },
                    "preparation": {
                        "fold": 0,
                        "train_rows": 4,
                        "validation_rows": 2,
                        "excluded_ids": [],
                        "train_manifest": str(train_lines),
                        "validation_manifest": str(validation_lines),
                    },
                }
            ),
            encoding="utf-8",
        )
        evaluation = heldout.with_suffix(".evaluation.json")
        evaluation.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "weights_sha256": weights_hash,
                    "predictions_sha256": _sha256(heldout),
                    "inference_manifest_sha256": _sha256(heldout_manifest),
                    "signature": {
                        "fold": 0,
                        "expected_rows": 2,
                        "input_sha256": truth_hash,
                        "training_manifest_sha256": _sha256(training_manifest),
                        "config_sha256": config_hash,
                        "preserve_internal_spaces": True,
                        "constrain_charset": False,
                        "checkpoint": "checkpoint_07-0.9000.ckpt",
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "truth": truth,
            "predictions": all_predictions,
            "inference": all_manifest,
            "training": training_manifest,
            "heldout": heldout,
            "evaluation": evaluation,
        }

    def _verify(self, fixture: dict[str, Path], family: str = "model22"):
        return verify_fold_model_lineage(
            fixture["truth"],
            fixture["predictions"],
            fixture["inference"],
            fixture["training"],
            fixture["heldout"],
            fixture["evaluation"],
            0,
            7,
            family,
        )

    def test_lineage_binds_same_model_full_and_signed_oof_predictions(self) -> None:
        lineage = self._verify(self._lineage_fixture())
        self.assertTrue(lineage["all_row_and_oof_heldout_predictions_identical"])
        self.assertEqual(lineage["train_rows"], 4)
        self.assertEqual(lineage["heldout_rows"], 2)

    def test_lineage_accepts_native_height_model71_family(self) -> None:
        fixture = self._lineage_fixture("model71")
        lineage = self._verify(fixture, "model71")
        self.assertEqual(lineage["model_family"], "model71")
        self.assertIn("model71", SUPPORTED_MODEL_FAMILIES)

    def test_runpod_manifest_path_relocates_after_artifact_recovery(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        expected = (project_root / "model71" / "readme.txt").resolve()
        actual = _manifest_path(
            "/workspace/road/model71/readme.txt", "recovered Model71 readme"
        )
        self.assertEqual(actual, expected)

    def test_lineage_rejects_full_prediction_that_differs_on_heldout(self) -> None:
        fixture = self._lineage_fixture()
        table = read_table(fixture["predictions"])
        rows = [(row["ID"], row["prediction"]) for row in table.rows]
        rows[0] = (rows[0][0], "tampered heldout value")
        self._write_predictions(fixture["predictions"], rows)
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            self._verify(fixture)

    def test_declared_batch_difference_uses_signed_heldout_predictions(self) -> None:
        fixture = self._lineage_fixture()
        all_manifest = json.loads(fixture["inference"].read_text())
        heldout_manifest = json.loads(
            fixture["heldout"].with_suffix(".manifest.json").read_text()
        )
        all_manifest["batch_size"] = 1
        heldout_manifest["batch_size"] = 4
        fixture["inference"].write_text(json.dumps(all_manifest), encoding="utf-8")
        fixture["heldout"].with_suffix(".manifest.json").write_text(
            json.dumps(heldout_manifest), encoding="utf-8"
        )
        evaluation = json.loads(fixture["evaluation"].read_text())
        evaluation["inference_manifest_sha256"] = _sha256(
            fixture["heldout"].with_suffix(".manifest.json")
        )
        fixture["evaluation"].write_text(json.dumps(evaluation), encoding="utf-8")
        table = read_table(fixture["predictions"])
        rows = [(row["ID"], row["prediction"]) for row in table.rows]
        rows[0] = (rows[0][0], "batch-sensitive alternate")
        self._write_predictions(fixture["predictions"], rows)

        report = run_fold(
            fixture["truth"], fixture["predictions"], fixture["inference"],
            fixture["training"], fixture["heldout"], fixture["evaluation"],
            self.root / "batch_run", 0, 7, "fixed_epoch_7", 0, "model22",
        )
        self.assertFalse(
            report["lineage"]["all_row_and_oof_heldout_predictions_identical"]
        )
        self.assertEqual(
            report["lineage"]["batch_sensitive_heldout_difference_count"], 1
        )
        raw = read_table(report["scores"]["raw"]["prediction_artifact"])
        signed = read_table(fixture["heldout"])
        self.assertEqual(raw.rows, signed.rows)

    def test_fit_side_blank_is_audited_and_filled_without_heldout_labels(self) -> None:
        fixture = self._lineage_fixture()
        table = read_table(fixture["predictions"])
        rows = [(row["ID"], row["prediction"]) for row in table.rows]
        rows = [(identifier, "" if identifier == "a1" else value) for identifier, value in rows]
        self._write_predictions(fixture["predictions"], rows)

        report = run_fold(
            fixture["truth"],
            fixture["predictions"],
            fixture["inference"],
            fixture["training"],
            fixture["heldout"],
            fixture["evaluation"],
            self.root / "blank_fit_run",
            0,
            7,
            "fixed_epoch_7",
            0,
            "model22",
        )
        self.assertEqual(report["lineage"]["fit_side_blank_prediction_ids"], ["a1"])
        self.assertEqual(report["lineage"]["fit_side_blank_fill"], "?")
        fit = json.loads(Path(report["fit_recipe_artifact"]).read_text())
        self.assertEqual(fit["fit_side_blank_prediction_ids"], ["a1"])

    def test_lineage_rejects_blank_on_heldout_fold(self) -> None:
        fixture = self._lineage_fixture()
        table = read_table(fixture["predictions"])
        rows = [(row["ID"], row["prediction"]) for row in table.rows]
        rows = [(identifier, "" if identifier == "a0" else value) for identifier, value in rows]
        self._write_predictions(fixture["predictions"], rows)
        with self.assertRaisesRegex(SchemaError, "blank prediction on heldout fold"):
            self._verify(fixture)

    def test_lineage_rejects_wrong_training_fold(self) -> None:
        fixture = self._lineage_fixture()
        training = json.loads(fixture["training"].read_text())
        training["preparation"]["fold"] = 1
        fixture["training"].write_text(json.dumps(training), encoding="utf-8")
        # Re-signing the changed manifest still cannot defeat the semantic fold check.
        evaluation = json.loads(fixture["evaluation"].read_text())
        evaluation["signature"]["training_manifest_sha256"] = _sha256(fixture["training"])
        fixture["evaluation"].write_text(json.dumps(evaluation), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "training preparation fold mismatch"):
            self._verify(fixture)

    def test_lineage_allows_declared_geometry_exclusion_from_internal_validation(self) -> None:
        fixture = self._lineage_fixture()
        training = json.loads(fixture["training"].read_text())
        training["preparation"]["validation_rows"] = 1
        training["preparation"]["excluded_ids"] = ["a0"]
        fixture["training"].write_text(json.dumps(training), encoding="utf-8")
        validation = Path(training["preparation"]["validation_manifest"])
        validation.write_text(f"{self.root / 'b0.jpg'}\n", encoding="utf-8")
        evaluation = json.loads(fixture["evaluation"].read_text())
        evaluation["signature"]["training_manifest_sha256"] = _sha256(fixture["training"])
        fixture["evaluation"].write_text(json.dumps(evaluation), encoding="utf-8")

        lineage = self._verify(fixture)
        self.assertEqual(lineage["heldout_rows"], 2)
        self.assertEqual(lineage["excluded_validation_truth_rows"], ["a0"])

    def test_lineage_rejects_undeclared_missing_internal_validation_row(self) -> None:
        fixture = self._lineage_fixture()
        training = json.loads(fixture["training"].read_text())
        training["preparation"]["validation_rows"] = 1
        fixture["training"].write_text(json.dumps(training), encoding="utf-8")
        validation = Path(training["preparation"]["validation_manifest"])
        validation.write_text(f"{self.root / 'b0.jpg'}\n", encoding="utf-8")
        evaluation = json.loads(fixture["evaluation"].read_text())
        evaluation["signature"]["training_manifest_sha256"] = _sha256(fixture["training"])
        fixture["evaluation"].write_text(json.dumps(evaluation), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "validation_rows mismatch"):
            self._verify(fixture)

    def test_heldout_labels_never_enter_stage_fit_or_transform(self) -> None:
        rows = self._rows()
        ids = tuple(row[0] for row in rows)
        predictions = tuple(f"prediction {identifier}" for identifier in ids)
        seen_sources: list[tuple[str, ...]] = []
        seen_references: list[tuple[str, ...]] = []

        def fit(source, references):
            seen_sources.append(tuple(source))
            seen_references.append(tuple(references))
            marker = references[0]
            transformed = [f"{value}|train:{marker}" for value in source]
            return StageFit(
                {"marker": marker},
                marker,
                transformed,
                [[] for _ in source],
                {"training_rows": len(source)},
            )

        def transform(values, state, _config):
            return [f"{value}|apply:{state}" for value in values], [[] for _ in values]

        stage = StageDefinition("fake", "fake", fit, transform)
        data_a = FoldModelData(
            "ID", ids, tuple(row[1] for row in rows), tuple(row[2] for row in rows), predictions
        )
        mutated_references = tuple(
            "secret changed heldout" if fold == 0 else target
            for _, target, fold in rows
        )
        data_b = FoldModelData(
            "ID", ids, mutated_references, tuple(row[2] for row in rows), predictions
        )
        result_a = fit_apply_fold_pipeline(data_a, 0, (stage,))
        result_b = fit_apply_fold_pipeline(data_b, 0, (stage,))
        self.assertEqual(result_a.stage_predictions, result_b.stage_predictions)
        self.assertEqual(result_a.stage_fits, result_b.stage_fits)
        expected_train_predictions = tuple(
            prediction for prediction, fold in zip(predictions, data_a.folds, strict=True) if fold != 0
        )
        self.assertEqual(seen_sources[0], expected_train_predictions)
        self.assertNotIn("alpha zero", seen_references[0])
        self.assertNotIn("beta zero", seen_references[0])

    def test_real_stage_runner_emits_only_heldout_scores(self) -> None:
        fixture = self._lineage_fixture()
        report = run_fold(
            fixture["truth"],
            fixture["predictions"],
            fixture["inference"],
            fixture["training"],
            fixture["heldout"],
            fixture["evaluation"],
            self.root / "run",
            0,
            7,
            "fixed_epoch_7",
            0,
            "model22",
        )
        self.assertEqual(report["stage_order"], ["raw", *(stage.name for stage in STAGES)])
        for stage in report["stage_order"]:
            self.assertEqual(report["scores"][stage]["scope"], "HELDOUT_FOLD_0_ONLY")
            self.assertNotIn("per_fold", report["scores"][stage])
        for stage in STAGES:
            fit = report["scores"].get(stage.name)
            self.assertIsNotNone(fit)
            diagnostic = report["lineage"]["train_rows"]
            self.assertEqual(diagnostic, 4)
        self.assertIn("not OOF", report["claim_scope"])

    def test_reserved_active_and_gpu_output_destinations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved active"):
            _prepare_output_dir(RESERVED_ACTIVE_OUTPUT)
        with self.assertRaisesRegex(ValueError, "GPU artifacts"):
            _prepare_output_dir(Path(__file__).resolve().parents[1] / "outputs" / "model36" / "forbidden")
        with self.assertRaisesRegex(ValueError, "GPU artifacts"):
            _prepare_output_dir(Path(__file__).resolve().parents[1] / "outputs" / "model71" / "forbidden")

    def test_combiner_accepts_one_verified_slice_per_fold(self) -> None:
        truth = self._write_truth()
        truth_sha = _sha256(truth)
        rows = self._rows()
        report_paths: list[Path] = []
        stage_order = ["raw", *(stage.name for stage in STAGES)]
        for fold in (0, 1, 2):
            fold_rows = [(identifier, target) for identifier, target, value in rows if value == fold]
            references = [target for _, target in fold_rows]
            stage_records = {}
            for position, stage in enumerate(stage_order):
                path = self.root / f"fold{fold}_{position}_{stage}.csv"
                self._write_predictions(path, fold_rows)
                stage_records[stage] = {
                    "scope": f"HELDOUT_FOLD_{fold}_ONLY",
                    "score": competition_error(references, references),
                    "prediction_artifact": str(path),
                    "prediction_sha256": _sha256(path),
                }
            report = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "heldout_fold": fold,
                "model_family": "model22" if fold == 0 else "model36",
                "checkpoint_epoch_zero_based": 7,
                "checkpoint_policy": "fixed_epoch_7",
                "policy_selection_fold": 0,
                "stage_order": stage_order,
                "scores": stage_records,
                "lineage": {"truth_sha256": truth_sha, "model_sha256": f"hash{fold}"},
            }
            report_path = self.root / f"fold{fold}_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            report_paths.append(report_path)
        combined = combine_fold_reports(truth, report_paths, self.root / "combined")
        self.assertEqual(combined["scores"]["supervised_errors"]["global"]["multi_score"], 1.0)
        output = read_table(
            combined["scores"]["supervised_errors"]["prediction_artifact"]
        )
        self.assertEqual([row["ID"] for row in output.rows], [row[0] for row in rows])


if __name__ == "__main__":
    unittest.main()
