from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from road_htr import model35_submission as m
from road_htr.metrics import competition_error
from road_htr.model35_deployment import DeploymentResult
from tests import test_model35_fold_postprocess as fold_fixtures


class Model35SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        policy = m._read_json(m.PROJECT_ROOT / "model35/terminal_policy.json")
        self.json("model35/terminal_policy.json", policy)
        self.addCleanup(patch.stopall)
        patch.object(m, "PROJECT_ROOT", self.root).start()
        self.truth = self.csv("truth.csv", ["ID", "Target", "fold"],
                              [(f"a{f}", "correct  text", f) for f in range(5)])
        self.fallback = self.csv("fallback.csv", ["ID", "prediction"],
                                 [(f"a{f}", "incorrect",) for f in range(5)])
        self.report = {"truth_sha256": m._sha256(self.truth), "folds": list(range(5)), "rows": 5,
                       "checkpoint_epoch_zero_based": 7, "checkpoint_policy": policy["checkpoint_policy"],
                       "policy_selection_fold": 0, "scores": {}}
        for stage, text in (("raw", "incorrect raw"), ("confusions", "correct  text")):
            path = self.csv(stage + ".csv", ["ID", "prediction"], [(f"a{f}", text) for f in range(5)])
            self.report["scores"][stage] = {
                "prediction_artifact": str(path), "prediction_sha256": m._sha256(path),
                "global": competition_error(["correct  text"] * 5, [text] * 5),
                "per_fold": {str(f): competition_error(["correct  text"], [text]) for f in range(5)},
            }

    def json(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def csv(self, name, columns, rows):
        path = self.root / name
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def final_fixture(self):
        config = {"train_csv": str(self.truth), "training": {"all_data_epochs": 8, "all_data_cosine_epochs": 10}}
        self.json("model36/config.yaml", config)  # JSON is a YAML subset.
        exclusion = self.root / "model22/exclude_ids.txt"
        exclusion.parent.mkdir()
        exclusion.write_text("# documented exclusion\n\nunused_id\n")
        train_list = self.root / "train.txt"
        train_list.write_text("".join(f"/images/a{f}.png\n" for f in range(5)))
        training = self.json("all/run_manifest.json", {"config": config, "preparation": {
            "fold": None, "train_rows": 5, "excluded_ids": ["unused_id"], "train_manifest": str(train_list)}})
        for epoch in range(8):
            (training.parent / f"checkpoint_{epoch:02d}-0.9.ckpt").write_bytes(str(epoch).encode())
        (training.parent / "best_0.9.safetensors").write_bytes(b"best")
        checkpoint = training.parent / "checkpoint_07-0.9.ckpt"
        weights = self.root / "epoch7.safetensors"
        weights.write_bytes(b"weights")
        conversion = self.json("conversion.json", {"checkpoint": str(checkpoint), "weights": str(weights),
                               "training_manifest": str(training), "training_manifest_sha256": m._sha256(training),
                               "checkpoint_sha256": m._sha256(checkpoint), "weights_sha256": m._sha256(weights)})
        test = self.csv("test.csv", ["ID"], [("t0",)])
        train_pred = self.csv("train_pred.csv", ["ID", "prediction"], [(f"a{f}", "train") for f in range(5)])
        test_pred = self.csv("test_pred.csv", ["ID", "prediction"], [("t0", "test  spaces")])
        for source, prediction, count in ((self.truth, train_pred, 5), (test, test_pred, 1)):
            self.json(prediction.with_suffix(".manifest.json").name, {
                "config": config, "fold": None, "image_variant": "full", "constrain_charset": False,
                "preserve_internal_spaces": True, "input_sha256": m._sha256(source),
                "predictions_sha256": m._sha256(prediction),
                "config_sha256": m._sha256(self.root / "model36/config.yaml"),
                "model_sha256": m._sha256(weights), "model": str(weights), "rows": count})
        return (self.truth, test, train_pred, test_pred, training, conversion)

    def test_complete_improvement_passes_and_separates_confirmation(self):
        result = m.check_oof_scores(self.truth, self.report, self.fallback)
        self.assertEqual(result["scores"]["confusions"]["confirmation_folds_1_4"]["multi_score"], 1.0)

    def test_missing_folds_rejected(self):
        self.report["folds"] = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            m.check_oof_scores(self.truth, self.report, self.fallback)

    def test_modified_prediction_rejected(self):
        path = Path(self.report["scores"]["confusions"]["prediction_artifact"])
        path.write_text(path.read_text().replace("correct", "altered"))
        with self.assertRaisesRegex(ValueError, "checksum"):
            m.check_oof_scores(self.truth, self.report, self.fallback)

    def test_reported_score_must_recompute(self):
        self.report["scores"]["confusions"]["global"]["multi_score"] = 0.7
        with self.assertRaisesRegex(ValueError, "global score"):
            m.check_oof_scores(self.truth, self.report, self.fallback)

    def test_no_improvement_rejected(self):
        candidate = Path(self.report["scores"]["confusions"]["prediction_artifact"])
        with self.assertRaisesRegex(ValueError, "does not improve fallback"):
            m.check_oof_scores(self.truth, self.report, candidate)

    def test_final_inference_checks_pass_with_commented_exclusions(self):
        args = self.final_fixture()
        result = m.verify_final_inference(*args)
        self.assertEqual(result["model_sha256"], m._sha256(self.root / "epoch7.safetensors"))

    def test_mixed_models_rejected(self):
        args = self.final_fixture()
        path = args[3].with_suffix(".manifest.json")
        value = m._read_json(path)
        value["model_sha256"] = "different"
        self.json(path.name, value)
        with self.assertRaisesRegex(ValueError, "model_sha256"):
            m.verify_final_inference(*args)

    def test_fold_model_and_incomplete_training_rejected(self):
        args = self.final_fixture()
        value = m._read_json(args[4])
        value["preparation"]["fold"] = 0
        self.json("all/run_manifest.json", value)
        with self.assertRaisesRegex(ValueError, "all-data"):
            m.verify_final_inference(*args)
        value["preparation"]["fold"] = None
        self.json("all/run_manifest.json", value)
        (args[4].parent / "checkpoint_06-0.9.ckpt").unlink()
        with self.assertRaisesRegex(ValueError, "completion artifacts"):
            m.verify_final_inference(*args)

    def test_final_prediction_tampering_rejected(self):
        args = self.final_fixture()
        args[3].write_text(args[3].read_text().replace("test  spaces", "modified text"))
        with self.assertRaisesRegex(ValueError, "predictions_sha256"):
            m.verify_final_inference(*args)

    def test_source_reports_required(self):
        with self.assertRaisesRegex(ValueError, "missing five-fold"):
            m.verify_oof_sources(self.truth, self.report)

    def test_recorded_lineage_catches_nonheldout_prediction_changes(self):
        helper = fold_fixtures.Model35FoldPostprocessTests()
        helper.setUp()
        self.addCleanup(helper.tearDown)
        fixture = helper._lineage_fixture()
        lineage = helper._verify(fixture)
        recipe = self.json("recipe.json", {"fitted": True})
        original = {"lineage": lineage, "heldout_fold": 0, "checkpoint_epoch_zero_based": 7,
                    "model_family": "model22", "fit_recipe_artifact": str(recipe),
                    "fit_recipe_sha256": m._sha256(recipe)}
        m.verify_recorded_fold_lineage(fixture["truth"], original)
        # Emulate older reports without the newly added diagnostic field.
        lineage.pop("excluded_validation_truth_rows")
        m.verify_recorded_fold_lineage(fixture["truth"], original)
        table = m.read_table(fixture["predictions"])
        changed = [(row["ID"], "changed training row" if row["ID"] == "a1" else row["prediction"])
                   for row in table.rows]
        helper._write_predictions(fixture["predictions"], changed)
        # The fold semantic check still passes: heldout rows were not changed.
        helper._verify(fixture)
        with self.assertRaisesRegex(ValueError, "changed since recipe fitting"):
            m.verify_recorded_fold_lineage(fixture["truth"], original)

    def test_cli_assembly_preserves_spaces_and_never_overwrites(self):
        args = self.final_fixture()
        report = self.json("oof_report.json", self.report)
        sample = self.csv("sample.csv", ["ID", "Target"], [("t0", "placeholder")])
        output = self.root / "outputs/model35/final_test"
        argv = ["build_submission.py", "--truth", str(args[0]), "--test", str(args[1]),
                "--sample", str(sample), "--fallback", str(self.fallback),
                "--oof-report", str(report), "--training-manifest", str(args[4]),
                "--training-predictions", str(args[2]), "--test-predictions", str(args[3]),
                "--conversion-receipt", str(args[5]), "--output-dir", str(output)]
        result = DeploymentResult(("test  spaces",), {"confusions": ("test  spaces",)}, {})
        # Source lineage has separate tests; avoid real training/expensive recipe fitting here.
        with patch("sys.argv", argv), patch.object(m, "verify_oof_sources"), \
                patch.object(m, "fit_apply_deployment", return_value=result) as fit:
            m.main()
            self.assertEqual(fit.call_args.args[2], ["test  spaces"])
            submission = m.read_table(output / "submission.csv")
            self.assertEqual(submission.fieldnames, ("ID", "Target"))
            self.assertEqual(submission.rows[0]["Target"], "test  spaces")
            self.assertEqual(m._read_json(output / "report.json")["status"], "validated_candidate_not_uploaded")
            before = (output / "submission.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                m.main()
            self.assertEqual((output / "submission.csv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
