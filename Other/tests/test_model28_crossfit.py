from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from road_htr.crossfit_postprocess import (
    PROTECTED_MODEL22_OUTPUT,
    STAGES,
    _prepare_output_dir,
    load_complete_oof,
    run_crossfit,
)
from road_htr.metrics import competition_error
from road_htr.schema import SchemaError, read_table


class Model28CrossfitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_truth(
        self, name: str, rows: list[tuple[str, str, int]]
    ) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Target", "fold"])
            writer.writeheader()
            for identifier, target, fold in rows:
                writer.writerow({"ID": identifier, "Target": target, "fold": fold})
        return path

    def _write_predictions(
        self, name: str, rows: list[tuple[str, str]]
    ) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
            writer.writeheader()
            for identifier, prediction in rows:
                writer.writerow({"ID": identifier, "prediction": prediction})
        return path

    @staticmethod
    def _rows() -> list[tuple[str, str, int]]:
        return [
            ("a0", "the house stands", 0),
            ("b0", "presence of land", 0),
            ("a1", "the house stands", 1),
            ("b1", "presence of land", 1),
            ("a2", "the house stands", 2),
            ("b2", "presence of land", 2),
        ]

    @staticmethod
    def _prediction_rows() -> list[tuple[str, str]]:
        return [
            ("b2", "presenceof land"),
            ("a1", "the hous stands"),
            ("a0", "the hous stands"),
            ("b0", "presenceof land"),
            ("a2", "the hous stands"),
            ("b1", "presenceof land"),
        ]

    def test_writes_ordered_sequential_scores_and_audit_artifacts(self) -> None:
        truth = self._write_truth("truth.csv", self._rows())
        predictions = self._write_predictions(
            "predictions.csv", self._prediction_rows()
        )
        output = self.root / "run"

        report = run_crossfit(truth, predictions, output)

        expected_order = ["raw", *(stage.name for stage in STAGES)]
        self.assertEqual(report["stage_order"], expected_order)
        self.assertEqual(set(report["scores"]), set(expected_order))
        for name in expected_order:
            score = report["scores"][name]
            self.assertIn("multi_score", score["global"])
            self.assertEqual(set(score["per_fold"]), {"0", "1", "2"})
            table = read_table(score["prediction_artifact"])
            self.assertEqual(
                [row["ID"] for row in table.rows],
                [row[0] for row in self._rows()],
            )
        prediction_map = dict(self._prediction_rows())
        expected_raw = competition_error(
            [target for _, target, _ in self._rows()],
            [prediction_map[identifier] for identifier, _, _ in self._rows()],
        )
        self.assertEqual(report["scores"]["raw"]["global"], expected_raw)
        for fold, fit in report["fold_fits"].items():
            self.assertNotIn(int(fold), fit["fit_folds"])
            self.assertTrue(Path(report["fold_fit_artifacts"][fold]).is_file())
            mapping = fit["stages"]["supervised_errors"]["mapping_artifact"]
            self.assertTrue(Path(mapping).is_file())
        self.assertTrue((output / "report.json").is_file())
        self.assertEqual(json.loads((output / "report.json").read_text())["rows"], 6)

    def test_heldout_outputs_do_not_depend_on_that_folds_labels(self) -> None:
        rows = self._rows()
        mutated = [
            (identifier, "zzzz qqqq" if fold == 0 else target, fold)
            for identifier, target, fold in rows
        ]
        truth_a = self._write_truth("truth_a.csv", rows)
        truth_b = self._write_truth("truth_b.csv", mutated)
        predictions = self._write_predictions(
            "predictions.csv", self._prediction_rows()
        )

        report_a = run_crossfit(truth_a, predictions, self.root / "run_a")
        report_b = run_crossfit(truth_b, predictions, self.root / "run_b")
        fold_zero_ids = {identifier for identifier, _, fold in rows if fold == 0}
        for name in report_a["stage_order"]:
            table_a = read_table(report_a["scores"][name]["prediction_artifact"])
            table_b = read_table(report_b["scores"][name]["prediction_artifact"])
            column_a = next(c for c in table_a.fieldnames if c != table_a.id_column)
            column_b = next(c for c in table_b.fieldnames if c != table_b.id_column)
            values_a = {
                row[table_a.id_column]: row[column_a]
                for row in table_a.rows
                if row[table_a.id_column] in fold_zero_ids
            }
            values_b = {
                row[table_b.id_column]: row[column_b]
                for row in table_b.rows
                if row[table_b.id_column] in fold_zero_ids
            }
            self.assertEqual(values_a, values_b, name)

    def test_rejects_incomplete_oof_and_single_fold_truth(self) -> None:
        truth = self._write_truth("truth.csv", self._rows())
        incomplete = self._write_predictions(
            "incomplete.csv", self._prediction_rows()[:-1]
        )
        with self.assertRaisesRegex(SchemaError, "ID mismatch"):
            load_complete_oof(truth, incomplete)

        one_fold = self._write_truth(
            "one_fold.csv",
            [("a", "one", 0), ("b", "two", 0)],
        )
        complete = self._write_predictions(
            "complete.csv", [("a", "one"), ("b", "two")]
        )
        with self.assertRaisesRegex(SchemaError, "at least two folds"):
            load_complete_oof(one_fold, complete)

    def test_refuses_nonempty_output_directory(self) -> None:
        truth = self._write_truth("truth.csv", self._rows())
        predictions = self._write_predictions(
            "predictions.csv", self._prediction_rows()
        )
        output = self.root / "occupied"
        output.mkdir()
        (output / "keep.txt").write_text("user data", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            run_crossfit(truth, predictions, output)
        self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "user data")

    def test_refuses_model22_output_destination(self) -> None:
        destination = PROTECTED_MODEL22_OUTPUT / "model28_forbidden_test"
        with self.assertRaisesRegex(ValueError, "active Model22 outputs"):
            _prepare_output_dir(destination)


if __name__ == "__main__":
    unittest.main()
