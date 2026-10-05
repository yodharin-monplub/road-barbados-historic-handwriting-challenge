from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from road_htr.crossfit_candidate_selector import (
    SelectorConfig,
    TextLanguageModel,
    _nested_training_features,
    apply_deployment,
    evaluate_crossfit,
    fit_deployment,
    load_labeled_rows,
    read_predictions,
    read_details,
    row_costs,
    stable_groups,
)
from road_htr.metrics import competition_error
from road_htr.schema import SchemaError, read_table


class Model33SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "single_fold_inner_splits": 3,
                    "nested_lm_splits": 2,
                    "seed": "fixture",
                    "char_order": 3,
                    "lm_alpha": 0.25,
                    "logistic_epochs": 35,
                    "logistic_learning_rate": 0.03,
                    "logistic_l2": 0.1,
                    "minimum_scale": 1e-6,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _truth(self, name: str, rows: list[tuple[str, str, int]]) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Target", "fold"])
            writer.writeheader()
            for identifier, target, fold in rows:
                writer.writerow({"ID": identifier, "Target": target, "fold": fold})
        return path

    def _predictions(self, name: str, rows: list[tuple[str, str]]) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
            writer.writeheader()
            for identifier, value in rows:
                writer.writerow({"ID": identifier, "prediction": value})
        return path

    @staticmethod
    def _rows() -> list[tuple[str, str, int]]:
        return [
            ("a0", "the house stands", 0),
            ("b0", "John owns land", 0),
            ("c0", "dated this day", 0),
            ("a1", "the house stands", 1),
            ("b1", "John owns land", 1),
            ("c1", "dated this day", 1),
            ("a2", "the house stands", 2),
            ("b2", "John owns land", 2),
            ("c2", "dated this day", 2),
        ]

    def _candidate_files(self) -> tuple[Path, Path]:
        rows = self._rows()
        candidate_a = self._predictions(
            "a.csv",
            [
                (identifier, target if identifier.startswith("a") else "bad output")
                for identifier, target, _ in rows
            ],
        )
        candidate_b = self._predictions(
            "b.csv",
            [
                (identifier, target if not identifier.startswith("a") else "other text")
                for identifier, target, _ in rows
            ],
        )
        return candidate_a, candidate_b

    def test_row_costs_sum_to_exact_metric_mean_error(self) -> None:
        truth = self._truth("truth.csv", self._rows())
        candidate_a, candidate_b = self._candidate_files()
        _, rows = load_labeled_rows(truth)
        values_a = read_predictions(candidate_a)
        values_b = read_predictions(candidate_b)
        costs = row_costs(rows, values_a, values_b)
        score_a = competition_error(
            [row.target for row in rows], [values_a[row.identifier] for row in rows]
        )
        score_b = competition_error(
            [row.target for row in rows], [values_b[row.identifier] for row in rows]
        )
        self.assertAlmostEqual(sum(value.a for value in costs.values()), score_a["mean_error"])
        self.assertAlmostEqual(sum(value.b for value in costs.values()), score_b["mean_error"])

    def test_prediction_and_detail_readers_preserve_internal_spaces(self) -> None:
        predictions = self._predictions("spaces.csv", [("a", "two  spaces")])
        details = self.root / "spaces.jsonl"
        details.write_text(
            json.dumps(
                {
                    "id": "a",
                    "candidates": ["two  spaces"],
                    "character_confidences": [0.9] * 11,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual(read_predictions(predictions)["a"], "two  spaces")
        self.assertEqual(read_details(details)["a"].source_text, "two  spaces")

    def test_complete_crossfit_heldout_choice_does_not_use_heldout_labels(self) -> None:
        rows = self._rows()
        truth_a = self._truth("truth_a.csv", rows)
        truth_b = self._truth(
            "truth_b.csv",
            [
                (identifier, "totally mutated fold zero" if fold == 0 else target, fold)
                for identifier, target, fold in rows
            ],
        )
        candidate_a, candidate_b = self._candidate_files()
        evaluate_crossfit(
            truth_a, candidate_a, candidate_b, self.root / "run_a", self.config
        )
        evaluate_crossfit(
            truth_b, candidate_a, candidate_b, self.root / "run_b", self.config
        )

        def fold_zero_audit(path: Path) -> dict[str, tuple[str, str]]:
            with path.open(encoding="utf-8", newline="") as handle:
                return {
                    row["ID"]: (row["selected_candidate"], row["probability_b"])
                    for row in csv.DictReader(handle)
                    if row["partition"] == "0"
                }

        self.assertEqual(
            fold_zero_audit(self.root / "run_a" / "selection_audit.csv"),
            fold_zero_audit(self.root / "run_b" / "selection_audit.csv"),
        )

    def test_single_fold_inner_group_excludes_its_labels(self) -> None:
        rows = [(f"id{i}", f"ordinary target number {i}", 0) for i in range(9)]
        truth_a = self._truth("truth_a.csv", rows)
        candidate_a = self._predictions(
            "a.csv", [(identifier, "ordinary target") for identifier, _, _ in rows]
        )
        candidate_b = self._predictions(
            "b.csv", [(identifier, "different words") for identifier, _, _ in rows]
        )
        groups = stable_groups([row[0] for row in rows], 3, "fixture:single-fold:0")
        heldout_group = 1
        truth_b = self._truth(
            "truth_b.csv",
            [
                (identifier, "mutated secret label" if groups[identifier] == heldout_group else target, fold)
                for identifier, target, fold in rows
            ],
        )
        evaluate_crossfit(
            truth_a,
            candidate_a,
            candidate_b,
            self.root / "inner_a",
            self.config,
            single_fold=0,
        )
        evaluate_crossfit(
            truth_b,
            candidate_a,
            candidate_b,
            self.root / "inner_b",
            self.config,
            single_fold=0,
        )

        def selected(path: Path) -> dict[str, tuple[str, str]]:
            with path.open(encoding="utf-8", newline="") as handle:
                return {
                    row["ID"]: (row["selected_candidate"], row["probability_b"])
                    for row in csv.DictReader(handle)
                    if int(row["partition"]) == heldout_group
                }

        self.assertEqual(
            selected(self.root / "inner_a" / "selection_audit.csv"),
            selected(self.root / "inner_b" / "selection_audit.csv"),
        )

    def test_nested_language_features_exclude_own_group_targets(self) -> None:
        rows = [(f"id{i}", f"rareword{i} common", i % 2) for i in range(8)]
        truth_a = self._truth("truth_a.csv", rows)
        _, labeled_a = load_labeled_rows(truth_a)
        assignments = stable_groups(
            [row.identifier for row in labeled_a], 2, "fixture:nested:lm"
        )
        changed_id = labeled_a[0].identifier
        changed_group = assignments[changed_id]
        mutated = [
            (row.identifier, "brandnewtoken" if row.identifier == changed_id else row.target, row.fold)
            for row in labeled_a
        ]
        truth_b = self._truth("truth_b.csv", mutated)
        _, labeled_b = load_labeled_rows(truth_b)
        candidate_a = {row.identifier: "common" for row in labeled_a}
        candidate_b = {row.identifier: "rareword alternate" for row in labeled_a}
        config = SelectorConfig.from_mapping(json.loads(self.config.read_text()))
        matrix_a = _nested_training_features(
            labeled_a, candidate_a, candidate_b, {}, {}, config, "nested"
        )
        matrix_b = _nested_training_features(
            labeled_b, candidate_a, candidate_b, {}, {}, config, "nested"
        )
        for index, row in enumerate(labeled_a):
            if assignments[row.identifier] == changed_group:
                self.assertEqual(matrix_a[index], matrix_b[index])

    def test_fit_and_apply_round_trip_without_test_labels(self) -> None:
        truth = self._truth("truth.csv", self._rows())
        candidate_a, candidate_b = self._candidate_files()
        model = self.root / "deployment.json"
        state = fit_deployment(
            truth, candidate_a, candidate_b, model, self.config
        )
        self.assertEqual(state["training_rows"], 9)
        test = self.root / "test.csv"
        test.write_text("ID\na0\nb0\n", encoding="utf-8")
        a_test = self._predictions("a_test.csv", [("a0", "the house"), ("b0", "bad")])
        b_test = self._predictions("b_test.csv", [("a0", "wrong"), ("b0", "John owns land")])
        result = apply_deployment(
            test, a_test, b_test, model, self.root / "test_output.csv"
        )
        self.assertEqual(result["rows"], 2)
        table = read_table(self.root / "test_output.csv")
        self.assertEqual([row["ID"] for row in table.rows], ["a0", "b0"])

    def test_submission_preserves_selected_internal_spaces(self) -> None:
        truth = self._truth("truth.csv", self._rows())
        candidate_a, candidate_b = self._candidate_files()
        model = self.root / "deployment.json"
        fit_deployment(truth, candidate_a, candidate_b, model, self.config)
        test = self.root / "test.csv"
        test.write_text("ID\na0\n", encoding="utf-8")
        sample = self.root / "sample.csv"
        sample.write_text("ID,Target\na0,placeholder\n", encoding="utf-8")
        a_test = self._predictions("a_test.csv", [("a0", "two  spaces")])
        b_test = self._predictions("b_test.csv", [("a0", "also  spaced")])
        submission = self.root / "submission.csv"
        apply_deployment(
            test,
            a_test,
            b_test,
            model,
            self.root / "test_output.csv",
            sample_path=sample,
            submission_path=submission,
        )
        raw = submission.read_text(encoding="utf-8")
        self.assertIn("  ", raw)

    def test_missing_candidate_id_is_rejected(self) -> None:
        truth = self._truth("truth.csv", self._rows())
        candidate_a, candidate_b = self._candidate_files()
        incomplete = self._predictions(
            "incomplete.csv", list(read_predictions(candidate_b).items())[:-1]
        )
        with self.assertRaisesRegex(SchemaError, "missing"):
            evaluate_crossfit(
                truth, candidate_a, incomplete, self.root / "bad", self.config
            )

    def test_language_model_round_trip(self) -> None:
        model = TextLanguageModel.fit(["John owns land", "the land stands"], 3, 0.25)
        restored = TextLanguageModel.from_dict(model.to_dict())
        self.assertEqual(model.qualities("John owns land"), restored.qualities("John owns land"))


if __name__ == "__main__":
    unittest.main()
