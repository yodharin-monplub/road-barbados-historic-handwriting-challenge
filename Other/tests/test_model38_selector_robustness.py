from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from road_htr import crossfit_candidate_selector as model33
from road_htr.model38_selector_robustness import (
    AuditVariant,
    _weighted_calibration,
    evaluate_seed,
    feature_indices,
    fit_logistic_numpy,
    load_verified_inputs,
)
from road_htr.schema import SchemaError


ROOT = Path(__file__).resolve().parents[1]


class Model38SelectorRobustnessTests(unittest.TestCase):
    def test_fixed_feature_sets_have_expected_unique_columns(self) -> None:
        expected = {
            "all": 54,
            "compact": 30,
            "no_lm": 45,
            "no_confidence": 36,
            "pair_only": 14,
            "pair_geometry": 9,
        }
        for name, count in expected.items():
            values = feature_indices(name)
            self.assertEqual(len(values), count)
            self.assertEqual(len(values), len(set(values)))

    def test_numpy_optimizer_matches_model33_reference_optimizer(self) -> None:
        generator = np.random.default_rng(38)
        matrix = generator.normal(size=(24, len(model33.feature_names()))).tolist()
        labels = [bool(value) for value in generator.integers(0, 2, size=24)]
        importance = generator.uniform(0.01, 2.0, size=24).tolist()
        config = replace(
            model33.SelectorConfig(),
            logistic_epochs=40,
            logistic_learning_rate=0.02,
        )
        reference = model33.fit_logistic(matrix, labels, importance, config)
        vectorized = fit_logistic_numpy(
            matrix,
            labels,
            importance,
            config,
            tuple(range(len(model33.feature_names()))),
        )
        self.assertEqual(vectorized.feature_names, reference.feature_names)
        np.testing.assert_allclose(vectorized.means, reference.means, rtol=0, atol=1e-14)
        np.testing.assert_allclose(vectorized.scales, reference.scales, rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            vectorized.coefficients, reference.coefficients, rtol=0, atol=1e-12
        )

    def test_held_partition_label_mutation_cannot_change_its_predictions(self) -> None:
        rows = [
            model33.LabeledRow(f"id{index}", f"target phrase {index}", 0)
            for index in range(15)
        ]
        candidate_a = {
            row.identifier: ("target phrase" if index % 2 else "other phrase")
            for index, row in enumerate(rows)
        }
        candidate_b = {
            row.identifier: ("different phrase" if index % 3 else "target phrase")
            for index, row in enumerate(rows)
        }
        seed = "model38-test-seed"
        base = replace(
            model33.SelectorConfig(),
            single_fold_inner_splits=3,
            nested_lm_splits=3,
            logistic_epochs=20,
        )
        assignments = model33.stable_groups(
            [row.identifier for row in rows], 3, f"{seed}:single-fold:0"
        )
        held = 1
        mutated = [
            replace(row, target="secret mutated label")
            if assignments[row.identifier] == held
            else row
            for row in rows
        ]
        variant = [AuditVariant("geometry", "pair_geometry", 0.08)]
        _, selections_a, _ = evaluate_seed(
            rows,
            candidate_a,
            candidate_b,
            {},
            {},
            base,
            variant,
            seed,
            outer_fold=0,
            calibration_bins=5,
        )
        _, selections_b, _ = evaluate_seed(
            mutated,
            candidate_a,
            candidate_b,
            {},
            {},
            base,
            variant,
            seed,
            outer_fold=0,
            calibration_bins=5,
        )

        def held_predictions(values: list[dict[str, object]]) -> dict[str, tuple[str, str]]:
            return {
                str(row["ID"]): (
                    str(row["probability_b"]), str(row["selected_candidate"])
                )
                for row in values
                if int(row["partition"]) == held
            }

        self.assertEqual(held_predictions(selections_a), held_predictions(selections_b))

    def test_weighted_calibration_is_zero_for_perfect_certainty(self) -> None:
        result = _weighted_calibration(
            [0.0, 1.0, 1.0, 0.0],
            [False, True, True, False],
            [1.0, 2.0, 3.0, 4.0],
            5,
        )
        self.assertEqual(result["weighted_brier"], 0.0)
        self.assertEqual(result["weighted_ece"], 0.0)
        self.assertEqual(result["weighted_choice_accuracy"], 1.0)

    def test_canonical_provenance_fails_on_source_hash_drift(self) -> None:
        config = json.loads((ROOT / "model38" / "config.json").read_text())
        config["artifacts"]["model33_source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(SchemaError, "model33_source SHA-256"):
            load_verified_inputs(config)


if __name__ == "__main__":
    unittest.main()
