from __future__ import annotations

import unittest

from model58.crossfold_error_repair import (
    build_outer_fold_maps,
    fit_mapping,
)
from road_htr.supervised_error_corrector import repair_supervised_errors


class Model58CrossfoldRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "maximum_distance": 2,
            "minimum_support": 2,
            "minimum_precision": 2 / 3,
            "minimum_length": 2,
        }

    def test_systematic_error_mapping_is_learned(self) -> None:
        mapping = fit_mapping(
            ["the ottur deed", "that ottur act", "one ottur grant"],
            ["the other deed", "that other act", "one other grant"],
            self.config,
        )
        self.assertEqual(mapping["ottur"]["target"], "other")
        result, changes = repair_supervised_errors("an ottur line", mapping)
        self.assertEqual(result, "an other line")
        self.assertEqual(len(changes), 1)

    def test_held_fold_pairs_are_excluded(self) -> None:
        predictions = ["ottur", "clean", "clean", "clean"]
        references = ["other", "clean", "clean", "clean"]
        folds = [0, 1, 2, 3]
        maps = build_outer_fold_maps(predictions, references, folds, self.config)
        self.assertNotIn("ottur", maps[0])
        # Only one occurrence exists even where fold 0 is eligible, below support 2.
        self.assertNotIn("ottur", maps[1])

    def test_ambiguous_source_is_rejected(self) -> None:
        mapping = fit_mapping(
            ["ottur", "ottur", "ottur"],
            ["other", "utter", "other"],
            self.config,
        )
        self.assertIn("ottur", mapping)
        strict = dict(self.config, minimum_precision=0.8)
        self.assertNotIn("ottur", fit_mapping(["ottur"] * 3, ["other", "utter", "other"], strict))


if __name__ == "__main__":
    unittest.main()
