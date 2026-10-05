from __future__ import annotations

import unittest

from road_htr.supervised_error_corrector import (
    SupervisedErrorModel,
    _aligned_token_pairs,
    repair_supervised_errors,
)


class SupervisedErrorCorrectorTests(unittest.TestCase):
    def test_alignment_ignores_insertion_and_keeps_substitutions(self) -> None:
        pairs = _aligned_token_pairs("the old hous", "the very old house")
        self.assertEqual(pairs, [("the", "the"), ("old", "old"), ("hous", "house")])

    def test_support_and_precision_gate_mapping(self) -> None:
        model = SupervisedErrorModel(
            [("hous", "house"), ("hous", "house"), ("hous", "hous")]
        )
        accepted = model.mappings(1, 2, 2 / 3, 2)
        rejected = model.mappings(1, 2, 0.8, 2)
        self.assertEqual(accepted["hous"]["target"], "house")
        self.assertNotIn("hous", rejected)

    def test_repair_preserves_case_and_punctuation(self) -> None:
        mapping = {
            "hous": {
                "target": "house",
                "support": 3,
                "observations": 3,
                "precision": 1.0,
                "distance": 1,
            }
        }
        value, changes = repair_supervised_errors("The HOUS,", mapping)
        self.assertEqual(value, "The HOUSE,")
        self.assertEqual(len(changes), 1)


if __name__ == "__main__":
    unittest.main()
