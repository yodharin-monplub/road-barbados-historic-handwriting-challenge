from __future__ import annotations

import unittest

from road_htr.approximate_boundary_corrector import (
    TrainingJoinedWordModel,
    repair_approximate_joins,
)


class ApproximateBoundaryCorrectorTests(unittest.TestCase):
    def test_noisy_join_is_split(self) -> None:
        model = TrainingJoinedWordModel(["in the presence of witnesses"] * 6)
        value, changes = repair_approximate_joins(
            "in the prsenceof witnesses", model, 2, 6, 5, 0.0, 16.0
        )
        self.assertEqual(value, "in the presence of witnesses")
        self.assertEqual(changes[0]["operation"], "approximate_split")

    def test_rare_bigram_is_rejected(self) -> None:
        model = TrainingJoinedWordModel(["presence of"])
        value, changes = repair_approximate_joins(
            "prsenceof", model, 2, 6, 5, 0.0, 16.0
        )
        self.assertEqual(value, "prsenceof")
        self.assertEqual(changes, [])

    def test_known_whole_word_is_untouched(self) -> None:
        model = TrainingJoinedWordModel(["there in"] * 6 + ["therein"])
        value, changes = repair_approximate_joins(
            "therein", model, 2, 6, 5, 0.0, 16.0
        )
        self.assertEqual(value, "therein")
        self.assertEqual(changes, [])


if __name__ == "__main__":
    unittest.main()
