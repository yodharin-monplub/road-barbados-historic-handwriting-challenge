from __future__ import annotations

import unittest

from road_htr.boundary_corrector import TrainingBoundaryModel, repair_boundaries


class BoundaryCorrectorTests(unittest.TestCase):
    def test_joined_word_is_split(self) -> None:
        model = TrainingBoundaryModel(["in the presence"] * 3)
        value, changes = repair_boundaries("inthe presence", model, 1, 1)
        self.assertEqual(value, "in the presence")
        self.assertEqual(changes[0]["operation"], "split")

    def test_spurious_space_is_merged(self) -> None:
        model = TrainingBoundaryModel(["these presents", "the grant"])
        value, changes = repair_boundaries("the se presents", model, 1, 1)
        self.assertEqual(value, "these presents")
        self.assertEqual(changes[0]["operation"], "merge")

    def test_known_whole_word_is_not_split(self) -> None:
        model = TrainingBoundaryModel(["another", "an other"])
        value, changes = repair_boundaries("another", model, 1, 1)
        self.assertEqual(value, "another")
        self.assertEqual(changes, [])

    def test_punctuation_boundary_is_not_merged(self) -> None:
        model = TrainingBoundaryModel(["these", "the grant"])
        value, changes = repair_boundaries("the, se", model, 1, 1)
        self.assertEqual(value, "the, se")
        self.assertEqual(changes, [])


if __name__ == "__main__":
    unittest.main()
