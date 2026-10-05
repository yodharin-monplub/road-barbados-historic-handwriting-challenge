from __future__ import annotations

import unittest

from road_htr.contextual_corrector import (
    TrainingLanguageModel,
    _deletion_neighborhood,
    correct_contextually,
)


class ContextualCorrectorTests(unittest.TestCase):
    def test_two_deletion_neighborhood(self) -> None:
        values = _deletion_neighborhood("abcd", 2)
        self.assertIn("abcd", values)
        self.assertIn("acd", values)
        self.assertIn("ad", values)

    def test_context_overrides_unigram_frequency(self) -> None:
        model = TrainingLanguageModel(
            ["the cat sat"] * 4 + ["a cot"] * 8
        )
        value, changes = correct_contextually(
            "the cut sat", model, 1, 16.0, 16.0, 1
        )
        self.assertEqual(value, "the cat sat")
        self.assertEqual(len(changes), 1)

    def test_distance_two_is_blocked_for_short_words(self) -> None:
        model = TrainingLanguageModel(["an an an"])
        self.assertEqual(model.candidates("zz"), ())

    def test_distance_two_repairs_long_word(self) -> None:
        model = TrainingLanguageModel(["the presence of"] * 3)
        value, changes = correct_contextually(
            "the prssnce of", model, 1, 4.0, 16.0, 1
        )
        self.assertEqual(value, "the presence of")
        self.assertEqual(changes[0]["distance"], 2)


if __name__ == "__main__":
    unittest.main()
