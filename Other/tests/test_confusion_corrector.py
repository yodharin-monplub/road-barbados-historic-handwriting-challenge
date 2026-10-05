from __future__ import annotations

import unittest

from road_htr.confusion_corrector import TrainingConfusionModel, repair_confusions


class ConfusionCorrectorTests(unittest.TestCase):
    def test_context_can_replace_a_known_word(self) -> None:
        model = TrainingConfusionModel(["the cat sat"] * 10 + ["a cot"] * 2)
        value, changes = repair_confusions("the cot sat", model, 2.0, 1.0)
        self.assertEqual(value, "the cat sat")
        self.assertEqual(len(changes), 1)

    def test_large_margin_keeps_source(self) -> None:
        model = TrainingConfusionModel(["the cat sat"] * 10 + ["a cot"] * 2)
        value, changes = repair_confusions("the cot sat", model, 2.0, 100.0)
        self.assertEqual(value, "the cot sat")
        self.assertEqual(changes, [])


if __name__ == "__main__":
    unittest.main()
