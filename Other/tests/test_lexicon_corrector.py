from __future__ import annotations

import unittest

from road_htr.lexicon_corrector import TrainingLexicon, correct_text


class LexiconCorrectorTests(unittest.TestCase):
    def test_one_edit_repair_preserves_case_and_punctuation(self) -> None:
        lexicon = TrainingLexicon(["twelve twelve presence presence presence"])
        value, changes = correct_text("Twelre, prsence", lexicon, 1, 1.0)
        self.assertEqual(value, "Twelve, presence")
        self.assertEqual(changes, 2)

    def test_known_historic_spelling_is_untouched(self) -> None:
        lexicon = TrainingLexicon(["publique publique public"])
        value, changes = correct_text("publique", lexicon, 1, 1.0)
        self.assertEqual(value, "publique")
        self.assertEqual(changes, 0)


if __name__ == "__main__":
    unittest.main()

