from __future__ import annotations

import unittest

from road_htr.confidence_distribution_selector import ConfidenceCandidate
from road_htr.token_confidence_fusion import _alignment, confident_tokens, fuse_tokens


class TokenConfidenceFusionTests(unittest.TestCase):
    def test_alignment_skips_donor_insertion(self) -> None:
        base = ConfidenceCandidate("the house", (0.8,) * 9)
        donor = ConfidenceCandidate("the old house", (0.8,) * 13)
        self.assertEqual(_alignment(confident_tokens(base), confident_tokens(donor)), [0, 2])

    def test_higher_confidence_close_word_is_replaced(self) -> None:
        base = ConfidenceCandidate("the hous", (0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2))
        donor = ConfidenceCandidate("the house", (0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9))
        value, changes = fuse_tokens(base, donor, "mean", 0.05, 0.5)
        self.assertEqual(value, "the house")
        self.assertEqual(len(changes), 1)

    def test_distance_gate_blocks_unrelated_word(self) -> None:
        base = ConfidenceCandidate("house", (0.2,) * 5)
        donor = ConfidenceCandidate("grant", (0.9,) * 5)
        value, changes = fuse_tokens(base, donor, "mean", 0.0, 0.25)
        self.assertEqual(value, "house")
        self.assertEqual(changes, [])


if __name__ == "__main__":
    unittest.main()
