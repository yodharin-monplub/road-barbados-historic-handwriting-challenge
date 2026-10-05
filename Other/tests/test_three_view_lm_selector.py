from __future__ import annotations

import unittest

from road_htr.confidence_distribution_selector import ConfidenceCandidate
from road_htr.three_view_lm_selector import select_index


class ThreeViewLanguageModelSelectorTests(unittest.TestCase):
    def test_best_combined_quality_is_selected(self) -> None:
        candidates = (
            ConfidenceCandidate("full text", (0.8,)),
            ConfidenceCandidate("center text", (0.82,)),
            ConfidenceCandidate("enhanced text", (0.81,)),
        )
        self.assertEqual(select_index(candidates, (-2.0, -1.0, -1.5), 0.1, 0.0, 0.0, 0.8), 1)

    def test_length_guard_falls_back_to_full(self) -> None:
        candidates = (
            ConfidenceCandidate("a complete line", (0.8,)),
            ConfidenceCandidate("short", (0.99,)),
            ConfidenceCandidate("tiny", (0.99,)),
        )
        self.assertEqual(select_index(candidates, (-3.0, -1.0, -1.0), 1.0, 0.0, 0.0, 0.9), 0)


if __name__ == "__main__":
    unittest.main()
