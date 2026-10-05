from __future__ import annotations

import unittest

from road_htr.confidence_distribution_selector import ConfidenceCandidate
from road_htr.ctc_decode import CharNGramLM
from road_htr.lm_view_selector import _language_quality, choose_alternate


class LanguageModelViewSelectorTests(unittest.TestCase):
    def test_seen_text_has_higher_language_quality(self) -> None:
        model = CharNGramLM(order=3, smoothing=0.2).fit(["the deed"] * 5)
        self.assertGreater(
            _language_quality("the deed", model),
            _language_quality("zxq vvv", model),
        )

    def test_language_score_can_override_confidence(self) -> None:
        primary = ConfidenceCandidate("noise", (0.80,) * 5)
        alternate = ConfidenceCandidate("deed", (0.79,) * 4)
        self.assertTrue(
            choose_alternate(primary, alternate, -4.0, -1.0, 0.1, 0.0, 0.8)
        )

    def test_length_guard_still_applies(self) -> None:
        primary = ConfidenceCandidate("a complete line", (0.8,) * 10)
        alternate = ConfidenceCandidate("line", (0.99,) * 4)
        self.assertFalse(
            choose_alternate(primary, alternate, -5.0, -1.0, 1.0, 0.0, 0.9)
        )


if __name__ == "__main__":
    unittest.main()
