from __future__ import annotations

import unittest

from road_htr.ctc_decode import CharNGramLM, greedy_decode, logadd


class CtcDecodeTests(unittest.TestCase):
    def test_greedy_collapse(self) -> None:
        self.assertEqual(greedy_decode([0, 1, 1, 0, 1, 2, 2], ["a", "b"]), "aab")

    def test_logadd(self) -> None:
        self.assertAlmostEqual(logadd(0.0, 0.0), 0.6931471805599453)

    def test_language_model_prefers_seen_transition(self) -> None:
        model = CharNGramLM(order=2, smoothing=0.1).fit(["the", "the", "thy"])
        self.assertGreater(model.log_probability("th", "e"), model.log_probability("th", "z"))


if __name__ == "__main__":
    unittest.main()

