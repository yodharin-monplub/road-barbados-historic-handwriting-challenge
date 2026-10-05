from __future__ import annotations

import math
import unittest

from road_htr.metrics import competition_error, levenshtein, weighted_cer, weighted_wer


class MetricTests(unittest.TestCase):
    def test_levenshtein(self) -> None:
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein([], []), 0)

    def test_perfect_is_zero(self) -> None:
        refs = ["the quick fox", "go home"]
        self.assertEqual(weighted_wer(refs, refs), 0.0)
        self.assertEqual(weighted_cer(refs, refs), 0.0)
        self.assertEqual(competition_error(refs, refs)["mean_error"], 0.0)

    def test_official_weighting_example(self) -> None:
        refs = ["the quick brown fox jumps over the lazy dog today", "go home"]
        hyps = [refs[0], "go"]
        expected = math.sqrt(2) / (math.sqrt(10) + math.sqrt(2))
        self.assertAlmostEqual(weighted_wer(refs, hyps), expected)
        self.assertAlmostEqual(expected, 0.309016994, places=8)

    def test_role_tokens_are_removed(self) -> None:
        self.assertEqual(weighted_cer(["deed"], ["<|assistant|>deed"]), 0.0)
        self.assertEqual(weighted_cer(["deed"], ["assistant: deed"]), 0.0)

    def test_multi_metric_normalization(self) -> None:
        result = competition_error(["a b"], [""])
        expected_error = (2 / 12 + 3 / 55) / 2
        self.assertAlmostEqual(result["mean_error"], expected_error)
        self.assertAlmostEqual(result["multi_score"], 1 - expected_error)

    def test_published_benchmark_normalization(self) -> None:
        wer = 11.5289665
        cer = 53.81177457
        score = 1 - 0.5 * (wer / 12 + cer / 55)
        self.assertAlmostEqual(score, 0.030428445, places=9)

    def test_count_mismatch_fails(self) -> None:
        with self.assertRaises(ValueError):
            weighted_wer(["one"], [])


if __name__ == "__main__":
    unittest.main()
