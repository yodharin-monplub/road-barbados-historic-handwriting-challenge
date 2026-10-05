from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from road_htr.confidence_distribution_selector import (
    ConfidenceCandidate,
    _uses_alternate,
    confidence_quality,
    read_confidence_candidates,
)


class ConfidenceDistributionSelectorTests(unittest.TestCase):
    def test_tail_quality_penalizes_single_low_character(self) -> None:
        stable = (0.8, 0.8, 0.8, 0.8)
        unstable = (0.99, 0.99, 0.99, 0.01)
        self.assertGreater(confidence_quality(stable, "q25"), confidence_quality(unstable, "q25"))

    def test_length_guard_blocks_short_alternate(self) -> None:
        primary = ConfidenceCandidate("a complete line", (0.8, 0.8))
        alternate = ConfidenceCandidate("short", (0.99, 0.99))
        rule = {
            "kind": "distribution",
            "quality_kind": "mean",
            "quality_margin": 0.0,
            "minimum_length_ratio": 0.9,
        }
        self.assertFalse(_uses_alternate(rule, primary, alternate))

    def test_geometric_quality_is_below_arithmetic_mean(self) -> None:
        values = (0.2, 0.8)
        self.assertLess(confidence_quality(values, "geometric"), confidence_quality(values, "mean"))

    def test_empty_ocr_candidate_gets_zero_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text(
                '{"id":"row","candidates":[""],"scores":[null],'
                '"character_confidences":[]}\n',
                encoding="utf-8",
            )
            candidate = read_confidence_candidates(path)["row"]
            self.assertEqual(candidate.text, "")
            self.assertEqual(candidate.confidences, (0.0,))


if __name__ == "__main__":
    unittest.main()
