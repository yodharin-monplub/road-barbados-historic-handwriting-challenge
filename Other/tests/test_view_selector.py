from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from road_htr.view_selector import choose_alternate, read_confident_candidates


class ViewSelectorTests(unittest.TestCase):
    def test_higher_confidence_selects_alternate(self) -> None:
        self.assertTrue(choose_alternate("abc", 0.8, "abd", 0.9, 0.0, 0.0))
        self.assertFalse(choose_alternate("abc", 0.9, "abd", 0.8, 0.0, 0.0))

    def test_length_guard(self) -> None:
        self.assertFalse(choose_alternate("abcdefghij", 0.7, "abc", 0.9, 0.0, 0.9))

    def test_empty_ocr_candidate_gets_sentinel_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text(
                '{"id":"row","candidates":[""],"scores":[null],'
                '"character_confidences":[]}\n',
                encoding="utf-8",
            )
            self.assertEqual(read_confident_candidates(path)["row"], ("", -1.0))


if __name__ == "__main__":
    unittest.main()
