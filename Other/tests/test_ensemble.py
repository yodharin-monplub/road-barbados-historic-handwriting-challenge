from __future__ import annotations

import unittest

from road_htr.ensemble import _normalized_distance


class EnsembleTests(unittest.TestCase):
    def test_normalized_distances(self) -> None:
        self.assertEqual(_normalized_distance("the deed", "the deed", "word"), 0.0)
        self.assertEqual(_normalized_distance("the deed", "the", "word"), 0.5)
        self.assertGreater(_normalized_distance("deed", "dead", "char"), 0.0)


if __name__ == "__main__":
    unittest.main()

