from __future__ import annotations

import unittest

from model42.evaluate import _feature_indices, _fixed_geometry_matrix
from road_htr import crossfit_candidate_selector as selector


class Model42StaticTests(unittest.TestCase):
    def test_feature_sets_are_fixed_and_compact_is_strict_subset(self) -> None:
        compact = _feature_indices("compact")
        pair_only = _feature_indices("pair_only")
        geometry = _feature_indices("pair_geometry")
        self.assertEqual(len(compact), 30)
        self.assertEqual(len(pair_only), 14)
        self.assertTrue(set(pair_only).issubset(compact))
        self.assertEqual(len(geometry), 9)
        self.assertTrue(set(geometry).issubset(pair_only))
        self.assertEqual(len(selector.feature_names()), 54)

    def test_groups_depend_only_on_ids_and_salt(self) -> None:
        ids = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        first = selector.stable_groups(ids, 5, "model42-frozen-confusions-v1:fold0")
        second = selector.stable_groups(list(reversed(ids)), 5, "model42-frozen-confusions-v1:fold0")
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), set(range(5)))

    def test_geometry_features_do_not_depend_on_targets(self) -> None:
        configuration = selector.SelectorConfig(logistic_epochs=1)
        first = [selector.LabeledRow("a", "secret label one", 0)]
        second = [selector.LabeledRow("a", "wholly different label", 0)]
        candidate_a = {"a": "The same candidate"}
        candidate_b = {"a": "The other candidate"}
        self.assertEqual(
            _fixed_geometry_matrix(first, candidate_a, candidate_b, {}, {}, configuration, "x"),
            _fixed_geometry_matrix(second, candidate_a, candidate_b, {}, {}, configuration, "x"),
        )


if __name__ == "__main__":
    unittest.main()
