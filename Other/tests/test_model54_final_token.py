from __future__ import annotations

import unittest

from model54.restore_final_tokens import (
    build_final_map,
    build_fold_maps,
    restore_final_token,
    token_key,
)


class Model54FinalTokenTests(unittest.TestCase):
    def test_key_strips_only_edge_nonletters_and_casefolds(self) -> None:
        self.assertEqual(token_key("(Deed.),"), "deed")
        self.assertEqual(token_key("W^m"), "w^m")

    def test_unanimous_high_support_mapping(self) -> None:
        mapping = build_final_map(["a Deed."] * 15 + ["middle deed token"], 15, 1.0)
        self.assertEqual(mapping["deed"]["target"], "Deed.")

    def test_nonunanimous_mapping_is_rejected(self) -> None:
        mapping = build_final_map(["a Deed."] * 15 + ["a Deed"], 15, 1.0)
        self.assertNotIn("deed", mapping)

    def test_only_final_span_changes_and_spacing_is_preserved(self) -> None:
        mapping = {"deed": {"target": "Deed.", "support": 15, "observations": 15, "ratio": 1.0}}
        result, change = restore_final_token("the  final\tdeed  ", mapping)
        self.assertEqual(result, "the  final\tDeed.  ")
        self.assertIsNotNone(change)

    def test_fold_map_excludes_heldout_labels(self) -> None:
        rows = [
            {"Target": "a Deed.", "fold": "0"},
            {"Target": "a Deed", "fold": "1"},
            {"Target": "a Deed", "fold": "2"},
        ]
        maps = build_fold_maps(rows, "Target", "fold", 2, 1.0)
        self.assertEqual(maps[0]["deed"]["target"], "Deed")
        self.assertNotIn("deed", maps[1])


if __name__ == "__main__":
    unittest.main()
