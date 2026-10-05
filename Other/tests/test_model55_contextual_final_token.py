from __future__ import annotations

import unittest

from model55.restore_contextual_final_tokens import (
    build_contextual_final_map,
    build_fold_maps,
    restore_contextual_final_token,
)


class Model55ContextualFinalTokenTests(unittest.TestCase):
    def test_context_can_separate_same_final_core(self) -> None:
        labels = ["a public Act."] * 6 + ["the said act"] * 6
        mapping = build_contextual_final_map(labels, 6, 0.8)
        self.assertEqual(mapping[("public", "act")]["target"], "Act.")
        self.assertEqual(mapping[("said", "act")]["target"], "act")

    def test_below_ratio_is_rejected(self) -> None:
        labels = ["public Act."] * 6 + ["public act"] * 2
        mapping = build_contextual_final_map(labels, 6, 0.8)
        self.assertNotIn(("public", "act"), mapping)

    def test_only_final_span_changes_and_spacing_is_preserved(self) -> None:
        mapping = {
            ("public", "act"): {
                "target": "Act.",
                "support": 6,
                "observations": 6,
                "ratio": 1.0,
                "previous_core": "public",
                "final_core": "act",
            }
        }
        result, change = restore_contextual_final_token("the  public\tact  ", mapping)
        self.assertEqual(result, "the  public\tAct.  ")
        self.assertIsNotNone(change)

    def test_single_token_line_is_unchanged(self) -> None:
        result, change = restore_contextual_final_token("Act", {})
        self.assertEqual(result, "Act")
        self.assertIsNone(change)

    def test_fold_map_excludes_heldout_labels(self) -> None:
        rows = [
            {"Target": "public Act.", "fold": "0"},
            {"Target": "public act", "fold": "1"},
            {"Target": "public act", "fold": "2"},
        ]
        maps = build_fold_maps(rows, "Target", "fold", 2, 1.0)
        self.assertEqual(maps[0][("public", "act")]["target"], "act")
        self.assertNotIn(("public", "act"), maps[1])


if __name__ == "__main__":
    unittest.main()
