from __future__ import annotations

import unittest

from model56.restore_contextual_initial_tokens import (
    build_contextual_initial_map,
    build_fold_maps,
    restore_contextual_initial_token,
)


class Model56ContextualInitialTokenTests(unittest.TestCase):
    def test_context_can_separate_same_initial_core(self) -> None:
        labels = ["By this Act"] * 20 + ["by virtue thereof"] * 20
        mapping = build_contextual_initial_map(labels, 20, 0.9)
        self.assertEqual(mapping[("by", "this")]["target"], "By")
        self.assertEqual(mapping[("by", "virtue")]["target"], "by")

    def test_below_ratio_is_rejected(self) -> None:
        labels = ["By this Act"] * 20 + ["by this deed"] * 3
        mapping = build_contextual_initial_map(labels, 20, 0.9)
        self.assertNotIn(("by", "this"), mapping)

    def test_only_initial_span_changes_and_spacing_is_preserved(self) -> None:
        mapping = {
            ("by", "this"): {
                "target": "By",
                "support": 20,
                "observations": 20,
                "ratio": 1.0,
                "initial_core": "by",
                "next_core": "this",
            }
        }
        result, change = restore_contextual_initial_token("  by\tthis  Act", mapping)
        self.assertEqual(result, "  By\tthis  Act")
        self.assertIsNotNone(change)

    def test_single_token_line_is_unchanged(self) -> None:
        result, change = restore_contextual_initial_token("Act", {})
        self.assertEqual(result, "Act")
        self.assertIsNone(change)

    def test_fold_map_excludes_heldout_labels(self) -> None:
        rows = [
            {"Target": "By this", "fold": "0"},
            {"Target": "by this", "fold": "1"},
            {"Target": "by this", "fold": "2"},
        ]
        maps = build_fold_maps(rows, "Target", "fold", 2, 1.0)
        self.assertEqual(maps[0][("by", "this")]["target"], "by")
        self.assertNotIn(("by", "this"), maps[1])


if __name__ == "__main__":
    unittest.main()
