from __future__ import annotations

import unittest

from model57.restore_contextual_medial_case import (
    build_fold_maps,
    build_medial_map,
    change_kind,
    restore_medial_case,
)


class Model57ContextualMedialCaseTests(unittest.TestCase):
    def test_change_kind_categories(self) -> None:
        self.assertEqual(change_kind("act", "Act"), "case")
        self.assertEqual(change_kind("act", "Act,"), "case_and_edge_punctuation")
        self.assertEqual(change_kind("act", "act,"), "edge_punctuation_only")

    def test_context_separates_same_core(self) -> None:
        labels = ["this public Act aforesaid"] * 10 + ["the said act was"] * 10
        mapping = build_medial_map(labels, 10, 1.0)
        self.assertEqual(mapping[("public", "act", "aforesaid")]["target"], "Act")
        self.assertEqual(mapping[("said", "act", "was")]["target"], "act")

    def test_case_change_and_spacing_are_preserved(self) -> None:
        mapping = {
            ("public", "act", "aforesaid"): {
                "target": "Act",
                "support": 10,
                "observations": 10,
                "ratio": 1.0,
                "left_core": "public",
                "current_core": "act",
                "right_core": "aforesaid",
            }
        }
        output, changes = restore_medial_case("this  public\tact  aforesaid", mapping)
        self.assertEqual(output, "this  public\tAct  aforesaid")
        self.assertEqual(len(changes), 1)

    def test_punctuation_only_change_is_rejected(self) -> None:
        mapping = {
            ("public", "act", "aforesaid"): {
                "target": "act,",
                "support": 10,
                "observations": 10,
                "ratio": 1.0,
                "left_core": "public",
                "current_core": "act",
                "right_core": "aforesaid",
            }
        }
        source = "this public act aforesaid"
        output, changes = restore_medial_case(source, mapping)
        self.assertEqual(output, source)
        self.assertEqual(changes, [])

    def test_fold_map_excludes_heldout_labels(self) -> None:
        rows = [
            {"Target": "this public Act now", "fold": "0"},
            {"Target": "this public act now", "fold": "1"},
            {"Target": "this public act now", "fold": "2"},
        ]
        maps = build_fold_maps(rows, "Target", "fold", 2, 1.0)
        self.assertEqual(maps[0][("public", "act", "now")]["target"], "act")
        self.assertNotIn(("public", "act", "now"), maps[1])


if __name__ == "__main__":
    unittest.main()
