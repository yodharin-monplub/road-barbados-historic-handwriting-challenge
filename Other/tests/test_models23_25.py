import unittest

from model23.normalize_ampersands import normalize_ampersands
from model24.restore_carets import (
    build_caret_map,
    build_fold_maps as build_caret_fold_maps,
    restore as restore_carets,
)
from model25.restore_unambiguous_case import (
    build_case_map,
    build_fold_maps as build_case_fold_maps,
    restore as restore_case,
)


class AmpersandNormalizationTests(unittest.TestCase):
    def test_replaces_only_plus_characters(self) -> None:
        self.assertEqual(
            normalize_ampersands("grant + assign & deliver"),
            "grant & assign & deliver",
        )


class CaretRestorationTests(unittest.TestCase):
    def test_restores_dominant_variant_without_moving_existing_carets(self) -> None:
        mapping = build_caret_map(["W^m M^r", "W^m Mr"], 1, 0.5)
        self.assertEqual(restore_carets("Wm M^r ordinary", mapping), "W^m M^r ordinary")

    def test_does_not_insert_when_caretless_variant_is_dominant(self) -> None:
        mapping = build_caret_map(["Wm", "Wm", "W^m"], 1, 0.5)
        self.assertNotIn("Wm", mapping)

    def test_fold_map_excludes_held_out_labels(self) -> None:
        rows = (
            {"Target": "W^m", "fold": "0"},
            {"Target": "Wm", "fold": "1"},
            {"Target": "Wm", "fold": "2"},
        )
        maps = build_caret_fold_maps(rows, "Target", "fold", 1, 0.5)
        self.assertNotIn("Wm", maps[0])
        self.assertEqual(maps[1]["Wm"], "W^m")


class CaseRestorationTests(unittest.TestCase):
    def test_restores_only_unambiguous_high_support_case(self) -> None:
        mapping = build_case_map(
            ["John met Lord", "John saw Lord", "john thanked Lord"],
            minimum_count=2,
            minimum_ratio=1.0,
        )
        self.assertEqual(restore_case("john met lord", mapping), "john met Lord")
        self.assertNotIn("john", mapping)

    def test_fold_map_excludes_held_out_labels(self) -> None:
        rows = (
            {"Target": "John", "fold": "0"},
            {"Target": "john", "fold": "1"},
            {"Target": "john", "fold": "2"},
        )
        maps = build_case_fold_maps(rows, "Target", "fold", 2, 1.0)
        self.assertEqual(maps[0]["john"], "john")
        self.assertNotIn("john", maps[1])


if __name__ == "__main__":
    unittest.main()
