from __future__ import annotations

import unittest

from model68.exact_line_memory import fit_mapping, load_config, repair_exact_line


class Model68ExactLineMemoryTests(unittest.TestCase):
    def test_config_is_frozen(self) -> None:
        config, _ = load_config("model68/config.json")
        self.assertEqual(config["minimum_precision"], 1.0)

    def test_unanimous_mapping_is_learned(self) -> None:
        config, _ = load_config("model68/config.json")
        mapping = fit_mapping(["Sealed in prsence"], ["Sealed in presence"], config)
        self.assertEqual(mapping["Sealed in prsence"]["target"], "Sealed in presence")

    def test_ambiguous_mapping_is_rejected(self) -> None:
        config, _ = load_config("model68/config.json")
        mapping = fit_mapping(["same line", "same line"], ["first", "second"], config)
        self.assertNotIn("same line", mapping)

    def test_match_is_exact_and_case_sensitive(self) -> None:
        mapping = {"same line": {"target": "correct line", "support": 1, "observations": 1, "precision": 1.0}}
        result, changes = repair_exact_line("Same line", mapping)
        self.assertEqual(result, "Same line")
        self.assertEqual(changes, [])

    def test_exact_match_is_replaced(self) -> None:
        mapping = {"same line": {"target": "correct line", "support": 1, "observations": 1, "precision": 1.0}}
        result, changes = repair_exact_line("same line", mapping)
        self.assertEqual(result, "correct line")
        self.assertEqual(changes[0]["support"], 1)


if __name__ == "__main__":
    unittest.main()
