from __future__ import annotations

import unittest

from model62.context_insertion_repair import (
    fit_mapping,
    load_config,
    repair_context_insertions,
)


class Model62ContextInsertionRepairTests(unittest.TestCase):
    def test_config_is_frozen(self) -> None:
        config, _ = load_config("model62/config.json")
        self.assertEqual(config["maximum_inserted_tokens"], 2)
        self.assertEqual(config["minimum_support"], 3)
        self.assertEqual(config["minimum_precision"], 0.5)

    def test_repeated_context_insertion_is_learned(self) -> None:
        config, _ = load_config("model62/config.json")
        mapping = fit_mapping(
            ["sett hand"] * 4,
            ["sett my hand"] * 3 + ["sett hand"],
            config,
        )
        self.assertEqual(mapping[("sett", "hand")]["target"], ("my",))
        self.assertEqual(mapping[("sett", "hand")]["observations"], 4)

    def test_low_precision_insertion_is_rejected(self) -> None:
        config, _ = load_config("model62/config.json")
        mapping = fit_mapping(
            ["sett hand"] * 7,
            ["sett my hand"] * 3 + ["sett hand"] * 4,
            config,
        )
        self.assertNotIn(("sett", "hand"), mapping)

    def test_insertion_preserves_surrounding_tokens(self) -> None:
        mapping = {
            ("sett", "hand"): {
                "target": ("my",),
                "support": 3,
                "observations": 4,
                "precision": 0.75,
            }
        }
        result, changes = repair_context_insertions("I sett hand here", mapping)
        self.assertEqual(result, "I sett my hand here")
        self.assertEqual(changes[0]["inserted"], ["my"])


if __name__ == "__main__":
    unittest.main()
