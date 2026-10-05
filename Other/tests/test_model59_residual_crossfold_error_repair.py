from __future__ import annotations

import unittest

from model58.crossfold_error_repair import fit_mapping
from model59.residual_crossfold_error_repair import load_config


class Model59ResidualRepairTests(unittest.TestCase):
    def test_config_is_frozen_to_long_unanimous_errors(self) -> None:
        config, _ = load_config("model59/config.json")
        self.assertEqual(config["minimum_length"], 6)
        self.assertEqual(config["minimum_precision"], 1.0)
        self.assertEqual(config["minimum_support"], 2)

    def test_long_unanimous_error_is_learned(self) -> None:
        config, _ = load_config("model59/config.json")
        mapping = fit_mapping(
            ["delivored", "delivored"],
            ["delivered", "delivered"],
            config,
        )
        self.assertEqual(mapping["delivored"]["target"], "delivered")

    def test_short_error_is_rejected(self) -> None:
        config, _ = load_config("model59/config.json")
        mapping = fit_mapping(["teh", "teh"], ["the", "the"], config)
        self.assertNotIn("teh", mapping)


if __name__ == "__main__":
    unittest.main()
