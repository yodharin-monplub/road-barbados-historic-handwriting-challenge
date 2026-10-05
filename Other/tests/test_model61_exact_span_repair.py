from __future__ import annotations

import unittest

from model61.exact_span_repair import fit_mapping, load_config, repair_exact_spans


class Model61ExactSpanRepairTests(unittest.TestCase):
    def test_config_is_frozen(self) -> None:
        config, _ = load_config("model61/config.json")
        self.assertEqual(config["maximum_source_tokens"], 2)
        self.assertEqual(config["minimum_support"], 3)
        self.assertEqual(config["minimum_precision"], 0.5)

    def test_repeated_split_error_is_learned(self) -> None:
        config, _ = load_config("model61/config.json")
        mapping = fit_mapping(
            ["with in", "with in", "with in", "with in"],
            ["within", "within", "within", "with in"],
            config,
        )
        self.assertEqual(mapping[("with", "in")]["target"], ("within",))
        self.assertEqual(mapping[("with", "in")]["observations"], 4)

    def test_low_precision_error_is_rejected(self) -> None:
        config, _ = load_config("model61/config.json")
        mapping = fit_mapping(
            ["with in"] * 7,
            ["within"] * 3 + ["with in"] * 4,
            config,
        )
        self.assertNotIn(("with", "in"), mapping)

    def test_longest_span_is_applied_first(self) -> None:
        mapping = {
            ("with",): {"target": ("width",), "support": 3, "observations": 3, "precision": 1.0},
            ("with", "in"): {"target": ("within",), "support": 3, "observations": 3, "precision": 1.0},
        }
        result, changes = repair_exact_spans("paid with in hand", mapping, 2)
        self.assertEqual(result, "paid within hand")
        self.assertEqual(changes[0]["before"], ["with", "in"])


if __name__ == "__main__":
    unittest.main()
