from __future__ import annotations

import unittest

from model67.contextual_deletion_repair import (
    fit_mapping,
    load_config,
    repair_contextual_deletions,
)


class Model67ContextualDeletionTests(unittest.TestCase):
    def test_config_is_frozen(self) -> None:
        config, _ = load_config("model67/config.json")
        self.assertEqual(config["minimum_precision"], 0.8)

    def test_context_mapping_is_learned(self) -> None:
        config, _ = load_config("model67/config.json")
        mapping = fit_mapping(["paid the the sum"] * 2, ["paid the sum"] * 2, config)
        self.assertIn(("the", "the", "sum"), mapping)

    def test_same_token_outside_context_is_preserved(self) -> None:
        mapping = {("the", "the", "sum"): {"support": 2, "observations": 2, "precision": 1.0}}
        result, changes = repair_contextual_deletions("the other record", mapping)
        self.assertEqual(result, "the other record")
        self.assertEqual(changes, [])

    def test_contextual_deletion(self) -> None:
        mapping = {("the", "the", "sum"): {"support": 2, "observations": 2, "precision": 1.0}}
        result, changes = repair_contextual_deletions("paid the the sum today", mapping)
        self.assertEqual(result, "paid the sum today")
        self.assertEqual(changes[0]["before"], "the")

    def test_single_token_never_becomes_blank(self) -> None:
        mapping = {("<s>", "noise", "</s>"): {"support": 2, "observations": 2, "precision": 1.0}}
        result, changes = repair_contextual_deletions("noise", mapping)
        self.assertEqual(result, "noise")
        self.assertEqual(changes, [])


if __name__ == "__main__":
    unittest.main()
