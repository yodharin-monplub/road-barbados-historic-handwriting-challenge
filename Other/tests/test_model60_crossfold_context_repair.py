from __future__ import annotations

import unittest

from model60.crossfold_context_repair import (
    build_outer_fold_models,
    load_config,
    transform,
)


class Model60CrossfoldContextRepairTests(unittest.TestCase):
    def test_config_is_frozen_to_context_only_recipe(self) -> None:
        config, _ = load_config("model60/config.json")
        self.assertEqual(config["context_weight"], 16.0)
        self.assertEqual(config["distance_penalty"], 32.0)
        self.assertNotIn("boundary", config)
        self.assertNotIn("confusion", config)

    def test_heldout_label_is_excluded(self) -> None:
        models = build_outer_fold_models(
            ["uniquezero", "one word", "two word"], [0, 1, 2]
        )
        self.assertNotIn("uniquezero", models[0].frequencies)
        self.assertIn("uniquezero", models[1].frequencies)

    def test_context_repairs_supported_misspelling(self) -> None:
        config, _ = load_config("model60/config.json")
        references = ["the presence of"] * 3 + ["another unrelated line"]
        folds = [0, 0, 0, 1]
        models = build_outer_fold_models(references, folds)
        values, operations = transform(
            ["the presense of"], {1: models[1]}, [1], config
        )
        self.assertEqual(values, ["the presence of"])
        self.assertEqual(len(operations[0]), 1)


if __name__ == "__main__":
    unittest.main()
