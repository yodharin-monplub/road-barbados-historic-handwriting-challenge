from __future__ import annotations

import unittest

from model64.refit_exact_span_repair import load_config as load64
from model65.refit_context_insertion_repair import load_config as load65


class Model64And65WrapperTests(unittest.TestCase):
    def test_model64_source_and_number(self) -> None:
        config, _ = load64("model64/config.json")
        self.assertEqual((config["model_number"], config["source_model"]), (64, "model63"))

    def test_model65_source_and_number(self) -> None:
        config, _ = load65("model65/config.json")
        self.assertEqual((config["model_number"], config["source_model"]), (65, "model64"))


if __name__ == "__main__": unittest.main()
