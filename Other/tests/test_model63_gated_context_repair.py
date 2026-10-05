from __future__ import annotations

import unittest

from model63.gated_context_repair import apply_gate, load_config


class Model63GatedContextRepairTests(unittest.TestCase):
    def test_config_is_frozen(self) -> None:
        config, _ = load_config("model63/config.json")
        self.assertEqual(config["minimum_candidate_frequency"], 2)

    def test_gate_accepts_only_rows_whose_every_change_is_supported(self) -> None:
        baseline = ["a", "b", "c"]
        proposals = ["x", "y", "z"]
        operations = [[{"frequency": 2}], [{"frequency": 3}, {"frequency": 1}], []]
        values, accepted = apply_gate(baseline, proposals, operations, 2)
        self.assertEqual(values, ["x", "b", "c"])
        self.assertEqual(accepted, [True, False, False])


if __name__ == "__main__": unittest.main()
