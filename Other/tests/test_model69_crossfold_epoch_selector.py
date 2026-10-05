from __future__ import annotations

import unittest

from model69.select_crossfold_epochs import load_config, select_epoch


class Model69CrossfoldEpochSelectorTests(unittest.TestCase):
    def test_config_is_frozen(self) -> None:
        config = load_config(__import__("pathlib").Path("model69/config.json"))
        self.assertEqual(config["eligible_epochs_zero_based"], list(range(8)))

    def test_heldout_fold_does_not_control_selection(self) -> None:
        scores = {
            0: {0: 0.99, 1: 0.01},
            1: {0: 0.10, 1: 0.90},
            2: {0: 0.10, 1: 0.90},
            3: {0: 0.10, 1: 0.90},
            4: {0: 0.10, 1: 0.90},
        }
        epoch, _ = select_epoch(scores, [1, 2, 3, 4], [0, 1])
        self.assertEqual(epoch, 1)

    def test_tie_selects_earlier_epoch(self) -> None:
        scores = {fold: {0: 0.8, 1: 0.8} for fold in range(5)}
        epoch, _ = select_epoch(scores, [0, 1, 2, 3], [0, 1])
        self.assertEqual(epoch, 0)


if __name__ == "__main__":
    unittest.main()
