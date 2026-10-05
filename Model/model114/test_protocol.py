"""CPU-only regression checks for the frozen seed pilot setup."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model114"))
import pilot


class ProtocolTests(unittest.TestCase):
    def test_frozen_inputs_and_runtime(self):
        protocol = pilot.check()
        self.assertEqual(len(protocol["train_ids"]), 3276)
        self.assertEqual(len(protocol["validation_ids"]), 819)
        self.assertFalse(set(protocol["train_ids"]) & set(protocol["validation_ids"]))

    def test_only_seed_and_output_differ(self):
        configs = [json.loads((ROOT / f"model{n}/config.json").read_text()) for n in (114, 115)]
        self.assertNotEqual(configs[0].pop("seed"), configs[1].pop("seed"))
        self.assertNotEqual(configs[0].pop("output_dir"), configs[1].pop("output_dir"))
        self.assertEqual(configs[0], configs[1])
        training = configs[0]["training"]
        self.assertEqual(training["batch_size"] * training["gradient_accumulation"], 16)
        self.assertEqual(training["max_steps"], 400)
        self.assertEqual(training["save_steps"], 400)
        self.assertEqual(training["eval_steps"], 400)

    def test_smoke_is_separate_and_fold_isolated(self):
        table = pilot.read_table(ROOT / "outputs/model114/trainer_smoke/rows.csv", require_target=True)
        protocol = pilot.check()
        train = {r[table.id_column] for r in table.rows if int(r["fold"]) != 0}
        val = {r[table.id_column] for r in table.rows if int(r["fold"]) == 0}
        self.assertEqual(len(train), 32)
        self.assertEqual(len(val), 3)
        self.assertLessEqual(train, set(protocol["train_ids"]))
        self.assertLessEqual(val, set(protocol["validation_ids"]))
        smoke = json.loads((ROOT / "model114/smoke_config.json").read_text())
        self.assertNotEqual(smoke["output_dir"], "outputs/model114/fold{fold}")
        self.assertEqual(smoke["training"]["max_steps"], 2)

    def test_unchanged_voter_reproduces_existing_ensemble(self):
        _, truth = pilot.canonical()
        source, _ = pilot.sources(truth)
        anchor = pilot.read_predictions(ROOT / "outputs/model104/full_oof/crossfit_predictions.csv")
        old = pilot.read_predictions(ROOT / "outputs/model112/diagnostic_family_word_oof.csv")
        for i in pilot.check()["validation_ids"]:
            result = pilot.fuse(anchor[i], {n: source[n][i] for n in pilot.NAMES}, True, False)
            self.assertEqual(result, old[i], i)


if __name__ == "__main__":
    unittest.main()
