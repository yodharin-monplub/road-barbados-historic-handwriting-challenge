import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from model46 import transfer


class Model46Tests(unittest.TestCase):
    def setUp(self):
        self.config, _ = transfer.load_config("model46/config.json")

    def test_recipe_and_terminal_are_immutable(self):
        self.assertEqual(self.config["terminal_stage"], "confusions")
        self.assertEqual(self.config["frozen_recipe"]["confusions"]["score_margin"], 12.0)
        bad = copy.deepcopy(self.config)
        bad["stage_order"].append("approximate_joins")
        with self.assertRaises(transfer.Model46GateError):
            transfer.validate_config(bad)

    def test_pins_and_dry_audit_verify_nonheldout_state_fit(self):
        audit = transfer.dry_audit("model46/config.json")
        self.assertEqual(audit["status"], "cpu_dry_audit_passed_no_candidate_score_claimed")
        self.assertEqual(audit["state_fit_audit"]["fit_folds"], [1, 2, 3, 4])
        self.assertEqual(audit["state_fit_audit"]["fit_rows"], 3276)
        self.assertFalse(audit["state_fit_audit"]["fit_uses_candidate_predictions"])

    def test_candidate_requires_complete_exact_fold0_ids(self):
        _, all_ids, _, folds = transfer.load_truth(self.config)
        ids = [item for item, fold in zip(all_ids, folds, strict=True) if fold == 0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
                writer.writeheader()
                writer.writerow({"ID": ids[0], "prediction": "kept  internal spaces"})
            with self.assertRaises(transfer.Model46GateError):
                transfer.load_candidate(self.config, path, ids)

    def test_optional_receipt_must_bind_path_or_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.csv"
            candidate.write_text("ID,prediction\na,x\n", encoding="utf-8")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"output": str(candidate)}), encoding="utf-8")
            result = transfer.verify_optional_lineage(receipt, {"path": str(candidate), "sha256": transfer.sha256(candidate)})
            self.assertEqual(result["status"], "receipt_binds_candidate")


if __name__ == "__main__":
    unittest.main()
