from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from model74.evaluate import evaluate
from road_htr.vlm_cli import _sha256, _tree_sha256


class Model74EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.truth = self.root / "truth.csv"
        with self.truth.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Target", "fold"])
            writer.writeheader()
            writer.writerows(
                [{"ID": "a", "Target": "alpha", "fold": 0}, {"ID": "b", "Target": "beta", "fold": 1}]
            )
        self.predictions = self.root / "predictions.csv"
        with self.predictions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
            writer.writeheader()
            writer.writerow({"ID": "a", "prediction": "alpha"})
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter.safetensors").write_bytes(b"adapter")
        self.manifest = self.predictions.with_suffix(".manifest.json")
        self.manifest.write_text(
            json.dumps(
                {
                    "output_sha256": _sha256(self.predictions),
                    "input_sha256": _sha256(self.truth),
                    "fold": 0,
                    "adapter": str(self.adapter),
                    "adapter_files_sha256": _tree_sha256(self.adapter),
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_verified_exact_score_receipt(self) -> None:
        receipt = evaluate(
            self.truth, self.predictions, self.manifest, self.root / "receipt.json", 0
        )
        self.assertEqual(receipt["score"]["multi_score"], 1.0)
        self.assertEqual(receipt["empty_prediction_ids"], [])

    def test_tampered_prediction_fails_closed(self) -> None:
        with self.predictions.open("a", encoding="utf-8") as handle:
            handle.write("tamper\n")
        with self.assertRaisesRegex(ValueError, "prediction checksum"):
            evaluate(
                self.truth, self.predictions, self.manifest, self.root / "receipt.json", 0
            )


if __name__ == "__main__":
    unittest.main()
