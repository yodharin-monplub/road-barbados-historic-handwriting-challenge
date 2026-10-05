from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from road_htr.vlm_cli import _sha256


SCRIPT = Path(__file__).resolve().parents[1] / "model76" / "evaluate_oof.py"
SPEC = importlib.util.spec_from_file_location("model76_evaluate_oof", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Model76OofTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.truth = self.root / "truth.csv"
        with self.truth.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Target", "fold"])
            writer.writeheader()
            for fold in range(5):
                writer.writerow({"ID": f"id{fold}", "Target": f"truth{fold}", "fold": fold})
        (self.root / "model73").mkdir()
        (self.root / "model74").mkdir()
        for fold in range(5):
            prediction = self.root / "model73" / f"fold{fold}_oof_step400.csv"
            prediction.write_text(
                f"ID,prediction\nid{fold},truth{fold}\n", encoding="utf-8"
            )
            manifest = prediction.with_suffix(".manifest.json")
            manifest.write_text(
                json.dumps(
                    {
                        "fold": fold,
                        "input_sha256": _sha256(self.truth),
                        "output_sha256": _sha256(prediction),
                    }
                ),
                encoding="utf-8",
            )
            signed = self.root / "model74" / f"fold{fold}_step400.evaluation.json"
            signed.write_text(
                json.dumps(
                    {
                        "fold": fold,
                        "score": {"mean_error": 0.0},
                        "predictions_sha256": _sha256(prediction),
                        "inference_manifest_sha256": _sha256(manifest),
                    }
                ),
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_combines_all_folds_in_truth_order(self) -> None:
        receipt = MODULE.evaluate(
            self.truth,
            self.root,
            self.root / "combined.csv",
            self.root / "combined.evaluation.json",
        )
        self.assertEqual(receipt["rows"], 5)
        self.assertEqual(receipt["empty_prediction_ids"], [])
        self.assertEqual(receipt["score"]["mean_error"], 0.0)

    def test_rejects_tampered_fold_prediction(self) -> None:
        prediction = self.root / "model73" / "fold3_oof_step400.csv"
        prediction.write_text("ID,prediction\nid3,tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            MODULE.evaluate(
                self.truth,
                self.root,
                self.root / "combined.csv",
                self.root / "combined.evaluation.json",
            )


if __name__ == "__main__":
    unittest.main()
