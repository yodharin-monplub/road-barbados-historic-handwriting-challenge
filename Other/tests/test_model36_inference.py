import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from road_htr import model36_inference as m


class Model36InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "predictions.csv"
        self.output.write_text("ID,prediction\na,text\n", encoding="utf-8")
        self.manifest = self.output.with_suffix(".manifest.json")
        self.manifest.write_text(json.dumps({"rows": 1}) + "\n", encoding="utf-8")

    def test_output_argument_accepts_split_and_equals_forms(self):
        self.assertEqual(
            m.output_argument(["--input", "in.csv", "--output", str(self.output)]),
            self.output,
        )
        self.assertEqual(
            m.output_argument([f"--output={self.output}", "--batch-size", "1"]),
            self.output,
        )

    def test_authenticates_exact_prediction_bytes(self):
        result = m.authenticate_prediction_manifest(self.output)
        payload = json.loads(result.read_text(encoding="utf-8"))
        self.assertEqual(payload["predictions_sha256"], m._sha256(self.output))
        self.assertFalse(result.with_suffix(result.suffix + ".tmp").exists())

    def test_existing_matching_hash_is_idempotent(self):
        m.authenticate_prediction_manifest(self.output)
        m.authenticate_prediction_manifest(self.output)
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["predictions_sha256"], m._sha256(self.output))

    def test_conflicting_hash_fails_closed_without_rewriting(self):
        original = {"rows": 1, "predictions_sha256": "0" * 64}
        self.manifest.write_text(json.dumps(original) + "\n", encoding="utf-8")
        before = self.manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, "conflicting"):
            m.authenticate_prediction_manifest(self.output)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_signed_entrypoint_runs_inference_before_authentication(self):
        self.output.unlink()
        self.manifest.unlink()

        def fake_infer(default_config):
            self.assertEqual(default_config, self.root / "config.yaml")
            self.output.write_text("ID,prediction\na,new text\n", encoding="utf-8")
            self.manifest.write_text(json.dumps({"rows": 1}) + "\n", encoding="utf-8")

        argv = ["infer.py", "--input", "input.csv", "--output", str(self.output)]
        with patch.object(m, "infer_main", side_effect=fake_infer) as infer, patch.object(
            sys, "argv", argv
        ):
            m.signed_infer_main(self.root / "config.yaml")
        infer.assert_called_once()
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["predictions_sha256"], m._sha256(self.output))


if __name__ == "__main__":
    unittest.main()
