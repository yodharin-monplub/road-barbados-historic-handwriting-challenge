import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from road_htr import model36_export as m


class Model36ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(m, "PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.training = self.root / "outputs/model36/foldall"
        self.training.mkdir(parents=True)
        config = {"training": {"all_data_epochs": 8, "all_data_cosine_epochs": 10}}
        config_file = self.root / "model36/config.yaml"
        config_file.parent.mkdir()
        config_file.write_text(json.dumps(config))
        self.manifest = self.training / "run_manifest.json"
        self.manifest.write_text(json.dumps({"config": config, "preparation": {"fold": None}}))
        for epoch in range(8):
            (self.training / f"checkpoint_{epoch:02d}-0.9.ckpt").write_bytes(str(epoch).encode())
        self.best = self.training / "best_0.99.safetensors"
        self.best.write_bytes(b"not the selected weights")
        self.output = self.root / "outputs/model36/export"

    @staticmethod
    def converter(command, **kwargs):
        Path(command[3]).write_bytes(b"converted epoch seven")

    def test_selects_only_epoch_seven_and_records_receipt(self):
        with patch.object(m.subprocess, "run", side_effect=self.converter) as run:
            weights = m.export_fixed_checkpoint(self.training, self.output)
        command = run.call_args.args[0]
        self.assertEqual(Path(command[-1]).name, "checkpoint_07-0.9.ckpt")
        self.assertEqual(len(command), 5)
        receipt = json.loads((self.output / "epoch07.conversion.json").read_text())
        self.assertEqual(receipt["weights_sha256"], m._sha256(weights))
        self.assertEqual(receipt["checkpoint_epoch_zero_based"], 7)
        with self.assertRaises(FileExistsError):
            m.export_fixed_checkpoint(self.training, self.output)

    def test_missing_checkpoint_and_stale_completion_marker_rejected(self):
        checkpoint = self.training / "checkpoint_06-0.9.ckpt"
        checkpoint.unlink()
        with self.assertRaisesRegex(ValueError, "not complete"):
            m.export_fixed_checkpoint(self.training, self.output)
        checkpoint.write_bytes(b"6")
        os.utime(self.best, ns=(1, 1))
        with self.assertRaisesRegex(ValueError, "predates"):
            m.export_fixed_checkpoint(self.training, self.output)
        self.assertFalse(self.output.exists())

    def test_fold_training_rejected(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["preparation"]["fold"] = 0
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "all-data"):
            m.export_fixed_checkpoint(self.training, self.output)

    def test_changed_source_during_conversion_has_no_receipt(self):
        def converter(command, **kwargs):
            self.converter(command, **kwargs)
            Path(command[-1]).write_bytes(b"changed")
        with patch.object(m.subprocess, "run", side_effect=converter), self.assertRaisesRegex(RuntimeError, "changed"):
            m.export_fixed_checkpoint(self.training, self.output)
        self.assertFalse((self.output / "epoch07.conversion.json").exists())


if __name__ == "__main__":
    unittest.main()
