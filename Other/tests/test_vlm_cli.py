import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from road_htr.vlm_cli import _verify_local_checkpoint


class LocalVlmCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "weights.bin").write_bytes(b"weights")
        manifest = {
            "sha": "abc123",
            "cardData": {"license": "apache-2.0"},
        }
        (self.root / "upstream_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        self.config = {
            "revision": "abc123",
            "license": "Apache-2.0",
            "expected_files_sha256": {
                name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                for name in ("weights.bin", "upstream_manifest.json")
            },
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_local_checkpoint_passes(self):
        _verify_local_checkpoint(self.config, self.root)

    def test_byte_drift_and_unexpected_file_fail_closed(self):
        (self.root / "weights.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            _verify_local_checkpoint(self.config, self.root)
        (self.root / "weights.bin").write_bytes(b"weights")
        (self.root / "extra.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "file set mismatch"):
            _verify_local_checkpoint(self.config, self.root)

    def test_revision_and_license_are_verified(self):
        self.config["revision"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
            _verify_local_checkpoint(self.config, self.root)
        self.config["revision"] = "abc123"
        self.config["license"] = "other"
        with self.assertRaisesRegex(RuntimeError, "license mismatch"):
            _verify_local_checkpoint(self.config, self.root)


if __name__ == "__main__":
    unittest.main()
