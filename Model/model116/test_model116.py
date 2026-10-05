"""CPU regression tests for shuffled-word labels, isolation, and provenance."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model116"))
import data


class ShuffledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((data.OUT / "manifest.json").read_text())
        cls.bank = json.loads(data.BANK.read_text())
        cls.table, cls.eligible = data.eligible_rows()

    def validate(self, manifest=None, bank=None, eligible=None):
        return data.validate_rows(manifest if manifest is not None else self.manifest,
                                  bank if bank is not None else self.bank,
                                  eligible if eligible is not None else self.eligible,
                                  self.table.target_column)

    def test_complete_dataset(self):
        rows = self.validate()
        self.assertEqual(len(rows), 819)
        self.assertEqual({len(r["donors"]) for r in rows}, {3, 4})

    def test_changed_label_rejected(self):
        m = copy.deepcopy(self.manifest)
        m["rows"][0]["target"] += " extra"
        with self.assertRaises(ValueError): self.validate(m)

    def test_heldout_donor_rejected(self):
        eligible = dict(self.eligible)
        del eligible[self.manifest["rows"][0]["donors"][0]["source_id"]]
        with self.assertRaises(ValueError): self.validate(eligible=eligible)

    def test_wrong_length_rejected(self):
        m = copy.deepcopy(self.manifest)
        m["rows"][0]["donors"] = m["rows"][0]["donors"][:2]
        with self.assertRaises(ValueError): self.validate(m)

    def test_duplicate_row_rejected(self):
        m = copy.deepcopy(self.manifest)
        m["rows"][1] = m["rows"][0]
        with self.assertRaises(ValueError): self.validate(m)

    def test_bad_hash_rejected(self):
        m = copy.deepcopy(self.manifest)
        m["rows"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): self.validate(m)

    def test_leaky_aligner_rejected(self):
        b = copy.deepcopy(self.bank)
        b["lineage"]["training_excluded_folds"] = []
        with self.assertRaises(ValueError): self.validate(bank=b)

    def test_exact_pixel_recomposition(self):
        from PIL import Image, ImageChops
        for row in self.manifest["rows"][:4]:
            with Image.open(data.absolute(row["image"])) as saved:
                self.assertIsNone(ImageChops.difference(saved.convert("RGB"), data.compose(row["donors"])).getbbox())

    def test_validation_cannot_receive_synthetic_rows(self):
        from road_htr.vlm_data import build_vlm_dataset
        with self.assertRaisesRegex(ValueError, "never enter validation"):
            build_vlm_dataset(data.TRUTH, ROOT / "data/raw/images", fold=0,
                              fold_mode="validation", synthetic_manifest=data.OUT / "manifest.json")


if __name__ == "__main__":
    unittest.main()
