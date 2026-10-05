import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from road_htr import paragraph_synthetic as ps
from road_htr import stackmix


class ParagraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [patch.object(ps, "ROOT", self.root), patch.object(stackmix, "ROOT", self.root)]
        for p in self.patches: p.start()
        Image.new("RGB", (100, 30), "white").save(self.root / "line.png")
        self.eligible = {"a": {"Target": "SYM Esq. hath"}, "b": {"Target": "Esq. hath Land"}}
        self.words = {str(i): {"target": t, "style": 0} for i, t in enumerate("SYM Esq. hath Land".split())}
        self.accepted = set(self.words)
        self.rows = [{"id": "synthetic", "paragraph_id": "p0", "line_index": 0, "style": 0,
                      "word_ids": list(self.words), "target": "SYM Esq. hath Land", "image": "line.png",
                      "sha256": stackmix.sha256(self.root / "line.png"), "transitions": [
                          {"source_id": "a", "token_index": 0}, {"source_id": "a", "token_index": 1},
                          {"source_id": "b", "token_index": 1}], "previous_line_transition": None}]

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def validate(self):
        return ps.validate_lines(self.rows, self.words, self.accepted, self.eligible, "Target")

    def test_new_text_preserves_historical_spelling(self):
        self.assertEqual(self.validate()[0]["target"], "SYM Esq. hath Land")

    def test_changed_spelling_rejected(self):
        self.rows[0]["target"] = "SIM Esq. has Land"
        with self.assertRaises(ValueError): self.validate()

    def test_heldout_transition_rejected(self):
        self.rows[0]["transitions"][0]["source_id"] = "heldout"
        with self.assertRaisesRegex(ValueError, "held-out"): self.validate()

    def test_invented_transition_rejected(self):
        self.rows[0]["transitions"][0]["token_index"] = 1
        with self.assertRaises(ValueError): self.validate()

    def test_unaccepted_word_rejected(self):
        self.accepted.remove("0")
        with self.assertRaisesRegex(ValueError, "unaccepted"): self.validate()

    def test_wrong_style_rejected(self):
        self.words["0"]["style"] = 1
        with self.assertRaisesRegex(ValueError, "style"): self.validate()

    def test_duplicate_rejected(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"): self.validate()

    def test_original_line_rejected(self):
        self.eligible["c"] = {"Target": self.rows[0]["target"]}
        with self.assertRaisesRegex(ValueError, "newly composed"): self.validate()

    def test_changed_pixels_rejected(self):
        Image.new("RGB", (100, 30), "black").save(self.root / "line.png")
        with self.assertRaisesRegex(ValueError, "hash"): self.validate()

    def test_missing_edges_rejected(self):
        self.rows[0]["transitions"].pop()
        with self.assertRaisesRegex(ValueError, "Missing"): self.validate()

    def test_wrong_paragraph_index_rejected(self):
        self.rows[0]["line_index"] = 1
        with self.assertRaisesRegex(ValueError, "grouping"): self.validate()


if __name__ == "__main__":
    unittest.main()
