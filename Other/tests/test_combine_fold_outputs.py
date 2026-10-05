from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "combine_fold_outputs.py"
SPEC = importlib.util.spec_from_file_location("combine_fold_outputs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CombineFoldOutputsTests(unittest.TestCase):
    def test_csv_is_restored_to_truth_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "truth.csv"
            truth.write_text("ID,fold\na,0\nb,1\nc,0\n", encoding="utf-8")
            fold0 = root / "f0.csv"
            fold0.write_text("ID,prediction\nc,C\na,A\n", encoding="utf-8")
            fold1 = root / "f1.csv"
            fold1.write_text("ID,prediction\nb,B\n", encoding="utf-8")
            id_column, order, expected = MODULE._expected_ids(truth)
            output = root / "combined.csv"
            MODULE.combine_csv(
                id_column, order, expected, {0: fold0, 1: fold1}, output
            )
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["ID"] for row in rows], ["a", "b", "c"])

    def test_jsonl_rejects_wrong_fold_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "combined.jsonl"
            fold0 = root / "f0.jsonl"
            fold0.write_text(json.dumps({"id": "wrong"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.combine_jsonl(["a"], {0: {"a"}}, {0: fold0}, output)


if __name__ == "__main__":
    unittest.main()
