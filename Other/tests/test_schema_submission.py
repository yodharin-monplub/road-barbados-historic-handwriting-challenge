from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from road_htr.schema import SchemaError, read_table
from road_htr.submission import validate_submission, write_submission


class SchemaSubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_csv(self, name: str, rows: list[dict[str, str]]) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_schema_inference(self) -> None:
        path = self.write_csv("train.csv", [{"ID": "abc", "Target": "a deed"}])
        table = read_table(path, require_target=True)
        self.assertEqual(table.id_column, "ID")
        self.assertEqual(table.target_column, "Target")

    def test_submission_round_trip(self) -> None:
        sample = self.write_csv(
            "sample.csv",
            [{"ID": "b", "Target": "x"}, {"ID": "a", "Target": "x"}],
        )
        output = self.root / "submission.csv"
        write_submission(sample, output, {"a": " alpha\n", "b": "beta "})
        self.assertEqual(validate_submission(output, sample)["rows"], 2)
        table = read_table(output)
        self.assertEqual([row["ID"] for row in table.rows], ["b", "a"])

    def test_empty_prediction_fails(self) -> None:
        sample = self.write_csv("sample.csv", [{"ID": "a", "Target": "x"}])
        with self.assertRaises(SchemaError):
            write_submission(sample, self.root / "bad.csv", {"a": "  "})


if __name__ == "__main__":
    unittest.main()

