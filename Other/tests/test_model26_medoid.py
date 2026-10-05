from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from road_htr.deployment_medoid import (
    EXPECTED_SOURCE_COUNT,
    FIXED_CHAR_MIX,
    apply,
    parse_five_sources,
)
from road_htr.schema import SchemaError, read_table
from road_htr.submission import validate_submission


class Model26MedoidTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_csv(self, name: str, rows: list[tuple[str, str]]) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
            writer.writeheader()
            for identifier, prediction in rows:
                writer.writerow({"ID": identifier, "prediction": prediction})
        return path

    def _write_jsonl(
        self,
        name: str,
        rows: list[tuple[str, str, float | None]],
    ) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8") as handle:
            for identifier, prediction, score in rows:
                record: dict[str, object] = {
                    "id": identifier,
                    "candidates": [prediction],
                }
                if score is not None:
                    record["scores"] = [score]
                handle.write(json.dumps(record) + "\n")
        return path

    @staticmethod
    def _specifications(paths: list[Path]) -> list[str]:
        return [f"fold{index}={path}" for index, path in enumerate(paths)]

    def test_majority_sample_order_diagnostics_and_submission(self) -> None:
        test = self._write_csv("test.csv", [("a", "unused"), ("b", "unused")])
        sample = self.root / "sample.csv"
        with sample.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Target"])
            writer.writeheader()
            writer.writerow({"ID": "b", "Target": "placeholder"})
            writer.writerow({"ID": "a", "Target": "placeholder"})

        fold_rows = [
            [("a", "the deed"), ("b", "first version")],
            [("a", "the deed"), ("b", "second version")],
            [("a", "the deed"), ("b", "second version")],
            [("a", "a deed"), ("b", "second version")],
            [("a", "deed"), ("b", "other version")],
        ]
        paths = [self._write_csv(f"fold{i}.csv", rows) for i, rows in enumerate(fold_rows)]
        output = self.root / "predictions.csv"
        diagnostics = self.root / "agreement.jsonl"
        submission = self.root / "submission.csv"

        result = apply(
            test,
            self._specifications(paths),
            output,
            diagnostics,
            sample,
            submission,
        )

        table = read_table(output)
        self.assertEqual([row["ID"] for row in table.rows], ["b", "a"])
        self.assertEqual([row["prediction"] for row in table.rows], ["second version", "the deed"])
        diagnostic_rows = [
            json.loads(line) for line in diagnostics.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([row["id"] for row in diagnostic_rows], ["b", "a"])
        self.assertTrue(all(row["selection_reason"] == "exact_majority" for row in diagnostic_rows))
        self.assertEqual(diagnostic_rows[0]["source_count"], EXPECTED_SOURCE_COUNT)
        self.assertEqual(diagnostic_rows[0]["fixed_char_mix"], FIXED_CHAR_MIX)
        self.assertEqual(result["weights"], {f"fold{i}": 0.2 for i in range(5)})
        self.assertEqual(result["submission_validation"]["rows"], 2)
        self.assertEqual(validate_submission(submission, sample)["rows"], 2)

    def test_confidence_breaks_only_a_medoid_tie(self) -> None:
        test = self._write_csv("test.csv", [("row", "unused")])
        values = [
            ("a", 0.10),
            ("a", 0.20),
            ("b", 0.90),
            ("b", 0.80),
            ("c", 1.00),
        ]
        paths = [
            self._write_jsonl(f"fold{i}.jsonl", [("row", text, confidence)])
            for i, (text, confidence) in enumerate(values)
        ]
        output = self.root / "predictions.csv"

        result = apply(test, self._specifications(paths), output)

        table = read_table(output)
        self.assertEqual(table.rows[0]["prediction"], "b")
        diagnostic_path = Path(result["agreement_output"])
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        self.assertEqual(diagnostic["selection_reason"], "confidence_tiebreak")
        self.assertTrue(diagnostic["confidence_tiebreak_used"])
        self.assertEqual(diagnostic["tied_medoid_count"], 2)
        # Candidate c has the strongest individual confidence but a worse medoid
        # cost, so confidence cannot promote it into the tie.
        self.assertNotEqual(diagnostic["prediction"], "c")

    def test_internal_spaces_survive_sources_medoid_and_submission(self) -> None:
        test = self._write_csv("test.csv", [("row", "unused")])
        sample = self.root / "sample.csv"
        with sample.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Target"])
            writer.writeheader()
            writer.writerow({"ID": "row", "Target": "placeholder"})
        values = ["seal  and", "seal  and", "seal  and", "seal and", "seal and"]
        paths = [
            self._write_csv(f"spaces{i}.csv", [("row", value)])
            for i, value in enumerate(values)
        ]
        output = self.root / "spaces_output.csv"
        submission = self.root / "spaces_submission.csv"

        apply(
            test,
            self._specifications(paths),
            output,
            sample_path=sample,
            submission_path=submission,
        )

        self.assertEqual(read_table(output).rows[0]["prediction"], "seal  and")
        self.assertEqual(read_table(submission).rows[0]["Target"], "seal  and")

    def test_lexical_fallback_is_independent_of_source_order(self) -> None:
        test = self._write_csv("test.csv", [("row", "unused")])
        values = ["b", "a", "b", "a", "c"]
        paths = [
            self._write_csv(f"fold{i}.csv", [("row", value)])
            for i, value in enumerate(values)
        ]
        forward = self.root / "forward.csv"
        reverse = self.root / "reverse.csv"

        apply(test, self._specifications(paths), forward)
        reverse_specs = [
            f"fold{index}={path}"
            for index, path in enumerate(reversed(paths))
        ]
        apply(test, reverse_specs, reverse)

        self.assertEqual(read_table(forward).rows[0]["prediction"], "a")
        self.assertEqual(read_table(reverse).rows[0]["prediction"], "a")

    def test_requires_exactly_five_unique_sources(self) -> None:
        path = self._write_csv("fold.csv", [("row", "text")])
        with self.assertRaisesRegex(ValueError, "exactly 5 sources"):
            parse_five_sources([f"fold{i}={path}" for i in range(4)])
        with self.assertRaisesRegex(ValueError, "path is repeated"):
            parse_five_sources([f"fold{i}={path}" for i in range(5)])

    def test_rejects_id_mismatch_and_empty_prediction(self) -> None:
        test = self._write_csv("test.csv", [("a", "unused"), ("b", "unused")])
        complete = [
            self._write_csv(f"complete{i}.csv", [("a", "one"), ("b", "two")])
            for i in range(4)
        ]
        incomplete = self._write_csv("incomplete.csv", [("a", "one")])
        with self.assertRaisesRegex(SchemaError, "ID mismatch"):
            apply(
                test,
                self._specifications([*complete, incomplete]),
                self.root / "mismatch.csv",
            )

        empty = self._write_csv("empty_prediction.csv", [("a", "one"), ("b", "")])
        with self.assertRaisesRegex(SchemaError, "empty prediction"):
            apply(
                test,
                self._specifications([*complete, empty]),
                self.root / "empty.csv",
            )

    def test_rejects_empty_source_and_invalid_confidence(self) -> None:
        empty = self.root / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(SchemaError, "source is empty"):
            parse_five_sources(
                [
                    f"empty={empty}",
                    *[
                        f"fold{i}={self._write_csv(f'valid{i}.csv', [('row', 'text')])}"
                        for i in range(4)
                    ],
                ]
            )

        bad = self._write_jsonl("bad.jsonl", [("row", "text", 1.5)])
        valid = [
            self._write_csv(f"more{i}.csv", [("row", "text")]) for i in range(4)
        ]
        with self.assertRaisesRegex(SchemaError, r"finite and in \[0, 1\]"):
            parse_five_sources([f"bad={bad}", *self._specifications(valid)])


if __name__ == "__main__":
    unittest.main()
