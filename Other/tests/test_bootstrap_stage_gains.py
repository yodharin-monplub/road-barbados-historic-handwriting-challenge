import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import bootstrap_stage_gains as m
from road_htr.metrics import competition_error


class BootstrapStageGainsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.references = ["selection reference", "assistant: one  two", "three four"]
        self.before = ["wrong", "one two", "three wrong"]
        self.after = ["much worse unrelated selection", "<|assistant|>one  two", "three four"]
        for name, columns, rows in (
            ("truth", ["ID", "Target", "fold"], [(str(i), value, i) for i, value in enumerate(self.references)]),
            ("before", ["ID", "prediction"], [(str(i), value) for i, value in enumerate(self.before)]),
            ("after", ["ID", "prediction"], [(str(i), value) for i, value in enumerate(self.after)]),
        ):
            with (self.root / f"{name}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerows(rows)

    def run_cli(self, *extra):
        argv = ["bootstrap_stage_gains.py", "--truth", str(self.root / "truth.csv"),
                "--prediction", f"before={self.root / 'before.csv'}",
                "--prediction", f"after={self.root / 'after.csv'}", "--samples", "20", *extra]
        output = io.StringIO()
        with patch("sys.argv", argv), contextlib.redirect_stdout(output):
            m.main()
        return json.loads(output.getvalue())

    def test_excluding_selection_fold_matches_exact_confirmation_metric(self):
        report = self.run_cli("--exclude-fold", "0")
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["included_folds"], [1, 2])
        self.assertEqual(report["excluded_folds"], [0])
        before = competition_error(self.references[1:], self.before[1:])["multi_score"]
        after = competition_error(self.references[1:], self.after[1:])["multi_score"]
        self.assertAlmostEqual(report["observed_scores"]["before"], before, places=14)
        self.assertAlmostEqual(report["comparisons"][0]["observed_delta"], after - before, places=14)
        self.assertEqual(report["observed_scores"]["after"], 1.0)

    def test_legacy_single_fold_and_all_rows_still_work(self):
        self.assertEqual(self.run_cli("--fold", "1")["rows"], 1)
        report = self.run_cli()
        self.assertEqual(report["rows"], 3)
        self.assertAlmostEqual(report["observed_scores"]["after"],
                               competition_error(self.references, self.after)["multi_score"], places=14)

    def test_unknown_and_empty_fold_selections_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            self.run_cli("--exclude-fold", "7")
        with self.assertRaisesRegex(ValueError, "no evaluation rows"):
            self.run_cli("--exclude-fold", "0", "--exclude-fold", "1", "--exclude-fold", "2")

    def test_filtered_comparison_rejects_unknown_prediction_ids(self):
        with (self.root / "after.csv").open("a") as handle:
            handle.write("unrelated,bad\n")
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            self.run_cli("--exclude-fold", "0")

    def test_output_writes_reproducible_json_and_refuses_overwrite(self):
        destination = self.root / "evidence" / "bootstrap.json"
        report = self.run_cli("--exclude-fold", "0", "--output", str(destination))
        self.assertEqual(json.loads(destination.read_text()), report)
        with self.assertRaises(FileExistsError):
            self.run_cli("--exclude-fold", "0", "--output", str(destination))


if __name__ == "__main__":
    unittest.main()
