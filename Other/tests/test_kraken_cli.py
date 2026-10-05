import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

from road_htr.kraken_cli import (
    _align_kraken_sequence_lengths_with_network,
    _prediction_cleaner,
    prepare_training_data,
    train,
)
from road_htr.vlm_data import read_id_file


class KrakenAllDataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_dir = self.root / "images"
        self.image_dir.mkdir()
        self.rows = [
            {"ID": f"row{index}", "Target": f"sample text {index}", "fold": str(index % 2)}
            for index in range(6)
        ]
        for row in self.rows:
            Image.new("RGB", (80, 24), "white").save(
                self.image_dir / f"{row['ID']}.jpg"
            )
        self.train_csv = self.root / "train.csv"
        self.write_rows()
        self.exclude_ids = self.root / "excluded.txt"
        self.exclude_ids.write_text("row5\n", encoding="utf-8")

    def write_rows(self):
        with self.train_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Target", "fold"])
            writer.writeheader()
            writer.writerows(self.rows)

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, evaluation_rows=2):
        return {
            "train_csv": str(self.train_csv),
            "image_dir": str(self.image_dir),
            "exclude_ids": str(self.exclude_ids),
            "prepared_dir": str(self.root / "prepared"),
            "image_variant": "full",
            "seed": 17,
            "training": {"all_data_evaluation_rows": evaluation_rows},
        }

    @staticmethod
    def manifest_lines(path):
        return Path(path).read_text(encoding="utf-8").splitlines()

    def test_all_data_retains_every_eligible_row_and_overlaps_evaluation(self):
        train_manifest, evaluation_manifest, details = prepare_training_data(
            self.config(), None
        )

        train_paths = self.manifest_lines(train_manifest)
        evaluation_paths = self.manifest_lines(evaluation_manifest)
        self.assertEqual(len(train_paths), 5)
        self.assertEqual(len(set(train_paths)), 5)
        self.assertEqual(len(evaluation_paths), 2)
        self.assertTrue(set(evaluation_paths).issubset(train_paths))
        self.assertIn("_overlap_", evaluation_manifest.name)
        self.assertTrue(details["validation_overlaps_training"])
        self.assertFalse(details["validation_metric_for_selection"])

    def test_all_data_evaluation_selection_is_deterministic(self):
        _, first_manifest, _ = prepare_training_data(self.config(), None)
        first = self.manifest_lines(first_manifest)
        _, second_manifest, _ = prepare_training_data(self.config(), None)
        self.assertEqual(first, self.manifest_lines(second_manifest))

    def test_all_data_evaluation_count_is_validated(self):
        for value in (None, 0, -1, True, 1.5, "2", 6):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_training_data(self.config(value), None)

    def test_fold_validation_remains_disjoint_and_ignores_all_data_setting(self):
        train_manifest, validation_manifest, details = prepare_training_data(
            self.config(0), 0
        )
        train_paths = set(self.manifest_lines(train_manifest))
        validation_paths = set(self.manifest_lines(validation_manifest))
        self.assertEqual(len(train_paths), 2)
        self.assertEqual(len(validation_paths), 3)
        self.assertTrue(train_paths.isdisjoint(validation_paths))
        self.assertFalse(details["validation_overlaps_training"])
        self.assertTrue(details["validation_metric_for_selection"])

    def test_generated_gt_strips_edges_but_preserves_internal_whitespace(self):
        self.rows[0]["Target"] = " \talpha  beta \t"
        self.write_rows()
        config = self.config()
        config["training"]["strip_target_edges"] = True

        _, _, details = prepare_training_data(config, 0)

        target = self.root / "prepared" / "lines_full" / "row0.gt.txt"
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha  beta")
        self.assertTrue(details["strip_target_edges"])
        with self.train_csv.open(encoding="utf-8", newline="") as handle:
            source = {row["ID"]: row for row in csv.DictReader(handle)}
        self.assertEqual(source["row0"]["Target"], " \talpha  beta \t")

    def test_ctc_preflight_reports_incompatible_validation_id_and_geometry(self):
        self.rows[0]["Target"] = "A" * 30
        self.write_rows()
        config = self.config()
        config["training"].update(
            {
                "codec_resize": "new",
                "ctc_subsampling": 8,
                "height": 96,
                "padding": 16,
                "max_width": 3584,
                "normalization": "NFD",
            }
        )

        for fold in (0, 1):
            with self.subTest(fold=fold), self.assertRaisesRegex(
                ValueError, "row0"
            ) as raised:
                prepare_training_data(config, fold)

            message = str(raised.exception)
            self.assertIn("minimum_ctc_alignment=59", message)
            self.assertIn("output_steps=44", message)
            self.assertIn("source=80x24", message)
            self.assertIn("height=96, padding=16, subsampling=8", message)
            self.assertIn("Kraken would silently replace", message)
            split = f"fold{fold}"
            self.assertFalse(
                (self.root / "prepared" / f"train_{split}_full.txt").exists()
            )
            self.assertFalse(
                (self.root / "prepared" / f"validation_{split}_full.txt").exists()
            )

    def test_ctc_preflight_passes_after_bad_id_is_excluded(self):
        self.rows[0]["Target"] = "A" * 30
        self.write_rows()
        self.exclude_ids.write_text("row0\nrow5\n", encoding="utf-8")
        config = self.config()
        config["training"].update(
            {
                "codec_resize": "new",
                "ctc_subsampling": 8,
                "height": 96,
                "padding": 16,
                "max_width": 3584,
                "normalization": "NFD",
            }
        )

        train_manifest, _, details = prepare_training_data(config, None)

        self.assertEqual(len(self.manifest_lines(train_manifest)), 4)
        self.assertEqual(details["ctc_preflight"]["checked_rows"], 4)
        self.assertEqual(details["ctc_preflight"]["subsampling"], 8)

    def test_all_data_command_uses_explicit_evaluation_and_augmentation(self):
        checkpoint = self.root / "pretrained.safetensors"
        checkpoint.write_bytes(b"weights")
        config = self.config(2)
        config.update(
            {
                "pretrained_model": str(checkpoint),
                "output_dir": str(self.root / "outputs" / "fold{fold}"),
                "runtime": {"device": "cuda:0", "precision": "bf16-mixed"},
            }
        )
        config["training"].update(
            {
                "all_data_epochs": 3,
                "all_data_cosine_epochs": 3,
                "augment": True,
            }
        )
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        with patch("road_htr.kraken_cli._ketos_binary", return_value="ketos"):
            train(config_path, all_data=True, dry_run=True)

        manifest = json.loads(
            (self.root / "outputs" / "foldall" / "run_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        command = manifest["command"]
        self.assertIn("--evaluation-data", command)
        evaluation_path = Path(command[command.index("--evaluation-data") + 1])
        self.assertEqual(len(self.manifest_lines(evaluation_path)), 2)
        self.assertEqual(manifest["preparation"]["train_rows"], 5)
        self.assertEqual(manifest["preparation"]["validation_rows"], 2)
        self.assertEqual(command[command.index("--quit") + 1], "fixed")
        self.assertEqual(command[-1], "--augment")


class Model22DataPolicyTests(unittest.TestCase):
    def test_recognition_width_tensor_moves_to_prepared_network_device(self):
        calls = []

        class Parameter:
            device = "cuda:0"

        class NeuralNetwork:
            @staticmethod
            def parameters():
                return iter([Parameter()])

        class Lengths:
            def to(self, *, device):
                calls.append(device)
                return ("moved", device)

        class RecognitionNetwork:
            nn = NeuralNetwork()

            @staticmethod
            def _rec_predict(line, lens=None):
                return line, lens

        class TaskModel:
            net = RecognitionNetwork()

        model = TaskModel()
        self.assertTrue(_align_kraken_sequence_lengths_with_network(model))
        self.assertEqual(model.net._rec_predict("pixels", Lengths()), ("pixels", ("moved", "cuda:0")))
        self.assertEqual(calls, ["cuda:0"])
        self.assertFalse(_align_kraken_sequence_lengths_with_network(model))

    def test_model22_has_exact_model_specific_exclusions_and_preflight(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "model22" / "config.yaml").read_text(encoding="utf-8")
        )
        exclusion_path = root / config["exclude_ids"]

        self.assertEqual(
            read_id_file(exclusion_path),
            {
                "79tMUVyfIdy3GzkG",
                "rh8o7bdCGOIBFPwH",
                "F8DYDDp2AvW9Dytw",
                "EcxuqKeZl7OQexfB",
                "gP7kGWfDFYMbZ8nO",
                "PO7QQLWIFOT65BTz",
            },
        )
        self.assertEqual(config["training"]["ctc_subsampling"], 8)
        self.assertIs(config["training"]["strip_target_edges"], True)
        self.assertIs(config["inference"]["preserve_internal_spaces"], True)

    def test_model22_cleaner_preserves_internal_spaces_and_normalizes_crlf(self):
        cleaner, preserve = _prediction_cleaner(
            {"inference": {"preserve_internal_spaces": True}}
        )

        self.assertTrue(preserve)
        self.assertEqual(cleaner(" \talpha  beta\r\ngamma  "), "alpha  beta gamma")

    def test_legacy_kraken_cleaner_still_collapses_whitespace_by_default(self):
        cleaner, preserve = _prediction_cleaner({})

        self.assertFalse(preserve)
        self.assertEqual(cleaner(" \talpha  beta\r\ngamma  "), "alpha beta gamma")

    def test_canonical_folds_stay_complete_and_only_geometry_ids_are_present(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "data" / "processed" / "train_folds.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 4095)
        self.assertEqual(
            {fold: sum(row["fold"] == fold for row in rows) for fold in "01234"},
            {fold: 819 for fold in "01234"},
        )
        present = {
            row["ID"]
            for row in rows
            if row["ID"]
            in {
                "79tMUVyfIdy3GzkG",
                "rh8o7bdCGOIBFPwH",
                "F8DYDDp2AvW9Dytw",
                "EcxuqKeZl7OQexfB",
                "gP7kGWfDFYMbZ8nO",
                "PO7QQLWIFOT65BTz",
            }
        }
        self.assertEqual(
            present,
            {"EcxuqKeZl7OQexfB", "gP7kGWfDFYMbZ8nO", "PO7QQLWIFOT65BTz"},
        )


if __name__ == "__main__":
    unittest.main()
