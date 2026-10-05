from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

from model30.train import prepare_run
from model30.weighted_manifest import (
    TrainingExample,
    WeightingPolicy,
    build_weighting_plan,
    canonical_mode,
    policy_from_config,
)


def _example(
    identifier: str,
    *,
    characters: int,
    words: int,
    ctc_length: int,
) -> TrainingExample:
    return TrainingExample(
        identifier=identifier,
        image_path=Path(f"/{identifier}.jpg"),
        target="x" * ctc_length,
        metric_char_length=characters,
        metric_word_length=words,
        ctc_target_length=ctc_length,
    )


class WeightingPlanTests(unittest.TestCase):
    def test_uniform_is_a_byte_order_identity_control(self) -> None:
        examples = tuple(
            _example(
                f"row{index}",
                characters=5 + index,
                words=1 + index,
                ctc_length=5 + index,
            )
            for index in range(4)
        )
        policy = policy_from_config(
            {"seed": 17, "weighting": {"repetition_budget": 9}}, "identity"
        )

        plan = build_weighting_plan(examples, policy, fold=0)

        self.assertEqual(canonical_mode("identity"), "uniform")
        self.assertEqual(policy.repetition_budget, 1.0)
        self.assertEqual(plan.repetitions, (1, 1, 1, 1))
        self.assertEqual(
            plan.expanded_paths, tuple(example.image_path for example in examples)
        )
        self.assertTrue(all(value == 1.0 for value in plan.clipped_importance))

    def test_metric_plan_is_clipped_complete_exact_budget_and_deterministic(self) -> None:
        examples = tuple(
            _example(
                f"row{index}",
                characters=value,
                words=max(1, value // 5),
                ctc_length=value + index,
            )
            for index, value in enumerate((1, 3, 8, 16, 32, 80))
        )
        policy = WeightingPolicy(
            mode="metric_clipped",
            clip_min=0.5,
            clip_max=2.0,
            repetition_budget=2.0,
            seed=29,
        )

        first = build_weighting_plan(examples, policy, fold=3)
        second = build_weighting_plan(examples, policy, fold=3)

        self.assertEqual(first, second)
        self.assertEqual(len(first.expanded_paths), 2 * len(examples))
        self.assertEqual(sum(first.repetitions), 2 * len(examples))
        self.assertGreaterEqual(min(first.repetitions), 1)
        self.assertGreaterEqual(min(first.clipped_importance), 0.5)
        self.assertLessEqual(max(first.clipped_importance), 2.0)
        self.assertTrue(math.isclose(math.fsum(first.clipped_importance), 6.0))

    def test_raw_importance_compensates_ctc_target_normalization(self) -> None:
        examples = (
            _example("short", characters=10, words=2, ctc_length=4),
            _example("long", characters=10, words=2, ctc_length=12),
        )
        policy = WeightingPolicy("metric_clipped", 0.1, 10.0, 10.0, 7)

        plan = build_weighting_plan(examples, policy, fold=1)

        self.assertAlmostEqual(plan.raw_importance[1] / plan.raw_importance[0], 3.0)

    def test_policy_rejects_budget_that_can_drop_a_training_row(self) -> None:
        with self.assertRaisesRegex(ValueError, "every training row remains present"):
            policy_from_config(
                {
                    "weighting": {
                        "mode": "metric_clipped",
                        "clip_min": 0.4,
                        "clip_max": 2.0,
                        "repetition_budget": 2.0,
                    }
                }
            )


class Model30PreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.rows = [
            {"ID": "held0", "Target": "held alpha", "fold": "0"},
            {"ID": "held1", "Target": "held beta", "fold": "0"},
            {"ID": "train0", "Target": "short", "fold": "1"},
            {"ID": "train1", "Target": "some longer words", "fold": "1"},
            {"ID": "train2", "Target": "another training line", "fold": "2"},
            {"ID": "train3", "Target": "tiny line", "fold": "2"},
        ]
        for row in self.rows:
            Image.new("RGB", (120, 32), "white").save(
                self.images / f"{row['ID']}.jpg"
            )
        self.train_csv = self.root / "train.csv"
        self._write_rows()
        self.exclusions = self.root / "exclude_ids.txt"
        self.exclusions.write_text("# none\n", encoding="utf-8")
        self.pretrained = self.root / "pretrained.safetensors"
        self.pretrained.write_bytes(b"fixed weights")
        self.config_path = self.root / "config.yaml"
        self.config = {
            "pretrained_model": str(self.pretrained),
            "train_csv": str(self.train_csv),
            "image_dir": str(self.images),
            "exclude_ids": str(self.exclusions),
            "prepared_dir": str(self.root / "prepared"),
            "output_dir": str(self.root / "unused" / "fold{fold}"),
            "validation_fold": 0,
            "seed": 23,
            "image_variant": "full",
            "runtime": {
                "device": "cpu",
                "precision": "32-true",
                "workers": 0,
                "threads": 1,
            },
            "training": {
                "strip_target_edges": True,
                "codec_resize": "new",
                "ctc_subsampling": 8,
                "height": 96,
                "max_width": 3584,
                "padding": 16,
                "batch_size": 1,
                "gradient_accumulation": 2,
                "epochs": 4,
                "cosine_epochs": 4,
                "min_epochs": 1,
                "stop": "fixed",
                "normalization": "NFD",
                "augment": False,
            },
            "weighting": {
                "mode": "metric_clipped",
                "clip_min": 0.5,
                "clip_max": 2.0,
                "repetition_budget": 2.0,
                "output_dir": str(self.root / "outputs" / "{mode}" / "fold{fold}"),
            },
        }
        self._write_config()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_rows(self) -> None:
        with self.train_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Target", "fold"])
            writer.writeheader()
            writer.writerows(self.rows)

    def _write_config(self) -> None:
        self.config_path.write_text(
            yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8"
        )

    def _prepare(self, mode: str = "metric_clipped"):
        with patch("model30.train._ketos_binary", return_value="ketos"):
            return prepare_run(self.config_path, fold_override=0, mode_override=mode)

    @staticmethod
    def _lines(path: Path) -> list[str]:
        return path.read_text(encoding="utf-8").splitlines()

    def test_weighting_never_duplicates_or_reorders_validation(self) -> None:
        manifest, command, output_dir = self._prepare()

        validation_path = Path(command[command.index("--evaluation-data") + 1])
        validation_ids = [Path(line).stem for line in self._lines(validation_path)]
        self.assertEqual(validation_ids, ["held0", "held1"])
        self.assertEqual(len(validation_ids), len(set(validation_ids)))
        self.assertEqual(manifest["preparation"]["validation_rows"], 2)
        self.assertEqual(manifest["preparation"]["unique_train_rows"], 4)
        self.assertEqual(manifest["preparation"]["train_rows"], 8)
        self.assertEqual(manifest["weighting"]["base_optimizer_steps_estimate"], 8)
        self.assertEqual(manifest["weighting"]["weighted_optimizer_steps_estimate"], 8)
        self.assertEqual(command[command.index("--epochs") + 1], "2")
        self.assertTrue((output_dir / "weighting_receipt.json").is_file())
        self.assertTrue((output_dir / "run_manifest.json").is_file())
        self.assertFalse((output_dir / "training_started.json").exists())

    def test_held_out_label_change_cannot_change_training_plan(self) -> None:
        first, first_command, _ = self._prepare()
        first_manifest = Path(
            first_command[first_command.index("--training-data") + 1]
        ).read_bytes()
        first_digest = first["weighting"]["training_only_sha256"]

        self.rows[0]["Target"] = "a completely different held label"
        self.rows[1]["Target"] = "changed validation text"
        self._write_rows()
        second, second_command, _ = self._prepare()
        second_manifest = Path(
            second_command[second_command.index("--training-data") + 1]
        ).read_bytes()

        self.assertEqual(first_digest, second["weighting"]["training_only_sha256"])
        self.assertEqual(first_manifest, second_manifest)

    def test_train_side_label_change_is_recorded(self) -> None:
        first, _, _ = self._prepare()
        self.rows[2]["Target"] = "a changed and much longer training label"
        self._write_rows()
        second, _, _ = self._prepare()

        self.assertNotEqual(
            first["weighting"]["training_only_sha256"],
            second["weighting"]["training_only_sha256"],
        )
        self.assertNotEqual(first["train_csv_sha256"], second["train_csv_sha256"])

    def test_uniform_mode_preserves_base_training_manifest(self) -> None:
        manifest, command, _ = self._prepare("uniform")

        base = Path(manifest["preparation"]["base_train_manifest"]).read_bytes()
        weighted = Path(command[command.index("--training-data") + 1]).read_bytes()
        self.assertEqual(base, weighted)
        self.assertEqual(manifest["preparation"]["unique_train_rows"], 4)
        self.assertEqual(manifest["preparation"]["train_rows"], 4)
        self.assertEqual(manifest["weighting"]["weighted_epochs"], 4)

    def test_invalid_competition_fold_is_rejected_before_preparation(self) -> None:
        with self.assertRaisesRegex(ValueError, "competition fold"):
            prepare_run(self.config_path, fold_override=5)


if __name__ == "__main__":
    unittest.main()
