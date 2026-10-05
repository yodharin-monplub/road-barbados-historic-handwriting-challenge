from __future__ import annotations

import copy
import inspect
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from road_htr import qwen2_vl_starter_highres as highres
from road_htr.qwen2_vl_starter_midres import (
    EXPECTED_TARGET_MODULES,
    EXPECTED_TRAINABLE_PARAMETERS,
    EXPECTED_TRAINABLE_TENSORS,
    GENERATION_CONTEXT_CAP,
    LANGUAGE_PROJECTIONS,
    MAX_IMAGE_TOKENS,
    MIN_IMAGE_TOKENS,
    MODEL_NUMBER,
    ORGANIZER_PROMPT,
    TARGET_IMAGE_TOKENS,
    TRAINING_SEQUENCE_CAP,
    deterministic_optimizer_schedule,
    expected_target_shapes,
    load_config,
    plan_letterbox,
    qwen_grid_tokens,
    split_outer_fold,
    train,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model48" / "config.yaml"


class Model48ConfigurationTests(unittest.TestCase):
    def test_frozen_midres_profile_is_the_only_change_from_model43(self) -> None:
        config, path = load_config(CONFIG_PATH)
        self.assertEqual(path, CONFIG_PATH)
        self.assertEqual(MODEL_NUMBER, 48)
        self.assertEqual(config["experiment"]["parent_model"], 43)
        self.assertEqual(config["prompt"], ORGANIZER_PROMPT)
        self.assertEqual(config["lora"]["rank"], 16)
        self.assertEqual(config["lora"]["alpha"], 32)
        self.assertEqual(
            config["lora"]["language_projections"], list(LANGUAGE_PROJECTIONS)
        )
        self.assertFalse(config["lora"]["include_merger"])
        self.assertFalse(config["lora"]["include_visual_blocks"])
        self.assertEqual(config["training"]["expected_total_updates"], 615)
        self.assertEqual(config["generation"]["max_new_tokens"], 48)
        self.assertEqual(config["generation"]["repetition_penalty"], 1.05)
        self.assertEqual(config["generation"]["no_repeat_ngram_size"], 4)

    def test_exact_visual_and_context_envelopes(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        image = config["image"]
        self.assertEqual(MIN_IMAGE_TOKENS, 512)
        self.assertEqual(TARGET_IMAGE_TOKENS, 640)
        self.assertEqual(MAX_IMAGE_TOKENS, 640)
        self.assertEqual(image["min_pixels"], 512 * 28 * 28)
        self.assertEqual(image["max_pixels"], 640 * 28 * 28)
        self.assertEqual(image["max_sequence_tokens"], TRAINING_SEQUENCE_CAP)
        self.assertEqual(
            image["max_generation_context_tokens"], GENERATION_CONTEXT_CAP
        )
        self.assertEqual(GENERATION_CONTEXT_CAP, TRAINING_SEQUENCE_CAP + 48)
        self.assertLess(MAX_IMAGE_TOKENS, 1024)
        self.assertGreater(MAX_IMAGE_TOKENS, 320)

    def test_all_protocol_drift_fails_closed(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        changes = (
            (("experiment", "model_number"), 43),
            (("image", "min_effective_image_tokens"), 511),
            (("image", "max_image_tokens"), 641),
            (("image", "max_sequence_tokens"), 769),
            (("lora", "rank"), 8),
            (("lora", "include_visual_blocks"), True),
            (("training", "expected_total_updates"), 612),
            (("training", "weight_decay"), 0.0),
            (("generation", "max_new_tokens"), 49),
            (("generation", "no_repeat_ngram_size"), 8),
            (("smoke", "max_reserved_gib"), 5.31),
        )
        for keys, value in changes:
            with self.subTest(keys=keys):
                changed = copy.deepcopy(config)
                changed[keys[0]][keys[1]] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)
        changed = copy.deepcopy(config)
        changed["prompt"] = changed["prompt"] + " Read all lines."
        with self.assertRaises(ValueError):
            validate_config(changed)


class Model48ArchitectureAndScheduleTests(unittest.TestCase):
    def test_architecture_is_the_exact_model43_language_ledger(self) -> None:
        shapes = expected_target_shapes()
        self.assertIs(expected_target_shapes, highres.expected_target_shapes)
        self.assertEqual(len(shapes), EXPECTED_TARGET_MODULES)
        self.assertEqual(EXPECTED_TARGET_MODULES, 196)
        self.assertEqual(EXPECTED_TRAINABLE_PARAMETERS, 18_464_768)
        self.assertEqual(EXPECTED_TRAINABLE_TENSORS, 392)
        self.assertFalse(any(name.startswith("visual.") for name in shapes))

    def test_exact_ceil_schedule_keeps_every_row(self) -> None:
        schedule = deterministic_optimizer_schedule(3276, 16, 3)
        self.assertEqual(schedule.remainder_micro_batches, 12)
        self.assertEqual(schedule.optimizer_updates_per_epoch, 205)
        self.assertEqual(schedule.total_optimizer_updates, 615)
        self.assertEqual(schedule.total_row_presentations, 9828)

    def test_outer_fold_split_is_disjoint_and_complete(self) -> None:
        records = tuple(
            SimpleNamespace(identifier=f"id-{fold}-{row}", fold=fold)
            for fold in range(5)
            for row in range(2)
        )
        training, validation = split_outer_fold(records, 0)
        self.assertEqual(len(training), 8)
        self.assertEqual(len(validation), 2)
        self.assertFalse(
            {record.identifier for record in training}
            & {record.identifier for record in validation}
        )


class Model48GeometryAndWiringTests(unittest.TestCase):
    def test_factor_aligned_complete_image_grid_is_midres_bounded(self) -> None:
        for width, height in ((1600, 300), (300, 1600), (997, 997), (2227, 413)):
            with self.subTest(size=(width, height)):
                plan = plan_letterbox(
                    width,
                    height,
                    minimum_tokens=MIN_IMAGE_TOKENS,
                    maximum_tokens=MAX_IMAGE_TOKENS,
                )
                self.assertEqual(plan.canvas_width % 28, 0)
                self.assertEqual(plan.canvas_height % 28, 0)
                self.assertGreaterEqual(plan.planned_image_tokens, 512)
                self.assertLessEqual(plan.planned_image_tokens, 640)
                self.assertEqual(
                    qwen_grid_tokens(plan.canvas_width, plan.canvas_height),
                    plan.planned_image_tokens,
                )
                self.assertLessEqual(plan.relative_content_aspect_error, 0.005)

    def test_training_writes_model48_manifest_without_delegating_to_train43(self) -> None:
        source = inspect.getsource(train)
        self.assertIn('"model_number": MODEL_NUMBER', source)
        self.assertIn("max_steps=schedule.total_optimizer_updates", source)
        self.assertIn("dataloader_drop_last=False", source)
        self.assertIn("highres.load_training_model", source)
        self.assertNotIn("highres.train(", source)

    def test_wrappers_use_the_isolated_midres_entrypoints(self) -> None:
        for filename, entrypoint in (
            ("train.py", "train_main"),
            ("infer.py", "infer_main"),
            ("audit.py", "audit_main"),
        ):
            text = (ROOT / "model48" / filename).read_text(encoding="utf-8")
            self.assertIn("qwen2_vl_starter_midres", text)
            self.assertIn(entrypoint, text)
            self.assertNotIn("qwen2_vl_starter_highres import", text)

    def test_full_cpu_audit_receipts_actual_geometry_and_worst_sequence(self) -> None:
        report = json.loads(
            (ROOT / "outputs" / "model48" / "dry_run.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["status"], "cpu_audits_passed_gpu_smoke_not_run")
        self.assertEqual(report["model_number"], 48)
        self.assertEqual(
            report["geometry"]["effective_tokens"],
            {"minimum": 520, "median": 636, "maximum": 640},
        )
        self.assertEqual(report["sequence"]["observed_max_training_tokens"], 731)
        self.assertEqual(report["processor_sample"]["sequence_tokens"], 728)
        self.assertEqual(report["processor_sample"]["processor_image_tokens"], 640)
        self.assertLessEqual(
            report["sequence"]["generation_upper_bound"],
            report["sequence"]["generation_context_cap"],
        )


if __name__ == "__main__":
    unittest.main()
