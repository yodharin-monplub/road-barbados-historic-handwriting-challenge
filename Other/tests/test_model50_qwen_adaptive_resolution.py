from __future__ import annotations

import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from road_htr.qwen2_vl_adaptive_resolution import (
    HARD_COMPACT_ASPECT_MAX,
    HARD_ELONGATED_ASPECT_MIN,
    HIGH_MAX_TOKENS,
    HIGH_MIN_TOKENS,
    LOW_MAX_TOKENS,
    LOW_MIN_TOKENS,
    NATIVE_LONG_SIDE_THRESHOLD_PX,
    POLICY_NAME,
    audit_geometry,
    load_adaptive_letterboxed_rgb,
    load_config,
    plan_adaptive_letterbox,
    qwen_grid_tokens,
    select_resolution_tier,
    train,
    validate_config,
)
from road_htr.qwen2_vl_starter_highres import (
    EXPECTED_TARGET_MODULES,
    EXPECTED_TRAINABLE_PARAMETERS,
    LANGUAGE_PROJECTIONS,
    OptimizerSchedule,
    deterministic_optimizer_schedule,
    split_outer_fold,
    verify_configured_schedule,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model50" / "config.yaml"


class Model50TierTests(unittest.TestCase):
    def test_selector_signature_cannot_receive_row_or_model_information(self) -> None:
        parameters = set(inspect.signature(select_resolution_tier).parameters)
        self.assertEqual(
            parameters,
            {
                "width",
                "height",
                "native_long_side_threshold_px",
                "hard_compact_aspect_max",
                "hard_elongated_aspect_min",
                "low_min_tokens",
                "low_max_tokens",
                "high_min_tokens",
                "high_max_tokens",
            },
        )
        self.assertFalse(
            parameters
            & {"label", "identifier", "fold", "prediction", "confidence", "split"}
        )

    def test_preregistered_geometry_boundaries_are_exact(self) -> None:
        ordinary = select_resolution_tier(1200, 80)
        self.assertEqual((ordinary.tier, ordinary.reason), ("low", "ordinary_resized_low"))
        self.assertEqual((ordinary.minimum_tokens, ordinary.maximum_tokens), (512, 640))

        native = select_resolution_tier(1600, 100)
        self.assertEqual((native.tier, native.reason), ("high", "native_high_resolution"))
        compact = select_resolution_tier(600, 100)
        self.assertEqual((compact.tier, compact.reason), ("high", "hard_compact_geometry"))
        elongated = select_resolution_tier(1000, 40)
        self.assertEqual((elongated.tier, elongated.reason), ("high", "hard_elongated_geometry"))
        both = select_resolution_tier(2500, 100)
        self.assertEqual((both.tier, both.reason), ("high", "native_and_hard_geometry"))

    def test_selector_rejects_invalid_geometry_or_budget(self) -> None:
        for size in ((0, 10), (10, -1), (True, 10), (10.5, 10)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                select_resolution_tier(*size)
        with self.assertRaises(ValueError):
            select_resolution_tier(100, 10, low_max_tokens=900)
        with self.assertRaises(ValueError):
            select_resolution_tier(100, 10, hard_compact_aspect_max=30)

    def test_tier_plans_are_factor_aligned_full_image_letterboxes(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        cases = (
            ((1100, 70), "low", LOW_MIN_TOKENS, LOW_MAX_TOKENS),
            ((3200, 220), "high", HIGH_MIN_TOKENS, HIGH_MAX_TOKENS),
            ((1000, 35), "high", HIGH_MIN_TOKENS, HIGH_MAX_TOKENS),
        )
        for (width, height), expected_tier, lower, upper in cases:
            with self.subTest(size=(width, height)):
                tier, plan = plan_adaptive_letterbox(width, height, config)
                self.assertEqual(tier.tier, expected_tier)
                self.assertEqual(plan.canvas_width % 28, 0)
                self.assertEqual(plan.canvas_height % 28, 0)
                self.assertGreaterEqual(plan.planned_image_tokens, lower)
                self.assertLessEqual(plan.planned_image_tokens, upper)
                self.assertEqual(
                    qwen_grid_tokens(plan.canvas_width, plan.canvas_height),
                    plan.planned_image_tokens,
                )
                self.assertLessEqual(plan.relative_content_aspect_error, 0.005)

    def test_image_loader_preserves_geometry_and_emits_tier_receipt(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "line.jpg"
            Image.new("RGB", (1100, 70), (240, 238, 230)).save(path)
            image, receipt = load_adaptive_letterboxed_rgb(path, config)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(receipt["policy"], POLICY_NAME)
        self.assertEqual(receipt["tier"], "low")
        self.assertEqual(image.size, (receipt["canvas_width"], receipt["canvas_height"]))
        self.assertEqual(qwen_grid_tokens(*image.size), receipt["planned_image_tokens"])
        self.assertLessEqual(receipt["relative_content_aspect_error"], 0.005)


class Model50ProtocolTests(unittest.TestCase):
    def test_config_freezes_architecture_prompt_generation_and_policy(self) -> None:
        config, path = load_config(CONFIG_PATH)
        self.assertEqual(path, CONFIG_PATH)
        self.assertEqual(config["image"]["selection_policy"], POLICY_NAME)
        self.assertEqual(config["test_csv"], "data/raw/Test.csv")
        self.assertEqual(config["image"]["native_long_side_threshold_px"], 1600)
        self.assertEqual(config["image"]["hard_compact_aspect_max"], 6.0)
        self.assertEqual(config["image"]["hard_elongated_aspect_min"], 25.0)
        self.assertEqual(config["architecture"]["expected_target_modules"], 196)
        self.assertEqual(config["architecture"]["expected_trainable_parameters"], 18_464_768)
        self.assertEqual(config["lora"]["rank"], 16)
        self.assertEqual(config["lora"]["language_projections"], list(LANGUAGE_PROJECTIONS))
        self.assertFalse(config["lora"]["include_merger"])
        self.assertFalse(config["lora"]["include_visual_blocks"])
        self.assertEqual(config["generation"]["max_new_tokens"], 48)
        self.assertIn("Transcribe EXACTLY as seen", config["prompt"])

    def test_contract_fails_closed_on_resolution_architecture_or_schedule_drift(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        changes = (
            (("image", "native_long_side_threshold_px"), 1599),
            (("image", "hard_compact_aspect_max"), 5.9),
            (("image", "low_max_image_tokens"), 641),
            (("image", "high_min_effective_image_tokens"), 895),
            (("image", "selection_policy"), "tuned_after_oof"),
            (("lora", "rank"), 8),
            (("lora", "include_visual_blocks"), True),
            (("training", "expected_total_updates"), 612),
            (("generation", "max_new_tokens"), 64),
        )
        for keys, value in changes:
            with self.subTest(keys=keys):
                changed = copy.deepcopy(config)
                changed[keys[0]][keys[1]] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)

    def test_exact_fold_and_optimizer_gate_matches_model43(self) -> None:
        schedule = deterministic_optimizer_schedule(3276, 16, 3)
        self.assertEqual(
            schedule,
            OptimizerSchedule(
                train_rows=3276,
                gradient_accumulation=16,
                epochs=3,
                full_accumulation_groups=204,
                remainder_micro_batches=12,
                optimizer_updates_per_epoch=205,
                total_optimizer_updates=615,
                transformer_floor_updates_per_epoch=204,
                transformer_floor_total_updates=612,
                total_row_presentations=9828,
            ),
        )
        config, _ = load_config(CONFIG_PATH)
        verify_configured_schedule(config, schedule)
        records = tuple(
            SimpleNamespace(identifier=f"id-{fold}-{row}", fold=fold)
            for fold in range(5)
            for row in range(2)
        )
        training, validation = split_outer_fold(records, 0)
        self.assertEqual((len(training), len(validation)), (8, 2))
        self.assertFalse({row.identifier for row in training} & {row.identifier for row in validation})

    def test_training_wires_adaptive_collator_without_post_inference_routing(self) -> None:
        source = inspect.getsource(train)
        self.assertIn("_collator(config, processor)", source)
        self.assertIn("max_steps=schedule.total_optimizer_updates", source)
        self.assertIn("dataloader_drop_last=False", source)
        self.assertIn("validation_records", source)
        self.assertNotIn("selector", source.casefold())
        self.assertNotIn("confidence", source.casefold())
        self.assertEqual(EXPECTED_TARGET_MODULES, 196)
        self.assertEqual(EXPECTED_TRAINABLE_PARAMETERS, 18_464_768)

    def test_real_geometry_audit_is_complete_and_saves_tokens(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        report = audit_geometry(config["image_dir"], config)
        self.assertEqual(report["images"], 5472)
        self.assertEqual(
            report["groups"]["high"]["rows"] + report["groups"]["low"]["rows"],
            5472,
        )
        self.assertEqual(report["groups"]["native_high_resolution_rows"], 1378)
        self.assertGreater(report["groups"]["high"]["proportion"], 0.25)
        self.assertLess(report["groups"]["high"]["proportion"], 0.35)
        self.assertGreater(
            report["estimated_token_compute_reduction"]["visual_token_linear_percent"],
            20.0,
        )
        self.assertLessEqual(report["maximum_content_aspect_error"], 0.005)
        self.assertEqual(
            report["selection_inputs"], ["source_width_px", "source_height_px"]
        )

    def test_policy_constants_are_preregistered_values(self) -> None:
        self.assertEqual(NATIVE_LONG_SIDE_THRESHOLD_PX, 1600)
        self.assertEqual(HARD_COMPACT_ASPECT_MAX, 6.0)
        self.assertEqual(HARD_ELONGATED_ASPECT_MIN, 25.0)
        self.assertEqual((LOW_MIN_TOKENS, LOW_MAX_TOKENS), (512, 640))
        self.assertEqual((HIGH_MIN_TOKENS, HIGH_MAX_TOKENS), (896, 1024))


if __name__ == "__main__":
    unittest.main()
