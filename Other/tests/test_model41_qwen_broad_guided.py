from __future__ import annotations

import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from road_htr import qwen2_vl_broad_guided as model41
from road_htr.qwen2_vl_challenger import SafetyGateError
from road_htr.qwen2_vl_ocr_guided import build_guided_prompt


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model41" / "config.yaml"


class FakeCollator:
    def encode_training(self, record):
        return {}, {
            "image_tokens": record.image_tokens,
            "sequence_tokens": record.sequence_tokens,
            "supervised_tokens": record.supervised_tokens,
        }

    def encode_prompt(self, record):
        return {}, {
            "image_tokens": record.image_tokens,
            "sequence_tokens": record.sequence_tokens,
        }


class Model41ConfigTests(unittest.TestCase):
    def test_exact_inheritance_and_prompt_only_experiment(self) -> None:
        config = model41.load_config(CONFIG_PATH)
        effective = config.effective_model39
        self.assertEqual(effective["output_dir"], "outputs/model41/fold{fold}")
        self.assertEqual(effective["image"]["max_image_tokens"], 320)
        self.assertEqual(effective["image"]["max_sequence_tokens"], 512)
        self.assertEqual(effective["lora"]["rank"], 8)
        self.assertEqual(effective["architecture"]["expected_target_modules"], 230)
        self.assertEqual(effective["training"]["expected_updates_per_epoch"], 205)
        self.assertEqual(effective["training"]["expected_total_updates"], 615)
        self.assertEqual(effective["generation"]["max_new_tokens"], 48)
        self.assertEqual(effective["generation"]["repetition_penalty"], 1.05)
        self.assertEqual(effective["generation"]["no_repeat_ngram_size"], 4)
        self.assertIn("{draft}", config.raw["prompt_template"])
        self.assertEqual(config.raw["draft_source"]["receipt_schema_version"], 2)
        self.assertEqual(
            config.raw["draft_source"]["empty_draft_policy"],
            model41.guided.EMPTY_DRAFT_POLICY,
        )

    def test_all_dependency_hash_pins_match_disk(self) -> None:
        config = model41.load_config(CONFIG_PATH)
        self.assertEqual(
            model41.guided.stable_sha256(config.base_config_path),
            model41.MODEL39_CONFIG_SHA256,
        )
        self.assertEqual(
            model41.guided.stable_sha256(config.base_implementation_path),
            model41.MODEL39_IMPLEMENTATION_SHA256,
        )
        self.assertEqual(
            model41.guided.stable_sha256(config.draft_config_path),
            model41.MODEL32_CONFIG_SHA256,
        )
        self.assertEqual(
            model41.guided.stable_sha256(config.draft_implementation_path),
            model41.MODEL32_IMPLEMENTATION_SHA256,
        )

    def test_weakened_sequence_or_safety_contract_fails(self) -> None:
        original = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = (
            (("sequence_preflight", "max_image_tokens"), 319),
            (("safety_assertions", "expected_lora_targets"), 58),
            (("safety_assertions", "max_new_tokens"), 64),
            (("cross_validation", "forbid_per_row_oracle"), False),
            (("draft_source", "empty_draft_policy"), "drop_empty_rows"),
        )
        for keys, value in mutations:
            with self.subTest(keys=keys), tempfile.TemporaryDirectory() as directory:
                changed = copy.deepcopy(original)
                changed[keys[0]][keys[1]] = value
                path = Path(directory) / "config.yaml"
                path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
                with self.assertRaises(ValueError):
                    model41.load_config(path)

    def test_literal_draft_replacement_cannot_reformat_braces(self) -> None:
        config = model41.load_config(CONFIG_PATH)
        draft = "Estate {target} No. 7"
        prompt = build_guided_prompt(config.raw["prompt_template"], draft)
        self.assertIn(draft, prompt)
        self.assertEqual(prompt.count("Estate {target} No. 7"), 1)

    def test_outputs_are_isolated(self) -> None:
        accepted = model41.require_model41_output("outputs/model41/audit.json")
        self.assertTrue(str(accepted).endswith("outputs/model41/audit.json"))
        with self.assertRaises(SafetyGateError):
            model41.require_model41_output("outputs/model39/foreign.json")
        with self.assertRaises(SafetyGateError):
            model41.require_model41_submission("submissions/model32/foreign.csv")


class Model41ProtocolTests(unittest.TestCase):
    def test_missing_receipt_is_reported_but_runtime_fails_closed(self) -> None:
        config = model41.load_config(CONFIG_PATH)
        missing_root = Path("/tmp/model41-deliberately-missing")
        draft = missing_root / "raw.csv"
        receipt = missing_root / "raw.receipt.json"
        with patch.object(model41, "_draft_paths", return_value=(draft, receipt)):
            readiness = model41.inspect_draft_readiness(config, "train")
        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["status"], "missing_or_incomplete")
        self.assertEqual(len(readiness["missing"]), 5)
        with patch.object(
            model41,
            "inspect_draft_readiness",
            return_value={"ready": False, "missing": ["receipt"]},
        ):
            with self.assertRaisesRegex(SafetyGateError, "receipt is not ready"):
                model41.require_draft_artifact(config, "train")

    def test_ceil_schedule_is_inherited_exactly(self) -> None:
        config = model41.load_config(CONFIG_PATH)
        schedule = model41.broad.deterministic_optimizer_schedule(3276, 16, 3)
        model41.broad.verify_configured_schedule(config.effective_model39, schedule)
        self.assertEqual(schedule.remainder_micro_batches, 12)
        self.assertEqual(schedule.optimizer_updates_per_epoch, 205)
        self.assertEqual(schedule.total_optimizer_updates, 615)

    def test_exact_sequence_preflight_visits_every_selected_row(self) -> None:
        train = (
            SimpleNamespace(identifier="train-a", label="a", image_tokens=300, sequence_tokens=470, supervised_tokens=20),
            SimpleNamespace(identifier="train-b", label="b", image_tokens=320, sequence_tokens=500, supervised_tokens=36),
        )
        validation = (
            SimpleNamespace(identifier="valid-a", image_tokens=315, sequence_tokens=455),
        )
        report = model41.preflight_sequence_budgets(
            FakeCollator(),
            (
                ("outer_train_with_target", train, True),
                ("outer_validation_prompt", validation, False),
            ),
        )
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["groups"]["outer_train_with_target"]["rows"], 2)
        self.assertEqual(report["maxima"]["image_tokens"]["value"], 320)
        self.assertEqual(report["maxima"]["sequence_tokens"]["record_id"], "train-b")
        self.assertEqual(report["maxima"]["supervised_tokens"]["value"], 36)
        selected = model41.select_exact_smoke_record(train, report)
        self.assertEqual(selected.identifier, "train-b")

    def test_sequence_preflight_rejects_over_cap_row(self) -> None:
        record = SimpleNamespace(
            identifier="too-long",
            image_tokens=320,
            sequence_tokens=513,
            supervised_tokens=1,
        )
        with self.assertRaisesRegex(SafetyGateError, "512 sequence"):
            model41.preflight_sequence_budgets(
                FakeCollator(), (("outer_train_with_target", (record,), True),)
            )

    def test_training_orders_sequence_gate_before_cuda_model_load(self) -> None:
        source = inspect.getsource(model41.train)
        self.assertLess(
            source.index("preflight_sequence_budgets"),
            source.index("broad.load_training_model"),
        )
        self.assertIn("ExactEpochScheduleMixin", source)
        self.assertIn("max_steps=schedule.total_optimizer_updates", source)
        self.assertIn("challenger.enforce_memory_limits", source)

    def test_generation_and_lineage_gates_are_explicit(self) -> None:
        callback = inspect.getsource(model41._make_validation_callback)
        self.assertIn("repetition_penalty=repetition_penalty", callback)
        self.assertIn("no_repeat_ngram_size=no_repeat_ngram_size", callback)
        self.assertIn("compare_fold_to_draft", callback)
        lineage = inspect.getsource(model41.verify_adapter_lineage)
        self.assertIn("[205, 410, 615]", lineage)
        self.assertIn("EXPECTED_TARGET_MODULES", lineage)
        self.assertIn("require_draft_artifact", lineage)

    def test_numbered_model_has_readme(self) -> None:
        self.assertTrue((ROOT / "model41" / "readme.txt").is_file())


if __name__ == "__main__":
    unittest.main()
