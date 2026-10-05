from __future__ import annotations

import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from PIL import Image

from road_htr import qwen2_vl_visual_guided as model45
from road_htr.qwen2_vl_challenger import SafetyGateError


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model45" / "config.yaml"


class FakeTokenizer:
    image_pad_id = 99

    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        text_parts = []
        for message in messages:
            for item in message.get("content", []):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
        suffix = " generation" if add_generation_prompt else " answer"
        return " ".join(text_parts) + suffix

    @classmethod
    def encode(cls, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        # One explicit image placeholder plus a deterministic word proxy.
        return [cls.image_pad_id] + [1] * max(1, len(str(text).split()))

    @classmethod
    def convert_tokens_to_ids(cls, token):
        self_check = token == model45.visual.IMAGE_PAD_TOKEN
        assert self_check
        return cls.image_pad_id


class FakeProcessorCollator:
    def encode_training(self, record):
        return {}, {
            "image_tokens": record.image_tokens,
            "sequence_tokens": record.sequence_tokens,
            "prompt_tokens": record.prompt_tokens,
            "supervised_tokens": record.supervised_tokens,
        }

    def encode_prompt(self, record):
        return {}, {
            "image_tokens": record.image_tokens,
            "sequence_tokens": record.sequence_tokens,
        }


class Model45ConfigTests(unittest.TestCase):
    def test_inherits_finalized_model40_without_unproved_cap_raise(self) -> None:
        config = model45.load_config(CONFIG_PATH)
        effective = config.effective_model40
        self.assertEqual(effective["output_dir"], "outputs/model45/fold{fold}")
        self.assertEqual(effective["architecture"]["expected_target_modules"], 326)
        self.assertEqual(effective["architecture"]["expected_trainable_parameters"], 14_610_432)
        self.assertEqual(effective["lora"]["adapted_vision_blocks"], list(range(32)))
        self.assertEqual(effective["image"]["max_image_tokens"], 640)
        self.assertEqual(effective["image"]["max_sequence_tokens"], 768)
        self.assertEqual(effective["image"]["max_generation_context_tokens"], 816)
        self.assertFalse(config.raw["sequence_cap_decision"]["increased_from_model40"])
        self.assertEqual(effective["training"]["expected_total_updates"], 615)
        self.assertEqual(effective["generation"]["max_new_tokens"], 48)
        self.assertEqual(effective["generation"]["repetition_penalty"], 1.05)
        self.assertEqual(effective["generation"]["no_repeat_ngram_size"], 4)
        self.assertEqual(config.raw["draft_source"]["receipt_schema_version"], 2)
        self.assertEqual(
            config.raw["draft_source"]["empty_draft_policy"],
            model45.guided.EMPTY_DRAFT_POLICY,
        )

    def test_dependency_hash_pins_match_finalized_sources(self) -> None:
        config = model45.load_config(CONFIG_PATH)
        checks = (
            (config.base_config_path, model45.MODEL40_CONFIG_SHA256),
            (config.base_implementation_path, model45.MODEL40_IMPLEMENTATION_SHA256),
            (config.draft_config_path, model45.MODEL32_CONFIG_SHA256),
            (config.draft_implementation_path, model45.MODEL32_IMPLEMENTATION_SHA256),
        )
        for path, expected in checks:
            self.assertEqual(model45.guided.stable_sha256(path), expected)

    def test_context_target_and_oracle_safety_cannot_be_weakened(self) -> None:
        original = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        changes = (
            (("sequence_cap_decision", "configured_training_cap"), 896),
            (("sequence_cap_decision", "increased_from_model40"), True),
            (("safety_assertions", "expected_lora_targets"), 230),
            (("safety_assertions", "max_allocated_gib_exclusive"), 5.5),
            (("cross_validation", "forbid_per_row_oracle"), False),
            (("draft_source", "receipt_schema_version"), 1),
        )
        for keys, value in changes:
            with self.subTest(keys=keys), tempfile.TemporaryDirectory() as directory:
                changed = copy.deepcopy(original)
                changed[keys[0]][keys[1]] = value
                path = Path(directory) / "config.yaml"
                path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
                with self.assertRaises(ValueError):
                    model45.load_config(path)

    def test_collator_composes_model40_letterbox_with_guided_text(self) -> None:
        self.assertTrue(issubclass(model45.FullVisualGuidedCollator, model45.visual.FullVisualQwenCollator))
        source = inspect.getsource(model45.FullVisualGuidedCollator._encode_text)
        self.assertIn("build_guided_messages", source)
        self.assertIn("record.draft", source)
        self.assertIn("add_generation_prompt=not include_label", source)

    def test_outputs_are_isolated(self) -> None:
        accepted = model45.require_model45_output("outputs/model45/dry.json")
        self.assertTrue(str(accepted).endswith("outputs/model45/dry.json"))
        with self.assertRaises(SafetyGateError):
            model45.require_model45_output("outputs/model40/dry.json")
        with self.assertRaises(SafetyGateError):
            model45.require_model45_submission("submissions/model41/test.csv")


class Model45AuditTests(unittest.TestCase):
    def test_exact_letterbox_tokens_are_used_in_guided_static_audit(self) -> None:
        config = model45.load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "wide.jpg"
            Image.new("RGB", (1000, 100), "white").save(image)
            records = (
                model45.guided.GuidedOcrRecord("row-a", image, "Barbados", 0, "Barbadoes"),
                model45.guided.GuidedOcrRecord("row-b", image, "No. 7", 1, "No {7}"),
            )
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                report = model45.audit_guided_sequence_budget(
                    config, records, require_targets=True
                )
        self.assertEqual(report["rows"], 2)
        self.assertTrue(report["exact_geometry_per_row"])
        self.assertTrue(report["exact_pinned_tokenizer_per_row"])
        self.assertGreaterEqual(report["image_tokens"]["minimum"], 512)
        self.assertLessEqual(report["image_tokens"]["maximum"], 640)
        self.assertEqual(report["cap_increase_from_model40"]["training_tokens"], 0)

    def test_missing_receipt_is_a_dry_status_and_runtime_blocker(self) -> None:
        config = model45.load_config(CONFIG_PATH)
        missing = Path("/tmp/model45-intentionally-missing")
        with patch.object(
            model45,
            "_draft_paths",
            return_value=(missing / "draft.csv", missing / "draft.receipt.json"),
        ):
            readiness = model45.inspect_draft_readiness(config, "train")
        self.assertFalse(readiness["ready"])
        self.assertEqual(len(readiness["missing"]), 5)
        with patch.object(
            model45,
            "inspect_draft_readiness",
            return_value={"ready": False, "missing": ["receipt"]},
        ):
            with self.assertRaisesRegex(SafetyGateError, "receipt is not ready"):
                model45.require_draft_artifact(config, "train")

    def test_dry_audit_reports_exact_cap_shortfall_but_runtime_enforces(self) -> None:
        config = model45.load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "wide.jpg"
            Image.new("RGB", (1000, 100), "white").save(image)
            record = model45.guided.GuidedOcrRecord(
                "over-cap",
                image,
                "short label",
                0,
                " ".join(["draft"] * 1000),
            )
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                report = model45.audit_guided_sequence_budget(
                    config,
                    (record,),
                    require_targets=True,
                    enforce_configured_caps=False,
                )
            self.assertFalse(report["fits_configured_caps"])
            self.assertGreater(
                report["additional_tokens_proved_necessary"]["training_sequence_tokens"],
                0,
            )
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                with self.assertRaisesRegex(SafetyGateError, "proved configured caps insufficient"):
                    model45.audit_guided_sequence_budget(
                        config, (record,), require_targets=True
                    )

    def test_real_processor_preflight_and_exact_smoke_selection(self) -> None:
        train = (
            SimpleNamespace(identifier="a", label="a", image_tokens=620, sequence_tokens=730, prompt_tokens=680, supervised_tokens=10),
            SimpleNamespace(identifier="b", label="b", image_tokens=640, sequence_tokens=760, prompt_tokens=710, supervised_tokens=20),
        )
        validation = (
            SimpleNamespace(identifier="v", image_tokens=636, sequence_tokens=740),
        )
        report = model45.preflight_processor_sequences(
            FakeProcessorCollator(),
            (
                ("outer_train_with_target", train, True),
                ("outer_validation_prompt", validation, False),
            ),
        )
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["maxima"]["image_tokens"]["value"], 640)
        self.assertEqual(report["maxima"]["sequence_tokens"]["record_id"], "b")
        self.assertEqual(model45.select_exact_smoke_record(train, report).identifier, "b")

    def test_processor_preflight_rejects_context_overrun(self) -> None:
        row = SimpleNamespace(identifier="long", image_tokens=640, sequence_tokens=769)
        with self.assertRaises(SafetyGateError):
            model45.preflight_processor_sequences(
                FakeProcessorCollator(), (("test_prompt", (row,), False),)
            )


class Model45ProtocolTests(unittest.TestCase):
    def test_corrected_schedule_is_exact(self) -> None:
        config = model45.load_config(CONFIG_PATH)
        schedule = model45.visual.deterministic_optimizer_schedule(3276, 16, 3)
        model45.visual.verify_configured_schedule(config.effective_model40, schedule)
        self.assertEqual(schedule.remainder_micro_batches, 12)
        self.assertEqual(schedule.optimizer_updates_per_epoch, 205)
        self.assertEqual(schedule.total_optimizer_updates, 615)

    def test_training_gates_run_before_cuda_and_keep_memory_check(self) -> None:
        source = inspect.getsource(model45.train)
        self.assertLess(source.index("require_draft_artifact"), source.index("visual.load_training_model"))
        self.assertLess(source.index("audit_guided_sequence_budget"), source.index("visual.load_training_model"))
        self.assertLess(source.index("preflight_processor_sequences"), source.index("visual.load_training_model"))
        self.assertIn("max_steps=schedule.total_optimizer_updates", source)
        self.assertIn("ExactEpochScheduleMixin", source)
        self.assertIn("challenger.enforce_memory_limits", source)

    def test_generation_and_lineage_are_fully_bound(self) -> None:
        callback = inspect.getsource(model45._make_validation_callback)
        self.assertIn("repetition_penalty=repetition_penalty", callback)
        self.assertIn("no_repeat_ngram_size=no_repeat_ngram_size", callback)
        self.assertIn("compare_fold_to_draft", callback)
        lineage = inspect.getsource(model45.verify_adapter_lineage)
        self.assertIn("[205, 410, 615]", lineage)
        self.assertIn("14_610_432", lineage)
        self.assertIn("require_draft_artifact", lineage)

    def test_readme_states_no_score(self) -> None:
        text = (ROOT / "model45" / "readme.txt").read_text(encoding="utf-8")
        self.assertIn("Model45 has no OOF", text)
        self.assertIn("No CUDA process", text)


if __name__ == "__main__":
    unittest.main()
