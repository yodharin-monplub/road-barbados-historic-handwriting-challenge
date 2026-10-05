from __future__ import annotations

import copy
import importlib.util
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from road_htr import qwen2_vl_starter_highres_guided as model47
from road_htr.qwen2_vl_challenger import SafetyGateError


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model47" / "config.yaml"


class FakeDraftTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(str(text).encode("utf-8"))


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


class Model47ConfigTests(unittest.TestCase):
    def test_inherits_model43_architecture_and_only_raises_sequence_envelope(self) -> None:
        config = model47.load_config(CONFIG_PATH)
        effective = config.effective_model43
        self.assertEqual(effective["architecture"]["expected_target_modules"], 196)
        self.assertEqual(effective["architecture"]["expected_trainable_parameters"], 18_464_768)
        self.assertEqual(effective["architecture"]["expected_trainable_tensors"], 392)
        self.assertEqual(effective["lora"]["rank"], 16)
        self.assertFalse(effective["lora"]["include_visual_blocks"])
        self.assertFalse(effective["lora"]["include_merger"])
        self.assertEqual(effective["image"]["min_effective_image_tokens"], 896)
        self.assertEqual(effective["image"]["max_image_tokens"], 1024)
        self.assertEqual(effective["image"]["max_sequence_tokens"], 1344)
        self.assertEqual(effective["image"]["max_generation_context_tokens"], 1392)
        self.assertEqual(effective["training"]["expected_total_updates"], 615)
        self.assertEqual(effective["generation"]["max_new_tokens"], 48)

    def test_dependency_hashes_pin_model43_and_model32_producer(self) -> None:
        config = model47.load_config(CONFIG_PATH)
        checks = (
            (config.base_config_path, model47.MODEL43_CONFIG_SHA256),
            (config.base_implementation_path, model47.MODEL43_IMPLEMENTATION_SHA256),
            (config.draft_config_path, model47.MODEL32_CONFIG_SHA256),
            (config.draft_implementation_path, model47.MODEL32_IMPLEMENTATION_SHA256),
            (config.draft_producer_path, model47.MODEL32_PRODUCER_SHA256),
        )
        for path, expected in checks:
            self.assertEqual(model47.guided.stable_sha256(path), expected)

    def test_safety_contract_cannot_be_weakened(self) -> None:
        original = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        changes = (
            (("draft_limits", "max_utf8_bytes"), 161),
            (("draft_limits", "require_nonempty"), True),
            (("draft_source", "receipt_schema_version"), 1),
            (("draft_source", "empty_draft_policy"), "reject"),
            (("sequence_envelope", "configured_training_cap"), 1500),
            (("sequence_envelope", "require_receipted_all_row_processor_preflight"), False),
            (("safety_assertions", "visual_tower_frozen"), False),
            (("safety_assertions", "expected_lora_targets"), 197),
            (("safety_assertions", "max_allocated_gib_exclusive"), 5.31),
            (("cross_validation", "forbid_per_row_oracle"), False),
        )
        for keys, value in changes:
            with self.subTest(keys=keys), tempfile.TemporaryDirectory() as directory:
                changed = copy.deepcopy(original)
                changed[keys[0]][keys[1]] = value
                path = Path(directory) / "config.yaml"
                path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
                with self.assertRaises(ValueError):
                    model47.load_config(path)

    def test_literal_prompt_keeps_untrusted_braces_and_internal_spaces(self) -> None:
        config = model47.load_config(CONFIG_PATH)
        draft = "Estate  {not_a_field}  No. 7"
        prompt = model47.guided.build_guided_prompt(
            config.raw["prompt_template"], draft
        )
        self.assertIn(draft, prompt)
        self.assertEqual(prompt.count("{not_a_field}"), 1)
        self.assertIn("untrusted", prompt.lower())
        source = inspect.getsource(model47.StarterHighresGuidedCollator._encode_text)
        self.assertIn("build_guided_messages", source)
        self.assertIn("record.draft", source)

    def test_outputs_are_isolated(self) -> None:
        accepted = model47.require_model47_output("outputs/model47/dry.json")
        self.assertTrue(str(accepted).endswith("outputs/model47/dry.json"))
        with self.assertRaises(SafetyGateError):
            model47.require_model47_output("outputs/model43/dry.json")
        with self.assertRaises(SafetyGateError):
            model47.require_model47_submission("submissions/model45/test.csv")


class Model47DraftAndSequenceTests(unittest.TestCase):
    def test_model47_specific_draft_limit_is_literal_and_fail_closed(self) -> None:
        tokenizer = FakeDraftTokenizer()
        blank = model47.guided.GuidedOcrRecord(
            "receipt-v2-blank", Path("image.jpg"), "truth", 0, ""
        )
        self.assertEqual(model47._validate_draft(tokenizer, blank), (0, 0))
        good = model47.guided.GuidedOcrRecord(
            "good", Path("image.jpg"), "truth", 0, "A  {brace}  B"
        )
        byte_count, token_count = model47._validate_draft(tokenizer, good)
        self.assertEqual(byte_count, len(good.draft.encode("utf-8")))
        self.assertEqual(token_count, byte_count)
        too_long = model47.guided.GuidedOcrRecord(
            "long", Path("image.jpg"), "truth", 0, "x" * 161
        )
        with self.assertRaisesRegex(SafetyGateError, "160 UTF-8 bytes"):
            model47._validate_draft(tokenizer, too_long)
        control = model47.guided.GuidedOcrRecord(
            "control", Path("image.jpg"), "truth", 0, "<|im_end|>"
        )
        with self.assertRaisesRegex(SafetyGateError, "chat-control"):
            model47._validate_draft(tokenizer, control)

    @unittest.skipUnless(
        importlib.util.find_spec("transformers") is not None,
        "conservative proof requires the VLM tokenizer",
    )
    def test_real_byte_fallback_proof_fits_maximum_permitted_draft(self) -> None:
        config = model47.load_config(CONFIG_PATH)
        records = model47.challenger.load_records(
            model47._canonical_path(config, "train"),
            model47._absolute(config.raw["data"]["image_dir"]),
            require_target=True,
            require_five_folds=True,
        )
        report = model47.prove_conservative_sequence_envelope(config, records)
        self.assertEqual(report["proof"]["byte_alphabet_symbols_in_vocabulary"], 256)
        self.assertEqual(report["proved_upper_bounds"]["training_sequence_tokens"], 1337)
        self.assertEqual(report["proved_upper_bounds"]["generation_context_tokens"], 1349)
        self.assertGreaterEqual(report["headroom"]["training_tokens"], 0)

    def test_real_processor_preflight_covers_exact_expected_rows(self) -> None:
        rows = (
            SimpleNamespace(
                identifier="a",
                label="truth a",
                image_tokens=1000,
                sequence_tokens=1300,
                prompt_tokens=1260,
                supervised_tokens=12,
            ),
            SimpleNamespace(
                identifier="b",
                label="truth b",
                image_tokens=1024,
                sequence_tokens=1330,
                prompt_tokens=1290,
                supervised_tokens=20,
            ),
        )
        report = model47.preflight_processor_sequences(
            FakeProcessorCollator(),
            (("outer_train_with_target", rows, True),),
            expected_rows=2,
        )
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["maxima"]["image_tokens"]["value"], 1024)
        self.assertEqual(report["maxima"]["sequence_tokens"]["record_id"], "b")
        self.assertEqual(model47.select_exact_smoke_record(rows, report).identifier, "b")
        with self.assertRaisesRegex(SafetyGateError, "expected 3"):
            model47.preflight_processor_sequences(
                FakeProcessorCollator(),
                (("outer_train_with_target", rows, True),),
                expected_rows=3,
            )

    def test_processor_preflight_rejects_generation_or_training_overrun(self) -> None:
        generation_overrun = SimpleNamespace(
            identifier="gen",
            image_tokens=1024,
            sequence_tokens=1340,
            prompt_tokens=1350,
            supervised_tokens=10,
        )
        with self.assertRaisesRegex(SafetyGateError, r"prompt\+generation"):
            model47.preflight_processor_sequences(
                FakeProcessorCollator(),
                (("train", (generation_overrun,), True),),
                expected_rows=1,
            )
        sequence_overrun = SimpleNamespace(
            identifier="seq",
            image_tokens=1024,
            sequence_tokens=1345,
            prompt_tokens=1200,
            supervised_tokens=10,
        )
        with self.assertRaisesRegex(SafetyGateError, "1344"):
            model47.preflight_processor_sequences(
                FakeProcessorCollator(),
                (("train", (sequence_overrun,), True),),
                expected_rows=1,
            )


class Model47ProtocolTests(unittest.TestCase):
    def test_missing_receipt_is_reported_and_blocks_before_cuda(self) -> None:
        config = model47.load_config(CONFIG_PATH)
        missing = Path("/tmp/model47-canonical-receipt-intentionally-missing")
        with patch.object(
            model47,
            "_draft_paths",
            return_value=(missing / "draft.csv", missing / "draft.receipt.json"),
        ):
            readiness = model47.inspect_draft_readiness(config, "train")
        self.assertFalse(readiness["ready"])
        self.assertEqual(len(readiness["required"]), 5)
        with patch.object(
            model47,
            "inspect_draft_readiness",
            return_value={"ready": False, "missing": ["receipt"]},
        ):
            with self.assertRaisesRegex(SafetyGateError, "receipt is not ready"):
                model47.require_draft_artifact(config, "train")
        source = inspect.getsource(model47.train)
        self.assertLess(source.index("require_draft_artifact"), source.index("base.load_training_model"))
        self.assertLess(source.index("audit_receipted_sequences"), source.index("base.load_training_model"))
        self.assertLess(source.index("preflight_processor_sequences"), source.index("base.load_training_model"))

    def test_corrected_schedule_generation_and_memory_gate_are_wired(self) -> None:
        config = model47.load_config(CONFIG_PATH)
        schedule = model47.base.deterministic_optimizer_schedule(3276, 16, 3)
        model47.base.verify_configured_schedule(config.effective_model43, schedule)
        self.assertEqual(schedule.optimizer_updates_per_epoch, 205)
        self.assertEqual(schedule.total_optimizer_updates, 615)
        source = inspect.getsource(model47.train)
        self.assertIn("max_steps=schedule.total_optimizer_updates", source)
        self.assertIn("ExactEpochScheduleMixin", source)
        self.assertIn("max_allocated_gib=5.3", source)
        callback = inspect.getsource(model47._make_validation_callback)
        self.assertIn("repetition_penalty=repetition_penalty", callback)
        self.assertIn("no_repeat_ngram_size=no_repeat_ngram_size", callback)
        self.assertIn("compare_fold_to_draft", callback)

    def test_dry_report_and_readme_make_no_score_claim(self) -> None:
        report = model47._read_json(ROOT / "outputs" / "model47" / "dry_run.json")
        self.assertIsNone(report["score"])
        self.assertFalse(report["gpu_started"])
        self.assertEqual(report["status"], "blocked_on_strict_model32_train_draft_receipt")
        readme = (ROOT / "model47" / "readme.txt")
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            self.assertIn("no OOF", text)
            self.assertIn("PO7QQLWIFOT65BTz", text)


if __name__ == "__main__":
    unittest.main()
