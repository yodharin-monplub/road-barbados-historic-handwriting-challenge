from __future__ import annotations

import copy
import inspect
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from road_htr import qwen2_vl_starter_highres as highres
from road_htr.qwen2_vl_challenger import SafetyGateError
from road_htr.qwen2_vl_starter_highres_efficient import (
    MAX_SUPERVISED_TOKENS,
    MODEL43_SOURCE_SHA256,
    MODEL_NUMBER,
    PROJECTED_HIDDEN_TOKENS,
    VOCAB_SIZE,
    audit_installed_qwen_forward,
    audit_supervised_suffix,
    estimate_logit_materialization,
    load_config,
    prepare_trailing_labels,
    tail_projected_forward,
    train,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model51" / "config.yaml"


class ToyCausalLM(nn.Module):
    """Small stock-shifted causal head with the same relevant Qwen semantics."""

    def __init__(self, hidden_size: int = 7, vocabulary: int = 19) -> None:
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocabulary, bias=False, dtype=torch.float64)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, hidden_states, labels):
        logits = self.lm_head(hidden_states)
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)


def suffix_labels(sequence: int, supervised: int, vocabulary: int = 19) -> torch.Tensor:
    labels = torch.full((1, sequence), -100, dtype=torch.long)
    labels[0, -supervised:] = torch.arange(supervised) % vocabulary
    return labels


class Model51ExactLossTests(unittest.TestCase):
    def test_fixed_window_needs_maximum_supervision_plus_one_for_shift(self) -> None:
        self.assertEqual(MAX_SUPERVISED_TOKENS, 36)
        self.assertEqual(PROJECTED_HIDDEN_TOKENS, 37)
        labels = suffix_labels(80, 36)
        tail, receipt = prepare_trailing_labels(labels)
        self.assertEqual(tuple(tail.shape), (1, 37))
        self.assertEqual(tail[0, 0].item(), -100)
        self.assertEqual(receipt.supervised_tokens, 36)
        self.assertEqual(receipt.first_supervised_position, 44)
        self.assertTrue(torch.equal(tail[:, 1:], labels[:, -36:]))

    def test_tail_projection_matches_full_loss_logits_and_gradients(self) -> None:
        for supervised in (1, 8, 17, 36):
            with self.subTest(supervised=supervised):
                torch.manual_seed(5100 + supervised)
                full_model = ToyCausalLM()
                efficient_model = copy.deepcopy(full_model)
                full_hidden = torch.randn(
                    1, 80, 7, dtype=torch.float64, requires_grad=True
                )
                efficient_hidden = full_hidden.detach().clone().requires_grad_(True)
                labels = suffix_labels(80, supervised)

                full = full_model(full_hidden, labels)
                tail_labels, _ = prepare_trailing_labels(labels)
                efficient, receipt = tail_projected_forward(
                    efficient_model,
                    {"hidden_states": efficient_hidden, "labels": tail_labels},
                )
                self.assertEqual(receipt.original_hidden_tokens, 80)
                self.assertEqual(receipt.projected_hidden_tokens, 37)
                self.assertEqual(receipt.head_calls, 1)
                self.assertEqual(tuple(efficient.logits.shape), (1, 37, 19))
                torch.testing.assert_close(
                    efficient.logits,
                    full.logits[:, -37:, :],
                    rtol=1e-12,
                    atol=1e-14,
                )
                torch.testing.assert_close(efficient.loss, full.loss, rtol=1e-6, atol=1e-7)

                full.loss.backward()
                efficient.loss.backward()
                torch.testing.assert_close(
                    efficient_hidden.grad, full_hidden.grad, rtol=1e-6, atol=1e-7
                )
                torch.testing.assert_close(
                    efficient_model.lm_head.weight.grad,
                    full_model.lm_head.weight.grad,
                    rtol=1e-6,
                    atol=1e-7,
                )

    def test_hook_is_removed_and_normal_eval_uses_the_full_head(self) -> None:
        model = ToyCausalLM()
        hidden = torch.randn(1, 80, 7, dtype=torch.float64)
        labels = suffix_labels(80, 12)
        tail, _ = prepare_trailing_labels(labels)
        projected, _ = tail_projected_forward(
            model, {"hidden_states": hidden, "labels": tail}
        )
        self.assertEqual(projected.logits.shape[-2], 37)
        self.assertEqual(len(model.lm_head._forward_pre_hooks), 0)
        model.eval()
        ordinary = model(hidden, labels)
        self.assertEqual(ordinary.logits.shape[-2], 80)
        self.assertEqual(len(model.lm_head._forward_pre_hooks), 0)

    def test_invalid_label_layouts_fail_closed(self) -> None:
        with self.assertRaisesRegex(SafetyGateError, "batch size one"):
            prepare_trailing_labels(torch.full((2, 80), -100, dtype=torch.long))
        with self.assertRaisesRegex(SafetyGateError, "empty"):
            prepare_trailing_labels(torch.full((1, 80), -100, dtype=torch.long))
        with self.assertRaisesRegex(SafetyGateError, "cap is 36"):
            prepare_trailing_labels(suffix_labels(80, 37))
        noncontiguous = suffix_labels(80, 8)
        noncontiguous[0, -4] = -100
        with self.assertRaisesRegex(SafetyGateError, "contiguous terminal suffix"):
            prepare_trailing_labels(noncontiguous)
        prefix = suffix_labels(80, 8)
        prefix[0, 10] = 3
        with self.assertRaisesRegex(SafetyGateError, "contiguous terminal suffix"):
            prepare_trailing_labels(prefix)
        with self.assertRaisesRegex(SafetyGateError, "exactly 37 trailing labels"):
            tail_projected_forward(
                ToyCausalLM(),
                {
                    "hidden_states": torch.randn(1, 80, 7, dtype=torch.float64),
                    "labels": suffix_labels(80, 8),
                },
            )


class Model51RuntimeAndDatasetAudits(unittest.TestCase):
    def test_installed_qwen_has_no_native_logit_window(self) -> None:
        report = audit_installed_qwen_forward()
        self.assertFalse(report["supports_logits_to_keep"])
        self.assertFalse(report["supports_num_logits_to_keep"])
        parameters = report["forward_parameters"]
        self.assertNotIn("logits_to_keep", parameters)
        self.assertNotIn("num_logits_to_keep", parameters)
        self.assertIn("labels", parameters)
        self.assertIn("pixel_values", parameters)

    def test_all_real_labels_fit_the_proven_suffix(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        records = highres._training_records(config)
        report = audit_supervised_suffix(config, records)
        self.assertEqual(report["rows"], 4095)
        self.assertTrue(report["all_templates_have_exact_prompt_prefix"])
        self.assertEqual(report["supervised_tokens"]["minimum"], 8)
        self.assertEqual(report["supervised_tokens"]["maximum"], 36)
        self.assertAlmostEqual(
            report["supervised_tokens"]["mean"], 17.63833943833944
        )
        self.assertEqual(report["projected_hidden_tokens"], 37)

    def test_static_materialization_counts_are_exact_but_not_runtime_claims(self) -> None:
        report = estimate_logit_materialization(1115)
        self.assertEqual(report["vocabulary_size"], VOCAB_SIZE)
        self.assertEqual(report["stock_projected_positions"], 1115)
        self.assertEqual(report["efficient_projected_positions"], 37)
        self.assertEqual(report["efficient_shifted_loss_positions"], 36)
        self.assertEqual(
            report["stock_logits_bytes"]["float32"], 1115 * VOCAB_SIZE * 4
        )
        self.assertEqual(
            report["efficient_logits_bytes"]["float32"], 37 * VOCAB_SIZE * 4
        )
        self.assertGreater(report["position_reduction_fraction"], 0.96)
        self.assertIn("not a CUDA", report["scope_note"])


class Model51IsolationAndConfigurationTests(unittest.TestCase):
    def test_only_controlled_loss_projection_differs_from_model43(self) -> None:
        config, path = load_config(CONFIG_PATH)
        self.assertEqual(path, CONFIG_PATH)
        self.assertEqual(config["experiment"]["model_number"], MODEL_NUMBER)
        self.assertEqual(config["experiment"]["parent_model"], 43)
        self.assertEqual(config["image"]["min_effective_image_tokens"], 896)
        self.assertEqual(config["image"]["max_image_tokens"], 1024)
        self.assertEqual(config["lora"]["rank"], 16)
        self.assertEqual(config["architecture"]["expected_target_modules"], 196)
        self.assertTrue(config["training"]["gradient_checkpointing"])
        self.assertEqual(config["training"]["expected_total_updates"], 615)
        self.assertEqual(config["efficiency"]["maximum_supervised_tokens"], 36)
        self.assertEqual(config["efficiency"]["projected_hidden_tokens"], 37)
        self.assertTrue(config["efficiency"]["gradient_checkpointing_unchanged"])
        self.assertTrue(config["efficiency"]["full_determinism_unchanged"])

    def test_model43_parent_source_is_byte_identical_and_pinned(self) -> None:
        self.assertEqual(highres._sha256(Path(highres.__file__)), MODEL43_SOURCE_SHA256)

    def test_configuration_drift_fails_closed(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        for keys, value in (
            (("experiment", "model_number"), 43),
            (("efficiency", "maximum_supervised_tokens"), 35),
            (("efficiency", "projected_hidden_tokens"), 36),
            (("efficiency", "native_logits_to_keep"), True),
            (("efficiency", "gradient_checkpointing_unchanged"), False),
            (("training", "gradient_checkpointing"), False),
            (("image", "max_image_tokens"), 1023),
            (("lora", "rank"), 8),
            (("smoke", "max_reserved_gib"), 5.31),
        ):
            with self.subTest(keys=keys):
                changed = copy.deepcopy(config)
                changed[keys[0]][keys[1]] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)

    def test_training_is_isolated_and_uses_tail_smoke_and_loss(self) -> None:
        source = inspect.getsource(train)
        self.assertIn("smoke = run_memory_smoke(", source)
        self.assertIn("TailProjectionTrainerMixin", source)
        self.assertIn("gradient_checkpointing=True", source)
        self.assertIn("full_determinism=True", source)
        self.assertIn('"model_number": MODEL_NUMBER', source)
        self.assertNotIn("highres.train(", source)

    def test_wrappers_point_only_to_model51_module(self) -> None:
        for filename, entrypoint in (
            ("train.py", "train_main"),
            ("infer.py", "infer_main"),
            ("audit.py", "audit_main"),
        ):
            text = (ROOT / "model51" / filename).read_text(encoding="utf-8")
            self.assertIn("qwen2_vl_starter_highres_efficient", text)
            self.assertIn(entrypoint, text)
            self.assertNotIn("qwen2_vl_starter_highres import", text)

    def test_dry_run_receipt_when_present(self) -> None:
        path = ROOT / "outputs" / "model51" / "dry_run.json"
        if not path.exists():
            self.skipTest("dry run is created after the unit-level checks")
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "cpu_audits_passed_gpu_smoke_not_run")
        self.assertEqual(report["model_number"], 51)
        self.assertEqual(report["supervised_suffix"]["rows"], 4095)
        self.assertEqual(report["processor_sample"]["projected_label_tokens"], 37)
        self.assertFalse(report["qwen_forward"]["supports_logits_to_keep"])


if __name__ == "__main__":
    unittest.main()
