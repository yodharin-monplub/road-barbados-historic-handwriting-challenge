from __future__ import annotations

import copy
import csv
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from road_htr.byt5_corrector import (
    EXPECTED_ATTENTION_TARGETS,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    GatePolicy,
    Model31Error,
    OcrPair,
    _load_neural_runtime,
    _prepare_run_directory,
    adapter_files,
    apply_gate,
    build_fold_plan,
    build_synthetic_corruptions,
    canonical_sha256,
    discover_lora_targets,
    enable_offline_mode,
    evaluate_crossfit,
    fit_gate,
    load_config,
    load_oof_pairs,
    reject_model22_destination,
    sha256_file,
    validate_byte_limits,
    validate_config,
    verify_adapter_receipt,
    verify_local_snapshot,
    write_adapter_receipt,
)
from road_htr.byt5_corrector import _torch_device
from road_htr.metrics import competition_error
from road_htr.schema import SchemaError, read_table


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model31" / "config.yaml"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class FakeModel:
    def named_modules(self):
        yield "", object()
        for block in range(12):
            for projection in ("q", "v"):
                yield f"encoder.block.{block}.layer.0.SelfAttention.{projection}", object()
        for block in range(4):
            for layer, attention in ((0, "SelfAttention"), (1, "EncDecAttention")):
                for projection in ("q", "v"):
                    yield (
                        f"decoder.block.{block}.layer.{layer}.{attention}.{projection}",
                        object(),
                    )


class Model31ConfigTests(unittest.TestCase):
    def test_torch_device_initializes_the_configured_cuda_allocator(self) -> None:
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 2
        fake_torch.device.return_value = "device-object"
        config = {"runtime": {"device": "cuda:1"}}

        device, index = _torch_device(config, fake_torch)

        self.assertEqual((device, index), ("device-object", 1))
        fake_torch.cuda.set_device.assert_called_once_with(1)
        fake_torch.device.assert_called_once_with("cuda:1")

    def test_config_pins_offline_safe_model_and_bounded_lora(self) -> None:
        config, resolved = load_config(CONFIG_PATH)
        self.assertEqual(resolved, CONFIG_PATH)
        self.assertEqual(config["model"]["id"], MODEL_ID)
        self.assertEqual(config["model"]["revision"], MODEL_REVISION)
        self.assertEqual(config["model"]["license"], MODEL_LICENSE)
        self.assertTrue(config["model"]["local_files_only"])
        self.assertFalse(config["model"]["trust_remote_code"])
        self.assertEqual(config["sequence"]["maximum_utf8_bytes"], 120)
        self.assertEqual(config["lora"]["expected_target_modules"], 40)
        self.assertFalse(config["synthetic"]["enabled"])

    def test_safety_critical_config_drift_fails_closed(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        mutations = (
            (("model", "revision"), "main"),
            (("model", "local_files_only"), False),
            (("model", "trust_remote_code"), True),
            (("sequence", "maximum_utf8_bytes"), 256),
            (("sequence", "training_overlength"), "truncate"),
            (("lora", "rank"), 16),
            (("lora", "expected_target_modules"), 39),
            (("training", "gradient_checkpointing"), False),
            (("generation", "do_sample"), True),
        )
        for keys, value in mutations:
            with self.subTest(keys=keys):
                changed = copy.deepcopy(config)
                changed[keys[0]][keys[1]] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)

    def test_loader_source_has_no_network_or_automatic_device_map(self) -> None:
        source = inspect.getsource(_load_neural_runtime)
        self.assertIn("local_files_only=True", source)
        self.assertIn("trust_remote_code=False", source)
        self.assertIn("use_safetensors=True", source)
        self.assertNotIn('device_map="auto"', source)
        enable_offline_mode()
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_exact_attention_target_discovery(self) -> None:
        targets = discover_lora_targets(FakeModel())
        self.assertEqual(len(targets), EXPECTED_ATTENTION_TARGETS)
        broken = FakeModel()
        modules = list(broken.named_modules())[:-1]
        broken.named_modules = lambda: iter(modules)  # type: ignore[method-assign]
        with self.assertRaisesRegex(Model31Error, "architecture drift"):
            discover_lora_targets(broken)


class Model31LeakageTests(unittest.TestCase):
    @staticmethod
    def records() -> tuple[OcrPair, ...]:
        return tuple(
            OcrPair(
                identifier=f"id{fold}_{index}",
                source="cot" if index == 0 else "house",
                target="cat" if index == 0 else "house",
                fold=fold,
            )
            for fold in range(5)
            for index in range(2)
        )

    def test_outer_target_mutation_cannot_change_training_or_inner_selection(self) -> None:
        records = self.records()
        mutated = tuple(
            OcrPair(row.identifier, row.source, "PRIVATE OUTER LABEL", row.fold)
            if row.fold == 0
            else row
            for row in records
        )
        plan_a = build_fold_plan(records, 0)
        plan_b = build_fold_plan(mutated, 0)
        self.assertEqual(plan_a.selection_train, plan_b.selection_train)
        self.assertEqual(plan_a.selection_validation, plan_b.selection_validation)
        self.assertEqual(plan_a.refit_train, plan_b.refit_train)
        self.assertNotEqual(plan_a.outer_validation, plan_b.outer_validation)
        self.assertEqual({row.fold for row in plan_a.selection_train}, {2, 3, 4})
        self.assertEqual({row.fold for row in plan_a.selection_validation}, {1})
        self.assertEqual({row.fold for row in plan_a.refit_train}, {1, 2, 3, 4})

    def test_synthetic_corruptions_are_deterministic_and_train_side_only(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        config = copy.deepcopy(config)
        config["synthetic"]["enabled"] = True
        plan = build_fold_plan(self.records(), 0)
        augmented_a, report_a = build_synthetic_corruptions(
            plan.selection_train, config, seed=123
        )
        augmented_b, report_b = build_synthetic_corruptions(
            plan.selection_train, config, seed=123
        )
        self.assertEqual(augmented_a, augmented_b)
        self.assertEqual(report_a, report_b)
        self.assertGreater(report_a["synthetic_rows"], 0)
        self.assertTrue(all(row.fold in {2, 3, 4} for row in augmented_a))
        self.assertTrue(
            all(
                row.identifier.startswith(tuple(item.identifier for item in plan.selection_train))
                for row in augmented_a
            )
        )

    def test_gate_is_global_inner_fit_with_identity_tie_break(self) -> None:
        sources = ["correct", "hous"]
        references = ["correct", "house"]
        generated = ["correct", "house"]
        policy, report = fit_gate(sources, references, generated, [0.1, 1.0])
        self.assertTrue(policy.enabled)
        self.assertGreater(report["selected_gain"], 0)
        selected, _audit = apply_gate(sources, generated, policy)
        self.assertEqual(selected, references)

        identity, tied = fit_gate(sources, sources, generated, [1.0])
        self.assertFalse(identity.enabled)
        self.assertEqual(tied["selected_policy"], {"enabled": False, "maximum_edit_ratio": None})

    def test_gate_never_emits_empty_or_overlength_neural_text(self) -> None:
        sources = ["keep one", "keep two", "keep three"]
        generated = ["", "x" * 121, "kept three"]
        selected, audit = apply_gate(sources, generated, GatePolicy(True, 1.0))
        self.assertEqual(selected[:2], sources[:2])
        self.assertEqual(audit["empty_fallbacks"], 1)
        self.assertEqual(audit["overlength_fallbacks"], 1)

    def test_training_byte_limit_never_truncates(self) -> None:
        records = list(self.records())
        self.assertEqual(
            validate_byte_limits(records, 120, require_targets=True)["rows"], 10
        )
        records[0] = OcrPair("too-long", "x" * 121, "safe", 0)
        with self.assertRaisesRegex(Model31Error, "truncation is forbidden"):
            validate_byte_limits(records, 120, require_targets=True)


class Model31DataAndArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_oof_schema_and_fold_set_are_enforced(self) -> None:
        truth = self.root / "truth.csv"
        source = self.root / "source.csv"
        write_csv(
            truth,
            ["ID", "Target", "fold"],
            [
                {"ID": f"id{fold}", "Target": f"word {fold}", "fold": fold}
                for fold in range(5)
            ],
        )
        write_csv(
            source,
            ["ID", "prediction"],
            [{"ID": f"id{fold}", "prediction": f"wrod {fold}"} for fold in range(5)],
        )
        self.assertEqual(len(load_oof_pairs(truth, source)), 5)
        with source.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))[:-1]
        write_csv(source, ["ID", "prediction"], rows)
        with self.assertRaisesRegex(SchemaError, "ID mismatch"):
            load_oof_pairs(truth, source)

    def test_local_snapshot_architecture_hash_and_weight_size_are_checked(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        config = copy.deepcopy(config)
        model_dir = self.root / "base"
        model_dir.mkdir()
        architecture = config["model"]["expected_architecture"]
        write_json(model_dir / "config.json", architecture)
        for name in (
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ):
            write_json(model_dir / name, {})
        (model_dir / "model.safetensors").write_bytes(b"safe")
        digest = sha256_file(model_dir / "model.safetensors")
        config["model"]["local_path"] = str(model_dir)
        config["model"]["expected_files_sha256"] = {"model.safetensors": digest}
        with mock.patch("road_htr.byt5_corrector.MODEL_WEIGHT_BYTES", 4), mock.patch(
            "road_htr.byt5_corrector.MODEL_WEIGHT_SHA256", digest
        ):
            receipt = verify_local_snapshot(config, self.root)
            self.assertEqual(receipt["files_sha256"]["model.safetensors"], digest)
            (model_dir / "model.safetensors").write_bytes(b"evil")
            with self.assertRaisesRegex(Model31Error, "hash mismatch"):
                verify_local_snapshot(config, self.root)

    def test_adapter_receipt_detects_any_byte_change(self) -> None:
        adapter = self.root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        write_json(adapter / "adapter_config.json", {"r": 8})
        receipt = write_adapter_receipt(
            adapter,
            run_lineage_sha256="run",
            stage="refit",
            epoch=2,
            base_snapshot_sha256="base",
        )
        self.assertEqual(
            verify_adapter_receipt(
                adapter,
                receipt,
                expected_base_snapshot_sha256="base",
                expected_run_lineage_sha256="run",
            )["epoch"],
            2,
        )
        (adapter / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(Model31Error, "differ"):
            verify_adapter_receipt(adapter, receipt)

    def test_resume_requires_exact_lineage_and_refuses_model22(self) -> None:
        output = self.root / "run"
        lineage = {"source": "abc", "fold": 0}
        _manifest, digest = _prepare_run_directory(output, lineage, resume=False)
        self.assertEqual(digest, canonical_sha256(lineage))
        _prepare_run_directory(output, lineage, resume=True)
        with self.assertRaisesRegex(Model31Error, "lineage"):
            _prepare_run_directory(output, {"source": "changed"}, resume=True)
        with self.assertRaisesRegex(Model31Error, "Model22"):
            reject_model22_destination(ROOT / "outputs" / "model22" / "forbidden")

    def test_five_fold_evaluator_recomputes_exact_identity_and_corrected_scores(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        config_path = self.root / "config.yaml"
        config_path.write_bytes(CONFIG_PATH.read_bytes())
        truth = self.root / "truth.csv"
        source = self.root / "source.csv"
        truth_rows = [
            {"ID": f"id{fold}", "Target": f"word {fold}", "fold": fold}
            for fold in range(5)
        ]
        source_rows = [
            {"ID": f"id{fold}", "prediction": f"wrod {fold}"}
            for fold in range(5)
        ]
        write_csv(truth, ["ID", "Target", "fold"], truth_rows)
        write_csv(source, ["ID", "prediction"], source_rows)
        runs: dict[int, Path] = {}
        for fold in range(5):
            run = self.root / f"fold{fold}"
            run.mkdir()
            prediction = run / "outer_predictions.csv"
            write_csv(
                prediction,
                ["ID", "prediction"],
                [{"ID": f"id{fold}", "prediction": f"word {fold}"}],
            )
            lineage = {
                "truth_sha256": sha256_file(truth),
                "source_sha256": sha256_file(source),
                "config_sha256": sha256_file(config_path),
                "fold_plan": {"outer_fold": fold},
            }
            lineage_digest = canonical_sha256(lineage)
            write_json(
                run / "run_manifest.json",
                {"lineage": lineage, "lineage_sha256": lineage_digest},
            )
            reference = [f"word {fold}"]
            write_json(
                run / "report.json",
                {
                    "run_lineage_sha256": lineage_digest,
                    "outer_predictions": str(prediction),
                    "outer_predictions_sha256": sha256_file(prediction),
                    "selected_epoch": 1,
                    "gate_policy": {"enabled": True, "maximum_edit_ratio": 1.0},
                    "fold_scores": {
                        "identity": competition_error(reference, [f"wrod {fold}"]),
                        "gated": competition_error(reference, reference),
                    },
                },
            )
            runs[fold] = run
        report = evaluate_crossfit(
            config,
            config_path,
            truth_path=truth,
            source_path=source,
            run_dirs=runs,
            output_dir=self.root / "combined",
        )
        self.assertEqual(report["scores"]["corrected"]["multi_score"], 1.0)
        self.assertGreater(report["multi_score_gain"], 0)
        table = read_table(report["oof_path"])
        self.assertEqual([row["ID"] for row in table.rows], [f"id{x}" for x in range(5)])


if __name__ == "__main__":
    unittest.main()
