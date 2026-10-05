from __future__ import annotations

import copy
import csv
import inspect
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from road_htr.qwen2_vl_challenger import (
    EXPECTED_TARGET_MODULES,
    MERGER_TARGETS,
    MODEL_REVISION,
    SafetyGateError,
    _decode_generated,
    audit_cuda_residency,
    audit_injected_lora_targets,
    audit_nf4_modules,
    audit_trainable_parameters,
    build_messages,
    discover_lora_targets,
    enable_offline_mode,
    exact_lora_target_pattern,
    enforce_memory_limits,
    exact_metrics,
    greedy_predict,
    image_tokens_from_grid,
    load_config,
    load_full_rgb,
    load_records,
    load_training_model,
    reject_model22_destination,
    require_new_output_dir,
    trim_label_edges,
    validate_config,
    verify_local_checkpoint,
    verify_starter_provenance,
    write_predictions,
)
from road_htr.schema import SchemaError


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model27" / "config.yaml"


class Linear:
    def __init__(self, in_features: int = 32, out_features: int = 32):
        self.in_features = in_features
        self.out_features = out_features


class Linear4bit(Linear):
    pass


class FakeDevice:
    def __init__(self, kind: str):
        self.type = kind

    def __str__(self) -> str:
        return self.type


class FakeParameter:
    def __init__(
        self,
        count: int = 16,
        *,
        trainable: bool = False,
        device: str = "cuda",
    ):
        self._count = count
        self.requires_grad = trainable
        self.device = FakeDevice(device)

    def numel(self) -> int:
        return self._count


def architecture_modules() -> list[tuple[str, object]]:
    modules: list[tuple[str, object]] = [("", object())]
    for layer in range(28):
        root = f"model.layers.{layer}.self_attn"
        modules.append((f"{root}.q_proj", Linear4bit()))
        modules.append((f"{root}.v_proj", Linear4bit(32, 8)))
    modules.extend((name, Linear4bit(128, 64)) for name in MERGER_TARGETS)
    return modules


class ArchitectureModel:
    def __init__(self, modules: list[tuple[str, object]] | None = None):
        self.modules = architecture_modules() if modules is None else modules

    def named_modules(self):
        return iter(self.modules)


class FakeAdapterConfig:
    def __init__(self, target_modules: object):
        self.target_modules = target_modules


class InjectedModel:
    def __init__(self, targets: tuple[str, ...], configured: object | None = None):
        self.peft_config = {
            "default": FakeAdapterConfig(
                exact_lora_target_pattern(targets) if configured is None else configured
            )
        }
        self.base_model = type(
            "FakeTuner", (), {"targeted_module_names": list(targets)}
        )()


class ResidencyModel(ArchitectureModel):
    def __init__(
        self,
        parameters: list[tuple[str, FakeParameter]],
        *,
        device_map: dict[str, object] | None = None,
        quantized_modules: int = 100,
    ):
        super().__init__()
        self._parameters = parameters
        self._buffers = [("cache", FakeParameter(device="cuda"))]
        self.hf_device_map = {"": 0} if device_map is None else device_map
        self.is_loaded_in_4bit = True
        self.modules.extend(
            (f"quantized.{index}", Linear4bit()) for index in range(quantized_modules)
        )

    def named_parameters(self):
        return iter(self._parameters)

    def named_buffers(self):
        return iter(self._buffers)


class Model27ConfigTests(unittest.TestCase):
    def test_pinned_config_and_starter_provenance(self) -> None:
        config, resolved = load_config(CONFIG_PATH)
        self.assertEqual(resolved, CONFIG_PATH)
        self.assertEqual(config["model"]["revision"], MODEL_REVISION)
        self.assertEqual(config["model"]["license"], "Apache-2.0")
        self.assertEqual(config["image"]["max_image_tokens"], 320)
        self.assertEqual(config["image"]["max_pixels"], 320 * 28 * 28)
        provenance = verify_starter_provenance(config, ROOT)
        self.assertEqual(provenance["archive_sha256"], config["provenance"]["starter_archive_sha256"])
        checkpoint = verify_local_checkpoint(config, ROOT)
        self.assertEqual(checkpoint["revision"], MODEL_REVISION)
        self.assertEqual(len(checkpoint["files_sha256"]), 13)
        self.assertEqual(checkpoint["architecture"]["num_hidden_layers"], 28)

    def test_safety_critical_config_cannot_drift(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        mutations = (
            (("model", "revision"), "main"),
            (("model", "attn_implementation"), "eager"),
            (("quantization", "type"), "fp4"),
            (("quantization", "compute_dtype"), "bfloat16"),
            (("training", "batch_size"), 2),
            (("training", "gradient_checkpointing"), False),
            (("lora", "rank"), 32),
            (("generation", "num_beams"), 2),
            (("smoke", "minimum_4bit_modules"), 1),
            (("smoke", "max_allocated_gib"), 5.31),
            (("smoke", "max_reserved_gib"), 5.31),
        )
        for keys, value in mutations:
            with self.subTest(keys=keys):
                changed = copy.deepcopy(config)
                changed[keys[0]][keys[1]] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)

    def test_runtime_source_uses_required_kbit_and_explicit_cuda_contract(self) -> None:
        source = inspect.getsource(load_training_model)
        self.assertIn("prepare_model_for_kbit_training", source)
        self.assertIn('bnb_4bit_quant_type="nf4"', source)
        self.assertIn('attn_implementation="sdpa"', source)
        self.assertIn('device_map={"": device_index}', source)
        self.assertIn("local_files_only=True", source)
        self.assertNotIn('device_map="auto"', source)
        self.assertLess(
            source.index("prepare_model_for_kbit_training("),
            source.index("freeze_vision_backbone("),
        )

    def test_offline_mode_and_both_memory_peaks_are_hard_gates(self) -> None:
        enable_offline_mode()
        self.assertEqual(__import__("os").environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(__import__("os").environ["TRANSFORMERS_OFFLINE"], "1")
        enforce_memory_limits(
            5 * 1024**3,
            5 * 1024**3,
            max_allocated_gib=5.3,
            max_reserved_gib=5.3,
        )
        with self.assertRaisesRegex(SafetyGateError, "allocated peak"):
            enforce_memory_limits(
                int(5.3 * 1024**3),
                5 * 1024**3,
                max_allocated_gib=5.3,
                max_reserved_gib=5.3,
            )
        with self.assertRaisesRegex(SafetyGateError, "reserved peak"):
            enforce_memory_limits(
                5 * 1024**3,
                int(5.3 * 1024**3),
                max_allocated_gib=5.3,
                max_reserved_gib=5.3,
            )


class Model27DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.csv_path = self.root / "folds.csv"
        rows = []
        for fold in range(5):
            identifier = f"item{fold}"
            Image.new("RGB", (100 + fold, 20 + fold), "white").save(
                self.images / f"{identifier}.jpg"
            )
            rows.append(
                {"ID": identifier, "Target": f"  word  {fold}  ", "fold": fold}
            )
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Target", "fold"])
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_edge_trim_preserves_internal_spaces_and_messages(self) -> None:
        self.assertEqual(trim_label_edges(" \tA  B\nC  "), "A  B\nC")
        messages = build_messages("prompt", "A  B")
        self.assertEqual(messages[-1]["content"][0]["text"], "A  B")

    def test_decode_removes_chat_role_without_collapsing_internal_spaces(self) -> None:
        class Processor:
            @staticmethod
            def batch_decode(*_args, **_kwargs):
                return [" assistant:  seal  and  "]

        self.assertEqual(_decode_generated(Processor(), object()), "seal  and")

    def test_greedy_generation_forwards_deterministic_repetition_controls(self) -> None:
        class Tensor:
            shape = (1, 7)

            def to(self, _device):
                return self

            def __getitem__(self, _key):
                return self

        class Model:
            def __init__(self):
                self.training = True
                self.config = type("Config", (), {"use_cache": False})()
                self.kwargs = None

            @staticmethod
            def parameters():
                return iter((FakeParameter(device="cuda"),))

            def eval(self):
                self.training = False

            def train(self, mode=True):
                self.training = mode

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return Tensor()

        class Processor:
            @staticmethod
            def batch_decode(*_args, **_kwargs):
                return ["answer"]

        class Collator:
            @staticmethod
            def encode_prompt(_record):
                return {"input_ids": Tensor()}, {}

        model = Model()
        output = greedy_predict(
            model,
            Processor(),
            Collator(),
            [object()],
            max_new_tokens=48,
            repetition_penalty=1.05,
            no_repeat_ngram_size=4,
        )
        self.assertEqual(output, ["answer"])
        self.assertEqual(model.kwargs["max_new_tokens"], 48)
        self.assertEqual(model.kwargs["repetition_penalty"], 1.05)
        self.assertEqual(model.kwargs["no_repeat_ngram_size"], 4)
        self.assertFalse(model.kwargs["do_sample"])
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)

    def test_load_complete_five_fold_records_without_geometric_edit(self) -> None:
        records = load_records(
            self.csv_path,
            self.images,
            require_target=True,
            require_five_folds=True,
        )
        self.assertEqual([record.fold for record in records], list(range(5)))
        self.assertEqual(records[0].label, "word  0")
        with Image.open(records[4].image_path) as original:
            expected_size = original.size
        self.assertEqual(load_full_rgb(records[4].image_path).size, expected_size)

    def test_missing_fold_is_rejected(self) -> None:
        rows = self.csv_path.read_text(encoding="utf-8").splitlines()
        self.csv_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(SchemaError, "folds must be exactly"):
            load_records(
                self.csv_path,
                self.images,
                require_target=True,
                require_five_folds=True,
            )

    def test_image_token_grid_and_exact_space_metric(self) -> None:
        self.assertEqual(image_tokens_from_grid([[1, 16, 80]], 2), 320)
        records = load_records(
            self.csv_path,
            self.images,
            require_target=True,
            require_five_folds=True,
        )
        predictions = [record.label or "" for record in records]
        self.assertEqual(exact_metrics(records, predictions)["multi_score"], 1.0)
        predictions[0] = "word 0"
        self.assertLess(exact_metrics(records, predictions)["multi_score"], 1.0)

    def test_prediction_artifact_preserves_order_and_internal_spaces(self) -> None:
        records = load_records(
            self.csv_path,
            self.images,
            require_target=True,
            require_five_folds=True,
        )
        output = self.root / "predictions.csv"
        write_predictions(output, records, [f" x  {i} " for i in range(5)])
        with output.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            written = list(reader)
            self.assertEqual(reader.fieldnames, ["ID", "prediction", "fold"])
        self.assertEqual([row["ID"] for row in written], [f"item{i}" for i in range(5)])
        self.assertEqual(written[0]["prediction"], "x  0")

    def test_test_prediction_artifact_has_one_non_id_column(self) -> None:
        test_csv = self.root / "test.csv"
        with test_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID"])
            writer.writeheader()
            writer.writerows({"ID": f"item{i}"} for i in range(5))
        records = load_records(test_csv, self.images, require_target=False)
        output = self.root / "test_predictions.csv"
        write_predictions(output, records, [f"value  {i}" for i in range(5)])
        with output.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            written = list(reader)
            self.assertEqual(reader.fieldnames, ["ID", "prediction"])
        self.assertEqual(written[0]["prediction"], "value  0")


class Model27ArchitectureGateTests(unittest.TestCase):
    def test_exact_58_module_target_audit(self) -> None:
        audit = discover_lora_targets(ArchitectureModel())
        self.assertEqual(len(audit.target_modules), EXPECTED_TARGET_MODULES)
        self.assertEqual(audit.q_modules, 28)
        self.assertEqual(audit.v_modules, 28)
        self.assertEqual(audit.merger_modules, 2)
        self.assertTrue(all("visual.blocks" not in name for name in audit.target_modules))
        pattern = exact_lora_target_pattern(audit.target_modules)
        self.assertTrue(all(__import__("re").fullmatch(pattern, name) for name in audit.target_modules))
        self.assertIsNone(__import__("re").fullmatch(pattern, "visual.blocks.0.attn.q_proj"))
        report = audit_injected_lora_targets(
            InjectedModel(audit.target_modules), audit.target_modules
        )
        self.assertEqual(report["injected_modules"], EXPECTED_TARGET_MODULES)

    def test_peft_target_condensation_or_injection_drift_fails_closed(self) -> None:
        targets = discover_lora_targets(ArchitectureModel()).target_modules
        condensed = InjectedModel(targets, configured={"q_proj", "v_proj"})
        with self.assertRaisesRegex(SafetyGateError, "exact-path regex"):
            audit_injected_lora_targets(condensed, targets)
        missing = InjectedModel(targets)
        missing.base_model.targeted_module_names.pop()
        with self.assertRaisesRegex(SafetyGateError, "injected LoRA target mismatch"):
            audit_injected_lora_targets(missing, targets)

    def test_missing_language_or_extra_merger_linear_fails_closed(self) -> None:
        missing = [
            item
            for item in architecture_modules()
            if item[0] != "model.layers.27.self_attn.v_proj"
        ]
        with self.assertRaisesRegex(SafetyGateError, "language module audit"):
            discover_lora_targets(ArchitectureModel(missing))
        extra = architecture_modules() + [("visual.merger.extra", Linear4bit())]
        with self.assertRaisesRegex(SafetyGateError, "unexpected visual merger"):
            discover_lora_targets(ArchitectureModel(extra))

    def test_only_target_lora_parameters_are_trainable(self) -> None:
        targets = discover_lora_targets(ArchitectureModel()).target_modules
        parameters = []
        for target in targets:
            for branch in ("lora_A", "lora_B"):
                parameters.append(
                    (
                        f"base_model.model.{target}.{branch}.default.weight",
                        FakeParameter(trainable=True),
                    )
                )
        model = ResidencyModel(parameters)
        report = audit_trainable_parameters(
            model, targets, max_trainable_parameters=4_000_000
        )
        self.assertEqual(report["target_modules"], 58)
        self.assertEqual(report["trainable_tensors"], 116)
        model._parameters.append(
            ("base_model.model.visual.blocks.0.weight", FakeParameter(trainable=True))
        )
        with self.assertRaisesRegex(SafetyGateError, "non-LoRA"):
            audit_trainable_parameters(
                model, targets, max_trainable_parameters=4_000_000
            )

    def test_nf4_and_no_offload_gates(self) -> None:
        model = ResidencyModel([("frozen", FakeParameter())])
        self.assertEqual(audit_nf4_modules(model, 100)["linear4bit_modules"], 158)
        self.assertEqual(audit_cuda_residency(model)["parameter_tensors"], 1)
        offloaded = ResidencyModel(
            [("frozen", FakeParameter(device="cpu"))], device_map={"layer": "cpu"}
        )
        with self.assertRaisesRegex(SafetyGateError, "hf_device_map"):
            audit_cuda_residency(offloaded)

    def test_model22_destinations_and_nonempty_outputs_are_rejected(self) -> None:
        protected = ROOT / "outputs" / "model22" / "do_not_write.csv"
        with self.assertRaisesRegex(SafetyGateError, "may not write"):
            reject_model22_destination(protected)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fold0"
            output.mkdir()
            (output / "existing").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                require_new_output_dir(output)
            self.assertEqual((output / "existing").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
