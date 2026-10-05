from __future__ import annotations

import copy
import importlib.util
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from road_htr.qwen2_vl_challenger import (
    SafetyGateError,
    audit_injected_lora_targets,
    exact_lora_target_pattern,
)
from road_htr.qwen2_vl_starter_highres import (
    EXPECTED_TARGET_MODULES,
    EXPECTED_TRAINABLE_PARAMETERS,
    EXPECTED_TRAINABLE_TENSORS,
    LANGUAGE_LAYERS,
    LANGUAGE_PROJECTIONS,
    LANGUAGE_SHAPES,
    LORA_RANK,
    MERGER_SHAPES,
    ExactEpochScheduleMixin,
    OptimizerSchedule,
    _language_path,
    _template_tokens,
    audit_checkpoint_target_shapes,
    audit_runtime_versions,
    audit_trainable_parameters,
    calculate_lora_parameters,
    deterministic_optimizer_schedule,
    discover_language_targets,
    expected_target_shapes,
    load_config,
    load_letterboxed_rgb,
    load_processor,
    load_training_model,
    make_schedule_callback,
    plan_letterbox,
    qwen_grid_tokens,
    split_outer_fold,
    train,
    validate_config,
    verify_configured_schedule,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model43" / "config.yaml"


class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features


def architecture_modules() -> list[tuple[str, object]]:
    modules: list[tuple[str, object]] = [("", object())]
    for layer in range(LANGUAGE_LAYERS):
        for projection in LANGUAGE_PROJECTIONS:
            modules.append(
                (_language_path(layer, projection), Linear(*LANGUAGE_SHAPES[projection]))
            )
    modules.extend((name, Linear(*shape)) for name, shape in MERGER_SHAPES.items())
    return modules


class ArchitectureModel:
    def __init__(self, modules: list[tuple[str, object]] | None = None):
        self._modules = architecture_modules() if modules is None else modules

    def named_modules(self):
        return iter(self._modules)


class FakeAdapterConfig:
    def __init__(self, targets: tuple[str, ...]):
        self.target_modules = exact_lora_target_pattern(targets)


class InjectedModel:
    def __init__(self, targets: tuple[str, ...]):
        self.peft_config = {"default": FakeAdapterConfig(targets)}
        self.base_model = SimpleNamespace(targeted_module_names=list(targets))


class FakeParameter:
    def __init__(self, count: int, trainable: bool = True):
        self._count = count
        self.requires_grad = trainable

    def numel(self) -> int:
        return self._count


class ParameterModel:
    def __init__(self, parameters: list[tuple[str, FakeParameter]]):
        self._parameters = parameters

    def named_parameters(self):
        return iter(self._parameters)


class FakeTokenizer:
    image_pad_id = 77

    @staticmethod
    def apply_chat_template(messages, tokenize, add_generation_prompt):
        del messages, tokenize, add_generation_prompt
        return "template"

    @staticmethod
    def encode(text, add_special_tokens):
        del text, add_special_tokens
        return [11, 77, 12, 13]

    @staticmethod
    def convert_tokens_to_ids(token):
        if token != "<|image_pad|>":
            raise AssertionError(token)
        return 77


class Model43ArchitectureTests(unittest.TestCase):
    def test_exact_starter_language_targets_and_derived_parameter_count(self) -> None:
        shapes = expected_target_shapes()
        self.assertEqual(len(shapes), 28 * 7)
        manual = sum(
            LORA_RANK * (inputs + outputs) for inputs, outputs in shapes.values()
        )
        self.assertEqual(calculate_lora_parameters(shapes, LORA_RANK), manual)
        self.assertEqual(manual, EXPECTED_TRAINABLE_PARAMETERS)
        self.assertEqual(EXPECTED_TRAINABLE_PARAMETERS, 18_464_768)
        self.assertEqual(
            sum(inputs * outputs for inputs, outputs in shapes.values()),
            1_310_195_712,
        )

    def test_fake_architecture_discovers_only_all_language_paths(self) -> None:
        audit = discover_language_targets(ArchitectureModel())
        self.assertEqual(len(audit.target_modules), EXPECTED_TARGET_MODULES)
        self.assertEqual(audit.language_modules, 196)
        self.assertEqual(audit.merger_modules, 0)
        self.assertEqual(audit.visual_modules, 0)
        self.assertEqual(audit.trainable_parameters, EXPECTED_TRAINABLE_PARAMETERS)
        self.assertEqual(audit.trainable_tensors, EXPECTED_TRAINABLE_TENSORS)
        self.assertFalse(any(name.startswith("visual.") for name in audit.target_modules))
        injection = audit_injected_lora_targets(
            InjectedModel(audit.target_modules), audit.target_modules
        )
        self.assertEqual(injection["injected_modules"], 196)

    def test_missing_projection_wrong_shape_or_rank_fails_closed(self) -> None:
        missing = [
            entry
            for entry in architecture_modules()
            if entry[0] != "model.layers.27.mlp.down_proj"
        ]
        with self.assertRaisesRegex(SafetyGateError, "language layout mismatch"):
            discover_language_targets(ArchitectureModel(missing))
        wrong = architecture_modules()
        for index, (name, _) in enumerate(wrong):
            if name == "model.layers.0.self_attn.k_proj":
                wrong[index] = (name, Linear(1536, 257))
                break
        with self.assertRaisesRegex(SafetyGateError, "projection shape mismatch"):
            discover_language_targets(ArchitectureModel(wrong))
        with self.assertRaisesRegex(SafetyGateError, "LoRA count mismatch"):
            discover_language_targets(ArchitectureModel(), rank=8)

    def test_trainable_ledger_rejects_visual_or_non_lora_parameters(self) -> None:
        audit = discover_language_targets(ArchitectureModel())
        parameters: list[tuple[str, FakeParameter]] = []
        for target, (inputs, outputs) in audit.shapes.items():
            parameters.extend(
                (
                    (
                        f"base_model.model.{target}.lora_A.default.weight",
                        FakeParameter(LORA_RANK * inputs),
                    ),
                    (
                        f"base_model.model.{target}.lora_B.default.weight",
                        FakeParameter(LORA_RANK * outputs),
                    ),
                )
            )
        report = audit_trainable_parameters(
            ParameterModel(parameters), audit, maximum=19_000_000
        )
        self.assertEqual(report["trainable_parameters"], 18_464_768)
        self.assertEqual(report["trainable_tensors"], 392)
        bad = parameters + [("base_model.model.visual.foo.lora_A.weight", FakeParameter(1))]
        with self.assertRaisesRegex(SafetyGateError, "unique language owner"):
            audit_trainable_parameters(ParameterModel(bad), audit, maximum=19_000_000)
        bad = parameters + [("base_model.model.lm_head.weight", FakeParameter(1))]
        with self.assertRaisesRegex(SafetyGateError, "non-LoRA"):
            audit_trainable_parameters(ParameterModel(bad), audit, maximum=19_000_000)

    def test_real_checkpoint_headers_match_exact_196_path_ledger(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        audit = audit_checkpoint_target_shapes(config, ROOT)
        self.assertEqual(audit.language_modules, 196)
        self.assertEqual(audit.merger_modules, 0)
        self.assertEqual(audit.trainable_parameters, 18_464_768)
        self.assertEqual(
            sum(inputs * outputs for inputs, outputs in audit.shapes.values()),
            1_310_195_712,
        )


class Model43ResolutionTests(unittest.TestCase):
    def test_factor_aligned_letterbox_stays_inside_visual_budget(self) -> None:
        for width, height in ((1600, 300), (300, 1600), (997, 997), (2227, 413)):
            with self.subTest(size=(width, height)):
                plan = plan_letterbox(
                    width, height, minimum_tokens=896, maximum_tokens=1024
                )
                self.assertEqual(plan.canvas_width % 28, 0)
                self.assertEqual(plan.canvas_height % 28, 0)
                self.assertGreaterEqual(plan.planned_image_tokens, 896)
                self.assertLessEqual(plan.planned_image_tokens, 1024)
                self.assertEqual(
                    qwen_grid_tokens(plan.canvas_width, plan.canvas_height),
                    plan.planned_image_tokens,
                )
                self.assertLessEqual(plan.relative_content_aspect_error, 0.005)
                self.assertLessEqual(plan.content_width, plan.canvas_width)
                self.assertLessEqual(plan.content_height, plan.canvas_height)

    def test_letterbox_load_preserves_complete_image_and_reports_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.png"
            Image.new("RGB", (333, 71), (240, 238, 230)).save(path)
            image, report = load_letterboxed_rgb(
                path, minimum_tokens=896, maximum_tokens=1024
            )
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (report["canvas_width"], report["canvas_height"]))
            self.assertEqual(
                qwen_grid_tokens(*image.size), report["planned_image_tokens"]
            )
            self.assertGreaterEqual(report["content_pixel_fraction"], 0.0)
            self.assertLessEqual(report["content_pixel_fraction"], 1.0)

    def test_static_chat_template_replaces_one_pad_with_visual_tokens(self) -> None:
        total = _template_tokens(
            FakeTokenizer(), [], generation_prompt=True, image_tokens=1024
        )
        self.assertEqual(total, 1027)


class Model43ProtocolTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("transformers") is not None,
        "exact runtime audit requires the VLM environment",
    )
    def test_exact_pinned_runtime_is_installed(self) -> None:
        self.assertEqual(
            audit_runtime_versions(),
            {
                "torch": "2.13.0",
                "torchvision": "0.28.0",
                "transformers": "4.49.0",
                "accelerate": "1.14.0",
                "bitsandbytes": "0.50.2",
                "peft": "0.14.0",
                "safetensors": "0.7.0",
                "Pillow": "12.3.0",
                "PyYAML": "6.0.3",
            },
        )

    def test_pinned_config_is_organizer_aligned_and_memory_bounded(self) -> None:
        config, path = load_config(CONFIG_PATH)
        self.assertEqual(path, CONFIG_PATH)
        self.assertEqual(config["lora"]["rank"], 16)
        self.assertEqual(config["lora"]["alpha"], 32)
        self.assertEqual(config["lora"]["language_projections"], list(LANGUAGE_PROJECTIONS))
        self.assertFalse(config["lora"]["include_merger"])
        self.assertFalse(config["lora"]["include_visual_blocks"])
        self.assertFalse(config["model"]["processor_use_fast"])
        self.assertEqual(config["image"]["min_pixels"], 896 * 28 * 28)
        self.assertEqual(config["image"]["max_image_tokens"], 1024)
        self.assertEqual(config["image"]["max_sequence_tokens"], 1152)
        self.assertEqual(config["training"]["learning_rate"], 3e-5)
        self.assertEqual(config["generation"]["max_new_tokens"], 48)
        self.assertEqual(config["generation"]["repetition_penalty"], 1.05)
        self.assertEqual(config["generation"]["no_repeat_ngram_size"], 4)
        self.assertIn("Transcribe EXACTLY as seen", config["prompt"])
        self.assertNotIn("adjacent", config["prompt"].lower())
        self.assertLessEqual(config["smoke"]["max_allocated_gib"], 5.3)
        self.assertLessEqual(config["smoke"]["max_reserved_gib"], 5.3)

    def test_immutable_contract_rejects_scope_resolution_and_memory_drift(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        changes = (
            (("lora", "rank"), 8),
            (("lora", "include_merger"), True),
            (("lora", "include_visual_blocks"), True),
            (("model", "processor_use_fast"), True),
            (("image", "min_pixels"), 4 * 28 * 28),
            (("image", "max_image_tokens"), 1025),
            (("image", "max_sequence_tokens"), 1100),
            (("training", "gradient_accumulation"), 8),
            (("training", "expected_total_updates"), 612),
            (("generation", "max_new_tokens"), 160),
            (("generation", "repetition_penalty"), 1.0),
            (("smoke", "max_reserved_gib"), 5.31),
            (("quantization", "load_in_4bit"), False),
        )
        for keys, value in changes:
            with self.subTest(keys=keys):
                changed = copy.deepcopy(config)
                changed[keys[0]][keys[1]] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)

    def test_exact_ceil_schedule_keeps_all_rows_in_three_epochs(self) -> None:
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
        for bad in ((0, 16, 3), (1, 0, 3), (1, 1, 0), (1.0, 1, 1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                deterministic_optimizer_schedule(*bad)

    def test_mixin_and_callback_fail_on_floor_schedule(self) -> None:
        class Upstream:
            @staticmethod
            def set_initial_training_values(*_args):
                return (4, 204, 3276, 9840, False, 3276, 615)

            @staticmethod
            def num_examples(dataloader):
                return len(dataloader)

        class Trainer(ExactEpochScheduleMixin, Upstream):
            pass

        schedule = deterministic_optimizer_schedule(3276, 16, 3)
        trainer = Trainer()
        trainer.exact_optimizer_schedule = schedule
        args = SimpleNamespace(
            gradient_accumulation_steps=16,
            max_steps=615,
            dataloader_drop_last=False,
        )
        self.assertEqual(
            trainer.set_initial_training_values(args, range(3276), 16),
            (3, 205, 3276, 9828, False, 3276, 615),
        )
        callback = make_schedule_callback(schedule, callback_base=object)
        for epoch, step in ((1.0, 205), (2.0, 410), (3.0, 615)):
            callback.on_epoch_end(
                args,
                SimpleNamespace(epoch=epoch, global_step=step),
                SimpleNamespace(),
            )
        callback.on_train_end(
            args,
            SimpleNamespace(epoch=3.0, global_step=615),
            SimpleNamespace(),
        )
        partial = make_schedule_callback(schedule, callback_base=object)
        with self.assertRaisesRegex(SafetyGateError, "partial or extra"):
            partial.on_epoch_end(
                args,
                SimpleNamespace(epoch=0.99, global_step=204),
                SimpleNamespace(),
            )

    def test_fold_split_is_complete_and_leakage_safe(self) -> None:
        records = tuple(
            SimpleNamespace(identifier=f"id-{fold}-{row}", fold=fold)
            for fold in range(5)
            for row in range(3)
        )
        training, validation = split_outer_fold(records, 3)
        self.assertEqual(len(training), 12)
        self.assertEqual(len(validation), 3)
        self.assertTrue(all(record.fold != 3 for record in training))
        self.assertTrue(all(record.fold == 3 for record in validation))
        duplicate = records + (SimpleNamespace(identifier="id-3-0", fold=0),)
        with self.assertRaisesRegex(SafetyGateError, "duplicate|overlap"):
            split_outer_fold(duplicate, 3)

    def test_runtime_wires_nf4_frozen_vision_exact_schedule_and_generation_guards(self) -> None:
        processor_source = inspect.getsource(load_processor)
        self.assertIn("use_fast=False", processor_source)
        self.assertIn("verify_processor_budget", processor_source)
        model_source = inspect.getsource(load_training_model)
        self.assertIn("prepare_model_for_kbit_training", model_source)
        self.assertIn('bnb_4bit_quant_type="nf4"', model_source)
        self.assertIn("freeze_vision_backbone", model_source)
        self.assertIn("exact_lora_target_pattern", model_source)
        self.assertNotIn('device_map="auto"', model_source)
        training_source = inspect.getsource(train)
        self.assertIn("max_steps=schedule.total_optimizer_updates", training_source)
        self.assertIn("dataloader_drop_last=False", training_source)
        self.assertIn("ExactTrainer", training_source)
        self.assertIn("repetition_penalty", training_source)
        self.assertIn("no_repeat_ngram_size", training_source)


if __name__ == "__main__":
    unittest.main()
