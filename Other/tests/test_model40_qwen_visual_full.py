from __future__ import annotations

import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from road_htr.qwen2_vl_challenger import (
    audit_injected_lora_targets,
    exact_lora_target_pattern,
)
from road_htr.qwen2_vl_visual_full import (
    ADAPTED_VISION_BLOCKS,
    EXPECTED_TARGET_MODULES,
    EXPECTED_TRAINABLE_PARAMETERS,
    EXPECTED_TRAINABLE_TENSORS,
    LANGUAGE_LAYERS,
    LANGUAGE_PROJECTIONS,
    LANGUAGE_SHAPES,
    LORA_RANK,
    MERGER_SHAPES,
    MODEL_CONTEXT_TOKENS,
    VISION_BLOCKS,
    VISION_PROJECTIONS,
    VISION_SHAPES,
    ExactEpochScheduleMixin,
    SafetyGateError,
    _language_path,
    _template_token_count,
    _vision_path,
    audit_checkpoint_target_shapes,
    audit_model_context,
    audit_trainable_parameters,
    calculate_lora_parameters,
    deterministic_optimizer_schedule,
    discover_full_visual_targets,
    expected_target_shapes,
    load_budget_filled_rgb,
    load_config,
    make_schedule_guard_callback,
    plan_aspect_preserving_resize,
    qwen_smart_grid,
    select_memory_smoke_record,
    train,
    validate_config,
    verify_configured_schedule,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model40" / "config.yaml"


class Linear:
    def __init__(self, inputs: int, outputs: int):
        self.in_features = inputs
        self.out_features = outputs


def architecture_modules() -> list[tuple[str, object]]:
    modules: list[tuple[str, object]] = [("", object())]
    for layer in range(LANGUAGE_LAYERS):
        for projection in LANGUAGE_PROJECTIONS:
            modules.append((_language_path(layer, projection), Linear(*LANGUAGE_SHAPES[projection])))
    for block in range(VISION_BLOCKS):
        for projection in VISION_PROJECTIONS:
            modules.append((_vision_path(block, projection), Linear(*VISION_SHAPES[projection])))
    modules.extend((name, Linear(*shape)) for name, shape in MERGER_SHAPES.items())
    return modules


class ArchitectureModel:
    def __init__(self, modules: list[tuple[str, object]] | None = None):
        self._modules = architecture_modules() if modules is None else modules

    def named_modules(self):
        return iter(self._modules)


class AdapterConfig:
    def __init__(self, targets: tuple[str, ...]):
        self.target_modules = exact_lora_target_pattern(targets)


class InjectedModel:
    def __init__(self, targets: tuple[str, ...]):
        self.peft_config = {"default": AdapterConfig(targets)}
        self.base_model = SimpleNamespace(targeted_module_names=list(targets))


class FakeParameter:
    def __init__(self, count: int):
        self._count = count
        self.requires_grad = True

    def numel(self) -> int:
        return self._count


class ParameterModel:
    def __init__(self, parameters: list[tuple[str, FakeParameter]]):
        self._parameters = parameters

    def named_parameters(self):
        return iter(self._parameters)


class Model40ArchitectureTests(unittest.TestCase):
    def test_all_32_vision_blocks_and_broad_language_are_exact(self) -> None:
        audit = discover_full_visual_targets(ArchitectureModel())
        self.assertEqual(ADAPTED_VISION_BLOCKS, tuple(range(32)))
        self.assertEqual(audit.language_modules, 196)
        self.assertEqual(audit.vision_modules, 128)
        self.assertEqual(audit.merger_modules, 2)
        self.assertEqual(len(audit.target_modules), EXPECTED_TARGET_MODULES)
        self.assertEqual(audit.trainable_parameters, EXPECTED_TRAINABLE_PARAMETERS)
        self.assertEqual(audit.trainable_tensors, EXPECTED_TRAINABLE_TENSORS)
        self.assertEqual(
            audit.parameters_by_family,
            {"language": 9_232_384, "vision": 5_242_880, "merger": 135_168},
        )
        result = audit_injected_lora_targets(
            InjectedModel(audit.target_modules), audit.target_modules
        )
        self.assertEqual(result["injected_modules"], 326)

    def test_parameter_count_comes_from_all_matrix_shapes(self) -> None:
        shapes = expected_target_shapes()
        count, families = calculate_lora_parameters(shapes, LORA_RANK)
        self.assertEqual(count, sum(LORA_RANK * sum(shape) for shape in shapes.values()))
        self.assertEqual(count, 14_610_432)
        self.assertEqual(sum(families.values()), count)
        self.assertEqual(len(shapes), 326)

    def test_missing_early_vision_block_or_shape_drift_fails(self) -> None:
        missing = [
            item for item in architecture_modules() if item[0] != "visual.blocks.0.attn.qkv"
        ]
        with self.assertRaisesRegex(SafetyGateError, "vision projection layout"):
            discover_full_visual_targets(ArchitectureModel(missing))
        wrong = architecture_modules()
        for index, (name, _) in enumerate(wrong):
            if name == "model.layers.27.mlp.down_proj":
                wrong[index] = (name, Linear(8959, 1536))
                break
        with self.assertRaisesRegex(SafetyGateError, "shape mismatch"):
            discover_full_visual_targets(ArchitectureModel(wrong))

    def test_real_checkpoint_headers_match_326_target_ledger(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        audit = audit_checkpoint_target_shapes(config, ROOT)
        self.assertEqual(len(audit.target_modules), 326)
        self.assertEqual(audit.trainable_parameters, 14_610_432)
        self.assertEqual(
            sum(inputs * outputs for inputs, outputs in audit.shapes.values()),
            1_973_420_032,
        )

    def test_trainable_tensors_have_exact_owners_and_sizes(self) -> None:
        audit = discover_full_visual_targets(ArchitectureModel())
        parameters: list[tuple[str, FakeParameter]] = []
        for target, (inputs, outputs) in audit.shapes.items():
            parameters.append(
                (f"base_model.model.{target}.lora_A.default.weight", FakeParameter(LORA_RANK * inputs))
            )
            parameters.append(
                (f"base_model.model.{target}.lora_B.default.weight", FakeParameter(LORA_RANK * outputs))
            )
        report = audit_trainable_parameters(
            ParameterModel(parameters), audit, max_trainable_parameters=15_000_000
        )
        self.assertEqual(report["trainable_parameters"], 14_610_432)
        self.assertEqual(report["trainable_tensors"], 652)


class Model40GeometryAndSequenceTests(unittest.TestCase):
    def test_letterbox_preserves_content_aspect_and_qwen_cap(self) -> None:
        target = 640 * 28 * 28
        for width, height in ((1000, 100), (267, 88), (5929, 1131), (1135, 28)):
            plan = plan_aspect_preserving_resize(
                width,
                height,
                minimum_tokens=512,
                maximum_tokens=640,
            )
            self.assertGreater(plan.uniform_scale, 0.0)
            self.assertLessEqual(plan.relative_content_aspect_error, 0.005)
            self.assertGreaterEqual(plan.planned_image_tokens, 512)
            self.assertLessEqual(plan.planned_image_tokens, 640)
            grid_width, grid_height, tokens = qwen_smart_grid(
                plan.canvas_width,
                plan.canvas_height,
                min_pixels=3136,
                max_pixels=target,
            )
            self.assertEqual((grid_width, grid_height), (plan.canvas_width, plan.canvas_height))
            self.assertGreaterEqual(tokens, 512)
            self.assertLessEqual(tokens, 640)

    def test_loaded_letterbox_is_factor_aligned_without_content_distortion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide.jpg"
            Image.new("RGB", (1000, 100), (240, 235, 220)).save(path)
            canvas, diagnostics = load_budget_filled_rgb(
                path, minimum_tokens=512, maximum_tokens=640
            )
            self.assertEqual(canvas.width % 28, 0)
            self.assertEqual(canvas.height % 28, 0)
            self.assertEqual(
                canvas.width * canvas.height // 28**2,
                diagnostics["planned_image_tokens"],
            )
            self.assertLessEqual(
                diagnostics["relative_content_aspect_error"], 0.005
            )
            self.assertGreater(diagnostics["content_pixel_fraction"], 0.8)

    def test_template_budget_replaces_exactly_one_image_pad(self) -> None:
        class Tokenizer:
            @staticmethod
            def apply_chat_template(*_args, **_kwargs):
                return "encoded"

            @staticmethod
            def encode(*_args, **_kwargs):
                return [10, 99, 11, 12]

            @staticmethod
            def convert_tokens_to_ids(_token):
                return 99

        tokens, pads = _template_token_count(
            Tokenizer(), [{"role": "user"}], add_generation_prompt=True, image_tokens=640
        )
        self.assertEqual(tokens, 643)
        self.assertEqual(pads, 1)

    def test_smoke_selector_uses_post_resize_workload(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for identifier, size, label in (
                ("short", (1200, 100), "a"),
                ("long", (300, 100), "a" * 80),
            ):
                path = root / f"{identifier}.jpg"
                Image.new("RGB", size, "white").save(path)
                records.append(
                    SimpleNamespace(identifier=identifier, image_path=path, label=label)
                )
            selected = select_memory_smoke_record(records, config)
            self.assertEqual(selected.identifier, "long")

    def test_model_context_metadata_and_static_caps(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        context = audit_model_context(config)
        self.assertEqual(context["model_max_position_embeddings"], MODEL_CONTEXT_TOKENS)
        self.assertEqual(config["image"]["max_sequence_tokens"], 768)
        self.assertEqual(config["image"]["max_generation_context_tokens"], 816)
        self.assertEqual(config["generation"]["max_new_tokens"], 48)


class Model40ProtocolTests(unittest.TestCase):
    def test_config_is_pinned_to_full_visual_memory_contract(self) -> None:
        config, path = load_config(CONFIG_PATH)
        self.assertEqual(path, CONFIG_PATH)
        self.assertIn("every complete handwritten line", config["prompt"])
        self.assertIn("Ignore only text fragments clipped", config["prompt"])
        self.assertEqual(config["generation"]["repetition_penalty"], 1.05)
        self.assertEqual(config["generation"]["no_repeat_ngram_size"], 4)
        changes = (
            (("lora", "adapted_vision_blocks"), list(range(1, 32))),
            (("image", "max_image_tokens"), 641),
            (("image", "min_effective_image_tokens"), 511),
            (("image", "max_sequence_tokens"), 640),
            (("image", "max_generation_context_tokens"), 817),
            (("generation", "repetition_penalty"), 1.0),
            (("smoke", "max_reserved_gib"), 5.31),
        )
        for keys, value in changes:
            with self.subTest(keys=keys):
                changed = copy.deepcopy(config)
                changed[keys[0]][keys[1]] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)

    def test_exact_205_by_3_schedule_and_floor_fix(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        schedule = deterministic_optimizer_schedule(3276, 16, 3)
        verify_configured_schedule(config, schedule)
        self.assertEqual(schedule.remainder_micro_batches, 12)
        self.assertEqual(schedule.optimizer_updates_per_epoch, 205)
        self.assertEqual(schedule.total_optimizer_updates, 615)
        self.assertEqual(schedule.transformer_floor_total_updates, 612)
        self.assertEqual(schedule.total_row_presentations, 9828)

    def test_mixin_and_boundary_callback_reject_floor_schedule(self) -> None:
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
        callback = make_schedule_guard_callback(schedule, callback_base=object)
        with self.assertRaisesRegex(SafetyGateError, "partial/extra"):
            callback.on_epoch_end(
                args,
                SimpleNamespace(epoch=0.99, global_step=204),
                SimpleNamespace(),
            )

    def test_training_source_wires_exact_schedule_and_generation_guards(self) -> None:
        source = inspect.getsource(train)
        self.assertIn("max_steps=schedule.total_optimizer_updates", source)
        self.assertIn("dataloader_drop_last=False", source)
        self.assertIn("ExactEpochTrainer", source)
        self.assertIn("repetition_penalty", source)
        self.assertIn("no_repeat_ngram_size", source)


if __name__ == "__main__":
    unittest.main()
