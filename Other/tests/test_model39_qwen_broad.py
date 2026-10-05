from __future__ import annotations

import copy
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

from road_htr.qwen2_vl_broad import (
    ADAPTED_VISION_BLOCKS,
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
    VISION_BLOCKS,
    VISION_PROJECTIONS,
    VISION_SHAPES,
    SafetyGateError,
    _language_path,
    _vision_path,
    audit_checkpoint_target_shapes,
    audit_trainable_parameters,
    calculate_lora_parameters,
    deterministic_optimizer_schedule,
    discover_broad_lora_targets,
    expected_target_shapes,
    load_config,
    load_training_model,
    make_schedule_guard_callback,
    split_outer_fold,
    train,
    validate_config,
    verify_configured_schedule,
)
from road_htr.qwen2_vl_challenger import audit_injected_lora_targets, exact_lora_target_pattern


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model39" / "config.yaml"


class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features


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


class Model39ArchitectureTests(unittest.TestCase):
    def test_exact_broad_target_and_parameter_audit(self) -> None:
        audit = discover_broad_lora_targets(ArchitectureModel())
        self.assertEqual(len(audit.target_modules), EXPECTED_TARGET_MODULES)
        self.assertEqual(audit.language_modules, 196)
        self.assertEqual(audit.vision_modules, 32)
        self.assertEqual(audit.merger_modules, 2)
        self.assertEqual(audit.trainable_parameters, EXPECTED_TRAINABLE_PARAMETERS)
        self.assertEqual(audit.trainable_tensors, EXPECTED_TRAINABLE_TENSORS)
        self.assertEqual(
            audit.parameters_by_family,
            {"language": 9_232_384, "vision": 1_310_720, "merger": 135_168},
        )
        self.assertTrue(all(block in range(24, 32) for block in ADAPTED_VISION_BLOCKS))
        injected = audit_injected_lora_targets(
            InjectedModel(audit.target_modules), audit.target_modules
        )
        self.assertEqual(injected["injected_modules"], 230)

    def test_parameter_count_is_derived_from_each_matrix_shape(self) -> None:
        shapes = expected_target_shapes()
        total, families = calculate_lora_parameters(shapes, LORA_RANK)
        manual = sum(LORA_RANK * (inputs + outputs) for inputs, outputs in shapes.values())
        self.assertEqual(total, manual)
        self.assertEqual(total, EXPECTED_TRAINABLE_PARAMETERS)
        changed = dict(shapes)
        first = next(iter(changed))
        inputs, outputs = changed[first]
        changed[first] = (inputs + 1, outputs)
        drifted, _ = calculate_lora_parameters(changed, LORA_RANK)
        self.assertEqual(drifted, total + LORA_RANK)
        self.assertEqual(sum(families.values()), total)

    def test_missing_or_wrong_projection_fails_closed(self) -> None:
        missing = [
            item
            for item in architecture_modules()
            if item[0] != "model.layers.27.mlp.down_proj"
        ]
        with self.assertRaisesRegex(SafetyGateError, "language projection layout"):
            discover_broad_lora_targets(ArchitectureModel(missing))
        wrong = architecture_modules()
        for index, (name, _) in enumerate(wrong):
            if name == "visual.blocks.31.attn.qkv":
                wrong[index] = (name, Linear(1280, 3839))
                break
        with self.assertRaisesRegex(SafetyGateError, "projection shape mismatch"):
            discover_broad_lora_targets(ArchitectureModel(wrong))

    def test_exact_trainable_tensor_count_and_owners(self) -> None:
        audit = discover_broad_lora_targets(ArchitectureModel())
        parameters: list[tuple[str, FakeParameter]] = []
        for target, (inputs, outputs) in audit.shapes.items():
            parameters.append(
                (f"base_model.model.{target}.lora_A.default.weight", FakeParameter(LORA_RANK * inputs))
            )
            parameters.append(
                (f"base_model.model.{target}.lora_B.default.weight", FakeParameter(LORA_RANK * outputs))
            )
        report = audit_trainable_parameters(
            ParameterModel(parameters), audit, max_trainable_parameters=11_000_000
        )
        self.assertEqual(report["trainable_parameters"], EXPECTED_TRAINABLE_PARAMETERS)
        self.assertEqual(report["trainable_tensors"], 460)
        parameters.append(("base_model.model.lm_head.weight", FakeParameter(1)))
        with self.assertRaisesRegex(SafetyGateError, "non-LoRA"):
            audit_trainable_parameters(
                ParameterModel(parameters), audit, max_trainable_parameters=11_000_000
            )

    def test_checkpoint_headers_match_the_calculation(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        audit = audit_checkpoint_target_shapes(config, ROOT)
        self.assertEqual(len(audit.target_modules), 230)
        self.assertEqual(audit.trainable_parameters, 10_678_272)


class Model39ProtocolTests(unittest.TestCase):
    def test_pinned_config_and_memory_contract(self) -> None:
        config, path = load_config(CONFIG_PATH)
        self.assertEqual(path, CONFIG_PATH)
        self.assertEqual(config["lora"]["rank"], 8)
        self.assertEqual(config["lora"]["adapted_vision_blocks"], list(range(24, 32)))
        self.assertEqual(config["image"]["max_image_tokens"], 320)
        self.assertEqual(config["generation"]["max_new_tokens"], 48)
        self.assertEqual(config["generation"]["repetition_penalty"], 1.05)
        self.assertEqual(config["generation"]["no_repeat_ngram_size"], 4)
        self.assertEqual(config["training"]["expected_updates_per_epoch"], 205)
        self.assertEqual(config["training"]["expected_total_updates"], 615)
        mutations = (
            (("lora", "rank"), 16),
            (("lora", "adapted_vision_blocks"), list(range(32))),
            (("architecture", "expected_trainable_parameters"), 1),
            (("training", "batch_size"), 2),
            (("training", "expected_total_updates"), 0),
            (("generation", "max_new_tokens"), 160),
            (("generation", "repetition_penalty"), 1.0),
            (("generation", "no_repeat_ngram_size"), 0),
            (("smoke", "max_allocated_gib"), 5.31),
            (("quantization", "load_in_4bit"), False),
        )
        for keys, value in mutations:
            with self.subTest(keys=keys):
                changed = copy.deepcopy(config)
                changed[keys[0]][keys[1]] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)

    def test_ceil_schedule_keeps_every_remainder_group(self) -> None:
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
        divisible = deterministic_optimizer_schedule(32, 16, 2)
        self.assertEqual(divisible.optimizer_updates_per_epoch, 2)
        self.assertEqual(divisible.remainder_micro_batches, 0)
        for bad in ((0, 16, 3), (1, 0, 3), (1, 1, 0), (1.0, 1, 1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                deterministic_optimizer_schedule(*bad)

    def test_configured_schedule_is_a_hard_gate(self) -> None:
        config, _ = load_config(CONFIG_PATH)
        schedule = deterministic_optimizer_schedule(3276, 16, 3)
        verify_configured_schedule(config, schedule)
        changed = copy.deepcopy(config)
        changed["training"]["expected_total_updates"] = 612
        with self.assertRaisesRegex(SafetyGateError, "schedule mismatch"):
            verify_configured_schedule(changed, schedule)

    def test_trainer_mixin_replaces_floor_bookkeeping(self) -> None:
        class Upstream:
            def set_initial_training_values(self, *_args):
                return (4, 204, 3276, 9840, False, 3276, 615)

            @staticmethod
            def num_examples(dataloader):
                return len(dataloader)

        class AuditedTrainer(ExactEpochScheduleMixin, Upstream):
            pass

        schedule = deterministic_optimizer_schedule(3276, 16, 3)
        trainer = AuditedTrainer()
        trainer.exact_optimizer_schedule = schedule
        args = SimpleNamespace(
            gradient_accumulation_steps=16,
            max_steps=615,
            dataloader_drop_last=False,
        )
        values = trainer.set_initial_training_values(args, range(3276), 16)
        self.assertEqual(values, (3, 205, 3276, 9828, False, 3276, 615))
        args.max_steps = 612
        with self.assertRaisesRegex(SafetyGateError, "max_steps"):
            trainer.set_initial_training_values(args, range(3276), 16)

    def test_schedule_callback_rejects_partial_epoch(self) -> None:
        schedule = deterministic_optimizer_schedule(3276, 16, 3)
        callback = make_schedule_guard_callback(schedule, callback_base=object)
        control = SimpleNamespace()
        args = SimpleNamespace()
        for epoch, step in ((1.0, 205), (2.0, 410), (3.0, 615)):
            callback.on_epoch_end(
                args,
                SimpleNamespace(epoch=epoch, global_step=step),
                control,
            )
        callback.on_train_end(
            args, SimpleNamespace(epoch=3.0, global_step=615), control
        )
        self.assertEqual(len(callback.epoch_boundaries), 3)
        partial = make_schedule_guard_callback(schedule, callback_base=object)
        with self.assertRaisesRegex(SafetyGateError, "optimizer boundary"):
            partial.on_epoch_end(
                args, SimpleNamespace(epoch=0.98, global_step=204), control
            )

    def test_fold_split_is_complete_and_id_disjoint(self) -> None:
        records = tuple(
            SimpleNamespace(identifier=f"id-{fold}-{row}", fold=fold)
            for fold in range(5)
            for row in range(3)
        )
        training, validation = split_outer_fold(records, 2)
        self.assertEqual(len(training), 12)
        self.assertEqual(len(validation), 3)
        self.assertTrue(all(record.fold != 2 for record in training))
        self.assertTrue(all(record.fold == 2 for record in validation))
        duplicate = records + (SimpleNamespace(identifier="id-2-0", fold=1),)
        with self.assertRaisesRegex(SafetyGateError, "overlap|duplicate"):
            split_outer_fold(duplicate, 2)

    def test_runtime_uses_nf4_exact_regex_and_explicit_cuda(self) -> None:
        source = inspect.getsource(load_training_model)
        self.assertIn("prepare_model_for_kbit_training", source)
        self.assertIn('bnb_4bit_quant_type="nf4"', source)
        self.assertIn('device_map={"": device_index}', source)
        self.assertIn("exact_lora_target_pattern", source)
        self.assertIn("freeze_vision_backbone", source)
        self.assertNotIn('device_map="auto"', source)
        training_source = inspect.getsource(train)
        self.assertIn("max_steps=schedule.total_optimizer_updates", training_source)
        self.assertIn("ExactEpochTrainer", training_source)
        self.assertIn("dataloader_drop_last=False", training_source)


if __name__ == "__main__":
    unittest.main()
