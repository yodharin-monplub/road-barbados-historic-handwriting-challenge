import copy
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model49 import decode


class Model49Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, _ = decode.load_config("model49/config.json")

    def test_registered_panel_differs_only_in_no_repeat_window(self) -> None:
        control, challenger = self.config["panel"]
        self.assertEqual(control["no_repeat_ngram_size"], 4)
        self.assertEqual(challenger["no_repeat_ngram_size"], 8)
        self.assertEqual(control["max_new_tokens"], challenger["max_new_tokens"])
        self.assertEqual(control["max_new_tokens"], 48)
        self.assertEqual(control["num_beams"], challenger["num_beams"])
        self.assertEqual(control["num_beams"], 1)
        self.assertFalse(control["do_sample"])
        self.assertFalse(challenger["do_sample"])
        differing = {
            key
            for key in set(control) | set(challenger)
            if key != "name" and control.get(key) != challenger.get(key)
        }
        self.assertEqual(differing, {"no_repeat_ngram_size"})

        bad = copy.deepcopy(self.config)
        bad["panel"][1]["max_new_tokens"] = 49
        with self.assertRaises(decode.Model49GateError):
            decode.validate_config(bad)

    def test_source_and_input_hash_ledgers_are_live(self) -> None:
        hashes = decode.source_receipt(self.config)
        self.assertEqual(
            set(hashes),
            {
                "model39_config",
                "model39_infer",
                "broad_runtime",
                "challenger_runtime",
                "schema",
                "submission",
            },
        )
        path, digest = decode.registered_input(
            self.config, "data/processed/train_folds.csv"
        )
        self.assertEqual(path, decode.resolve(self.config["input"]["path"]))
        self.assertEqual(digest, self.config["input"]["sha256"])
        with self.assertRaises(decode.Model49GateError):
            decode.registered_input(self.config, "data/raw/Test.csv")

    def test_label_constraint_audit_without_heavy_imports(self) -> None:
        config = copy.deepcopy(self.config)
        records = [
            SimpleNamespace(identifier="a", label="four", fold=0),
            SimpleNamespace(identifier="b", label="eight", fold=0),
            SimpleNamespace(identifier="c", label="plain", fold=2),
        ]
        encoded = {
            "four": [1, 2, 3, 4, 1, 2, 3, 4],
            "eight": list(range(8)) + list(range(8)),
            "plain": [9, 10, 11],
        }

        class FakeTokenizer:
            def encode(self, label, *, add_special_tokens):
                if add_special_tokens:
                    raise AssertionError("special tokens must remain disabled")
                return encoded[label]

        config["input"]["rows"] = len(records)
        config["label_constraint_audit"]["expected"] = {
            "4": {
                "count": 2,
                "by_fold": {"0": 2, "1": 0, "2": 0, "3": 0, "4": 0},
                "sorted_ids_sha256": decode._ids_digest(["a", "b"]),
            },
            "8": {
                "count": 1,
                "by_fold": {"0": 1, "1": 0, "2": 0, "3": 0, "4": 0},
                "sorted_ids_sha256": decode._ids_digest(["b"]),
            },
        }
        receipt = decode.audit_label_constraints(config, records, FakeTokenizer())
        self.assertEqual(
            receipt["fold0_interpretation"],
            {
                "ngram4_exact_truths_made_impossible": 2,
                "ngram8_exact_truths_made_impossible": 1,
            },
        )
        self.assertTrue(decode.has_repeated_ngram(encoded["four"], 4))
        self.assertFalse(decode.has_repeated_ngram(encoded["four"], 8))

    def test_adapter_requires_complete_receipt_and_binds_epoch(self) -> None:
        config = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model39"
            run_dir = root / "fold0"
            adapter = run_dir / "best_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            manifest = {
                "model_number": 39,
                "fold": 0,
                "train_csv_sha256": config["input"]["sha256"],
                "resolved_config": str(
                    decode.resolve(config["base_model39"]["config"])
                ),
            }
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            config["base_model39"]["adapter_root"] = str(root)

            with self.assertRaisesRegex(decode.Model49GateError, "complete.json"):
                decode.adapter_receipt(config, adapter)

            complete = {
                "fold": 0,
                "best_epoch": 2,
                "best_multi_score": 0.82,
                "best_adapter": str(adapter),
                "observed_optimizer_updates": 615,
                "optimizer_schedule": {"total_optimizer_updates": 615},
            }
            (run_dir / "complete.json").write_text(
                json.dumps(complete), encoding="utf-8"
            )
            receipt = decode.adapter_receipt(config, adapter)
            self.assertEqual(receipt["best_epoch"], 2)
            self.assertEqual(receipt["fold"], 0)
            self.assertIn("adapter_model.safetensors", receipt["files_sha256"])

    def test_predict_controls_are_fixed_except_ngram(self) -> None:
        class FakeTensor:
            shape = (1, 4)

            def __getitem__(self, _):
                return self

        class FakeModel:
            training = False

            class Config:
                use_cache = False

            config = Config()

            def __init__(self):
                self.calls = []

            def eval(self):
                return None

            def train(self, value):
                self.training = value

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return FakeTensor()

        class FakeCollator:
            def encode_prompt(self, record):
                return {"input_ids": FakeTensor()}, {}

        class FakeInferenceMode:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        class FakeTorch:
            @staticmethod
            def inference_mode():
                return FakeInferenceMode()

        model = FakeModel()
        with (
            patch.dict(sys.modules, {"torch": FakeTorch()}),
            patch("road_htr.qwen2_vl_challenger._model_device", return_value="cpu"),
            patch(
                "road_htr.qwen2_vl_challenger._to_device",
                side_effect=lambda batch, _: batch,
            ),
            patch(
                "road_htr.qwen2_vl_challenger._decode_generated",
                return_value="A  B",
            ),
            redirect_stdout(StringIO()),
        ):
            first = decode.predict_vector(
                model,
                object(),
                FakeCollator(),
                [object()],
                self.config["panel"][0],
            )
            second = decode.predict_vector(
                model,
                object(),
                FakeCollator(),
                [object()],
                self.config["panel"][1],
            )
        self.assertEqual(first, second)
        self.assertEqual(first, ["A  B"])
        left, right = model.calls
        left = {key: value for key, value in left.items() if key != "input_ids"}
        right = {key: value for key, value in right.items() if key != "input_ids"}
        differing = {
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        }
        self.assertEqual(differing, {"no_repeat_ngram_size"})
        self.assertEqual(left["no_repeat_ngram_size"], 4)
        self.assertEqual(right["no_repeat_ngram_size"], 8)

    def test_outputs_are_new_and_isolated(self) -> None:
        candidate = decode.new_output_dir(
            self.config, "outputs/model49/future_comparison"
        )
        self.assertEqual(candidate, decode.resolve("outputs/model49/future_comparison"))
        with self.assertRaises(decode.Model49GateError):
            decode.new_output_dir(self.config, "outputs/model44/not_model49")
        with self.assertRaises(decode.Model49GateError):
            decode.new_output_dir(self.config, "outputs/model49")


if __name__ == "__main__":
    unittest.main()
