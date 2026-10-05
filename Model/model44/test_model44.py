import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model44 import decode


class Model44Tests(unittest.TestCase):
    def setUp(self):
        self.config, self.config_path = decode.load_config("model44/config.json")

    def test_registered_panel_is_exact_and_has_no_tuned_eos_knobs(self):
        self.assertEqual([entry["name"] for entry in self.config["panel"]], ["greedy", "beam3_starter_aligned"])
        self.assertEqual(self.config["panel"][1]["num_beams"], 3)
        self.assertEqual(self.config["panel"][0]["max_new_tokens"], 48)
        self.assertEqual(self.config["frozen_omissions"], ["length_penalty", "early_stopping", "eos_token_id", "forced_eos_token_id"])
        bad = copy.deepcopy(self.config)
        bad["panel"][1]["max_new_tokens"] = 49
        with self.assertRaises(decode.Model44GateError):
            decode.validate_config(bad)

    def test_source_ledger_and_registered_inputs_are_pinned(self):
        hashes = decode.source_receipt(self.config)
        self.assertEqual(set(hashes), {"model39_config", "model39_infer", "broad_runtime", "challenger_runtime", "schema", "submission"})
        kind, _, digest = decode.registered_input(self.config, "data/processed/train_folds.csv", 0)
        self.assertEqual(kind, "fold_oof")
        self.assertEqual(digest, self.config["inputs"]["fold_oof"]["sha256"])
        with self.assertRaises(decode.Model44GateError):
            decode.registered_input(self.config, "data/processed/train_folds.csv", None)

    def test_adapter_must_be_exact_best_adapter_and_output_is_isolated(self):
        with self.assertRaises(decode.Model44GateError):
            decode.adapter_receipt(self.config, "outputs/model39/fold0/final_adapter", 0)
        with self.assertRaises(decode.Model44GateError):
            decode.new_output_dir(self.config, "outputs/not_model44/fold0")
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(decode.Model44GateError):
                decode.new_output_dir(self.config, temp)

    def test_predict_vector_runs_a_full_vector_and_keeps_internal_spaces(self):
        class FakeTensor:
            shape = (1, 4)
            def __getitem__(self, _): return self
        class FakeModel:
            training = False
            class Config: use_cache = False
            config = Config()
            def eval(self): pass
            def train(self, value): self.training = value
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return FakeTensor()
        class FakeCollator:
            def encode_prompt(self, record): return {"input_ids": FakeTensor()}, {}
        class FakeProcessor: pass
        records = [object(), object()]
        model = FakeModel()
        policy = self.config["panel"][1]
        class FakeInferenceMode:
            def __enter__(self): return self
            def __exit__(self, *_): return False
        class FakeTorch:
            @staticmethod
            def inference_mode(): return FakeInferenceMode()
        with patch.dict(sys.modules, {"torch": FakeTorch()}), patch("road_htr.qwen2_vl_challenger._model_device", return_value="cpu"), patch("road_htr.qwen2_vl_challenger._to_device", side_effect=lambda batch, _: batch), patch("road_htr.qwen2_vl_challenger._decode_generated", return_value="A  B"):
            predicted = decode.predict_vector(model, FakeProcessor(), FakeCollator(), records, policy)
        self.assertEqual(predicted, ["A  B", "A  B"])
        self.assertEqual(model.kwargs["num_beams"], 3)
        self.assertNotIn("length_penalty", model.kwargs)


if __name__ == "__main__":
    unittest.main()
