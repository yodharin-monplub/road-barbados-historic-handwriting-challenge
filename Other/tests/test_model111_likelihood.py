import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model111"))
from common import choose
from score import completion_inputs


class LikelihoodTests(unittest.TestCase):
    def test_lower_nll_wins(self):
        self.assertEqual(choose("a", "b", [(2,1), (3,2)]), "b")

    def test_tie_keeps_primary(self):
        self.assertEqual(choose("a", "b", [(1,2), (2,1)]), "a")

    def test_conservative_requires_both(self):
        self.assertEqual(choose("a", "b", [(3,1), (1,2)], conservative=True), "a")
        self.assertEqual(choose("a", "b", [(3,1), (1,2)]), "b")

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): choose("a", "b", [(float("nan"), 0)])

    def test_prompt_is_excluded_and_eos_is_predicted(self):
        import torch
        prefix = {"input_ids": torch.tensor([[10,11,12]]), "attention_mask": torch.ones((1,3),dtype=torch.long)}
        inputs, targets = completion_inputs(prefix, [20,21,99])
        self.assertEqual(inputs["input_ids"].tolist(), [[10,11,12,20,21]])
        self.assertEqual(targets.tolist(), [[20,21,99]])
        self.assertEqual(inputs["input_ids"][:, -targets.shape[1]:].tolist(), [[12,20,21]])
        self.assertEqual(prefix["input_ids"].tolist(), [[10,11,12]])
        self.assertEqual(inputs["attention_mask"].shape, inputs["input_ids"].shape)

    def test_empty_completion_rejected(self):
        import torch
        with self.assertRaises(ValueError): completion_inputs({"input_ids":torch.tensor([[1]])}, [])


if __name__ == "__main__": unittest.main()
