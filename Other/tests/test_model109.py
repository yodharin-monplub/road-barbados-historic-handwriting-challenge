import importlib.util
from pathlib import Path
import sys
import unittest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model109"))
from lora import LoRALinear, attach, adapter_state, apply_adapter


class FakeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.ModuleDict({"to_q": nn.Linear(4, 4), "to_k": nn.Linear(4, 4), "to_v": nn.Linear(4, 4)})
        self.text_encoder = nn.ModuleDict({"to_q": nn.Linear(4, 4)})
        self.style_lin = nn.Linear(4, 4)


class GeneratorAdapterTests(unittest.TestCase):
    def test_zero_adapter_preserves_base(self):
        base = nn.Linear(4, 3)
        layer = LoRALinear(base, 2, 4)
        x = torch.randn(2, 4)
        self.assertTrue(torch.equal(base(x), layer(x)))
        layer(x).sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(layer.lora_b.grad.abs().sum().item(), 0)

    def test_only_generator_attention_and_style_train(self):
        network = FakeNetwork()
        names = attach(network, 2, 4)
        self.assertEqual(len(names), 3)
        for name, param in network.named_parameters():
            self.assertEqual(param.requires_grad, "lora_" in name or name.startswith("style_lin."))

    def test_adapter_roundtrip_and_key_guard(self):
        network = FakeNetwork()
        names = attach(network, 2, 4)
        payload = {"rank": 2, "alpha": 4, "target_modules": names, "state": adapter_state(network)}
        restored = FakeNetwork()
        apply_adapter(restored, payload)
        self.assertFalse(any(p.requires_grad for p in restored.parameters()))
        broken = dict(payload, state={})
        with self.assertRaises(ValueError):
            apply_adapter(FakeNetwork(), broken)


if __name__ == "__main__":
    unittest.main()
