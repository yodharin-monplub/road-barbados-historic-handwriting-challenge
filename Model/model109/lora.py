"""Minimal attention LoRA; base weights stay frozen and are not saved twice."""
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + ((x @ self.lora_a.T) @ self.lora_b.T) * self.scale


def attach(network, rank=8, alpha=16):
    network.requires_grad_(False)
    names = []
    for name, layer in list(network.named_modules()):
        if name.startswith("text_encoder.") or not isinstance(layer, nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in {"to_q", "to_k", "to_v"}:
            continue
        parent_name, child = name.rsplit(".", 1)
        setattr(network.get_submodule(parent_name), child, LoRALinear(layer, rank, alpha))
        names.append(name)
    if not names:
        raise ValueError("No eligible attention projections found")
    network.style_lin.requires_grad_(True)
    return names


def adapter_state(network):
    return {name: tensor.detach().cpu() for name, tensor in network.state_dict().items()
            if "lora_a" in name or "lora_b" in name or name.startswith("style_lin.")}


def apply_adapter(network, payload):
    names = attach(network, payload["rank"], payload["alpha"])
    if names != payload["target_modules"]:
        raise ValueError("Adapter target modules differ from pinned model")
    expected = adapter_state(network)
    if set(expected) != set(payload["state"]):
        raise ValueError("Adapter parameter key mismatch")
    for name, tensor in payload["state"].items():
        if tensor.shape != expected[name].shape:
            raise ValueError("Adapter parameter shape mismatch")
    network.load_state_dict(payload["state"], strict=False)
    network.eval().requires_grad_(False)
