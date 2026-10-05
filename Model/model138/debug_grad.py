#!/usr/bin/env python3
"""Find out why the GRPO loss produces no gradient on the LoRA weights."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model122"))
sys.path.insert(0, str(ROOT / "Model" / "model138"))

from train import Collator, LineDataset, absolute, load_rows  # noqa: E402
from grpo import build_policy, sequence_logprob  # noqa: E402

cfg = json.loads((ROOT / "model138/config_grpo.json").read_text(encoding="utf-8-sig"))
cfg["_fold"] = "0"
device = torch.device("cuda")
model, processor = build_policy(cfg, device)

trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
print("trainable tensors:", len(trainable))
print("examples:", [n for n, _ in trainable[:4]])

rows = load_rows(cfg, "0", "train")[:1]
loader = DataLoader(LineDataset(rows, absolute(cfg["image_dir"]), cfg, train=False),
                    batch_size=1, collate_fn=Collator(processor, cfg, with_labels=False))
batch = next(iter(loader))
batch.pop("ids")
batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
batch["pixel_values"] = batch["pixel_values"].to(next(model.parameters()).dtype)
prompt_len = batch["input_ids"].shape[1]

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    sampled = model.generate(**batch, max_new_tokens=24, do_sample=True,
                             temperature=1.0, num_return_sequences=2)
print("sampled:", tuple(sampled.shape), "grad_enabled:", torch.is_grad_enabled())

expanded = {k: (v.expand(2, *v.shape[1:]) if torch.is_tensor(v) and v.shape[0] == 1 else v)
            for k, v in batch.items()}
model.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    logprobs, lengths = sequence_logprob(model, expanded, sampled.detach(), prompt_len,
                                         processor.tokenizer.eos_token_id)
print("logprobs.requires_grad:", logprobs.requires_grad, "values:", logprobs.tolist())
print("lengths:", lengths.tolist())

loss = -(torch.tensor([1.0, -1.0], device=device) * (logprobs / lengths.clamp(min=1))).mean()
print("loss:", float(loss), "requires_grad:", loss.requires_grad)
loss.backward()
with_grad = [(n, float(p.grad.abs().sum())) for n, p in trainable if p.grad is not None]
print("params with grad:", len(with_grad), "of", len(trainable))
print("largest:", sorted(with_grad, key=lambda kv: -kv[1])[:4])
