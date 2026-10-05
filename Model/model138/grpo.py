#!/usr/bin/env python3
"""Model138: GRPO fine-tuning against the competition metric itself.

Ordinary fine-tuning maximizes token likelihood; the challenge scores weighted
word and character edits, where one word edit costs about 4.6 character edits.
GRPO closes that gap: for each training line sample a group of transcriptions
from the current policy, score every sample with the exact competition cost,
turn the within-group spread into advantages, and push probability toward the
samples that scored best. A forum participant at "slightly above 0.91" reports
using GRPO, and it is the only technique in this challenge we have not tried.

Start from a supervised checkpoint (model122/train.py with save_model true) and
train a fresh LoRA on top, so the SFT behaviour is preserved and memory stays
modest.

  python model138/grpo.py --config model138/config_grpo.json --fold 0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model122"))

from road_htr.metrics import levenshtein  # noqa: E402
from road_htr.submission import clean_prediction  # noqa: E402
from train import (Collator, LineDataset, absolute, load_rows, predict,  # noqa: E402
                   prompt_text, score, seed_everything, write_preds)


def row_cost(reference: str, hypothesis: str) -> float:
    """Exactly the challenge's per-row weighted cost (lower is better)."""
    ref_words, hyp_words = reference.split(), hypothesis.split()
    word = levenshtein(ref_words, hyp_words) * (len(ref_words) ** 0.5)
    char = levenshtein(tuple(reference), tuple(hypothesis)) * (len(reference) ** 0.5)
    return word / 12.0 + char / 55.0


def build_policy(cfg, device):
    """Load the supervised checkpoint and attach a trainable LoRA."""
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    base = str(absolute(str(cfg["sft_model"]).format(fold=cfg["_fold"])))
    processor = AutoProcessor.from_pretrained(base, **cfg.get("processor_kwargs",
                                                             {"fix_mistral_regex": True}))
    model_class = getattr(transformers, cfg.get("model_class",
                                                "LightOnOcrForConditionalGeneration"))
    model = model_class.from_pretrained(base, dtype=torch.bfloat16)
    model.config.use_cache = True
    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0),
        bias="none", target_modules=lora["target_modules"], task_type="CAUSAL_LM"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.to(device)
    return model, processor


def sequence_logprob(model, batch, sequences, prompt_len, eos_id):
    """Sum log p(token) over the generated continuation, per sequence.

    The mask covers the generated tokens up to and including the first EOS;
    everything generate() padded after that must not enter the objective.
    """
    inputs = {k: v for k, v in batch.items() if k != "ids"}
    inputs["input_ids"] = sequences
    inputs["attention_mask"] = torch.ones_like(sequences)
    logits = model(**inputs).logits[:, :-1]
    targets = sequences[:, 1:]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    mask = torch.zeros_like(token_logprobs)
    mask[:, prompt_len - 1:] = 1.0
    if eos_id is not None:
        # The chat template puts <|im_end|> inside the PROMPT too, so the EOS
        # search must start at the first generated token - otherwise the
        # prompt's own EOS zeroes the whole mask and the loss is exactly 0.
        is_eos = (targets == eos_id).clone()
        is_eos[:, : prompt_len - 1] = False
        after_eos = is_eos.cumsum(dim=1) - is_eos.int()  # 0 up to and including EOS
        mask = mask * (after_eos == 0).float()
    return (token_logprobs * mask).sum(dim=1), mask.sum(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    cfg["_fold"] = args.fold
    seed_everything(int(cfg.get("seed", 20260919)))
    device = torch.device("cuda")
    out_dir = absolute(cfg["output_dir"]) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    train_rows = load_rows(cfg, args.fold, "train")
    val_rows = load_rows(cfg, args.fold, "val")
    if args.eval_limit:
        val_rows = val_rows[: args.eval_limit]
    model, processor = build_policy(cfg, device)

    group = int(cfg.get("group_size", 6))
    loader = DataLoader(
        LineDataset(train_rows, absolute(cfg["image_dir"]), cfg, train=False),
        batch_size=1, shuffle=True, num_workers=int(cfg.get("workers", 2)),
        collate_fn=Collator(processor, cfg, with_labels=False))
    targets = {row["ID"]: row["Target"] for row in train_rows}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["lr"]), weight_decay=0.0, betas=(0.9, 0.99))
    accum = int(cfg.get("grad_accum", 8))
    total_steps = args.max_steps or int(cfg.get("max_steps", 300))
    history, step, started = [], 0, time.time()

    baseline = None
    if val_rows and cfg.get("eval_before", True):
        preds = predict(model, processor, val_rows, cfg, device, beams=1, log_prefix="sft")
        baseline = score(val_rows, preds)
        baseline.update(stage="sft_start", step=0)
        history.append(baseline)
        print(json.dumps(baseline), flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro = groups_seen = groups_skipped = 0
    for batch in loader:
        identifier = batch.pop("ids")[0]
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        batch["pixel_values"] = batch["pixel_values"].to(next(model.parameters()).dtype)
        prompt_len = batch["input_ids"].shape[1]
        # no_grad, not inference_mode: the sampled ids are fed back through the
        # policy for their log-probabilities, and inference-mode tensors cannot
        # take part in autograd.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            sampled = model.generate(
                **batch, max_new_tokens=int(cfg.get("max_new_tokens", 56)),
                do_sample=True, temperature=float(cfg.get("temperature", 0.9)),
                top_p=float(cfg.get("top_p", 0.95)), num_return_sequences=group)
        texts = [clean_prediction(t) for t in
                 processor.batch_decode(sampled[:, prompt_len:], skip_special_tokens=True)]
        costs = torch.tensor([row_cost(targets[identifier], t) for t in texts], device=device)
        groups_seen += 1
        if float(costs.std()) < 1e-6:
            groups_skipped += 1  # every sample scored the same: no signal here
            continue
        advantages = (costs.mean() - costs) / (costs.std() + 1e-6)

        expanded = {k: (v.expand(group, *v.shape[1:]) if torch.is_tensor(v) and v.shape[0] == 1
                        else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logprobs, lengths = sequence_logprob(model, expanded, sampled.detach(),
                                                 prompt_len, processor.tokenizer.eos_token_id)
        loss = -(advantages * (logprobs / lengths.clamp(min=1))).mean() / accum
        loss.backward()
        micro += 1
        if micro % accum:
            continue
        trainable = [p for p in model.parameters() if p.requires_grad]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            trainable, float(cfg.get("max_grad_norm", 1.0))))
        optimizer.step()
        # Diagnostic: if the adapter never moves, the policy cannot change.
        adapter_norm = float(sum(p.detach().abs().sum() for p in trainable))
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % int(cfg.get("log_steps", 10)) == 0:
            print(f"step {step}/{total_steps} mean_cost {float(costs.mean()):.4f} "
                  f"best_in_group {float(costs.min()):.4f} "
                  f"no_signal {groups_skipped}/{groups_seen} "
                  f"gnorm {grad_norm:.3e} adapter {adapter_norm:.3e} "
                  f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}GiB "
                  f"{time.time()-started:.0f}s", flush=True)
        if step % int(cfg.get("eval_steps", 100)) == 0 or step >= total_steps:
            preds = predict(model, processor, val_rows, cfg, device, beams=1,
                            log_prefix=f"grpo{step}")
            metrics = score(val_rows, preds)
            metrics.update(stage="grpo", step=step, seconds=time.time() - started)
            history.append(metrics)
            print(json.dumps(metrics), flush=True)
            write_preds(out_dir / f"oof_grpo_step{step}.csv", preds)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            model.train()
        if step >= total_steps:
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if cfg.get("save_model", True):
        model.save_pretrained(out_dir / "adapter")
    best = max((h for h in history if h.get("stage") == "grpo"),
               key=lambda h: h["multi_score"], default=None)
    print("DONE", json.dumps({"sft_start": baseline, "best_grpo": best}), flush=True)


if __name__ == "__main__":
    main()
