#!/usr/bin/env python3
"""Model122: height-normalized LightOnOCR-2-1B fine-tuning (full or LoRA).

A self-contained PyTorch loop (no TRL) so the same file runs on the local
Windows RTX 4050 (LoRA) and on a rented Linux GPU (full fine-tuning).  Every
epoch-end checkpoint is scored with the exact competition metric on the held
out fold; OOF and test predictions are written as CSV.

  python model122/train.py --config model122/config_full.json --fold 0
  python model122/train.py --config model122/config_full.json --fold all --predict-test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))

from road_htr.metrics import competition_error  # noqa: E402
from road_htr.submission import clean_prediction  # noqa: E402
from road_htr.vlm_data import apply_variant, augment_document, read_id_file  # noqa: E402

ASSISTANT_HEADER = "<|im_start|>assistant\n"


def absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resize_line(image: Image.Image, target_height: int, max_width: int) -> Image.Image:
    """Resize to a fixed text-line height, preserving aspect ratio.

    target_height <= 0 keeps the native size (the processor still caps the
    longest edge).  This uses only the pixels themselves and is applied
    identically to every image, never as a prediction-selection signal.
    """
    if target_height <= 0:
        return image
    width = max(1, round(image.width * target_height / image.height))
    height = target_height
    if width > max_width:
        height = max(1, round(height * max_width / width))
        width = max_width
    return image.resize((width, height), Image.Resampling.BICUBIC)


def prompt_text(processor, cfg=None) -> str:
    instruction = (cfg or {}).get("prompt", "")
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


class LineDataset(Dataset):
    def __init__(self, rows, image_dir, cfg, train: bool):
        self.rows, self.image_dir, self.cfg, self.train = rows, Path(image_dir), cfg, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.image_dir / f"{row['ID']}.jpg") as handle:
            image = handle.convert("RGB")
        # "center" crops the middle band of the supplied line image, the
        # complementary view the old Model9/17/19 selectors fused with "full".
        image = apply_variant(image, self.cfg.get("image_variant", "full"))
        if self.train:
            image = augment_document(image, self.cfg.get("augmentation"))
            jitter = float(self.cfg.get("height_jitter", 0.0))
            height = int(self.cfg["target_height"])
            if jitter and height > 0:
                height = int(round(height * random.uniform(1 - jitter, 1 + jitter)))
        else:
            height = int(self.cfg["target_height"])
        image = resize_line(image, height, int(self.cfg["longest_edge"]))
        return {"id": row["ID"], "image": image, "target": row.get("Target", "")}


class Collator:
    def __init__(self, processor, cfg, with_labels: bool):
        self.processor, self.cfg, self.with_labels = processor, cfg, with_labels
        # GOT-OCR2's processor builds its own prompt and has no chat template.
        self.prompt = None if cfg.get("family") == "gotocr2" else prompt_text(processor, cfg)
        # LightOnOCR's Pixtral processor takes the cap per call; Qwen3-VL takes
        # min_pixels/max_pixels at construction, so it gets no per-call size.
        self.call_kwargs = ({} if cfg.get("family", "lighton") != "lighton"
                            else {"size": {"longest_edge": int(cfg["longest_edge"])}})
        tok = processor.tokenizer
        self.header_ids = tok.encode(ASSISTANT_HEADER, add_special_tokens=False)
        self.pad_id = tok.pad_token_id

    def _gotocr2(self, batch):
        """GOT-OCR2 has no chat template: its processor emits the whole prompt.

        Every image is resized to a fixed 1024x1024, so each prompt is exactly
        the same length and the target can simply be concatenated after it.
        """
        images = [item["image"] for item in batch]
        enc = self.processor(images, return_tensors="pt")
        tokenizer = self.processor.tokenizer
        prompt_ids = enc["input_ids"]
        prompt_len = prompt_ids.shape[1]
        if not self.with_labels:
            enc["ids"] = [item["id"] for item in batch]
            return enc
        suffix = tokenizer([item["target"] + "<|im_end|>" for item in batch],
                           add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = torch.cat([prompt_ids, suffix["input_ids"]], dim=1)
        attention = torch.cat([enc["attention_mask"], suffix["attention_mask"]], dim=1)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        labels[attention == 0] = -100
        enc["input_ids"], enc["attention_mask"], enc["labels"] = input_ids, attention, labels
        enc["ids"] = [item["id"] for item in batch]
        return enc

    def __call__(self, batch):
        if self.cfg.get("family") == "gotocr2":
            return self._gotocr2(batch)
        images = [item["image"] for item in batch]
        if self.with_labels:
            texts = [self.prompt + item["target"] + "\n<|im_end|>" for item in batch]
            self.processor.tokenizer.padding_side = "right"
        else:
            texts = [self.prompt] * len(batch)
            self.processor.tokenizer.padding_side = "left"
        enc = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True,
            **self.call_kwargs,
        )
        if self.with_labels:
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            header = torch.tensor(self.header_ids)
            n = len(self.header_ids)
            for row in range(labels.shape[0]):
                ids = enc["input_ids"][row]
                windows = ids.unfold(0, n, 1)
                hits = (windows == header).all(dim=1).nonzero()
                if len(hits) == 0:
                    raise RuntimeError("assistant header not found")
                labels[row, : int(hits[-1]) + n] = -100
            enc["labels"] = labels
        enc["ids"] = [item["id"] for item in batch]
        return enc


def read_pseudo_rows(cfg, fold):
    """Self-training rows: test images labeled by a model that never saw `fold`.

    Zindi's organizers explicitly permit automated pseudo-labeling
    (discussions/34459). The file for fold k must come from the fold-k model,
    so the held-out fold stays unseen and its score remains honest.
    """
    template = cfg.get("pseudo_csv")
    if not template:
        return []
    path = absolute(str(template).format(fold=fold))
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        column = [c for c in reader.fieldnames if c != "ID"][0]
        rows = [{"ID": row["ID"], "Target": row[column].strip(), "fold": -1}
                for row in reader if row[column].strip()]
    limit = int(cfg.get("pseudo_limit", 0))
    return rows[:limit] if limit else rows


def load_rows(cfg, fold, split):
    excluded = read_id_file(absolute(cfg["exclude_ids"]))
    if split == "test":
        with absolute(cfg["test_csv"]).open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    with absolute(cfg["train_csv"]).open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["ID"] not in excluded]
    if fold == "all":
        return rows + read_pseudo_rows(cfg, fold) if split == "train" else []
    fold = int(fold)
    if split == "train":
        return [r for r in rows if int(r["fold"]) != fold] + read_pseudo_rows(cfg, fold)
    return [r for r in rows if int(r["fold"]) == fold]


def build_model(cfg, device):
    import transformers
    from transformers import AutoProcessor

    base = str(absolute(cfg["base"]))
    processor = AutoProcessor.from_pretrained(base, **cfg.get("processor_kwargs",
                                                             {"fix_mistral_regex": True}))
    full = cfg["mode"] == "full"
    dtype = torch.float32 if full and cfg.get("master_fp32", True) else torch.bfloat16
    model_class = getattr(transformers, cfg.get("model_class",
                                                "LightOnOcrForConditionalGeneration"))
    model = model_class.from_pretrained(base, dtype=dtype)
    if cfg.get("conv3d_as_linear", False):
        patch_conv3d_as_linear(model)
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if not full:
        from peft import LoraConfig, get_peft_model

        lora = cfg["lora"]
        model = get_peft_model(model, LoraConfig(
            r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            bias="none", target_modules=lora["target_modules"], task_type="CAUSAL_LM",
        ))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        # A LoRA target list silently misses encoders with other layer names (Model141, Model143).
        vision_tags = ("vision_tower", "multi_modal_projector", "visual", "vision_model")
        n_vision = sum(1 for n, p in model.named_parameters()
                       if p.requires_grad and any(t in n for t in vision_tags))
        print(f"trainable vision tensors: {n_vision}", flush=True)
    model.to(device)
    return model, processor


def patch_conv3d_as_linear(model):
    """Run patch-embedding Conv3d layers as the equivalent matmul (Model143).

    Qwen-VL patch embeddings are a Conv3d whose kernel equals its input patch,
    and bf16 Conv3d under torch 2.9 + cuDNN is pathologically slow (>10x). The
    output is identical: one dot product per output channel per patch.
    """
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv3d):
            continue
        if (tuple(module.stride) != tuple(module.kernel_size) or any(module.padding)
                or module.groups != 1):
            continue

        def forward(x, conv=module):
            if tuple(x.shape[2:]) != tuple(conv.kernel_size):
                return torch.nn.Conv3d.forward(conv, x)
            w = conv.weight.reshape(conv.out_channels, -1)
            out = torch.nn.functional.linear(x.reshape(x.shape[0], -1).to(w.dtype), w, conv.bias)
            return out.view(x.shape[0], conv.out_channels, 1, 1, 1)

        module.forward = forward
        print(f"conv3d_as_linear: {name}", flush=True)


def param_groups(model, cfg):
    vision_lr = float(cfg.get("vision_lr", cfg["lr"]))
    decay, no_decay, vision = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(tag in name for tag in ("vision_tower", "multi_modal_projector", "visual")):
            vision.append(p)
        elif p.ndim < 2 or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    wd = float(cfg.get("weight_decay", 0.01))
    groups = [
        {"params": decay, "weight_decay": wd, "lr": float(cfg["lr"])},
        {"params": no_decay, "weight_decay": 0.0, "lr": float(cfg["lr"])},
        {"params": vision, "weight_decay": wd, "lr": vision_lr},
    ]
    return [g for g in groups if g["params"]]


@torch.inference_mode()
def predict(model, processor, rows, cfg, device, beams: int, log_prefix: str):
    ds = LineDataset(rows, absolute(cfg["image_dir"]), cfg, train=False)
    loader = DataLoader(ds, batch_size=int(cfg.get("eval_batch_size", 8)), shuffle=False,
                        num_workers=int(cfg.get("workers", 2)),
                        collate_fn=Collator(processor, cfg, with_labels=False))
    was_training = model.training
    model.eval()
    model.config.use_cache = True
    out = {}
    started = time.time()
    for step, batch in enumerate(loader):
        ids = batch.pop("ids")
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["pixel_values"] = batch["pixel_values"].to(next(model.parameters()).dtype)
        # GOT-OCR2's configured EOS is <|endoftext|>, but its chat format - and
        # therefore the training targets - end with <|im_end|>. Without this the
        # model never stops and every line runs to max_new_tokens.
        stop_id = (processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
                   if cfg.get("family") == "gotocr2" else None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(
                **batch, max_new_tokens=int(cfg.get("max_new_tokens", 96)), do_sample=False,
                num_beams=beams, repetition_penalty=float(cfg.get("repetition_penalty", 1.0)),
                early_stopping=beams > 1,
                **({"eos_token_id": stop_id} if stop_id is not None else {}),
            )
        new = gen[:, batch["input_ids"].shape[1]:]
        for identifier, text in zip(ids, processor.batch_decode(new, skip_special_tokens=True)):
            out[identifier] = clean_prediction(text)
        if step % 20 == 0:
            print(f"{log_prefix} predict {len(out)}/{len(ds)} {time.time()-started:.0f}s", flush=True)
    model.config.use_cache = False
    if was_training:
        model.train()
    return out


def score(rows, preds):
    refs = [r["Target"] for r in rows]
    hyps = [preds.get(r["ID"], "") for r in rows]
    return competition_error(refs, hyps)


def write_preds(path: Path, preds: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "prediction"])
        for key, value in preds.items():
            writer.writerow([key, value])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True, help="0-4 or all")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="smoke-test limit")
    parser.add_argument("--eval-limit", type=int, default=0)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    seed_everything(int(cfg.get("seed", 20260918)))
    device = torch.device("cuda")
    out_dir = absolute(cfg["output_dir"]) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    train_rows = load_rows(cfg, args.fold, "train")
    val_rows = load_rows(cfg, args.fold, "val")
    if args.eval_limit:
        val_rows = val_rows[: args.eval_limit]
    pseudo = sum(1 for row in train_rows if str(row.get("fold")) == "-1")
    print(f"fold {args.fold}: train {len(train_rows)} rows ({pseudo} pseudo-labeled), "
          f"val {len(val_rows)} rows", flush=True)
    model, processor = build_model(cfg, device)
    train_loader = DataLoader(
        LineDataset(train_rows, absolute(cfg["image_dir"]), cfg, train=True),
        batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True,
        num_workers=int(cfg.get("workers", 2)), persistent_workers=int(cfg.get("workers", 2)) > 0,
        collate_fn=Collator(processor, cfg, with_labels=True),
    )
    accum = int(cfg.get("grad_accum", 1))
    epochs = int(cfg["epochs"])
    total_steps = math.ceil(len(train_loader) / accum) * epochs
    if args.max_steps:
        total_steps = args.max_steps
    warmup = int(total_steps * float(cfg.get("warmup_ratio", 0.05)))
    optimizer = torch.optim.AdamW(param_groups(model, cfg), betas=(0.9, 0.98), eps=1e-8,
                                  fused=True)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def set_lr(step):
        if step < warmup:
            factor = (step + 1) / max(1, warmup)
        else:
            progress = (step - warmup) / max(1, total_steps - warmup)
            floor = float(cfg.get("min_lr_ratio", 0.05))
            factor = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr * factor

    eval_epochs = set(cfg.get("eval_epochs", [epochs]))
    history = []
    best = {"multi_score": -math.inf}
    step = 0
    started = time.time()
    model.train()
    done = False
    for epoch in range(1, epochs + 1):
        running, count = 0.0, 0
        for micro, batch in enumerate(train_loader):
            batch.pop("ids")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch["pixel_values"] = batch["pixel_values"].to(next(model.parameters()).dtype)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / accum
            loss.backward()
            running += loss.item() * accum
            count += 1
            if (micro + 1) % accum:
                continue
            set_lr(step)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           float(cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % int(cfg.get("log_steps", 20)) == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"epoch {epoch} step {step}/{total_steps} loss {running/count:.4f} "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} mem {mem:.2f}GiB "
                      f"{time.time()-started:.0f}s", flush=True)
                running, count = 0.0, 0
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        if val_rows and (epoch in eval_epochs or done):
            preds = predict(model, processor, val_rows, cfg, device, beams=1, log_prefix=f"ep{epoch}")
            metrics = score(val_rows, preds)
            metrics.update(epoch=epoch, step=step, seconds=time.time() - started, beams=1)
            history.append(metrics)
            print(json.dumps(metrics), flush=True)
            write_preds(out_dir / f"oof_ep{epoch}_greedy.csv", preds)
            if metrics["multi_score"] > best["multi_score"]:
                best = metrics
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if done:
            break

    # The final epoch is the deployment checkpoint (epochs are fixed in the
    # config, never chosen per fold), so beam decoding is run on it.
    final_beams = int(cfg.get("final_beams", 1))
    if val_rows and final_beams > 1:
        preds = predict(model, processor, val_rows, cfg, device, beams=final_beams, log_prefix="final")
        metrics = score(val_rows, preds)
        metrics.update(epoch=epoch, step=step, beams=final_beams, seconds=time.time() - started)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        write_preds(out_dir / f"oof_final_beam{final_beams}.csv", preds)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if args.predict_test:
        test_rows = load_rows(cfg, args.fold, "test")
        preds = predict(model, processor, test_rows, cfg, device, beams=max(1, final_beams), log_prefix="test")
        write_preds(out_dir / f"test_beam{max(1, final_beams)}.csv", preds)
    if cfg.get("save_model", False):
        model.save_pretrained(out_dir / "model")
        processor.save_pretrained(out_dir / "model")
    (out_dir / "DONE").write_text(json.dumps({"best": best, "seconds": time.time() - started}), encoding="utf-8")
    print("DONE", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
