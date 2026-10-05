"""Fit a small handwriting-generator adapter to strictly eligible competition crops."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "model108"))
from road_htr.stackmix import absolute, sha256, save_json, eligible_rows
from road_htr.vlm_data import read_id_file
from prepare import SUPPORTED, validate_requests


def prepare(config):
    bank_path = absolute(config["bank"])
    requests_path = absolute(config["pilot_requests"])
    bank = json.loads(bank_path.read_text())
    requests = json.loads(requests_path.read_text())
    validate_requests(requests)
    truth = ROOT / "data/processed/train_folds.csv"
    table, eligible = eligible_rows(truth, config["fold"], read_id_file(ROOT / "config/known_bad_train_ids.txt"))
    if bank["fold"] != config["fold"] or bank["lineage"]["training_excluded_folds"] != [config["fold"]]:
        raise ValueError("Generator aligner saw the OCR validation fold")
    if bank["signature"]["truth_sha256"] != sha256(truth):
        raise ValueError("Wrong crop-bank truth")
    reserved = {r["target_source_id"] for r in requests["rows"]}
    reserved.update(ref["source_id"] for row in requests["rows"] for ref in row["style_references"])
    groups = defaultdict(list)
    for word in bank["words"]:
        source = word["source_id"]
        if source in reserved:
            continue
        if source not in eligible or int(word["source_fold"]) == config["fold"]:
            raise ValueError("Generator crop belongs to a heldout/excluded line")
        a, b = word["text_span"]
        if eligible[source][table.target_column][a:b] != word["text"]:
            raise ValueError("Generator crop target mismatch")
        if sha256(absolute(word["image"])) != word["sha256"]:
            raise ValueError("Generator crop image hash mismatch")
        if 1 <= len(word["text"]) <= 18 and set(word["text"]) <= SUPPORTED:
            groups[source].append(word)
    groups = {source: words for source, words in groups.items() if len(words) >= 6}
    words = [word for source in sorted(groups) for word in groups[source]]
    if len(words) < 200:
        raise ValueError("Not enough fold-isolated generator training words")
    metadata = {"fold": config["fold"], "training_words": len(words), "training_source_lines": len(groups),
                "source_ids": sorted(groups), "reserved_pilot_source_ids": sorted(reserved),
                "truth_sha256": sha256(truth), "bank_sha256": sha256(bank_path),
                "pilot_requests_sha256": sha256(requests_path), "config": config}
    return words, groups, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "model109/config.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke_test:
        config = dict(config, steps=1, output_dir="outputs/model109/smoke")
    words, groups, metadata = prepare(config)
    output = absolute(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "preparation.json", metadata)
    print(f"Generator training: {len(words)} words, {len(groups)} eligible source lines; pilot sources excluded", flush=True)
    if args.prepare_only:
        return
    if (output / "adapter.pt").exists() or (output / "training_metrics.jsonl").exists():
        raise ValueError("Refusing to overwrite a previous generator run")
    import torch
    from runtime import load_models, image_tensor
    from lora import attach, adapter_state
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    network, style, vae, scheduler, tokenizer = load_models()
    modules = attach(network, config["rank"], config["alpha"])
    # Upstream checkpointing is not needed for a microbatch of one and its
    # custom backward can mishandle frozen inputs; use ordinary autograd.
    for module in network.modules():
        if hasattr(module, "checkpoint") and isinstance(module.checkpoint, bool):
            module.checkpoint = False
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False
    parameters = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    metadata.update(trainable_parameters=sum(p.numel() for p in parameters), target_modules=modules,
                    assets_sha256=sha256(ROOT / "checkpoints/model108/asset_manifest.json"))
    save_json(output / "run_manifest.json", metadata)
    network.train()
    network.text_encoder.eval()
    for step in range(1, config["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(config["gradient_accumulation"]):
            word = rng.choice(words)
            refs = rng.sample([w for w in groups[word["source_id"]] if w["image"] != word["image"]], 5)
            with torch.no_grad():
                pixels = image_tensor(word["image"]).unsqueeze(0).cuda()
                latent = vae.encode(pixels).latent_dist.sample() * 0.18215
                features = style(torch.stack([image_tensor(ref["image"]) for ref in refs]).cuda())
            tokens = tokenizer([word["text"]], padding="max_length", truncation=False, max_length=40, return_tensors="pt").to("cuda")
            noise = torch.randn_like(latent)
            timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device="cuda")
            noisy = scheduler.add_noise(latent, noise, timestep)
            prediction = network(noisy, timesteps=timestep, context=tokens, style_extractor=features)
            loss = torch.nn.functional.mse_loss(prediction, noise)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite generator loss")
            (loss / config["gradient_accumulation"]).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"])
        optimizer.step()
        metric = {"step": step, "loss": sum(losses) / len(losses), "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
        with (output / "training_metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metric) + "\n")
        if step % 10 == 0:
            print(json.dumps(metric), flush=True)
    path = output / "adapter.pt"
    torch.save({"rank": config["rank"], "alpha": config["alpha"], "state": adapter_state(network),
                "target_modules": modules, "assets_sha256": metadata["assets_sha256"],
                "run_manifest_sha256": sha256(output / "run_manifest.json")}, path)
    save_json(output / "complete.json", {"steps": config["steps"], "adapter_sha256": sha256(path), "competition_cv": None})
    print("Generator fine-tuning complete; OCR validation still required", flush=True)


if __name__ == "__main__":
    main()
