"""Private DiffusionPen inference; no upstream data loader or remote API inference."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import absolute, sha256, save_json
from prepare import validate_requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, default=ROOT / "outputs/model108/fold0/pilot/requests.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.requests = absolute(args.requests)
    manifest = json.loads(args.requests.read_text())
    rows = validate_requests(manifest)
    if args.limit:
        rows = rows[:args.limit]
    assets = ROOT / "checkpoints/model108"
    import torch
    from PIL import Image
    import numpy as np
    from runtime import load_models, image_tensor
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this bounded pilot")
    device = "cuda:0"
    torch.set_num_threads(2)
    network, style, vae, scheduler, tokenizer = load_models(device)
    adapter_hash = None
    if args.adapter:
        if not args.output or args.output.resolve() == args.requests.parent.resolve():
            raise ValueError("Fine-tuned generation needs a separate output directory")
        completion = json.loads((args.adapter.parent / "complete.json").read_text())
        adapter_hash = sha256(args.adapter)
        if completion["adapter_sha256"] != adapter_hash:
            raise ValueError("Generator adapter hash mismatch")
        payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
        run_path = args.adapter.parent / "run_manifest.json"
        if payload["run_manifest_sha256"] != sha256(run_path) or payload["assets_sha256"] != sha256(assets / "asset_manifest.json"):
            raise ValueError("Generator adapter provenance mismatch")
        training = json.loads(run_path.read_text())
        if training["fold"] != manifest["fold"] or training["pilot_requests_sha256"] != sha256(args.requests):
            raise ValueError("Adapter used a different fold/pilot")
        if set(training["source_ids"]) & set(training["reserved_pilot_source_ids"]):
            raise ValueError("Generator trained on reserved pilot sources")
        sys.path.insert(0, str(ROOT / "model109"))
        from lora import apply_adapter
        apply_adapter(network, payload)
    output = absolute(args.output) if args.output else args.requests.parent
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    generated = []
    with torch.inference_mode():
        for row in rows:
            tensors = [image_tensor(ref["image"]) for ref in row["style_references"]]
            style_features = style(torch.stack(tensors).to(device))
            tokens = tokenizer([row["target"]], padding="max_length", truncation=False, return_tensors="pt", max_length=40).to(device)
            generator = torch.Generator(device=device).manual_seed(row["seed"])
            latent = torch.randn((1, 4, 8, 32), generator=generator, device=device)
            scheduler.set_timesteps(50)
            for timestep in scheduler.timesteps:
                t = torch.full((1,), int(timestep), dtype=torch.long, device=device)
                residual = network(latent, timesteps=t, context=tokens, y=None, style_extractor=style_features)
                latent = scheduler.step(residual, timestep, latent).prev_sample
            pixels = (vae.decode(latent / 0.18215).sample / 2 + 0.5).clamp(0, 1)
            image = Image.fromarray((pixels[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
            path = images / f"{row['id']}.png"
            image.save(path)
            generated.append(dict(row, image=str(path.relative_to(ROOT)), sha256=sha256(path)))
            save_json(output / "generation_manifest.json", {"method": "diffusionpen", "fold": manifest["fold"],
                "requests_sha256": sha256(args.requests), "assets_sha256": sha256(assets / "asset_manifest.json"),
                "adapter_sha256": adapter_hash, "requests_path": str(args.requests.relative_to(ROOT)),
                "steps": 50, "rows": generated, "complete": len(generated) == len(manifest["rows"])})
            print(f"Generated {len(generated)}/{len(rows)}", flush=True)
    with (output / "labels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Target"])
        writer.writerows((row["id"], row["target"]) for row in generated)


if __name__ == "__main__":
    main()
