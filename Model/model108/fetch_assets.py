"""Download explicit pretrained weights/code only; never snapshot an upstream dataset."""
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import sha256, save_json


def main():
    from huggingface_hub import hf_hub_download
    config = json.loads((ROOT / "model108/config.json").read_text())
    if config["status"] != "user_confirmed_rules_proceed":
        raise ValueError("Rules gate not confirmed")
    dest = ROOT / "checkpoints/model108"
    code = dest / "code"
    code.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("unet.py", "feature_extractor.py", "LICENSE"):
        url = f"https://raw.githubusercontent.com/koninik/DiffusionPen/{config['code_revision']}/{name}"
        path = code / name
        urllib.request.urlretrieve(url, path)
        files[str(path.relative_to(ROOT))] = {"sha256": sha256(path), "source": url}
    assets = [
        ("konnik/DiffusionPen", config["revision"], "diffusionpen", [
            "README.md", "diffusionpen_iam_model_path/models/ema_ckpt.pt", "style_models/iam_style_diffusionpen.pth"]),
        ("google/canine-c", config["canine_revision"], "canine", [
            "README.md", "config.json", "model.safetensors", "special_tokens_map.json", "tokenizer_config.json"]),
        ("stable-diffusion-v1-5/stable-diffusion-v1-5", config["vae_revision"], "sd15", [
            "README.md", "vae/config.json", "vae/diffusion_pytorch_model.safetensors", "scheduler/scheduler_config.json"]),
    ]
    for repo, revision, folder, names in assets:
        for name in names:
            print(f"Allowlisted asset: {repo}/{name}", flush=True)
            path = Path(hf_hub_download(repo, name, revision=revision, local_dir=dest / folder))
            files[str(path.relative_to(ROOT))] = {"sha256": sha256(path), "repository": repo, "revision": revision, "filename": name}
    save_json(dest / "asset_manifest.json", {"files": files, "licenses": {
        "code": "MIT", "diffusionpen_weights": "MIT (model card)", "canine": "Apache-2.0", "vae_scheduler": "CreativeML OpenRAIL-M"},
        "external_datasets_downloaded": False})
    print("Asset verification ledger complete", flush=True)


if __name__ == "__main__":
    main()
