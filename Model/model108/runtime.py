"""Pinned DiffusionPen runtime shared by the pretrained and fine-tuned ablations."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import sha256, absolute


def load_models(device="cuda:0"):
    import torch
    from transformers import CanineModel, CanineTokenizer
    from diffusers import AutoencoderKL, DDIMScheduler
    assets = ROOT / "checkpoints/model108"
    ledger = json.loads((assets / "asset_manifest.json").read_text())
    for name, entry in ledger["files"].items():
        if sha256(absolute(name)) != entry["sha256"]:
            raise ValueError(f"Generator asset hash mismatch: {name}")
    sys.path.insert(0, str(assets / "code"))
    from unet import UNetModel
    from feature_extractor import ImageEncoder
    tokenizer = CanineTokenizer.from_pretrained(assets / "canine", local_files_only=True)
    encoder = torch.nn.DataParallel(CanineModel.from_pretrained(assets / "canine", local_files_only=True))
    network = UNetModel(image_size=(64, 256), in_channels=4, model_channels=320, out_channels=4,
                        num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=4,
                        num_classes=339, context_dim=320, vocab_size=78, text_encoder=encoder,
                        args=SimpleNamespace(interpolation=False, mix_rate=None))
    checkpoint = torch.load(assets / "diffusionpen/diffusionpen_iam_model_path/models/ema_ckpt.pt", map_location="cpu", weights_only=True)
    network.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint.items()}, strict=True)
    del checkpoint
    network.eval().requires_grad_(False).to(device)
    style = ImageEncoder(model_name="mobilenetv2_100", num_classes=0, pretrained=False, trainable=False)
    weights = torch.load(assets / "diffusionpen/style_models/iam_style_diffusionpen.pth", map_location="cpu", weights_only=True)
    expected = style.state_dict()
    subset = {k: v for k, v in weights.items() if k in expected and expected[k].shape == v.shape}
    style.load_state_dict(subset, strict=True)
    style.eval().to(device)
    vae = AutoencoderKL.from_pretrained(assets / "sd15", subfolder="vae", local_files_only=True).eval().requires_grad_(False).to(device)
    scheduler = DDIMScheduler.from_pretrained(assets / "sd15", subfolder="scheduler", local_files_only=True)
    return network, style, vae, scheduler, tokenizer


def image_tensor(path):
    import torch
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(absolute(path)) as handle:
        image = ImageOps.pad(handle.convert("RGB"), (256, 64), method=Image.Resampling.LANCZOS, color="white")
        array = np.asarray(image).copy().astype(np.float32) / 127.5 - 1
    return torch.from_numpy(array).permute(2, 0, 1)
