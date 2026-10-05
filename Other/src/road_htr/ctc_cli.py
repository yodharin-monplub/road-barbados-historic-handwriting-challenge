"""Train and infer the custom CRNN-CTC handwriting recognizer."""

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
from typing import Any

import yaml
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .ctc_decode import CharNGramLM, greedy_decode, prefix_beam_search
from .ctc_model import CRNN
from .metrics import competition_error
from .preprocess import center_band, load_rgb
from .schema import ImageIndex, read_table
from .submission import clean_prediction, write_submission
from .vlm_data import read_id_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    return config


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_labeled_rows(config: dict[str, Any], mode: str, fold: int | None):
    table = read_table(absolute(config["train_csv"]), require_target=True)
    if mode != "all" and (fold is None or "fold" not in table.fieldnames):
        raise ValueError("fold selection requires a fold column and fold number")
    excluded = read_id_file(absolute(config["exclude_ids"])) if config.get("exclude_ids") else set()
    assert table.target_column is not None
    rows: list[tuple[str, str]] = []
    for row in table.rows:
        identifier = row[table.id_column].strip()
        if identifier in excluded:
            continue
        if mode == "train" and int(row["fold"]) == fold:
            continue
        if mode == "validation" and int(row["fold"]) != fold:
            continue
        rows.append((identifier, row[table.target_column]))
    return rows, excluded


class CharacterVocabulary:
    def __init__(self, characters: list[str]):
        self.characters = characters
        self.to_index = {character: index + 1 for index, character in enumerate(characters)}

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharacterVocabulary":
        return cls(sorted(set("".join(texts))))

    def encode(self, text: str) -> list[int]:
        try:
            return [self.to_index[character] for character in text]
        except KeyError as exc:
            raise ValueError(f"character absent from training vocabulary: {exc.args[0]!r}") from exc

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"blank": 0, "characters": self.characters}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterVocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["characters"])

    def __len__(self) -> int:
        return len(self.characters) + 1


def _augment(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.7:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.65, 1.45))
    if rng.random() < 0.6:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.9)))
    if rng.random() < 0.2:
        image = ImageOps.grayscale(image).convert("RGB")
    if rng.random() < 0.6:
        image = image.rotate(
            rng.uniform(-1.5, 1.5),
            Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255),
        )
    return image


def _line_mix(first: Image.Image, second: Image.Image, gap: int = 12) -> Image.Image:
    height = max(first.height, second.height)
    mixed = Image.new("RGB", (first.width + gap + second.width, height), (255, 255, 255))
    mixed.paste(first, (0, (height - first.height) // 2))
    mixed.paste(second, (first.width + gap, (height - second.height) // 2))
    return mixed


def _fit_line(image: Image.Image, height: int, max_width: int) -> Image.Image:
    scale = min(height / image.height, max_width / image.width)
    width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize((width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(resized, (0, (height - resized_height) // 2))
    return canvas


class LineDataset:
    def __init__(
        self,
        rows: list[tuple[str, str | None]],
        image_index: ImageIndex,
        vocabulary: CharacterVocabulary | None,
        height: int,
        max_width: int,
        center_fraction: float,
        training: bool,
        line_mix_probability: float,
        seed: int,
    ):
        self.rows = rows
        self.image_index = image_index
        self.vocabulary = vocabulary
        self.height = height
        self.max_width = max_width
        self.center_fraction = center_fraction
        self.training = training
        self.line_mix_probability = line_mix_probability
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        from torchvision.transforms.functional import pil_to_tensor

        identifier, target = self.rows[index]
        rng = random.Random(self.seed + self.epoch * len(self.rows) + index)
        image = load_rgb(self.image_index.resolve(identifier))
        if self.center_fraction < 1:
            image = center_band(image, self.center_fraction)
        if self.training and rng.random() < self.line_mix_probability:
            second_index = rng.randrange(len(self.rows))
            second_id, second_target = self.rows[second_index]
            second = load_rgb(self.image_index.resolve(second_id))
            if self.center_fraction < 1:
                second = center_band(second, self.center_fraction)
            image = _line_mix(image, second)
            target = f"{target} {second_target}"
        if self.training:
            image = _augment(image, rng)
        image = _fit_line(image, self.height, self.max_width)
        tensor = pil_to_tensor(image).float().div_(255).sub_(0.5).div_(0.5)
        encoded = self.vocabulary.encode(target) if self.vocabulary and target is not None else []
        return {
            "image": tensor,
            "width": tensor.shape[-1],
            "encoded": encoded,
            "id": identifier,
            "target": target,
        }


def collate_lines(items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch_width = max(item["width"] for item in items)
    batch_width = max(16, int(math.ceil(batch_width / 4) * 4))
    height = items[0]["image"].shape[-2]
    images = torch.ones((len(items), 3, height, batch_width), dtype=torch.float32)
    widths = torch.tensor([item["width"] for item in items], dtype=torch.long)
    encoded_parts = [torch.tensor(item["encoded"], dtype=torch.long) for item in items]
    target_lengths = torch.tensor([len(part) for part in encoded_parts], dtype=torch.long)
    for position, item in enumerate(items):
        images[position, :, :, : item["width"]] = item["image"]
    targets = torch.cat(encoded_parts) if sum(target_lengths).item() else torch.empty(0, dtype=torch.long)
    return {
        "images": images,
        "widths": widths,
        "targets": targets,
        "target_lengths": target_lengths,
        "ids": [item["id"] for item in items],
        "texts": [item["target"] for item in items],
    }


def _validation_predictions(model, loader, vocabulary, device, use_amp: bool):
    import torch

    predictions: list[str] = []
    references: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                logits = model(images)
            indices = logits.argmax(-1).cpu()
            lengths = CRNN.output_lengths(batch["widths"])
            predictions.extend(
                greedy_decode(row[:length], vocabulary.characters)
                for row, length in zip(indices, lengths, strict=True)
            )
            references.extend(batch["texts"])
    return references, predictions


def train(config_path: str | Path, fold_override: int | None = None) -> None:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    config = load_config(config_path)
    seed = int(config.get("seed", 20260903))
    set_seed(seed)
    fold_value = fold_override if fold_override is not None else config.get("validation_fold")
    fold = int(fold_value) if fold_value is not None else None
    train_rows, excluded = select_labeled_rows(config, "train" if fold is not None else "all", fold)
    validation_rows = select_labeled_rows(config, "validation", fold)[0] if fold is not None else []
    # Character inventory may use the complete provided Train.csv. This avoids a
    # fold crashing on a rare symbol while the language model still sees only
    # the training portion of that fold.
    all_rows = select_labeled_rows(config, "all", None)[0]
    vocabulary = CharacterVocabulary.from_texts([target for _, target in all_rows])

    image_index = ImageIndex(absolute(config["image_dir"]))
    height = int(config.get("image_height", 96))
    max_width = int(config.get("max_width", 2048))
    center_fraction = float(config.get("center_fraction", 0.84))
    augmentation = config.get("augmentation", {})
    train_dataset = LineDataset(
        [(identifier, target) for identifier, target in train_rows],
        image_index,
        vocabulary,
        height,
        max_width,
        center_fraction,
        True,
        float(augmentation.get("line_mix_probability", 0.25)),
        seed,
    )
    validation_dataset = LineDataset(
        [(identifier, target) for identifier, target in validation_rows],
        image_index,
        vocabulary,
        height,
        max_width,
        center_fraction,
        False,
        0.0,
        seed,
    )
    training = config.get("training", {})
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(training.get("workers", 4)),
        pin_memory=True,
        collate_fn=collate_lines,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training.get("eval_batch_size", 12)),
        shuffle=False,
        num_workers=int(training.get("workers", 4)),
        pin_memory=True,
        collate_fn=collate_lines,
    )

    architecture = config.get("architecture", {})
    model = CRNN(
        len(vocabulary),
        hidden_size=int(architecture.get("hidden_size", 384)),
        lstm_layers=int(architecture.get("lstm_layers", 3)),
        dropout=float(architecture.get("dropout", 0.2)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.0008)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    epochs = int(training.get("epochs", 80))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(training.get("min_learning_rate", 1e-6))
    )
    criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    accumulation = int(training.get("gradient_accumulation", 1))
    clip_norm = float(training.get("clip_norm", 5.0))
    use_amp = bool(training.get("bf16", True) and device.type == "cuda")

    output_template = str(config.get("output_dir", "outputs/model4/fold{fold}"))
    output_dir = absolute(output_template.format(fold=fold if fold is not None else "all"))
    output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary.save(output_dir / "vocabulary.json")
    lm_config = config.get("decoder", {})
    CharNGramLM(
        int(lm_config.get("lm_order", 5)), float(lm_config.get("lm_smoothing", 0.2))
    ).fit(target for _, target in train_rows).save(output_dir / "char_lm.json")
    manifest = {
        "config": config,
        "fold": fold,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "excluded_ids": sorted(excluded),
        "characters": vocabulary.characters,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    best_error = float("inf")
    patience = 0
    max_patience = int(training.get("patience", 12))
    log_path = output_dir / "training_log.jsonl"
    for epoch in range(epochs):
        start = time.monotonic()
        train_dataset.epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            input_lengths = CRNN.output_lengths(batch["widths"]).to(device)
            if torch.any(target_lengths > input_lengths):
                raise ValueError("target is longer than available CTC timesteps")
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                loss = loss / accumulation
            loss.backward()
            running_loss += float(loss.detach()) * accumulation
            if step % accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(len(train_loader), 1),
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.monotonic() - start,
        }
        if validation_rows:
            references, predictions = _validation_predictions(
                model, validation_loader, vocabulary, device, use_amp
            )
            record.update(competition_error(references, predictions))
            current_error = record["mean_error"]
        else:
            current_error = record["train_loss"]
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

        if current_error < best_error:
            best_error = current_error
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_classes": len(vocabulary),
                    "architecture": architecture,
                    "epoch": epoch + 1,
                    "validation_error": best_error,
                },
                output_dir / "best.pt",
            )
        else:
            patience += 1
            if validation_rows and patience >= max_patience:
                print(f"early stop after {patience} non-improving epochs")
                break


def _load_model(checkpoint_dir: Path, device):
    import torch

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    architecture = checkpoint["architecture"]
    model = CRNN(
        checkpoint["num_classes"],
        hidden_size=int(architecture.get("hidden_size", 384)),
        lstm_layers=int(architecture.get("lstm_layers", 3)),
        dropout=float(architecture.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def infer(
    config_path: str | Path,
    input_csv: str | Path,
    checkpoint_dir: str | Path,
    output_csv: str | Path,
    fold: int | None = None,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> None:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    config = load_config(config_path)
    seed = int(config.get("seed", 20260903))
    set_seed(seed)
    checkpoint = absolute(checkpoint_dir)
    vocabulary = CharacterVocabulary.load(checkpoint / "vocabulary.json")
    lm = CharNGramLM.load(checkpoint / "char_lm.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint, device)

    table = read_table(absolute(input_csv), require_target=False)
    if fold is not None and "fold" not in table.fieldnames:
        raise ValueError("--fold needs an input fold column")
    selected = [row for row in table.rows if fold is None or int(row["fold"]) == fold]
    rows = [(row[table.id_column].strip(), None) for row in selected]
    dataset = LineDataset(
        rows,
        ImageIndex(absolute(config["image_dir"])),
        None,
        int(config.get("image_height", 96)),
        int(config.get("max_width", 2048)),
        float(config.get("center_fraction", 0.84)),
        False,
        0.0,
        seed,
    )
    decoder = config.get("decoder", {})
    loader = DataLoader(
        dataset,
        batch_size=int(decoder.get("batch_size", 12)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("workers", 4)),
        collate_fn=collate_lines,
        pin_memory=True,
    )
    mode = decoder.get("mode", "beam")
    use_amp = bool(config.get("training", {}).get("bf16", True) and device.type == "cuda")

    output = absolute(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail = output.with_suffix(".jsonl")
    predictions: dict[str, str] = {}
    with output.open("w", encoding="utf-8", newline="") as csv_handle, detail.open(
        "w", encoding="utf-8"
    ) as detail_handle, torch.inference_mode():
        writer = csv.DictWriter(csv_handle, fieldnames=[table.id_column, "prediction"])
        writer.writeheader()
        completed = 0
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                logits = model(images)
            log_probs = F.log_softmax(logits.float(), dim=-1).cpu()
            lengths = CRNN.output_lengths(batch["widths"])
            for position, identifier in enumerate(batch["ids"]):
                sample_probs = log_probs[position, : lengths[position]]
                if mode == "greedy":
                    candidates = [
                        (
                            greedy_decode(sample_probs.argmax(-1), vocabulary.characters),
                            float(sample_probs.max(-1).values.mean()),
                        )
                    ]
                elif mode == "beam":
                    candidates = prefix_beam_search(
                        sample_probs,
                        vocabulary.characters,
                        beam_width=int(decoder.get("beam_width", 32)),
                        token_prune=int(decoder.get("token_prune", 24)),
                        lm=lm,
                        lm_alpha=float(decoder.get("lm_alpha", 0.35)),
                        space_bonus=float(decoder.get("space_bonus", 0.0)),
                        nbest=int(decoder.get("nbest", 8)),
                    )
                else:
                    raise ValueError("decoder mode must be greedy or beam")
                cleaned = [(clean_prediction(text), score) for text, score in candidates]
                chosen = next((text for text, _ in cleaned if text), "")
                predictions[identifier] = chosen
                writer.writerow({table.id_column: identifier, "prediction": chosen})
                detail_handle.write(
                    json.dumps(
                        {
                            "id": identifier,
                            "candidates": [text for text, _ in cleaned],
                            "scores": [score for _, score in cleaned],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            completed += len(batch["ids"])
            print(f"decoded {completed}/{len(dataset)}", flush=True)

    if submission_path:
        if not sample_path:
            raise ValueError("--sample is required with --submission")
        write_submission(absolute(sample_path), absolute(submission_path), predictions)


def train_main(default_config: str | Path | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    train(args.config, args.fold)


def infer_main(default_config: str | Path | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    infer(
        args.config,
        args.input,
        args.checkpoint,
        args.output,
        fold=args.fold,
        sample_path=args.sample,
        submission_path=args.submission,
    )
