"""Local-only Hugging Face datasets for vision-language fine-tuning."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from .preprocess import center_band, enhance_document
from .schema import ImageIndex, read_table


DEFAULT_PROMPT = (
    "Transcribe only the complete handwritten text on the central main line. "
    "Ignore partial text from lines above or below. Return the transcription "
    "only, preserving the original capitalization, spelling, spaces, and punctuation."
)


def read_id_file(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    result: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            result.add(value)
    return result


def apply_variant(image: Any, variant: str):
    image = image.convert("RGB")
    if variant == "full":
        return image
    if variant == "center":
        return center_band(image)
    if variant == "enhanced":
        return enhance_document(image)
    if variant == "center_enhanced":
        return enhance_document(center_band(image))
    raise ValueError(f"unknown image variant {variant!r}")


def augment_document(image: Any, config: dict[str, Any] | None):
    """Apply mild, label-preserving line-image augmentation.

    Randomness comes from the DataLoader worker seed.  The defaults are kept
    deliberately small because the supplied crops are already tightly framed.
    """

    if not config or not bool(config.get("enabled", False)):
        return image
    from PIL import Image, ImageEnhance, ImageFilter

    probability = float(config.get("probability", 1.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("augmentation.probability must be between 0 and 1")
    if random.random() > probability:
        return image

    brightness = float(config.get("brightness", 0.0))
    contrast = float(config.get("contrast", 0.0))
    rotation = float(config.get("rotation_degrees", 0.0))
    translation = float(config.get("translate_y_fraction", 0.0))
    blur_probability = float(config.get("blur_probability", 0.0))
    blur_radius = float(config.get("blur_radius", 0.0))
    for name, value, maximum in (
        ("brightness", brightness, 0.5),
        ("contrast", contrast, 0.5),
        ("rotation_degrees", rotation, 3.0),
        ("translate_y_fraction", translation, 0.1),
        ("blur_probability", blur_probability, 1.0),
        ("blur_radius", blur_radius, 2.0),
    ):
        if not 0.0 <= value <= maximum:
            raise ValueError(f"augmentation.{name} must be between 0 and {maximum}")

    if brightness:
        image = ImageEnhance.Brightness(image).enhance(
            1.0 + random.uniform(-brightness, brightness)
        )
    if contrast:
        image = ImageEnhance.Contrast(image).enhance(
            1.0 + random.uniform(-contrast, contrast)
        )
    if rotation:
        image = image.rotate(
            random.uniform(-rotation, rotation),
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(255, 255, 255),
        )
    if translation:
        shift = random.uniform(-translation, translation) * image.height
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1.0, 0.0, 0.0, 0.0, 1.0, shift),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255),
        )
    if blur_radius and random.random() < blur_probability:
        image = image.filter(
            ImageFilter.GaussianBlur(random.uniform(0.1, blur_radius))
        )
    return image


def build_vlm_dataset(
    csv_path: str | Path,
    image_dir: str | Path,
    prompt: str = DEFAULT_PROMPT,
    fold: int | None = None,
    fold_mode: str = "all",
    exclude_ids: set[str] | None = None,
    image_variant: str = "full",
    augmentation: dict[str, Any] | None = None,
    synthetic_manifest: str | Path | None = None,
):
    """Build a local prompt-completion vision dataset understood by TRL.

    `fold_mode` is one of `all`, `train` (not equal to fold), or `validation`
    (equal to fold). No rows or images are uploaded.
    """

    from datasets import Dataset, Image as DatasetImage

    table = read_table(csv_path, require_target=True)
    assert table.target_column is not None
    if fold_mode not in {"all", "train", "validation"}:
        raise ValueError("fold_mode must be all, train, or validation")
    if fold_mode != "all" and (fold is None or "fold" not in table.fieldnames):
        raise ValueError("a fold column and fold number are required for CV selection")

    image_index = ImageIndex(image_dir)
    excluded = exclude_ids or set()
    payload: dict[str, list[Any]] = {
        "id": [],
        "image": [],
        "prompt": [],
        "completion": [],
    }
    for row in table.rows:
        identifier = row[table.id_column].strip()
        if identifier in excluded:
            continue
        if fold_mode != "all":
            row_fold = int(row["fold"])
            if fold_mode == "train" and row_fold == fold:
                continue
            if fold_mode == "validation" and row_fold != fold:
                continue
        payload["id"].append(identifier)
        payload["image"].append(str(image_index.resolve(identifier)))
        payload["prompt"].append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )
        payload["completion"].append(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": row[table.target_column]},
                    ],
                }
            ]
        )

    if synthetic_manifest is not None:
        if fold_mode == "validation":
            raise ValueError("Synthetic images must never enter validation")
        from .stackmix import absolute, validate_synthetic
        real_ids = set(payload["id"])
        eligible = {row[table.id_column]: row for row in table.rows
                    if row[table.id_column] in real_ids}
        synthetic = validate_synthetic(synthetic_manifest, csv_path, fold, eligible,
                                       table.target_column)
        # Appending N_real/4 examples yields the preregistered 80/20 dataset mix.
        if len(synthetic) != len(payload["id"]) // 4:
            raise ValueError("Synthetic row count must be floor(real rows / 4)")
        for row in synthetic:
            payload["id"].append(row["id"])
            payload["image"].append(str(absolute(row["image"])))
            payload["prompt"].append([{"role":"user", "content":[
                {"type":"image"}, {"type":"text", "text":prompt}]}])
            payload["completion"].append([{"role":"assistant", "content":[
                {"type":"text", "text":row["target"]}]}])
    dataset = Dataset.from_dict(payload).cast_column("image", DatasetImage(decode=True))

    def transform(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        batch["image"] = [
            augment_document(apply_variant(image, image_variant), augmentation)
            for image in batch["image"]
        ]
        return batch

    dataset.set_transform(transform)
    return dataset
