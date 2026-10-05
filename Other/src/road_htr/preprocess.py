"""Aspect-preserving image transforms shared by OCR candidates."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def center_band(image: Image.Image, fraction: float = 0.74) -> Image.Image:
    """Crop symmetric vertical context while retaining the centered target line."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1:
        return image.copy()
    target_height = max(1, round(image.height * fraction))
    top = max(0, (image.height - target_height) // 2)
    return image.crop((0, top, image.width, top + target_height))


def enhance_document(image: Image.Image, grayscale: bool = False) -> Image.Image:
    image = ImageOps.autocontrast(image, cutoff=(0.25, 0.25), preserve_tone=True)
    if grayscale:
        image = ImageOps.grayscale(image).convert("RGB")
    return image


def resize_to_height(
    image: Image.Image,
    height: int,
    max_width: int | None = None,
) -> Image.Image:
    if height <= 0:
        raise ValueError("height must be positive")
    width = max(1, round(image.width * height / image.height))
    if max_width is not None:
        width = min(width, max_width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def pad_width(
    image: Image.Image,
    width: int,
    fill: tuple[int, int, int] = (255, 255, 255),
    alignment: str = "left",
) -> Image.Image:
    if image.width > width:
        raise ValueError(f"image width {image.width} exceeds canvas width {width}")
    if alignment == "left":
        left = 0
    elif alignment == "center":
        left = (width - image.width) // 2
    else:
        raise ValueError("alignment must be 'left' or 'center'")
    canvas = Image.new("RGB", (width, image.height), fill)
    canvas.paste(image, (left, 0))
    return canvas


def letterbox(
    image: Image.Image,
    size: tuple[int, int],
    fill: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Fit an image into a (height, width) canvas without geometric distortion."""

    height, width = size
    if height <= 0 or width <= 0:
        raise ValueError("letterbox dimensions must be positive")
    scale = min(width / image.width, height / image.height)
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), fill)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas
