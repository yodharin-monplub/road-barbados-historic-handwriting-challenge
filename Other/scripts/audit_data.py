#!/usr/bin/env python3
"""Audit image integrity and authenticated competition metadata."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.schema import ImageIndex, find_file, read_table  # noqa: E402


def percentile(values: list[int | float], q: float) -> float:
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def image_audit(index: ImageIndex) -> dict[str, object]:
    widths: list[int] = []
    heights: list[int] = []
    modes: Counter[str] = Counter()
    bad: list[dict[str, str]] = []
    for path in index.paths():
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                widths.append(image.width)
                heights.append(image.height)
                modes[image.mode] += 1
        except Exception as exc:  # Pillow exposes several decoder error types.
            bad.append({"file": path.name, "error": str(exc)})

    aspects = [w / h for w, h in zip(widths, heights, strict=True)]

    def summary(values: list[int | float]) -> dict[str, float]:
        return {
            "min": float(min(values)),
            "p05": percentile(values, 0.05),
            "median": float(statistics.median(values)),
            "p95": percentile(values, 0.95),
            "max": float(max(values)),
        }

    return {
        "count": len(index),
        "bad_count": len(bad),
        "bad": bad,
        "modes": dict(sorted(modes.items())),
        "width": summary(widths),
        "height": summary(heights),
        "aspect_ratio": summary(aspects),
    }


def table_audit(path: Path, index: ImageIndex, require_target: bool) -> dict[str, object]:
    table = read_table(path, require_target=require_target)
    missing_images: list[str] = []
    image_names: set[str] = set()
    for row in table.rows:
        identifier = row[table.id_column].strip()
        try:
            image_names.add(index.resolve(identifier).name)
        except FileNotFoundError:
            missing_images.append(identifier)

    result: dict[str, object] = {
        "path": str(path),
        "rows": len(table.rows),
        "columns": list(table.fieldnames),
        "id_column": table.id_column,
        "target_column": table.target_column,
        "missing_image_count": len(missing_images),
        "missing_image_ids": missing_images[:50],
        "resolved_images": sorted(image_names),
    }
    if require_target and table.target_column:
        targets = [row[table.target_column] for row in table.rows]
        char_lengths = [len(value) for value in targets]
        word_lengths = [len(value.split()) for value in targets]
        result.update(
            {
                "character_length": {
                    "min": min(char_lengths),
                    "median": statistics.median(char_lengths),
                    "p95": percentile(char_lengths, 0.95),
                    "max": max(char_lengths),
                },
                "word_length": {
                    "min": min(word_lengths),
                    "median": statistics.median(word_lengths),
                    "p95": percentile(word_lengths, 0.95),
                    "max": max(word_lengths),
                },
                "characters": "".join(sorted(set("".join(targets)))),
                "unique_targets": len(set(targets)),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    image_dir = args.raw_dir / "images"
    index = ImageIndex(image_dir)
    report: dict[str, object] = {
        "raw_dir": str(args.raw_dir.resolve()),
        "images": image_audit(index),
        "metadata": {},
    }

    metadata: dict[str, object] = report["metadata"]  # type: ignore[assignment]
    specs = {
        "train": (("Train.csv", "train.csv"), True),
        "test": (("Test.csv", "test.csv"), False),
        "sample_submission": (
            ("SampleSubmission.csv", "sample_submission.csv"),
            False,
        ),
    }
    resolved_all: set[str] = set()
    for key, (names, require_target) in specs.items():
        try:
            path = find_file(args.raw_dir, names)
        except FileNotFoundError as exc:
            metadata[key] = {"status": "missing", "message": str(exc)}
            continue
        item = table_audit(path, index, require_target)
        resolved_all.update(item.pop("resolved_images"))  # type: ignore[arg-type]
        metadata[key] = item

    unreferenced = sorted(
        path.name for path in index.paths() if path.name not in resolved_all
    )
    report["images_not_referenced_by_train_or_test_count"] = len(unreferenced)
    report["images_not_referenced_by_train_or_test_examples"] = unreferenced[:50]
    report["referenced_image_count"] = len(resolved_all)

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
