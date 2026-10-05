#!/usr/bin/env python3
"""Reproducible diagnostics for labels, image geometry, and weak/blank scans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_htr.schema import ImageIndex, read_table  # noqa: E402
from road_htr.vlm_data import read_id_file  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "min": min(values),
        "p05": percentile(values, 0.05),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def image_signal(path: Path) -> tuple[int, int, float, float, float]:
    """Return geometry and low-cost grayscale ink/contrast statistics."""

    with Image.open(path) as source:
        image = source.convert("L")
        width, height = image.size
        image.thumbnail((512, 128), Image.Resampling.BILINEAR)
        histogram = image.histogram()
        pixels = sum(histogram)
        ink_fraction = sum(histogram[:220]) / pixels
        mean = ImageStat.Stat(image).mean[0]
        contrast = ImageStat.Stat(image).stddev[0]
    return width, height, ink_fraction, 255 - mean, contrast


def normalized_target(value: str) -> str:
    return " ".join(value.casefold().split())


def duplicate_groups(rows, id_column: str, target_column: str) -> list[dict[str, object]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[row[target_column]].append(row[id_column].strip())
    return [
        {"target": target, "ids": identifiers}
        for target, identifiers in sorted(
            grouped.items(), key=lambda item: (-len(item[1]), item[0])
        )
        if len(identifiers) > 1
    ]


def phrase_counts(targets: list[str], size: int, suffix: bool = False):
    counter: Counter[str] = Counter()
    for target in targets:
        words = target.split()
        if len(words) >= size:
            phrase = words[-size:] if suffix else words[:size]
            counter[" ".join(phrase)] += 1
    return [{"phrase": phrase, "count": count} for phrase, count in counter.most_common(20)]


def exact_hash_groups(index: ImageIndex, identifiers: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for identifier in identifiers:
        path = index.resolve(identifier)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        grouped[digest].append(identifier)
    return {digest: ids for digest, ids in grouped.items() if len(ids) > 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/raw/Train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data/raw/Test.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/raw/images")
    parser.add_argument(
        "--suspect-ids", type=Path, default=ROOT / "config/known_bad_train_ids.txt"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/dataset_diagnostics.json")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()

    train = read_table(args.train, require_target=True)
    test = read_table(args.test)
    assert train.target_column is not None
    image_index = ImageIndex(args.images)
    train_ids = [row[train.id_column].strip() for row in train.rows]
    test_ids = [row[test.id_column].strip() for row in test.rows]
    targets = [row[train.target_column] for row in train.rows]
    target_by_id = {
        row[train.id_column].strip(): row[train.target_column] for row in train.rows
    }

    image_rows: dict[str, dict[str, object]] = {}
    for split, identifiers in (("train", train_ids), ("test", test_ids)):
        for identifier in identifiers:
            width, height, ink_fraction, darkness, contrast = image_signal(
                image_index.resolve(identifier)
            )
            image_rows[identifier] = {
                "id": identifier,
                "split": split,
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "ink_fraction_below_220": ink_fraction,
                "darkness": darkness,
                "contrast": contrast,
            }

    split_geometry: dict[str, object] = {}
    for split, identifiers in (("train", train_ids), ("test", test_ids)):
        selected = [image_rows[identifier] for identifier in identifiers]
        split_geometry[split] = {
            key: numeric_summary([float(row[key]) for row in selected])
            for key in (
                "width",
                "height",
                "aspect_ratio",
                "ink_fraction_below_220",
                "darkness",
                "contrast",
            )
        }

    character_counts = Counter("".join(targets))
    character_examples: dict[str, list[str]] = defaultdict(list)
    for row in train.rows:
        identifier = row[train.id_column].strip()
        for character in set(row[train.target_column]):
            if len(character_examples[character]) < 5:
                character_examples[character].append(identifier)
    rare_characters = [
        {
            "character": character,
            "codepoint": f"U+{ord(character):04X}",
            "count": count,
            "example_ids": character_examples[character],
        }
        for character, count in sorted(character_counts.items(), key=lambda item: (item[1], item[0]))
        if count <= 20
    ]

    normalized: dict[str, list[str]] = defaultdict(list)
    for row in train.rows:
        normalized[normalized_target(row[train.target_column])].append(
            row[train.id_column].strip()
        )
    normalized_duplicates = [
        {"normalized_target": target, "ids": identifiers}
        for target, identifiers in sorted(
            normalized.items(), key=lambda item: (-len(item[1]), item[0])
        )
        if len(identifiers) > 1
    ]

    anomalies: list[dict[str, object]] = []
    for row in train.rows:
        identifier = row[train.id_column].strip()
        target = row[train.target_column]
        reasons: list[str] = []
        if target != target.strip():
            reasons.append("outer_whitespace")
        if "  " in target:
            reasons.append("repeated_space")
        if "\n" in target or "\r" in target:
            reasons.append("line_break")
        if any(ord(character) > 127 for character in target):
            reasons.append("non_ascii")
        if reasons:
            anomalies.append({"id": identifier, "target": target, "reasons": reasons})

    weakest = sorted(
        image_rows.values(),
        key=lambda row: (float(row["ink_fraction_below_220"]), float(row["contrast"])),
    )[:40]
    for row in weakest:
        if row["id"] in target_by_id:
            row["target"] = target_by_id[str(row["id"])]

    suspect_ids = read_id_file(args.suspect_ids) if args.suspect_ids.is_file() else set()
    suspect_rows = []
    for identifier in sorted(suspect_ids):
        if identifier not in target_by_id:
            suspect_rows.append({"id": identifier, "status": "not_in_train"})
            continue
        suspect_rows.append(
            {
                **image_rows[identifier],
                "target": target_by_id[identifier],
                "status": "present",
            }
        )

    char_lengths = [len(value) for value in targets]
    word_lengths = [len(value.split()) for value in targets]
    width_per_character = [
        float(image_rows[identifier]["width"]) / max(len(target), 1)
        for identifier, target in zip(train_ids, targets, strict=True)
    ]
    report: dict[str, object] = {
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
        "label_statistics": {
            "character_length": numeric_summary([float(value) for value in char_lengths]),
            "word_length": numeric_summary([float(value) for value in word_lengths]),
            "image_width_per_character": numeric_summary(width_per_character),
            "unique_exact": len(set(targets)),
            "unique_casefolded": len(normalized),
            "exact_duplicate_groups": duplicate_groups(
                train.rows, train.id_column, train.target_column
            ),
            "normalized_duplicate_groups": normalized_duplicates,
            "rare_characters": rare_characters,
            "anomalies": anomalies,
            "top_prefixes_3_words": phrase_counts(targets, 3),
            "top_suffixes_3_words": phrase_counts(targets, 3, suffix=True),
        },
        "image_statistics_by_split": split_geometry,
        "weakest_image_signals": weakest,
        "publicly_reported_suspect_rows": suspect_rows,
    }
    if not args.skip_hashes:
        duplicate_hashes = exact_hash_groups(image_index, train_ids + test_ids)
        cross_split = []
        train_set = set(train_ids)
        test_set = set(test_ids)
        for digest, identifiers in duplicate_hashes.items():
            if set(identifiers) & train_set and set(identifiers) & test_set:
                cross_split.append({"sha256": digest, "ids": identifiers})
        report["exact_image_duplicate_groups"] = duplicate_hashes
        report["exact_cross_split_image_duplicates"] = cross_split

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_rows": len(train_ids),
                "test_rows": len(test_ids),
                "duplicate_target_groups": len(
                    report["label_statistics"]["exact_duplicate_groups"]  # type: ignore[index]
                ),
                "label_anomalies": len(anomalies),
                "suspect_rows_present": sum(row["status"] == "present" for row in suspect_rows),
                "exact_duplicate_image_groups": len(
                    report.get("exact_image_duplicate_groups", {})
                ),
                "cross_split_exact_images": len(
                    report.get("exact_cross_split_image_duplicates", [])
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
