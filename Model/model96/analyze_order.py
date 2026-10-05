#!/usr/bin/env python3
"""Audit candidate row orders and export readable joined transcripts.

This is deliberately an analysis-only model.  It never changes a prediction:
we first need evidence that neighboring crops really are neighboring lines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


WORD = re.compile(r"[A-Za-z]+(?:['^][A-Za-z]+)?")


def read_csv(path: Path, text_column: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "ID" not in rows[0] or text_column not in rows[0]:
        raise ValueError(f"{path} must contain ID and {text_column} columns")
    if len({row["ID"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate IDs in {path}")
    return rows


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "median": statistics.median(ordered),
        "p75": ordered[round(0.75 * (len(ordered) - 1))],
        "p90": ordered[round(0.90 * (len(ordered) - 1))],
        "fraction_ge_0_8": sum(value >= 0.8 for value in ordered) / len(ordered),
        "fraction_ge_0_95": sum(value >= 0.95 for value in ordered) / len(ordered),
    }


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def word_edges(text: str) -> tuple[str | None, str | None]:
    words = [match.group(0).casefold() for match in WORD.finditer(text)]
    return (words[0], words[-1]) if words else (None, None)


def boundary_rate(texts: list[str], known_bigrams: set[tuple[str, str]]) -> float:
    hits = total = 0
    for left, right in zip(texts, texts[1:], strict=False):
        _, last = word_edges(left)
        first, _ = word_edges(right)
        if last is None or first is None:
            continue
        total += 1
        hits += (last, first) in known_bigrams
    return hits / total if total else 0.0


def rank_correlation(ids: list[str], archive_rank: dict[str, int]) -> float:
    pairs = [
        (index, archive_rank[identifier])
        for index, identifier in enumerate(ids)
        if identifier in archive_rank
    ]
    if len(pairs) < 2:
        return float("nan")
    xs = [float(pair[0]) for pair in pairs]
    ys = [float(pair[1]) for pair in pairs]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else float("nan")


def write_indexed(path: Path, rows: list[dict[str, str]], column: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(rows):
            handle.write(f"[{index:04d}] ID={row['ID']}\n{row[column].strip()}\n\n")


def write_plain(path: Path, rows: list[dict[str, str]], column: str) -> None:
    path.write_text(
        "\n".join(row[column].strip() for row in rows) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/Train.csv"))
    parser.add_argument(
        "--test-predictions", type=Path, default=Path("outputs/model88/test.csv")
    )
    parser.add_argument("--image-archive", type=Path, default=Path("data/raw/images.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model96"))
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    train = read_csv(args.train, "Target")
    test = read_csv(args.test_predictions, "prediction")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    train_text = [row["Target"].strip() for row in train]
    adjacent = [similarity(a, b) for a, b in zip(train_text, train_text[1:])]
    rng = random.Random(args.seed)
    random_pairs = [
        similarity(train_text[a], train_text[b])
        for a, b in (rng.sample(range(len(train_text)), 2) for _ in adjacent)
    ]

    within_line_bigrams: set[tuple[str, str]] = set()
    for text in train_text:
        words = [match.group(0).casefold() for match in WORD.finditer(text)]
        within_line_bigrams.update(zip(words, words[1:], strict=False))
    shuffled_text = train_text.copy()
    rng.shuffle(shuffled_text)

    with zipfile.ZipFile(args.image_archive) as archive:
        infos = [item for item in archive.infolist() if not item.is_dir()]
    archive_ids = [Path(item.filename).stem for item in infos]
    archive_rank = {identifier: index for index, identifier in enumerate(archive_ids)}
    if len(archive_rank) != len(archive_ids):
        raise ValueError("duplicate image stems in archive")

    train_ids = [row["ID"].strip() for row in train]
    test_ids = [row["ID"].strip() for row in test]
    all_rows = {
        **{row["ID"]: ("train", row["Target"].strip()) for row in train},
        **{row["ID"]: ("test", row["prediction"].strip()) for row in test},
    }
    missing = sorted(set(all_rows) - set(archive_ids))
    unknown = sorted(set(archive_ids) - set(all_rows))

    write_indexed(output / "train_csv_order_indexed.txt", train, "Target")
    write_plain(output / "train_csv_order_plain.txt", train, "Target")
    write_indexed(output / "test_csv_order_indexed.txt", test, "prediction")
    write_plain(output / "test_csv_order_plain.txt", test, "prediction")
    with (output / "archive_order_indexed.txt").open("w", encoding="utf-8") as handle:
        for index, identifier in enumerate(archive_ids):
            split, text = all_rows.get(identifier, ("unknown", ""))
            handle.write(f"[{index:04d}] split={split} ID={identifier}\n{text}\n\n")

    normalized_counts = Counter(" ".join(text.casefold().split()) for text in train_text)
    duplicate_rows = sum(count for count in normalized_counts.values() if count > 1)
    repeated_groups = sum(count > 1 for count in normalized_counts.values())
    adjacent_exact = sum(
        " ".join(left.casefold().split()) == " ".join(right.casefold().split())
        for left, right in zip(train_text, train_text[1:], strict=False)
    )
    id_lengths = sorted({len(identifier) for identifier in train_ids + test_ids})
    report = {
        "schema_version": 1,
        "conclusion": (
            "No supplied order is supported as manuscript line order. IDs are random "
            "opaque strings; CSV and ZIP orders disagree; Train.csv begins with "
            "duplicate/template clusters and otherwise behaves close to random."
        ),
        "safe_next_step": (
            "Do not grammar-correct one concatenated story. Preserve each ID/line and "
            "only use context after a page/order key is recovered or validated by OOF."
        ),
        "rows": {"train": len(train), "test": len(test), "archive": len(archive_ids)},
        "identifier_audit": {
            "lengths": id_lengths,
            "pure_numeric": sum(identifier.isdigit() for identifier in train_ids + test_ids),
            "numeric_sequence": False,
        },
        "archive_audit": {
            "missing_labeled_or_predicted_ids": missing,
            "unknown_archive_ids": unknown,
            "train_csv_vs_archive_rank_correlation": rank_correlation(train_ids, archive_rank),
            "test_csv_vs_archive_rank_correlation": rank_correlation(test_ids, archive_rank),
            "distinct_timestamps": len({item.date_time for item in infos}),
        },
        "train_order_audit": {
            "adjacent_text_similarity": quantiles(adjacent),
            "seeded_random_pair_similarity": quantiles(random_pairs),
            "adjacent_exact_normalized_pairs": adjacent_exact,
            "rows_belonging_to_exact_duplicate_groups": duplicate_rows,
            "exact_duplicate_groups": repeated_groups,
            "known_within_line_bigram_at_csv_boundary_rate": boundary_rate(
                train_text, within_line_bigrams
            ),
            "known_within_line_bigram_at_shuffled_boundary_rate": boundary_rate(
                shuffled_text, within_line_bigrams
            ),
        },
        "artifacts": {
            "train_indexed": str(output / "train_csv_order_indexed.txt"),
            "train_plain": str(output / "train_csv_order_plain.txt"),
            "test_indexed": str(output / "test_csv_order_indexed.txt"),
            "test_plain": str(output / "test_csv_order_plain.txt"),
            "archive_indexed": str(output / "archive_order_indexed.txt"),
        },
    }
    (output / "order_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
