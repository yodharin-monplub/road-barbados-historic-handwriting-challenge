#!/usr/bin/env python3
"""Merge targeted OCR JSONL reruns into a complete candidate file by ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> tuple[list[str], dict[str, dict]]:
    order: list[str] = []
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = str(record["id"]).strip()
            if not identifier or identifier in records:
                raise ValueError(f"blank/duplicate ID at {path}:{line_number}")
            order.append(identifier)
            records[identifier] = record
    if not records:
        raise ValueError(f"no records in {path}")
    return order, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    order, merged = read_jsonl(args.base)
    replaced: set[str] = set()
    for path in args.override:
        _, values = read_jsonl(path)
        unknown = set(values) - set(merged)
        if unknown:
            raise ValueError(f"override IDs absent from base: {sorted(unknown)[:5]}")
        overlap = replaced.intersection(values)
        if overlap:
            raise ValueError(f"IDs repeated across overrides: {sorted(overlap)[:5]}")
        merged.update(values)
        replaced.update(values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for identifier in order:
            handle.write(json.dumps(merged[identifier], ensure_ascii=False) + "\n")
    print(f"wrote {len(order)} records; replaced {len(replaced)}")


if __name__ == "__main__":
    main()

