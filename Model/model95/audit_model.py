#!/usr/bin/env python3
"""Download and fully hash-audit the pinned Model95 Qwen3-VL snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path

from huggingface_hub import snapshot_download


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    expected_shards = model["expected_shards"]
    required_payload = int(model["expected_weight_payload_bytes"])
    required_files = int(model["expected_weight_file_bytes"])
    disk_target = (args.cache_dir or Path.cwd()).resolve()
    disk_target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(disk_target).free
    if free < required_files + 20 * 1024**3:
        raise RuntimeError(
            f"insufficient free disk: {free / 1024**3:.1f} GiB; "
            f"need at least {(required_files + 20 * 1024**3) / 1024**3:.1f} GiB"
        )

    snapshot = Path(
        snapshot_download(
            repo_id=model["id"],
            revision=model["revision"],
            cache_dir=str(disk_target),
        )
    ).resolve()
    if snapshot.name != model["revision"]:
        raise RuntimeError(f"resolved snapshot is not pinned revision: {snapshot}")

    metadata = {}
    for name, expected_hash in model["expected_metadata_sha256"].items():
        path = snapshot / name
        actual_hash = digest(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"metadata hash mismatch for {name}: {actual_hash} != {expected_hash}"
            )
        metadata[name] = {"bytes": path.stat().st_size, "sha256": actual_hash}
    readme = (snapshot / "README.md").read_text(encoding="utf-8").lower()
    if not re.search(r"(?m)^license:\s*apache-2\.0\s*$", readme):
        raise RuntimeError("model card no longer declares Apache-2.0")

    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    if int(index["metadata"]["total_size"]) != required_payload:
        raise RuntimeError("weight index payload size differs from configuration")
    indexed_names = sorted(set(index["weight_map"].values()))
    if indexed_names != sorted(expected_shards):
        raise RuntimeError("weight index shard set differs from configuration")

    shards = []
    for number, name in enumerate(indexed_names, start=1):
        path = snapshot / name
        expected = expected_shards[name]
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"physical shard size mismatch for {name}")
        print(f"hashing shard {number}/{len(indexed_names)}: {name}", flush=True)
        actual_hash = digest(path)
        if actual_hash != expected["sha256"]:
            raise RuntimeError(f"weight shard hash mismatch for {name}")
        shards.append({"name": name, "bytes": path.stat().st_size, "sha256": actual_hash})
    if sum(row["bytes"] for row in shards) != required_files:
        raise RuntimeError("physical weight bytes differ from configuration")

    report = {
        "schema_version": 1,
        "kind": "model95_qwen3_vl_snapshot_audit",
        "complete": True,
        "model_id": model["id"],
        "revision": model["revision"],
        "license": model["license"],
        "snapshot": str(snapshot),
        "weight_payload_bytes": required_payload,
        "weight_file_bytes": required_files,
        "metadata": metadata,
        "shards": shards,
        "config": str(config_path),
        "config_sha256": digest(config_path),
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps({"snapshot": str(snapshot), "shards": len(shards)}, indent=2))


if __name__ == "__main__":
    main()
