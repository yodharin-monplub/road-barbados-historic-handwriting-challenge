#!/usr/bin/env python3
"""Download and hash-audit the immutable Model93 Qwen snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    required_payload = int(model["expected_weight_bytes"])
    required_files = int(model["expected_weight_file_bytes"])
    disk_target = args.cache_dir or Path.cwd()
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
            cache_dir=str(args.cache_dir.resolve()) if args.cache_dir else None,
        )
    ).resolve()
    if snapshot.name != model["revision"]:
        raise RuntimeError(f"resolved snapshot is not pinned revision: {snapshot}")

    metadata = {}
    for name, expected in model["expected_metadata_sha256"].items():
        path = snapshot / name
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"metadata hash mismatch for {name}: {actual} != {expected}")
        metadata[name] = {"bytes": path.stat().st_size, "sha256": actual}
    if "license: apache-2.0" not in (
        snapshot / "README.md"
    ).read_text(encoding="utf-8").lower():
        raise RuntimeError("model card no longer declares Apache-2.0")

    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    if int(index["metadata"]["total_size"]) != required_payload:
        raise RuntimeError("weight index total differs from configured byte count")
    shard_names = sorted(set(index["weight_map"].values()))
    if len(shard_names) != int(model["expected_shards"]):
        raise RuntimeError("weight shard count differs from configuration")
    shards = []
    for number, name in enumerate(shard_names, start=1):
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"hashing shard {number}/{len(shard_names)}: {name}", flush=True)
        shards.append(
            {"name": name, "bytes": path.stat().st_size, "sha256": digest(path)}
        )
    # Hugging Face total_size is tensor payload only. Physical safetensors files
    # additionally contain their headers, so both quantities are pinned.
    if sum(row["bytes"] for row in shards) != required_files:
        raise RuntimeError("physical shard bytes differ from configured file byte count")

    report = {
        "schema_version": 1,
        "kind": "model93_qwen_snapshot_audit",
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
    print(
        json.dumps(
            {
                "snapshot": str(snapshot),
                "weight_payload_bytes": required_payload,
                "weight_file_bytes": required_files,
                "shards": len(shards),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
