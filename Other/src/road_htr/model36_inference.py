"""Model36-specific inference wrapper that authenticates prediction bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .kraken_cli import infer_main


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_argument(arguments: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parsed, _ = parser.parse_known_args(arguments)
    return parsed.output.resolve()


def authenticate_prediction_manifest(output: Path) -> Path:
    output = output.resolve()
    if not output.is_file():
        raise FileNotFoundError(output)
    manifest = output.with_suffix(".manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checksum = _sha256(output)
    recorded = payload.get("predictions_sha256")
    if recorded is not None and recorded != checksum:
        raise ValueError("inference manifest already contains a conflicting prediction hash")
    payload["predictions_sha256"] = checksum
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(manifest)
    return manifest


def signed_infer_main(default_config: str | Path) -> None:
    output = output_argument(sys.argv[1:])
    infer_main(default_config)
    authenticate_prediction_manifest(output)

