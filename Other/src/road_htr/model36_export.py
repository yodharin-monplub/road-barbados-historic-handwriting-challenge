"""Convert only fixed epoch 7 from a completed all-data Model36 run."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

import yaml

from .fold_model_postprocess import PROJECT_ROOT, _absolute, _atomic_json, _read_json, _sha256


def export_fixed_checkpoint(training_dir: Path, output_dir: Path) -> Path:
    training_dir, output_dir = training_dir.resolve(), output_dir.resolve()
    allowed = PROJECT_ROOT / "outputs/model36"
    if not output_dir.is_relative_to(allowed) or output_dir == allowed:
        raise ValueError("export must use a new subdirectory of outputs/model36")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest_path = training_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    canonical = PROJECT_ROOT / "model36/config.yaml"
    if manifest["config"] != yaml.safe_load(canonical.read_text()):
        raise ValueError("training configuration differs from Model36")
    if manifest["preparation"].get("fold", "missing") is not None:
        raise ValueError("export requires completed all-data training")
    config = manifest["config"]
    if config["training"].get("all_data_epochs") != 8 or config["training"].get("all_data_cosine_epochs") != 10:
        raise ValueError("unexpected all-data training schedule")
    checkpoints = list(training_dir.glob("checkpoint_*.ckpt"))
    by_epoch = {}
    for checkpoint in checkpoints:
        match = re.fullmatch(r"checkpoint_(\d+)-.+\.ckpt", checkpoint.name)
        if match is None or int(match.group(1)) in by_epoch:
            raise ValueError("aborted, duplicate, or unrecognized checkpoint")
        by_epoch[int(match.group(1))] = checkpoint
    best = list(training_dir.glob("best_*.safetensors"))
    if set(by_epoch) != set(range(8)) or not best:
        raise ValueError("all-data training is not complete")
    if max(path.stat().st_mtime_ns for path in best) < max(
        path.stat().st_mtime_ns for path in [manifest_path, *checkpoints]
    ):
        raise ValueError("best-weights completion marker predates training artifacts")
    checkpoint = by_epoch[7]
    hashes = {"checkpoint_sha256": _sha256(checkpoint),
              "training_manifest_sha256": _sha256(manifest_path),
              "config_sha256": _sha256(canonical)}
    output_dir.mkdir(parents=True, exist_ok=False)
    weights = output_dir / "epoch07.safetensors"
    _atomic_json(output_dir / "status.json", {"status": "converting_not_ready"})
    command = [str(PROJECT_ROOT / ".venv-kraken/bin/ketos"), "convert", "--output", str(weights), str(checkpoint)]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if not weights.is_file() or weights.stat().st_size == 0:
        raise RuntimeError("converter did not produce nonempty weights")
    for path, key in ((checkpoint, "checkpoint_sha256"), (manifest_path, "training_manifest_sha256"),
                      (canonical, "config_sha256")):
        if _sha256(path) != hashes[key]:
            raise RuntimeError("source artifacts changed during checkpoint conversion")
    receipt = {"schema_version": 1, "checkpoint_epoch_zero_based": 7,
               "checkpoint": str(checkpoint), "weights": str(weights),
               "training_manifest": str(manifest_path), **hashes,
               "weights_sha256": _sha256(weights), "command": command,
               "selection": "fixed_epoch7_not_best_filename_or_training_metric"}
    _atomic_json(output_dir / "epoch07.conversion.json", receipt)
    _atomic_json(output_dir / "status.json", {"status": "converted_not_a_submission"})
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=PROJECT_ROOT / "outputs/model36/foldall")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/model36/deployment_epoch07")
    args = parser.parse_args()
    print(export_fixed_checkpoint(_absolute(args.training_dir), _absolute(args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
