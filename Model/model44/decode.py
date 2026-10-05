#!/usr/bin/env python3
"""Model44: an isolated, whole-vector decoding-policy challenger.

This deliberately reuses Model39's loading, preprocessing, output, and metric
code.  It adds only a two-member, pre-registered generation panel.  It never
selects a policy by row: each member emits and scores one complete vector.
Heavy libraries are imported only after all CPU lineage gates have passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


class Model44GateError(RuntimeError):
    """A frozen provenance, vector-integrity, or output-isolation gate failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def below(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve(path)
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise Model44GateError("Model44 config must be a JSON object")
    validate_config(config)
    return config, resolved


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("model_number") != 44:
        raise Model44GateError("model_number must be exactly 44")
    panel = config.get("panel")
    expected = [
        {"name": "greedy", "do_sample": False, "num_beams": 1, "num_return_sequences": 1, "max_new_tokens": 48, "repetition_penalty": 1.05, "no_repeat_ngram_size": 4, "use_cache": True},
        {"name": "beam3_starter_aligned", "do_sample": False, "num_beams": 3, "num_return_sequences": 1, "max_new_tokens": 48, "repetition_penalty": 1.05, "no_repeat_ngram_size": 4, "use_cache": True},
    ]
    if panel != expected:
        raise Model44GateError("panel is immutable: greedy plus the registered 3-beam vector only")
    if config.get("frozen_omissions") != ["length_penalty", "early_stopping", "eos_token_id", "forced_eos_token_id"]:
        raise Model44GateError("generation omissions must remain frozen")
    base = config.get("base_model39")
    if not isinstance(base, Mapping) or base.get("required_model_number") != 39:
        raise Model44GateError("only Model39-compatible broad adapters are allowed")
    if base.get("adapter_root") != "outputs/model39":
        raise Model44GateError("adapter root must remain outputs/model39")
    sources = base.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {"model39_infer", "broad_runtime", "challenger_runtime", "schema", "submission"}:
        raise Model44GateError("complete Model39 source hash ledger is required")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {"fold_oof", "test"}:
        raise Model44GateError("only the registered fold OOF and test inputs are allowed")
    if config.get("output_root") != "outputs/model44":
        raise Model44GateError("output root must remain isolated at outputs/model44")


def source_receipt(config: Mapping[str, Any]) -> dict[str, str]:
    base = config["base_model39"]
    files: dict[str, Mapping[str, str]] = {
        "model39_config": {"path": str(base["config"]), "sha256": str(base["config_sha256"])},
        **{str(name): value for name, value in base["sources"].items()},
    }
    actual: dict[str, str] = {}
    for name, spec in files.items():
        path = resolve(spec["path"])
        if not path.is_file() or path.is_symlink():
            raise Model44GateError(f"pinned source missing or symlinked: {path}")
        digest = sha256(path)
        if digest != spec["sha256"]:
            raise Model44GateError(f"source hash mismatch for {name}: {digest}")
        actual[name] = digest
    return actual


def registered_input(config: Mapping[str, Any], input_csv: str | Path, fold: int | None) -> tuple[str, Path, str]:
    path = resolve(input_csv)
    if not path.is_file() or path.is_symlink():
        raise Model44GateError(f"input is absent or symlinked: {path}")
    digest = sha256(path)
    for kind, spec in config["inputs"].items():
        registered = resolve(spec["path"])
        if path == registered and digest == spec["sha256"]:
            if kind == "fold_oof" and fold not in range(5):
                raise Model44GateError("fold OOF evaluation requires --fold in 0..4")
            if kind == "test" and fold is not None:
                raise Model44GateError("test evaluation must not filter by fold")
            return str(kind), path, digest
    raise Model44GateError("input path/hash is not a pre-registered Model44 vector")


def directory_receipt(directory: Path) -> dict[str, str]:
    if not directory.is_dir() or directory.is_symlink():
        raise Model44GateError(f"adapter directory is absent or symlinked: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise Model44GateError("adapter directory is empty")
    receipt: dict[str, str] = {}
    for path in files:
        if path.is_symlink():
            raise Model44GateError(f"adapter file is symlinked: {path}")
        receipt[str(path.relative_to(directory))] = sha256(path)
    if "adapter_config.json" not in receipt:
        raise Model44GateError("adapter_config.json missing from adapter receipt")
    if not any(name.startswith("adapter_model") for name in receipt):
        raise Model44GateError("adapter weights missing from adapter receipt")
    return receipt


def adapter_receipt(config: Mapping[str, Any], adapter_path: str | Path, fold: int | None) -> dict[str, Any]:
    if fold not in range(5):
        raise Model44GateError("adapter-backed runs require a registered fold")
    adapter = resolve(adapter_path)
    root = resolve(str(config["base_model39"]["adapter_root"]))
    expected = (root / f"fold{fold}" / "best_adapter").resolve()
    if adapter != expected or not below(adapter, root):
        raise Model44GateError("adapter must be exactly Model39 fold{fold}/best_adapter")
    run_dir = expected.parent
    run_manifest_path = run_dir / "run_manifest.json"
    complete_path = run_dir / "complete.json"
    if not run_manifest_path.is_file() or not complete_path.is_file():
        raise Model44GateError("Model39 run_manifest.json and complete.json are both required")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if run_manifest.get("model_number") != 39 or run_manifest.get("fold") != fold:
        raise Model44GateError("Model39 run manifest does not bind the requested fold")
    if run_manifest.get("train_csv_sha256") != config["inputs"]["fold_oof"]["sha256"]:
        raise Model44GateError("Model39 run manifest has an unregistered train CSV")
    if Path(str(run_manifest.get("resolved_config", ""))).resolve() != resolve(config["base_model39"]["config"]):
        raise Model44GateError("Model39 run manifest was not produced from the pinned config")
    if complete.get("fold") != fold or resolve(str(complete.get("best_adapter", ""))) != expected:
        raise Model44GateError("Model39 complete receipt does not bind best_adapter")
    return {"path": str(adapter), "files_sha256": directory_receipt(adapter), "run_manifest_sha256": sha256(run_manifest_path), "complete_sha256": sha256(complete_path)}


def cpu_audit(config_path: str | Path, input_csv: str | Path, fold: int | None, adapter: str | Path | None = None, *, verify_checkpoint: bool = True) -> dict[str, Any]:
    """Run CPU-only source/input/checkpoint checks.  It never loads an adapter."""
    config, config_path = load_config(config_path)
    sources = source_receipt(config)
    kind, input_path, input_hash = registered_input(config, input_csv, fold)
    from road_htr import qwen2_vl_broad as broad

    model39, _ = broad.load_config(resolve(config["base_model39"]["config"]))
    broad.enable_offline_mode()
    checkpoint: dict[str, Any] | None = None
    provenance: dict[str, str] | None = None
    if verify_checkpoint:
        provenance = broad.verify_starter_provenance(model39)
        checkpoint = broad.verify_local_checkpoint(model39)
    records = broad.load_records(input_path, resolve(model39["image_dir"]), require_target=kind == "fold_oof", require_five_folds=kind == "fold_oof")
    if kind == "fold_oof":
        records = tuple(record for record in records if record.fold == fold)
    if not records:
        raise Model44GateError("registered vector is empty")
    if len({record.identifier for record in records}) != len(records):
        raise Model44GateError("registered vector has duplicate IDs")
    receipt: dict[str, Any] = {
        "status": "cpu_audit_passed_adapter_not_loaded",
        "model_number": 44,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "model44_decode_sha256": sha256(Path(__file__).resolve()),
        "source_hashes": sources,
        "input": {"kind": kind, "path": str(input_path), "sha256": input_hash, "rows": len(records), "fold": fold},
        "panel": config["panel"],
        "frozen_omissions": config["frozen_omissions"],
        "whole_vector_only": True,
        "row_oracle_mixing": False,
        "checkpoint": checkpoint,
        "provenance": provenance,
    }
    if adapter is None:
        receipt["adapter"] = {"available": False, "reason": "not supplied; no score is claimed"}
    else:
        receipt["adapter"] = {"available": True, **adapter_receipt(config, adapter, fold)}
    return receipt


def predict_vector(model: Any, processor: Any, collator: Any, records: Sequence[Any], policy: Mapping[str, Any]) -> list[str]:
    """Generate exactly one prediction per input record for one frozen policy."""
    import torch
    from road_htr import qwen2_vl_challenger as challenger

    if not records:
        raise Model44GateError("cannot generate an empty vector")
    device = challenger._model_device(model)
    was_training = bool(model.training)
    old_cache = bool(getattr(model.config, "use_cache", False))
    model.eval()
    model.config.use_cache = True
    predictions: list[str] = []
    try:
        for record in records:
            batch, _ = collator.encode_prompt(record)
            batch = challenger._to_device(batch, device)
            prompt_tokens = int(batch["input_ids"].shape[-1])
            with torch.inference_mode():
                output = model.generate(**batch, **{key: value for key, value in policy.items() if key != "name"})
            generated = output[:, prompt_tokens:]
            predictions.append(challenger._decode_generated(processor, generated))
    finally:
        model.config.use_cache = old_cache
        model.train(was_training)
    if len(predictions) != len(records):
        raise Model44GateError("generation did not produce one complete vector")
    return predictions


def new_output_dir(config: Mapping[str, Any], output_dir: str | Path) -> Path:
    raw_root = ROOT / str(config["output_root"])
    if raw_root.exists() and raw_root.is_symlink():
        raise Model44GateError("isolated output root may not be a symlink")
    root = raw_root.resolve()
    path = resolve(output_dir)
    if not below(path, root) or path == root:
        raise Model44GateError("outputs must be isolated below outputs/model44")
    if path.exists():
        raise FileExistsError(f"output directory must be new: {path}")
    return path


def run_panel(config_path: str | Path, input_csv: str | Path, adapter_path: str | Path, output_dir: str | Path, *, fold: int | None) -> dict[str, Any]:
    """Run both policies; emit/score both complete vectors without selecting either."""
    started = time.monotonic()
    config, resolved_config = load_config(config_path)
    output = new_output_dir(config, output_dir)
    before = cpu_audit(resolved_config, input_csv, fold, adapter_path, verify_checkpoint=True)
    from road_htr import qwen2_vl_broad as broad

    model39, _ = broad.load_config(resolve(config["base_model39"]["config"]))
    records = broad.load_records(resolve(input_csv), resolve(model39["image_dir"]), require_target=before["input"]["kind"] == "fold_oof", require_five_folds=before["input"]["kind"] == "fold_oof")
    if fold is not None:
        records = tuple(record for record in records if record.fold == fold)
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / "receipt.initial.json", before)
    processor = broad.load_processor(model39)
    model, target_audit = broad._load_inference_model(model39, resolve(adapter_path))
    collator = broad.QwenOcrCollator(processor, str(model39["prompt"]), int(model39["image"]["max_image_tokens"]), int(model39["image"]["max_sequence_tokens"]))
    candidates: list[dict[str, Any]] = []
    for policy in config["panel"]:
        predictions = predict_vector(model, processor, collator, records, policy)
        csv_path = broad.write_predictions(output / f"{policy['name']}.csv", records, predictions)
        if len(predictions) != len(records):
            raise Model44GateError("refusing a partial vector output")
        entry: dict[str, Any] = {"policy": policy, "csv": str(csv_path), "csv_sha256": sha256(csv_path), "rows": len(predictions)}
        if all(record.label is not None for record in records):
            entry["whole_vector_metrics"] = broad.exact_metrics(records, predictions)
        candidates.append(entry)
    after = cpu_audit(resolved_config, input_csv, fold, adapter_path, verify_checkpoint=True)
    if any(before[key] != after[key] for key in ("config_sha256", "model44_decode_sha256", "source_hashes", "input", "adapter", "checkpoint", "provenance")):
        raise Model44GateError("TOCTOU: config, source, input, adapter, or checkpoint changed during generation")
    report = {"status": "complete", "model_number": 44, "runtime_seconds": time.monotonic() - started, "initial_receipt": "receipt.initial.json", "final_receipt": "receipt.final.json", "target_audit": vars(target_audit), "whole_vector_only": True, "row_oracle_mixing": False, "selection_performed": False, "candidates": candidates}
    json_write(output / "receipt.final.json", after)
    json_write(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Model44 whole-vector Model39 decoding-policy challenger")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-audit", action="store_true")
    args = parser.parse_args()
    if args.dry_audit:
        print(json.dumps(cpu_audit(args.config, args.input, args.fold, args.adapter), indent=2, ensure_ascii=False))
        return
    if args.adapter is None or args.output_dir is None:
        parser.error("--adapter and --output-dir are required unless --dry-audit is used")
    print(json.dumps(run_panel(args.config, args.input, args.adapter, args.output_dir, fold=args.fold), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
