#!/usr/bin/env python3
"""Model49: frozen whole-vector no-repeat-4 versus no-repeat-8 decoding.

The experiment is deliberately narrow: one completed Model39 fold-0 adapter
is loaded once and deterministic greedy decoding is run twice.  The sole
generation-policy difference is ``no_repeat_ngram_size``.  Heavy model imports
remain behind complete-adapter and provenance gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


class Model49GateError(RuntimeError):
    """A frozen Model49 lineage, policy, or vector-integrity gate failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def below(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def expected_panel() -> list[dict[str, Any]]:
    common = {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "max_new_tokens": 48,
        "repetition_penalty": 1.05,
        "use_cache": True,
    }
    return [
        {
            "name": "greedy_ngram4_control",
            **common,
            "no_repeat_ngram_size": 4,
        },
        {
            "name": "greedy_ngram8_challenger",
            **common,
            "no_repeat_ngram_size": 8,
        },
    ]


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve(path)
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise Model49GateError("Model49 config must be a JSON object")
    validate_config(config)
    return config, resolved


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("model_number") != 49:
        raise Model49GateError("model_number must be exactly 49")
    panel = config.get("panel")
    if panel != expected_panel():
        raise Model49GateError(
            "panel is immutable: greedy ngram-4 control plus greedy ngram-8 challenger"
        )
    left, right = panel
    differing = {
        key
        for key in set(left) | set(right)
        if key != "name" and left.get(key) != right.get(key)
    }
    if differing != {"no_repeat_ngram_size"}:
        raise Model49GateError("the policies may differ only in no_repeat_ngram_size")
    if config.get("only_policy_difference") != "no_repeat_ngram_size":
        raise Model49GateError("the sole policy difference must be explicitly registered")
    if config.get("frozen_omissions") != [
        "length_penalty",
        "early_stopping",
        "eos_token_id",
        "forced_eos_token_id",
    ]:
        raise Model49GateError("generation omissions must remain frozen")

    base = config.get("base_model39")
    if not isinstance(base, Mapping):
        raise Model49GateError("base_model39 ledger is required")
    if base.get("required_model_number") != 39 or base.get("registered_fold") != 0:
        raise Model49GateError("only the completed Model39 fold-0 adapter is registered")
    if base.get("required_optimizer_updates") != 615:
        raise Model49GateError("Model39 completion must record exactly 615 updates")
    if base.get("adapter_root") != "outputs/model39":
        raise Model49GateError("adapter root must remain outputs/model39")
    sources = base.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "model39_infer",
        "broad_runtime",
        "challenger_runtime",
        "schema",
        "submission",
    }:
        raise Model49GateError("complete Model39 source hash ledger is required")

    input_spec = config.get("input")
    if not isinstance(input_spec, Mapping):
        raise Model49GateError("registered fold input is required")
    if input_spec.get("rows") != 4095 or input_spec.get("fold0_rows") != 819:
        raise Model49GateError("registered vector sizes must remain 4095/819")

    label = config.get("label_constraint_audit")
    if not isinstance(label, Mapping):
        raise Model49GateError("label-constraint audit is required")
    if label.get("tokenizer_add_special_tokens") is not False:
        raise Model49GateError("label audit must use raw target tokens")
    if label.get("transformers_version") != "4.49.0":
        raise Model49GateError("label audit is pinned to transformers 4.49.0")
    expected = label.get("expected")
    if not isinstance(expected, Mapping) or set(expected) != {"4", "8"}:
        raise Model49GateError("label audit must register exactly n=4 and n=8")
    if expected["4"].get("count") != 26 or expected["4"].get("by_fold", {}).get("0") != 4:
        raise Model49GateError("registered no-repeat-4 label counts changed")
    if expected["8"].get("count") != 1 or expected["8"].get("by_fold", {}).get("0") != 0:
        raise Model49GateError("registered no-repeat-8 label counts changed")
    if config.get("output_root") != "outputs/model49":
        raise Model49GateError("output root must remain isolated at outputs/model49")


def source_receipt(config: Mapping[str, Any]) -> dict[str, str]:
    base = config["base_model39"]
    files: dict[str, Mapping[str, str]] = {
        "model39_config": {
            "path": str(base["config"]),
            "sha256": str(base["config_sha256"]),
        },
        **{str(name): spec for name, spec in base["sources"].items()},
    }
    actual: dict[str, str] = {}
    for name, spec in files.items():
        path = resolve(spec["path"])
        if not path.is_file() or path.is_symlink():
            raise Model49GateError(f"pinned source missing or symlinked: {path}")
        digest = sha256(path)
        if digest != spec["sha256"]:
            raise Model49GateError(f"source hash mismatch for {name}: {digest}")
        actual[name] = digest
    return actual


def registered_input(config: Mapping[str, Any], input_csv: str | Path) -> tuple[Path, str]:
    path = resolve(input_csv)
    spec = config["input"]
    if not path.is_file() or path.is_symlink():
        raise Model49GateError(f"input is absent or symlinked: {path}")
    digest = sha256(path)
    if path != resolve(spec["path"]) or digest != spec["sha256"]:
        raise Model49GateError("input path/hash is not the registered fold vector")
    return path, digest


def has_repeated_ngram(token_ids: Sequence[int], size: int) -> bool:
    if size <= 0:
        raise ValueError("ngram size must be positive")
    seen: set[tuple[int, ...]] = set()
    for start in range(len(token_ids) - size + 1):
        gram = tuple(int(value) for value in token_ids[start : start + size])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def _ids_digest(identifiers: Sequence[str]) -> str:
    payload = "\n".join(sorted(identifiers))
    if payload:
        payload += "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_label_constraints(
    config: Mapping[str, Any], records: Sequence[Any], tokenizer: Any | None = None
) -> dict[str, Any]:
    """Recompute the pre-registered tokenizer-level n-gram constraint counts."""
    if len(records) != int(config["input"]["rows"]):
        raise Model49GateError("label audit did not receive the complete registered table")
    if tokenizer is None:
        import transformers
        from transformers import AutoTokenizer

        required_version = str(config["label_constraint_audit"]["transformers_version"])
        if transformers.__version__ != required_version:
            raise Model49GateError(
                f"transformers version {transformers.__version__} != {required_version}"
            )
        model39, _ = _load_model39_config(config)
        tokenizer = AutoTokenizer.from_pretrained(
            str(resolve(model39["model"]["local_path"])),
            local_files_only=True,
            trust_remote_code=False,
        )

    actual: dict[str, Any] = {}
    add_special = bool(config["label_constraint_audit"]["tokenizer_add_special_tokens"])
    for raw_size in ("4", "8"):
        size = int(raw_size)
        affected: list[str] = []
        by_fold = {str(fold): 0 for fold in range(5)}
        for record in records:
            if record.label is None or record.fold not in range(5):
                raise Model49GateError("label audit requires labeled records with folds 0..4")
            token_ids = tokenizer.encode(record.label, add_special_tokens=add_special)
            if has_repeated_ngram(token_ids, size):
                affected.append(record.identifier)
                by_fold[str(record.fold)] += 1
        actual[raw_size] = {
            "count": len(affected),
            "by_fold": by_fold,
            "sorted_ids_sha256": _ids_digest(affected),
        }
    if actual != config["label_constraint_audit"]["expected"]:
        raise Model49GateError(
            "tokenizer label-constraint audit differs from the registered receipt"
        )
    return {
        "definition": config["label_constraint_audit"]["definition"],
        "tokenizer_add_special_tokens": add_special,
        "transformers_version": config["label_constraint_audit"]["transformers_version"],
        "counts": actual,
        "fold0_interpretation": {
            "ngram4_exact_truths_made_impossible": actual["4"]["by_fold"]["0"],
            "ngram8_exact_truths_made_impossible": actual["8"]["by_fold"]["0"],
        },
    }


def _load_model39_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    from road_htr import qwen2_vl_broad as broad

    return broad.load_config(resolve(config["base_model39"]["config"]))


def registered_records(config: Mapping[str, Any], input_csv: str | Path) -> tuple[Any, ...]:
    path, _ = registered_input(config, input_csv)
    from road_htr import qwen2_vl_broad as broad

    model39, _ = _load_model39_config(config)
    records = broad.load_records(
        path,
        resolve(model39["image_dir"]),
        require_target=True,
        require_five_folds=True,
    )
    if len(records) != int(config["input"]["rows"]):
        raise Model49GateError("registered input row count changed")
    if len({record.identifier for record in records}) != len(records):
        raise Model49GateError("registered input contains duplicate IDs")
    fold = int(config["base_model39"]["registered_fold"])
    selected = tuple(record for record in records if record.fold == fold)
    if len(selected) != int(config["input"]["fold0_rows"]):
        raise Model49GateError("registered fold-0 vector row count changed")
    return records


def directory_receipt(directory: Path) -> dict[str, str]:
    if not directory.is_dir() or directory.is_symlink():
        raise Model49GateError(f"adapter directory is absent or symlinked: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise Model49GateError("adapter directory is empty")
    receipt: dict[str, str] = {}
    for path in files:
        if path.is_symlink():
            raise Model49GateError(f"adapter file is symlinked: {path}")
        receipt[str(path.relative_to(directory))] = sha256(path)
    if "adapter_config.json" not in receipt:
        raise Model49GateError("adapter_config.json missing from adapter receipt")
    if not any(name.startswith("adapter_model") for name in receipt):
        raise Model49GateError("adapter weights missing from adapter receipt")
    return receipt


def adapter_receipt(config: Mapping[str, Any], adapter_path: str | Path) -> dict[str, Any]:
    fold = int(config["base_model39"]["registered_fold"])
    adapter = resolve(adapter_path)
    root = resolve(str(config["base_model39"]["adapter_root"]))
    expected = (root / f"fold{fold}" / "best_adapter").resolve()
    if adapter != expected or not below(adapter, root):
        raise Model49GateError("adapter must be exactly Model39 fold0/best_adapter")
    run_dir = expected.parent
    manifest_path = run_dir / "run_manifest.json"
    complete_path = run_dir / "complete.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise Model49GateError(
            "completed Model39 run_manifest.json and complete.json are both required"
        )
    if manifest_path.is_symlink() or complete_path.is_symlink():
        raise Model49GateError("Model39 completion receipts may not be symlinks")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if run_manifest.get("model_number") != 39 or run_manifest.get("fold") != fold:
        raise Model49GateError("Model39 run manifest does not bind fold 0")
    if run_manifest.get("train_csv_sha256") != config["input"]["sha256"]:
        raise Model49GateError("Model39 run manifest has an unregistered train CSV")
    manifest_config = Path(str(run_manifest.get("resolved_config", ""))).resolve()
    if manifest_config != resolve(config["base_model39"]["config"]):
        raise Model49GateError("Model39 run manifest was not produced from the pinned config")
    if complete.get("fold") != fold or resolve(str(complete.get("best_adapter", ""))) != expected:
        raise Model49GateError("Model39 complete receipt does not bind fold-0 best_adapter")
    best_epoch = complete.get("best_epoch")
    if not isinstance(best_epoch, int) or best_epoch not in {1, 2, 3}:
        raise Model49GateError("Model39 complete receipt has no valid best epoch")
    if not isinstance(complete.get("best_multi_score"), (int, float)):
        raise Model49GateError("Model39 complete receipt has no best-fold score")
    required_updates = int(config["base_model39"]["required_optimizer_updates"])
    if complete.get("observed_optimizer_updates") != required_updates:
        raise Model49GateError("Model39 training did not complete all registered updates")
    schedule = complete.get("optimizer_schedule")
    if not isinstance(schedule, Mapping) or schedule.get("total_optimizer_updates") != required_updates:
        raise Model49GateError("Model39 completion schedule is inconsistent")
    return {
        "path": str(adapter),
        "fold": fold,
        "best_epoch": best_epoch,
        "best_multi_score": float(complete["best_multi_score"]),
        "files_sha256": directory_receipt(adapter),
        "run_manifest_sha256": sha256(manifest_path),
        "complete_sha256": sha256(complete_path),
    }


def contract_audit(
    config_path: str | Path,
    input_csv: str | Path,
    *,
    verify_checkpoint: bool = True,
) -> dict[str, Any]:
    """Validate the static experiment contract without accepting an adapter."""
    config, resolved_config = load_config(config_path)
    sources = source_receipt(config)
    input_path, input_hash = registered_input(config, input_csv)
    from road_htr import qwen2_vl_broad as broad

    model39, _ = _load_model39_config(config)
    broad.enable_offline_mode()
    checkpoint: dict[str, Any] | None = None
    provenance: dict[str, str] | None = None
    if verify_checkpoint:
        provenance = broad.verify_starter_provenance(model39)
        checkpoint = broad.verify_local_checkpoint(model39)
    records = registered_records(config, input_path)
    label_audit = audit_label_constraints(config, records)
    fold = int(config["base_model39"]["registered_fold"])
    fold_rows = sum(record.fold == fold for record in records)
    return {
        "status": "static_contract_passed_adapter_not_checked",
        "model_number": 49,
        "score_claimed": False,
        "generation_possible": False,
        "config_path": str(resolved_config),
        "config_sha256": sha256(resolved_config),
        "model49_decode_sha256": sha256(Path(__file__).resolve()),
        "source_hashes": sources,
        "input": {
            "path": str(input_path),
            "sha256": input_hash,
            "rows": len(records),
            "fold": fold,
            "fold_rows": fold_rows,
        },
        "panel": config["panel"],
        "only_policy_difference": config["only_policy_difference"],
        "frozen_omissions": config["frozen_omissions"],
        "whole_vector_only": True,
        "row_oracle_mixing": False,
        "label_constraint_audit": label_audit,
        "checkpoint": checkpoint,
        "provenance": provenance,
        "adapter": {
            "checked": False,
            "reason": "static contract only; readiness and generation require a completed adapter",
        },
    }


def cpu_readiness_audit(
    config_path: str | Path,
    input_csv: str | Path,
    adapter_path: str | Path,
    *,
    verify_checkpoint: bool = True,
) -> dict[str, Any]:
    """CPU-only readiness gate; hashes but never loads the completed adapter."""
    receipt = contract_audit(
        config_path,
        input_csv,
        verify_checkpoint=verify_checkpoint,
    )
    config, _ = load_config(config_path)
    receipt["adapter"] = {"checked": True, **adapter_receipt(config, adapter_path)}
    receipt["status"] = "cpu_readiness_passed_completed_adapter_not_loaded"
    receipt["generation_possible"] = True
    return receipt


def predict_vector(
    model: Any,
    processor: Any,
    collator: Any,
    records: Sequence[Any],
    policy: Mapping[str, Any],
) -> list[str]:
    """Generate exactly one complete prediction vector under one frozen policy."""
    import torch
    from road_htr import qwen2_vl_challenger as challenger

    if not records:
        raise Model49GateError("cannot generate an empty vector")
    device = challenger._model_device(model)
    was_training = bool(model.training)
    old_cache = bool(getattr(model.config, "use_cache", False))
    model.eval()
    model.config.use_cache = True
    predictions: list[str] = []
    try:
        for position, record in enumerate(records, start=1):
            batch, _ = collator.encode_prompt(record)
            batch = challenger._to_device(batch, device)
            prompt_tokens = int(batch["input_ids"].shape[-1])
            generation = {key: value for key, value in policy.items() if key != "name"}
            with torch.inference_mode():
                output = model.generate(**batch, **generation)
            generated = output[:, prompt_tokens:]
            predictions.append(challenger._decode_generated(processor, generated))
            if position % 50 == 0 or position == len(records):
                print(f"{policy['name']}: generated {position}/{len(records)}", flush=True)
    finally:
        model.config.use_cache = old_cache
        model.train(was_training)
    if len(predictions) != len(records):
        raise Model49GateError("generation did not produce one complete vector")
    return predictions


def new_output_dir(config: Mapping[str, Any], output_dir: str | Path) -> Path:
    raw_root = ROOT / str(config["output_root"])
    if raw_root.exists() and raw_root.is_symlink():
        raise Model49GateError("isolated output root may not be a symlink")
    root = raw_root.resolve()
    path = resolve(output_dir)
    if not below(path, root) or path == root:
        raise Model49GateError("outputs must be isolated below outputs/model49")
    if path.exists():
        raise FileExistsError(f"output directory must be new: {path}")
    return path


def run_panel(
    config_path: str | Path,
    input_csv: str | Path,
    adapter_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate and score both full vectors from one in-memory adapter."""
    started = time.monotonic()
    config, resolved_config = load_config(config_path)
    output = new_output_dir(config, output_dir)
    before = cpu_readiness_audit(
        resolved_config,
        input_csv,
        adapter_path,
        verify_checkpoint=True,
    )
    from road_htr import qwen2_vl_broad as broad

    model39, _ = _load_model39_config(config)
    all_records = broad.load_records(
        resolve(input_csv),
        resolve(model39["image_dir"]),
        require_target=True,
        require_five_folds=True,
    )
    fold = int(config["base_model39"]["registered_fold"])
    records = tuple(record for record in all_records if record.fold == fold)
    if len(records) != int(config["input"]["fold0_rows"]):
        raise Model49GateError("refusing to decode a partial fold-0 vector")

    output.mkdir(parents=True, exist_ok=False)
    json_write(output / "receipt.initial.json", before)
    processor = broad.load_processor(model39)
    model, target_audit = broad._load_inference_model(model39, resolve(adapter_path))
    collator = broad.QwenOcrCollator(
        processor,
        str(model39["prompt"]),
        int(model39["image"]["max_image_tokens"]),
        int(model39["image"]["max_sequence_tokens"]),
    )
    candidates: list[dict[str, Any]] = []
    for policy in config["panel"]:
        predictions = predict_vector(model, processor, collator, records, policy)
        if len(predictions) != len(records):
            raise Model49GateError("refusing a partial vector output")
        csv_path = broad.write_predictions(
            output / f"{policy['name']}.csv", records, predictions
        )
        candidates.append(
            {
                "policy": policy,
                "csv": str(csv_path),
                "csv_sha256": sha256(csv_path),
                "rows": len(predictions),
                "whole_vector_metrics": broad.exact_metrics(records, predictions),
            }
        )

    after = cpu_readiness_audit(
        resolved_config,
        input_csv,
        adapter_path,
        verify_checkpoint=True,
    )
    immutable_keys = (
        "config_sha256",
        "model49_decode_sha256",
        "source_hashes",
        "input",
        "adapter",
        "checkpoint",
        "provenance",
        "label_constraint_audit",
    )
    if any(before[key] != after[key] for key in immutable_keys):
        raise Model49GateError(
            "TOCTOU: config, source, input, adapter, checkpoint, or label audit changed"
        )

    control_score = float(candidates[0]["whole_vector_metrics"]["multi_score"])
    challenger_score = float(candidates[1]["whole_vector_metrics"]["multi_score"])
    report = {
        "status": "complete",
        "model_number": 49,
        "runtime_seconds": time.monotonic() - started,
        "initial_receipt": "receipt.initial.json",
        "final_receipt": "receipt.final.json",
        "adapter_loaded_once": True,
        "same_checkpoint_and_epoch": True,
        "bound_best_epoch": before["adapter"]["best_epoch"],
        "target_audit": vars(target_audit),
        "whole_vector_only": True,
        "row_oracle_mixing": False,
        "submission_created": False,
        "only_policy_difference": "no_repeat_ngram_size",
        "candidates": candidates,
        "comparison": {
            "metric": "multi_score",
            "control": "greedy_ngram4_control",
            "challenger": "greedy_ngram8_challenger",
            "challenger_minus_control": challenger_score - control_score,
            "selection_performed": False,
        },
    }
    json_write(output / "receipt.final.json", after)
    json_write(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model49 frozen greedy no-repeat ngram decoding comparison"
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/train_folds.csv")
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--contract-audit", action="store_true")
    modes.add_argument("--dry-audit", action="store_true")
    args = parser.parse_args()
    if args.contract_audit:
        if args.adapter is not None or args.output_dir is not None:
            parser.error("--contract-audit does not accept an adapter or output directory")
        print(json.dumps(contract_audit(args.config, args.input), indent=2, ensure_ascii=False))
        return
    if args.dry_audit:
        if args.adapter is None:
            parser.error("--dry-audit requires the completed Model39 --adapter")
        if args.output_dir is not None:
            parser.error("--dry-audit does not write an output directory")
        print(
            json.dumps(
                cpu_readiness_audit(args.config, args.input, args.adapter),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.adapter is None or args.output_dir is None:
        parser.error("--adapter and --output-dir are required for generation")
    print(
        json.dumps(
            run_panel(args.config, args.input, args.adapter, args.output_dir),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
