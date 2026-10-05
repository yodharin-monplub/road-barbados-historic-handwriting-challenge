"""Audited, resumable direct-image OCR with Qwen3-VL for Model95."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import ImageIndex, read_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
MODEL_KIND = "model95_qwen3_vl_direct"


def absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = absolute(path)
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid Model95 configuration schema")
    model = config.get("model", {})
    if not re.fullmatch(r"[0-9a-f]{40}", str(model.get("revision", ""))):
        raise ValueError("Model95 model revision must be an immutable commit hash")
    if str(model.get("license", "")).casefold() != "apache-2.0":
        raise ValueError("Model95 only permits the audited Apache-2.0 checkpoint")
    if float(config["generation"]["temperature"]) != 0.0:
        raise ValueError("Model95 requires deterministic temperature-zero decoding")
    if not str(config.get("prompt", "")).strip():
        raise ValueError("Model95 prompt must not be blank")
    return config, resolved


def _module_path() -> Path:
    return Path(__file__).resolve()


def _manifest_path(records_path: Path) -> Path:
    return records_path.with_suffix(records_path.suffix + ".manifest.json")


def prepare_image_manifest(
    config_path: str | Path,
    split: str,
    output_path: str | Path,
    fold: int | None = None,
) -> dict[str, Any]:
    """Create an ID/image-only manifest; training targets never enter prompts."""

    config, resolved_config = load_config(config_path)
    if split == "oof":
        if fold is None:
            raise ValueError("--fold is required for OOF preparation")
        source_path = absolute(config["data"]["truth"])
        source = read_table(source_path, require_target=True)
        if "fold" not in source.fieldnames:
            raise ValueError(f"fold column missing from {source_path}")
        rows = [row for row in source.rows if int(row["fold"]) == fold]
    elif split == "test":
        if fold is not None:
            raise ValueError("test preparation does not accept --fold")
        source_path = absolute(config["data"]["test"])
        source = read_table(source_path)
        rows = list(source.rows)
    else:
        raise ValueError("split must be oof or test")
    if not rows:
        raise ValueError("selected image manifest is empty")

    images = ImageIndex(absolute(config["data"]["image_dir"]))
    records = []
    for row in rows:
        identifier = row[source.id_column].strip()
        image_path = images.resolve(identifier)
        records.append(
            {
                "id": identifier,
                "split": split,
                "fold": fold,
                "image_name": image_path.name,
                "image_sha256": sha256(image_path),
            }
        )
    output = absolute(output_path)
    body = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    atomic_text(output, body)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{MODEL_KIND}_image_manifest",
        "complete": True,
        "split": split,
        "fold": fold,
        "rows": len(records),
        "labels_in_records": False,
        "prompt_in_records": False,
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "image_dir": str(absolute(config["data"]["image_dir"])),
        "records": str(output),
        "records_sha256": sha256(output),
        "config": str(resolved_config),
        "config_sha256": sha256(resolved_config),
        "implementation_sha256": sha256(_module_path()),
    }
    atomic_json(_manifest_path(output), manifest)
    return manifest


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        required = {"id", "split", "fold", "image_name", "image_sha256"}
        if set(value) != required:
            raise ValueError(f"unexpected fields at {path}:{line_number}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(value["image_sha256"])):
            raise ValueError(f"invalid image hash at {path}:{line_number}")
        records.append(value)
    identifiers = [str(record["id"]) for record in records]
    if not records or len(identifiers) != len(set(identifiers)):
        raise ValueError("record file is empty or has duplicate IDs")
    return records


def verify_inputs(
    config_path: Path, records_path: Path, audit_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    config, resolved_config = load_config(config_path)
    records_path = absolute(records_path)
    manifest_path = _manifest_path(records_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("labels_in_records") is not False:
        raise RuntimeError("image manifest is incomplete or label-contaminated")
    expected = {
        "records_sha256": sha256(records_path),
        "config_sha256": sha256(resolved_config),
        "implementation_sha256": sha256(_module_path()),
    }
    for field, actual in expected.items():
        if manifest.get(field) != actual:
            raise RuntimeError(f"image manifest {field} mismatch")
    records = read_records(records_path)
    if int(manifest.get("rows", -1)) != len(records):
        raise RuntimeError("image manifest row count mismatch")

    audit = json.loads(absolute(audit_path).read_text(encoding="utf-8"))
    model = config["model"]
    if audit.get("complete") is not True:
        raise RuntimeError("checkpoint audit is incomplete")
    if audit.get("model_id") != model["id"] or audit.get("revision") != model["revision"]:
        raise RuntimeError("checkpoint audit model/revision mismatch")
    if audit.get("config_sha256") != sha256(resolved_config):
        raise RuntimeError("checkpoint audit config hash mismatch")
    snapshot = Path(audit["snapshot"]).resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    return config, records, snapshot


_WRAPPER = re.compile(r"<\|(?:im_start|im_end|assistant|user)\|>", re.IGNORECASE)
_PREFIX = re.compile(r"^\s*(?:transcription|text|output|answer)\s*:\s*", re.IGNORECASE)


def parse_generation(raw: str) -> str:
    """Remove response wrappers without modernizing or spell-correcting text."""

    value = _WRAPPER.sub("", str(raw)).strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```[^\n]*\n?", "", value)
        value = re.sub(r"\n?```$", "", value)
    value = _PREFIX.sub("", value.strip()).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return re.sub(r"\r\n|\r|\n", " ", value).strip()


def render_prompt(processor: Any, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _append_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        identifier = str(value.get("id", ""))
        if not identifier or identifier in completed:
            raise ValueError(f"invalid/duplicate resumed row at {path}:{number}")
        completed[identifier] = value
    return completed


def _write_predictions(path: Path, records: Sequence[Mapping[str, Any]], values: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
            writer.writeheader()
            for record in records:
                writer.writerow({"ID": record["id"], "prediction": values[record["id"]]})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_inference(
    config_path: str | Path,
    records_path: str | Path,
    audit_path: str | Path,
    output_dir: str | Path,
    batch_size: int | None = None,
) -> dict[str, Any]:
    config, records, snapshot = verify_inputs(
        absolute(config_path), absolute(records_path), absolute(audit_path)
    )
    output = absolute(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    details_path = output / "details.jsonl"
    prediction_path = output / "predictions.csv"
    completed = _load_completed(details_path)
    expected_ids = {str(record["id"]) for record in records}
    if set(completed) - expected_ids:
        raise RuntimeError("resume file contains IDs outside the current manifest")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    import vllm
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("Model95 requires CUDA")
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    required_memory = float(config["model"]["minimum_gpu_memory_gib"])
    if memory < required_memory:
        raise RuntimeError(f"GPU has {memory:.1f} GiB; Model95 requires {required_memory:.1f} GiB")

    image_config = config["image"]
    processor = AutoProcessor.from_pretrained(
        str(snapshot),
        trust_remote_code=False,
        local_files_only=True,
        min_pixels=int(image_config["min_pixels"]),
        max_pixels=int(image_config["max_pixels"]),
    )
    rendered = render_prompt(processor, config["prompt"])
    engine = LLM(
        model=str(snapshot),
        tokenizer=str(snapshot),
        trust_remote_code=False,
        dtype=str(config["model"]["dtype"]),
        max_model_len=int(config["model"]["max_model_len"]),
        gpu_memory_utilization=float(config["model"]["gpu_memory_utilization"]),
        seed=int(config["generation"]["seed"]),
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs={
            "min_pixels": int(image_config["min_pixels"]),
            "max_pixels": int(image_config["max_pixels"]),
        },
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=float(config["generation"]["top_p"]),
        max_tokens=int(config["generation"]["max_new_tokens"]),
        seed=int(config["generation"]["seed"]),
    )
    image_dir = absolute(config["data"]["image_dir"])
    pending = [record for record in records if record["id"] not in completed]
    size = batch_size or int(config.get("execution", {}).get("batch_size", 32))
    if size < 1:
        raise ValueError("batch size must be positive")
    for offset in range(0, len(pending), size):
        batch = pending[offset : offset + size]
        images = []
        requests = []
        try:
            for record in batch:
                image_path = image_dir / str(record["image_name"])
                if sha256(image_path) != record["image_sha256"]:
                    raise RuntimeError(f"image hash mismatch: {image_path}")
                image = Image.open(image_path).convert("RGB")
                images.append(image)
                requests.append(
                    {
                        "prompt": rendered,
                        "multi_modal_data": {"image": image},
                        "multi_modal_uuids": {"image": record["image_sha256"]},
                    }
                )
            generated = engine.generate(requests, sampling, use_tqdm=False)
            if len(generated) != len(batch):
                raise RuntimeError("vLLM returned an unexpected number of results")
            details = []
            for record, result in zip(batch, generated, strict=True):
                raw = result.outputs[0].text if result.outputs else ""
                parsed = parse_generation(raw)
                details.append(
                    {
                        "id": record["id"],
                        "image_sha256": record["image_sha256"],
                        "raw": raw,
                        "prediction": parsed,
                        "valid": bool(parsed)
                        and len(parsed) <= int(config["validity"]["maximum_characters"]),
                    }
                )
            _append_jsonl(details_path, details)
            completed.update({str(row["id"]): row for row in details})
            print(f"completed {len(completed)}/{len(records)}", flush=True)
        finally:
            for image in images:
                image.close()

    predictions = {identifier: str(row["prediction"]) for identifier, row in completed.items()}
    if set(predictions) != expected_ids:
        raise RuntimeError("inference did not produce every requested ID")
    _write_predictions(prediction_path, records, predictions)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": MODEL_KIND,
        "complete": True,
        "rows": len(records),
        "invalid_rows": sum(not bool(completed[str(row["id"])]["valid"]) for row in records),
        "prompt": config["prompt"],
        "model_id": config["model"]["id"],
        "revision": config["model"]["revision"],
        "snapshot": str(snapshot),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": memory,
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
        "records_sha256": sha256(absolute(records_path)),
        "details_sha256": sha256(details_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
    }
    atomic_json(output / "run.json", report)
    return report


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare label-free Model95 image records")
    parser.add_argument("--config", type=Path, default=Path("model95/config.json"))
    parser.add_argument("--split", choices=("oof", "test"), required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_image_manifest(args.config, args.split, args.output, args.fold), indent=2))


def inference_main() -> None:
    parser = argparse.ArgumentParser(description="Run audited direct Qwen3-VL OCR")
    parser.add_argument("--config", type=Path, default=Path("model95/config.json"))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    print(json.dumps(run_inference(args.config, args.records, args.audit, args.output_dir, args.batch_size), indent=2))
