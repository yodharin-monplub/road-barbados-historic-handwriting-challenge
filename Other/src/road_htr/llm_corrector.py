"""Competition-only prompt preparation and local-LLM OCR correction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics import levenshtein
from .schema import SchemaError, predictions_by_id, read_table
from .submission import clean_prediction_preserving_internal_spaces


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
SOURCE_NAMES = (
    "primary",
    "full",
    "enhanced",
    "center",
    "center_enhanced",
    "kraken",
)
SYSTEM_PROMPT = """You verify OCR transcriptions of seventeenth-century Barbados legal manuscripts.
The language is Early Modern English with period legal Latin, names, scribal abbreviations, and inconsistent historical spelling. Preserve forms such as publique, sayde, yee, pnts, prsents, w^ch, capitalization, punctuation, and spacing when supported by the OCR witnesses. Do not modernize, translate, expand abbreviations, add missing historical content, or explain your decision. Treat the supplied transcriptions as noisy witnesses of one handwritten line. Return exactly one corrected transcription enclosed in <transcription> and </transcription>."""


def absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, ids: Sequence[str], values: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
            writer.writeheader()
            for identifier in ids:
                writer.writerow({"ID": identifier, "prediction": values[identifier]})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = absolute(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid Model93 configuration schema")
    model = value.get("model", {})
    required_model = {"id", "revision", "license", "dtype", "max_model_len"}
    if not required_model <= set(model):
        raise ValueError("Model93 configuration is missing pinned model fields")
    if model["license"] != "Apache-2.0":
        raise ValueError("Model93 requires the audited Apache-2.0 checkpoint")
    if not re.fullmatch(r"[0-9a-f]{40}", str(model["revision"])):
        raise ValueError("model revision must be an immutable 40-character commit")
    generation = value.get("generation", {})
    if generation.get("thinking") is not False or float(
        generation.get("temperature", -1)
    ) != 0.0:
        raise ValueError("Model93 requires deterministic non-thinking generation")
    if value.get("validity", {}).get("ascii_only") is not True:
        raise ValueError("Model93 requires ASCII-only output validation")
    sources = value.get("sources", {})
    for split in ("oof", "test"):
        if set(sources.get(split, {})) != set(SOURCE_NAMES):
            raise ValueError(f"{split} sources must be exactly {SOURCE_NAMES}")
    return value, resolved


def _prediction_map(path: str | Path) -> tuple[Path, dict[str, str]]:
    resolved = absolute(path)
    table = read_table(resolved)
    column = table.target_column
    if column is None:
        candidates = [name for name in table.fieldnames if name != table.id_column]
        if len(candidates) != 1:
            raise SchemaError(f"prediction CSV must have one value column: {resolved}")
        column = candidates[0]
    values = {
        identifier: clean_prediction_preserving_internal_spaces(value)
        for identifier, value in predictions_by_id(table, column).items()
    }
    empty = [identifier for identifier, value in values.items() if not value]
    if empty:
        raise SchemaError(f"empty source predictions in {resolved}: {empty[:5]}")
    return resolved, values


def load_sources(
    config: Mapping[str, Any], split: str, required_ids: Sequence[str]
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    if split not in {"oof", "test"}:
        raise ValueError("split must be oof or test")
    expected = set(required_ids)
    sources: dict[str, dict[str, str]] = {}
    provenance: dict[str, dict[str, str]] = {}
    for name in SOURCE_NAMES:
        path, values = _prediction_map(config["sources"][split][name])
        missing = expected - set(values)
        unexpected = set(values) - expected
        if missing or unexpected:
            raise SchemaError(
                f"{split} {name} ID mismatch: missing={sorted(missing)[:5]}, "
                f"unexpected={sorted(unexpected)[:5]}"
            )
        sources[name] = values
        provenance[name] = {"path": str(path), "sha256": sha256(path)}
    return sources, provenance


def _normal(value: str) -> str:
    return " ".join(value.casefold().split())


def _character_ngrams(value: str, order: int = 3) -> frozenset[str]:
    normalized = f"  {_normal(value)}  "
    return frozenset(
        normalized[index : index + order]
        for index in range(max(len(normalized) - order + 1, 1))
    )


def _words(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z]+", _normal(value)))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@dataclass(frozen=True)
class RetrievalItem:
    identifier: str
    fold: int
    target: str
    primary: str
    correction: bool
    character_ngrams: frozenset[str]
    words: frozenset[str]


def _edit_ratio(source: str, target: str) -> float:
    return levenshtein(tuple(source), tuple(target)) / max(len(source), len(target), 1)


def _format_witnesses(sources: Mapping[str, Mapping[str, str]], identifier: str) -> str:
    labels = {
        "primary": "PRIMARY",
        "full": "FULL",
        "enhanced": "ENHANCED",
        "center": "CENTER",
        "center_enhanced": "CENTER_ENHANCED",
        "kraken": "KRAKEN",
    }
    return "\n".join(
        f"{labels[name]}: {sources[name][identifier]}" for name in SOURCE_NAMES
    )


def _select_examples(
    identifier: str,
    fold: int | None,
    primary: str,
    items: Sequence[RetrievalItem],
    correction_count: int,
    identity_count: int,
    character_weight: float,
    word_weight: float,
) -> list[tuple[RetrievalItem, float]]:
    query_characters = _character_ngrams(primary)
    query_words = _words(primary)
    ranked: list[tuple[float, RetrievalItem]] = []
    for item in items:
        if item.identifier == identifier or (fold is not None and item.fold == fold):
            continue
        score = character_weight * _jaccard(
            query_characters, item.character_ngrams
        ) + word_weight * _jaccard(query_words, item.words)
        ranked.append((score, item))
    ranked.sort(key=lambda value: (-value[0], value[1].identifier))

    chosen: list[tuple[RetrievalItem, float]] = []
    for wanted, count in ((True, correction_count), (False, identity_count)):
        matches = [(item, score) for score, item in ranked if item.correction == wanted]
        chosen.extend(matches[:count])
    chosen.sort(key=lambda value: (-value[1], value[0].identifier))
    if len(chosen) != correction_count + identity_count:
        raise RuntimeError("not enough fold-excluded correction/identity examples")
    return chosen


def _build_prompt(
    identifier: str,
    sources: Mapping[str, Mapping[str, str]],
    examples: Sequence[tuple[RetrievalItem, float]],
) -> str:
    sections = []
    for index, (example, _) in enumerate(examples, start=1):
        sections.append(
            f"EXAMPLE {index}\n{_format_witnesses(sources, example.identifier)}\n"
            f"CORRECT: {example.target}"
        )
    sections.append(
        "LINE TO TRANSCRIBE\n"
        + _format_witnesses(sources, identifier)
        + "\nCORRECT:"
    )
    return "\n\n".join(sections)


def prepare_prompts(
    config_path: str | Path,
    split: str,
    output_path: str | Path,
    fold: int | None = None,
) -> dict[str, Any]:
    config, resolved_config = load_config(config_path)
    truth_path = absolute(config["data"]["truth"])
    truth = read_table(truth_path, require_target=True)
    assert truth.target_column is not None
    truth_ids = [row[truth.id_column].strip() for row in truth.rows]
    truth_by_id = {row[truth.id_column].strip(): row for row in truth.rows}
    if split == "oof":
        if fold is not None and fold not in {int(row["fold"]) for row in truth.rows}:
            raise ValueError(f"fold {fold} is absent")
        selected_ids = [
            identifier
            for identifier in truth_ids
            if fold is None or int(truth_by_id[identifier]["fold"]) == fold
        ]
        required_ids = truth_ids
    elif split == "test":
        if fold is not None:
            raise ValueError("test prompt preparation does not accept --fold")
        test_path = absolute(config["data"]["test"])
        test = read_table(test_path)
        selected_ids = [row[test.id_column].strip() for row in test.rows]
        required_ids = selected_ids
    else:
        raise ValueError("split must be oof or test")

    split_sources, split_provenance = load_sources(config, split, required_ids)
    oof_sources, oof_provenance = (
        (split_sources, split_provenance)
        if split == "oof"
        else load_sources(config, "oof", truth_ids)
    )
    retrieval = config["retrieval"]
    maximum_example_edit_ratio = float(retrieval["maximum_example_edit_ratio"])
    character_weight = float(retrieval["character_similarity_weight"])
    word_weight = float(retrieval["word_similarity_weight"])
    if character_weight < 0 or word_weight < 0 or not math.isclose(
        character_weight + word_weight, 1.0
    ):
        raise ValueError("retrieval similarity weights must be nonnegative and sum to one")
    items = []
    for identifier in truth_ids:
        target = truth_by_id[identifier][truth.target_column]
        primary = oof_sources["primary"][identifier]
        ratio = _edit_ratio(primary, target)
        correction = primary != target and ratio <= maximum_example_edit_ratio
        # Very distant examples do not demonstrate controlled typo repair.
        if primary != target and not correction:
            continue
        items.append(
            RetrievalItem(
                identifier,
                int(truth_by_id[identifier]["fold"]),
                target,
                primary,
                correction,
                _character_ngrams(primary),
                _words(primary),
            )
        )

    records = []
    for identifier in selected_ids:
        row_fold = int(truth_by_id[identifier]["fold"]) if split == "oof" else None
        primary = split_sources["primary"][identifier]
        examples = _select_examples(
            identifier,
            row_fold,
            primary,
            items,
            int(retrieval["correction_examples"]),
            int(retrieval["identity_examples"]),
            character_weight,
            word_weight,
        )
        if split == "oof":
            user_prompt = _build_prompt(identifier, split_sources, examples)
        else:
            # Test prompts need training witnesses for their examples, not test witnesses.
            example_sections = []
            for index, (example, _) in enumerate(examples, start=1):
                example_sections.append(
                    f"EXAMPLE {index}\n{_format_witnesses(oof_sources, example.identifier)}\n"
                    f"CORRECT: {example.target}"
                )
            example_sections.append(
                "LINE TO TRANSCRIBE\n"
                + _format_witnesses(split_sources, identifier)
                + "\nCORRECT:"
            )
            user_prompt = "\n\n".join(example_sections)
        prompt_hash = hashlib.sha256(
            (SYSTEM_PROMPT + "\0" + user_prompt).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": identifier,
                "split": split,
                "fold": row_fold,
                "primary": primary,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "retrieved_examples": [
                    {
                        "id": example.identifier,
                        "fold": example.fold,
                        "kind": "correction" if example.correction else "identity",
                        "similarity": score,
                    }
                    for example, score in examples
                ],
                "prompt_sha256": prompt_hash,
            }
        )

    output = absolute(output_path)
    serialized = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in records)
    _atomic_text(output, serialized)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "model93_prompt_manifest",
        "split": split,
        "fold": fold,
        "rows": len(records),
        "config": str(resolved_config),
        "config_sha256": sha256(resolved_config),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "truth": str(truth_path),
        "truth_sha256": sha256(truth_path),
        "sources": split_provenance,
        "oof_retrieval_sources": oof_provenance,
        "prompt_file": str(output),
        "prompt_file_sha256": sha256(output),
        "fold_exclusion_enforced": split == "oof",
        "external_corpus_used": False,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


_TAG = re.compile(r"^\s*<transcription>(.*?)</transcription>\s*$", re.I | re.S)
_PREFIX = re.compile(r"^(?:correct|transcription|answer)\s*:\s*", re.I)


def parse_generation(
    raw_output: str,
    primary: str,
    *,
    maximum_characters: int,
    maximum_edit_ratio: float,
    minimum_word_ratio: float,
    maximum_word_ratio: float,
) -> tuple[str, str | None]:
    raw = str(raw_output).strip()
    if not raw:
        return primary, "blank_output"
    match = _TAG.fullmatch(raw)
    if match:
        value = match.group(1).strip()
    else:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines) != 1 or "<think>" in raw.casefold():
            return primary, "malformed_output"
        value = _PREFIX.sub("", lines[0]).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1].strip()
    if not value:
        return primary, "blank_output"
    if "\n" in value or "\r" in value:
        return primary, "multiline_output"
    if not value.isascii():
        return primary, "non_ascii_output"
    if len(value) > maximum_characters:
        return primary, "overlength_output"
    word_ratio = len(value.split()) / max(len(primary.split()), 1)
    if not minimum_word_ratio <= word_ratio <= maximum_word_ratio:
        return primary, "word_count_outlier"
    if _edit_ratio(primary, value) > maximum_edit_ratio:
        return primary, "edit_distance_outlier"
    return clean_prediction_preserving_internal_spaces(value), None


def _read_prompt_records(path: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SchemaError(f"invalid prompt JSON at {path}:{line_number}") from error
            identifier = str(row.get("id", "")).strip()
            if not identifier or identifier in seen:
                raise SchemaError(f"missing/duplicate prompt ID at {path}:{line_number}")
            expected_hash = hashlib.sha256(
                (row["messages"][0]["content"] + "\0" + row["messages"][1]["content"]).encode("utf-8")
            ).hexdigest()
            if row.get("prompt_sha256") != expected_hash:
                raise SchemaError(f"prompt hash mismatch for {identifier}")
            seen.add(identifier)
            records.append(row)
    if not records:
        raise SchemaError("prompt file is empty")
    return records


def verify_model_audit(
    model_audit_file: Path,
    config: Mapping[str, Any],
    resolved_config: Path,
) -> tuple[dict[str, Any], Path]:
    """Fail closed if the pinned local checkpoint no longer matches its audit."""

    model_audit = json.loads(model_audit_file.read_text(encoding="utf-8"))
    if not model_audit.get("complete"):
        raise RuntimeError("model snapshot audit is incomplete")
    if model_audit.get("config_sha256") != sha256(resolved_config):
        raise RuntimeError("model snapshot audit was produced from a different config")
    if model_audit.get("model_id") != config["model"]["id"] or model_audit.get(
        "revision"
    ) != config["model"]["revision"]:
        raise RuntimeError("model snapshot audit does not match configuration")
    model_snapshot = Path(model_audit["snapshot"])
    if not model_snapshot.is_dir():
        raise FileNotFoundError(model_snapshot)

    # Recheck the full snapshot before each inference process. This costs one
    # sequential 61 GB read, but prevents a corrupted or swapped shard from being
    # trusted merely because its filename and byte count stayed unchanged.
    metadata = model_audit.get("metadata", {})
    expected_metadata = config["model"]["expected_metadata_sha256"]
    if set(metadata) != set(expected_metadata):
        raise RuntimeError("model audit metadata ledger is incomplete")
    for name, expected_hash in expected_metadata.items():
        path = model_snapshot / name
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"audited model metadata changed: {name}")
        if int(metadata[name].get("bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"audited model metadata size changed: {name}")

    shards = model_audit.get("shards", [])
    if len(shards) != int(config["model"]["expected_shards"]):
        raise RuntimeError("model audit shard ledger is incomplete")
    actual_bytes = 0
    for shard in shards:
        path = model_snapshot / shard["name"]
        if not path.is_file() or path.stat().st_size != int(shard["bytes"]):
            raise RuntimeError(f"audited model shard missing or resized: {shard['name']}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(shard.get("sha256", ""))):
            raise RuntimeError(f"invalid audited shard digest: {shard['name']}")
        if sha256(path) != shard["sha256"]:
            raise RuntimeError(f"audited model shard changed: {shard['name']}")
        actual_bytes += path.stat().st_size
    if actual_bytes != int(config["model"]["expected_weight_file_bytes"]):
        raise RuntimeError("audited physical model bytes differ from configuration")
    return model_audit, model_snapshot


def run_local_llm(
    config_path: str | Path,
    prompts_path: str | Path,
    model_audit_path: str | Path,
    output_path: str | Path,
    details_path: str | Path,
    batch_size: int,
    resume: bool,
) -> dict[str, Any]:
    config, resolved_config = load_config(config_path)
    model_audit_file = absolute(model_audit_path)
    model_audit, model_snapshot = verify_model_audit(
        model_audit_file, config, resolved_config
    )
    prompts = absolute(prompts_path)
    prompt_manifest_path = prompts.with_suffix(prompts.suffix + ".manifest.json")
    prompt_manifest = json.loads(prompt_manifest_path.read_text(encoding="utf-8"))
    if prompt_manifest.get("prompt_file_sha256") != sha256(prompts):
        raise RuntimeError("prompt file differs from its manifest")
    if prompt_manifest.get("config_sha256") != sha256(resolved_config):
        raise RuntimeError("prompt manifest was produced from a different config")
    if prompt_manifest.get("implementation_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("prompt manifest was produced by a different implementation")
    records = _read_prompt_records(prompts)
    if int(prompt_manifest.get("rows", -1)) != len(records):
        raise RuntimeError("prompt row count differs from its manifest")
    if batch_size < 1:
        raise ValueError("batch size must be positive")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import torch
        import transformers
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError("Model93 inference requires the isolated vLLM environment") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Model93 inference requires CUDA")
    if torch.cuda.get_device_properties(0).total_memory < float(
        config["model"]["minimum_gpu_memory_gib"]
    ) * 1024**3:
        raise RuntimeError("Qwen3-30B-A3B BF16 requires an 80 GB-class GPU")

    output = absolute(output_path)
    details = absolute(details_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    details.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = details.with_suffix(details.suffix + ".manifest.json")
    run_identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": "model93_inference_manifest",
        "config_sha256": sha256(resolved_config),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "prompt_file_sha256": sha256(prompts),
        "model_audit_sha256": sha256(model_audit_file),
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "dtype": config["model"]["dtype"],
        "seed": int(config["generation"]["seed"]),
    }
    completed: dict[str, dict[str, Any]] = {}
    if details.exists():
        if not resume:
            raise FileExistsError(details)
        if not manifest_path.is_file():
            raise RuntimeError("cannot resume without an inference manifest")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(prior.get(key) != value for key, value in run_identity.items()):
            raise RuntimeError("resume manifest differs from this run")
        with details.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                completed[row["id"]] = row
    elif output.exists():
        raise FileExistsError(output)
    else:
        _atomic_text(manifest_path, json.dumps({**run_identity, "complete": False}, indent=2) + "\n")

    model = config["model"]
    generation = config["generation"]
    engine = LLM(
        model=str(model_snapshot),
        dtype=model["dtype"],
        trust_remote_code=False,
        tensor_parallel_size=1,
        max_model_len=int(model["max_model_len"]),
        gpu_memory_utilization=float(model["gpu_memory_utilization"]),
        seed=int(generation["seed"]),
    )
    sampling = SamplingParams(
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        max_tokens=int(generation["max_new_tokens"]),
        seed=int(generation["seed"]),
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_snapshot), local_files_only=True, trust_remote_code=False
    )
    remaining = [row for row in records if row["id"] not in completed]
    validity = config["validity"]
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start : start + batch_size]
        rendered_prompts = [
            tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in batch
        ]
        results = engine.generate(
            rendered_prompts, sampling_params=sampling, use_tqdm=False
        )
        if len(results) != len(batch):
            raise RuntimeError("vLLM returned the wrong batch size")
        with details.open("a", encoding="utf-8") as handle:
            for row, result in zip(batch, results, strict=True):
                raw = result.outputs[0].text
                prediction, reason = parse_generation(
                    raw,
                    row["primary"],
                    maximum_characters=int(validity["maximum_characters"]),
                    maximum_edit_ratio=float(validity["maximum_edit_ratio"]),
                    minimum_word_ratio=float(validity["minimum_word_ratio"]),
                    maximum_word_ratio=float(validity["maximum_word_ratio"]),
                )
                detail = {
                    "id": row["id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "primary": row["primary"],
                    "raw_output": raw,
                    "prediction": prediction,
                    "fallback_reason": reason,
                    "edit_ratio": _edit_ratio(row["primary"], prediction),
                }
                handle.write(json.dumps(detail, ensure_ascii=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed[row["id"]] = detail

    ids = [row["id"] for row in records]
    if set(completed) != set(ids):
        raise RuntimeError("inference ended with incomplete IDs")
    predictions = {identifier: completed[identifier]["prediction"] for identifier in ids}
    _atomic_csv(output, ids, predictions)
    final_manifest = {
        **run_identity,
        "complete": True,
        "rows": len(ids),
        "fallback_rows": sum(completed[identifier]["fallback_reason"] is not None for identifier in ids),
        "output": str(output),
        "output_sha256": sha256(output),
        "details": str(details),
        "details_sha256": sha256(details),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    _atomic_text(manifest_path, json.dumps(final_manifest, indent=2, sort_keys=True) + "\n")
    return final_manifest


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fold-safe Model93 prompts")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("oof", "test"), required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_prompts(args.config, args.split, args.output, args.fold)
    print(json.dumps(report, indent=2))


def inference_main() -> None:
    parser = argparse.ArgumentParser(description="Run local Qwen Model93 correction")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run_local_llm(
        args.config,
        args.prompts,
        args.model_audit,
        args.output,
        args.details,
        args.batch_size,
        args.resume,
    )
    print(json.dumps(report, indent=2))
