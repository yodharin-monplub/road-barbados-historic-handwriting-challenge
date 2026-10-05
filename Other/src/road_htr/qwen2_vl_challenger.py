"""Guarded QLoRA training for the organizer's Qwen2-VL 2B starter model.

Heavy dependencies are imported only inside runtime entry points.  This keeps
the configuration, provenance, schema, and architecture audits testable on a
CPU-only machine without downloading a checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image

from .metrics import competition_error, strip_conversation_tokens
from .schema import ImageIndex, SchemaError, read_table
from .submission import clean_prediction_preserving_internal_spaces, write_submission


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
MODEL_LICENSE = "Apache-2.0"
MODEL_CLASS = "Qwen2VLForConditionalGeneration"
LOCAL_CHECKPOINT_PATH = "checkpoints/model27/qwen2-vl-2b-instruct"
LOCAL_CHECKPOINT_FILES = {
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "README.md": "74fbe869df708593087ae2ee66a6605d33fc2025937644721dc1360963d037f4",
    "chat_template.json": "ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96",
    "config.json": "422adefa19e62dd175961cec85bc0400344fe5bf9b22bd1182e05aaae78556e0",
    "generation_config.json": "d2864bf1edea5863d331edfff48106b586a366f5a2c41aa77731fadc53aa25d2",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model-00001-of-00002.safetensors": (
        "994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9"
    ),
    "model-00002-of-00002.safetensors": (
        "92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833"
    ),
    "model.safetensors.index.json": (
        "260ab9fa1418d6d6ab79daa1d9da2c47264f3b72edb4630fc799077ac67d27c6"
    ),
    "preprocessor_config.json": (
        "b5eaad0c2815f07631535dcc58f3c462b0d73693638ad21d19f3c50820eae1cc"
    ),
    "tokenizer.json": "cb63a0a23eef3d5b01063a9880a1925a65aaf4d1591d519910ee3527852950a0",
    "tokenizer_config.json": (
        "ff5c4fd898fe8c39591eb70e5d39d2782802d4204d6ae9ba1223252f354842a0"
    ),
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}

STARTER_ARCHIVE = "data/raw/Starters.zip"
STARTER_ARCHIVE_SHA256 = (
    "38de286b4c3958fc2fbebe0135ed487e0a4b9889a203ff62658603d010ec4150"
)
STARTER_CONFIG_MEMBER = "Starters/VLM/config.yaml"
STARTER_CONFIG_SHA256 = (
    "10d5a991f5963c784fb01cb2b03b55bba53360df825551b692d4b8149a3fa4d7"
)
STARTER_TRAINER_MEMBER = "Starters/VLM/trainer.py"
STARTER_TRAINER_SHA256 = (
    "1b38ead00654ef0c5f7c6a1cc62860d63dceb27f7941adfa40e252a5ee75839e"
)

LANGUAGE_TARGET = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.self_attn\.(?P<projection>q_proj|v_proj)$"
)
MERGER_TARGETS = ("visual.merger.mlp.0", "visual.merger.mlp.2")
EXPECTED_LANGUAGE_LAYERS = 28
EXPECTED_TARGET_MODULES = EXPECTED_LANGUAGE_LAYERS * 2 + len(MERGER_TARGETS)
PROTECTED_MODEL22_OUTPUT = (PROJECT_ROOT / "outputs" / "model22").resolve()
GIB = 1024**3


class SafetyGateError(RuntimeError):
    """Raised when a pre-training safety or reproducibility gate fails."""


@dataclass(frozen=True)
class OcrRecord:
    identifier: str
    image_path: Path
    label: str | None
    fold: int | None


@dataclass(frozen=True)
class LoraTargetAudit:
    language_root: str
    language_layers: int
    q_modules: int
    v_modules: int
    merger_modules: int
    target_modules: tuple[str, ...]


def _absolute(path: str | Path, relative_to: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (relative_to / candidate).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_sha256(archive: Path, member: str) -> str:
    with zipfile.ZipFile(archive) as bundle:
        try:
            payload = bundle.read(member)
        except KeyError as exc:
            raise SafetyGateError(f"starter archive is missing {member!r}") from exc
    return hashlib.sha256(payload).hexdigest()


def trim_label_edges(value: object) -> str:
    """Remove only leading/trailing whitespace; preserve the interior verbatim."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {resolved}")
    validate_config(config)
    return config, resolved


def _nested(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"missing required config key: {'.'.join(keys)}")
        value = value[key]
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the immutable architecture and 6 GiB training contract."""

    immutable = {
        ("model", "id"): MODEL_ID,
        ("model", "revision"): MODEL_REVISION,
        ("model", "license"): MODEL_LICENSE,
        ("model", "class"): MODEL_CLASS,
        ("model", "trust_remote_code"): False,
        ("model", "local_path"): LOCAL_CHECKPOINT_PATH,
        ("model", "local_files_only"): True,
        ("model", "attn_implementation"): "sdpa",
        ("quantization", "load_in_4bit"): True,
        ("quantization", "type"): "nf4",
        ("quantization", "double_quant"): True,
        ("quantization", "compute_dtype"): "float16",
        ("training", "batch_size"): 1,
        ("training", "eval_batch_size"): 1,
        ("training", "gradient_checkpointing"): True,
        ("lora", "rank"): 16,
        ("lora", "include_verified_merger"): True,
        ("architecture", "expected_language_layers"): EXPECTED_LANGUAGE_LAYERS,
        ("architecture", "expected_target_modules"): EXPECTED_TARGET_MODULES,
        ("cross_validation", "folds"): 5,
        ("generation", "do_sample"): False,
        ("generation", "num_beams"): 1,
        ("smoke", "minimum_4bit_modules"): 100,
    }
    for keys, expected in immutable.items():
        actual = _nested(config, *keys)
        if actual != expected:
            raise ValueError(
                f"{'.'.join(keys)} must be {expected!r} for Model27, got {actual!r}"
            )

    projections = _nested(config, "lora", "language_projections")
    if projections != ["q_proj", "v_proj"]:
        raise ValueError("lora.language_projections must be exactly [q_proj, v_proj]")

    min_pixels = int(_nested(config, "image", "min_pixels"))
    max_pixels = int(_nested(config, "image", "max_pixels"))
    patch_unit = int(_nested(config, "image", "merged_patch_size"))
    max_image_tokens = int(_nested(config, "image", "max_image_tokens"))
    max_sequence_tokens = int(_nested(config, "image", "max_sequence_tokens"))
    if patch_unit != 28:
        raise ValueError("Qwen2-VL merged_patch_size must be 28")
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("image pixel bounds must be positive and ordered")
    if min_pixels % patch_unit**2 or max_pixels % patch_unit**2:
        raise ValueError("pixel bounds must be multiples of the merged patch area")
    if max_pixels != max_image_tokens * patch_unit**2:
        raise ValueError("max_pixels must exactly encode max_image_tokens")
    if not 1 <= max_image_tokens <= 320:
        raise ValueError("Model27 limits images to at most 320 visual tokens")
    if not max_image_tokens < max_sequence_tokens <= 512:
        raise ValueError("max_sequence_tokens must be above image tokens and at most 512")

    memory_gib = float(_nested(config, "smoke", "max_allocated_gib"))
    if memory_gib <= 0 or memory_gib > 5.3:
        raise ValueError("smoke.max_allocated_gib must be positive and at most 5.3")
    reserved_gib = float(_nested(config, "smoke", "max_reserved_gib"))
    if reserved_gib <= 0 or reserved_gib > 5.3:
        raise ValueError("smoke.max_reserved_gib must be positive and at most 5.3")
    if int(_nested(config, "smoke", "expected_target_modules")) != EXPECTED_TARGET_MODULES:
        raise ValueError(f"smoke.expected_target_modules must be {EXPECTED_TARGET_MODULES}")
    if int(_nested(config, "smoke", "max_trainable_parameters")) > 4_000_000:
        raise ValueError("max_trainable_parameters may not exceed 4,000,000")

    output_template = str(_nested(config, "output_dir"))
    if "{fold}" not in output_template:
        raise ValueError("output_dir must contain the deterministic {fold} placeholder")
    if int(_nested(config, "training", "epochs")) < 1:
        raise ValueError("training.epochs must be at least one")
    if int(_nested(config, "training", "gradient_accumulation")) < 1:
        raise ValueError("training.gradient_accumulation must be at least one")

    checkpoint_files = _nested(config, "model", "expected_files_sha256")
    if checkpoint_files != LOCAL_CHECKPOINT_FILES:
        raise ValueError("model.expected_files_sha256 differs from the pinned local ledger")


def verify_local_checkpoint(
    config: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Verify the complete offline checkpoint snapshot and architecture metadata."""

    checkpoint = _absolute(str(_nested(config, "model", "local_path")), project_root)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    expected = dict(_nested(config, "model", "expected_files_sha256"))
    actual_paths = {
        path.relative_to(checkpoint).as_posix(): path
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    if set(actual_paths) != set(expected):
        missing = sorted(set(expected) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected))
        raise SafetyGateError(
            f"local checkpoint file set mismatch: missing={missing}, extra={extra}"
        )
    hashes: dict[str, str] = {}
    for name in sorted(expected):
        path = actual_paths[name]
        if path.is_symlink():
            raise SafetyGateError(f"local checkpoint file may not be a symlink: {path}")
        digest = _sha256(path)
        if digest != expected[name]:
            raise SafetyGateError(
                f"local checkpoint SHA256 mismatch for {name}: {digest} != {expected[name]}"
            )
        hashes[name] = digest

    metadata = json.loads(actual_paths["config.json"].read_text(encoding="utf-8"))
    architecture = metadata.get("architectures")
    vision = metadata.get("vision_config") or {}
    required_metadata = {
        "architectures": [MODEL_CLASS],
        "model_type": "qwen2_vl",
        "num_hidden_layers": EXPECTED_LANGUAGE_LAYERS,
        "hidden_size": 1536,
        "vision_depth": 32,
        "vision_spatial_merge_size": 2,
    }
    actual_metadata = {
        "architectures": architecture,
        "model_type": metadata.get("model_type"),
        "num_hidden_layers": metadata.get("num_hidden_layers"),
        "hidden_size": metadata.get("hidden_size"),
        "vision_depth": vision.get("depth"),
        "vision_spatial_merge_size": vision.get("spatial_merge_size"),
    }
    if actual_metadata != required_metadata:
        raise SafetyGateError(
            f"local checkpoint architecture mismatch: {actual_metadata}"
        )
    return {
        "path": str(checkpoint),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "files_sha256": hashes,
        "architecture": actual_metadata,
    }


def enable_offline_mode() -> None:
    """Forbid accidental Hugging Face network access in every Model27 run."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def verify_starter_provenance(
    config: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> dict[str, str]:
    provenance = _nested(config, "provenance")
    archive = _absolute(str(_nested(provenance, "starter_archive")), project_root)
    expected_archive = str(_nested(provenance, "starter_archive_sha256"))
    if expected_archive != STARTER_ARCHIVE_SHA256:
        raise SafetyGateError("configured starter archive hash differs from Model27 ledger")
    if not archive.is_file():
        raise FileNotFoundError(archive)
    actual_archive = _sha256(archive)
    if actual_archive != expected_archive:
        raise SafetyGateError(
            f"starter archive SHA256 mismatch: {actual_archive} != {expected_archive}"
        )

    members = {
        STARTER_CONFIG_MEMBER: STARTER_CONFIG_SHA256,
        STARTER_TRAINER_MEMBER: STARTER_TRAINER_SHA256,
    }
    configured_members = _nested(provenance, "members")
    if configured_members != members:
        raise SafetyGateError("configured starter member hashes differ from Model27 ledger")
    for member, expected in members.items():
        actual = _member_sha256(archive, member)
        if actual != expected:
            raise SafetyGateError(f"starter member SHA256 mismatch for {member}: {actual}")
    return {
        "archive": str(archive),
        "archive_sha256": actual_archive,
        **{member: digest for member, digest in members.items()},
    }


def _is_below(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def reject_model22_destination(path: str | Path) -> Path:
    resolved = _absolute(path)
    if _is_below(resolved, PROTECTED_MODEL22_OUTPUT):
        raise SafetyGateError(f"Model27 may not write under {PROTECTED_MODEL22_OUTPUT}")
    return resolved


def require_new_output_dir(path: str | Path) -> Path:
    resolved = reject_model22_destination(path)
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {resolved}")
    return resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        return


def load_records(
    csv_path: str | Path,
    image_dir: str | Path,
    *,
    require_target: bool,
    require_five_folds: bool = False,
) -> tuple[OcrRecord, ...]:
    table = read_table(csv_path, require_target=require_target)
    if require_five_folds and "fold" not in table.fieldnames:
        raise SchemaError("Model27 training CSV requires a fold column")
    image_index = ImageIndex(image_dir)
    records: list[OcrRecord] = []
    for row in table.rows:
        identifier = row[table.id_column].strip()
        label = None
        if table.target_column is not None:
            label = trim_label_edges(row.get(table.target_column))
            if require_target and not label:
                raise SchemaError(f"blank target after edge trim for {identifier!r}")
        fold = int(row["fold"]) if "fold" in table.fieldnames else None
        records.append(OcrRecord(identifier, image_index.resolve(identifier), label, fold))
    if require_five_folds:
        present = {record.fold for record in records}
        if present != set(range(5)):
            raise SchemaError(f"folds must be exactly 0..4, got {sorted(present)}")
        for fold in range(5):
            if not any(record.fold == fold for record in records):
                raise SchemaError(f"fold {fold} is empty")
    return tuple(records)


class RecordDataset:
    """Minimal map-style dataset accepted by Transformers Trainer."""

    def __init__(self, records: Sequence[OcrRecord]):
        if not records:
            raise ValueError("dataset may not be empty")
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> OcrRecord:
        return self.records[index]


def load_full_rgb(path: str | Path) -> Image.Image:
    """Load the complete crop without geometric edits; the processor scales it."""

    with Image.open(path) as source:
        return source.convert("RGB")


def build_messages(prompt: str, label: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if label is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            }
        )
    return messages


def image_tokens_from_grid(grid_thw: Any, spatial_merge_size: int = 2) -> int:
    """Count Qwen2-VL post-merger image tokens from one or more [T,H,W] rows."""

    values = grid_thw.tolist() if hasattr(grid_thw, "tolist") else grid_thw
    if values and isinstance(values[0], (int, float)):
        values = [values]
    total = 0
    for row in values:
        if len(row) != 3:
            raise SafetyGateError(f"invalid image_grid_thw row: {row!r}")
        t, height, width = (int(item) for item in row)
        patches = t * height * width
        divisor = spatial_merge_size**2
        if min(t, height, width) <= 0 or patches % divisor:
            raise SafetyGateError(f"invalid Qwen2-VL image grid: {row!r}")
        total += patches // divisor
    return total


class QwenOcrCollator:
    """Batch-one collator with exact assistant-only masking and hard token gates."""

    def __init__(
        self,
        processor: Any,
        prompt: str,
        max_image_tokens: int,
        max_sequence_tokens: int,
        spatial_merge_size: int = 2,
    ):
        self.processor = processor
        self.prompt = prompt
        self.max_image_tokens = max_image_tokens
        self.max_sequence_tokens = max_sequence_tokens
        self.spatial_merge_size = spatial_merge_size

    def _encode_text(self, record: OcrRecord, include_label: bool) -> str:
        if include_label:
            if record.label is None:
                raise ValueError("training record has no label")
            messages = build_messages(self.prompt, record.label)
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        return self.processor.apply_chat_template(
            build_messages(self.prompt), tokenize=False, add_generation_prompt=True
        )

    def _processor_call(self, text: str, image: Image.Image) -> Mapping[str, Any]:
        return self.processor(
            text=[text],
            images=[image],
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

    def _diagnostics(self, batch: Mapping[str, Any]) -> dict[str, int]:
        if "input_ids" not in batch or "image_grid_thw" not in batch:
            raise SafetyGateError("processor did not return Qwen2-VL text/image tensors")
        sequence_tokens = int(batch["input_ids"].shape[-1])
        image_tokens = image_tokens_from_grid(
            batch["image_grid_thw"], self.spatial_merge_size
        )
        if image_tokens > self.max_image_tokens:
            raise SafetyGateError(
                f"processor emitted {image_tokens} image tokens; cap is {self.max_image_tokens}"
            )
        if sequence_tokens > self.max_sequence_tokens:
            raise SafetyGateError(
                f"processor emitted {sequence_tokens} total tokens; "
                f"cap is {self.max_sequence_tokens}"
            )
        return {"image_tokens": image_tokens, "sequence_tokens": sequence_tokens}

    def encode_prompt(self, record: OcrRecord) -> tuple[dict[str, Any], dict[str, int]]:
        image = load_full_rgb(record.image_path)
        prompt_text = self._encode_text(record, include_label=False)
        batch = dict(self._processor_call(prompt_text, image))
        return batch, self._diagnostics(batch)

    def encode_training(self, record: OcrRecord) -> tuple[dict[str, Any], dict[str, int]]:
        import torch

        image = load_full_rgb(record.image_path)
        full_text = self._encode_text(record, include_label=True)
        prompt_text = self._encode_text(record, include_label=False)
        full = dict(self._processor_call(full_text, image))
        prompt = dict(self._processor_call(prompt_text, image))
        diagnostics = self._diagnostics(full)
        prompt_tokens = int(prompt["input_ids"].shape[-1])
        full_tokens = int(full["input_ids"].shape[-1])
        if prompt_tokens >= full_tokens:
            raise SafetyGateError("assistant label produced no trainable tokens")
        if not torch.equal(
            full["input_ids"][0, :prompt_tokens], prompt["input_ids"][0]
        ):
            raise SafetyGateError("chat template prompt is not an exact training prefix")
        labels = full["input_ids"].clone()
        labels[:, :prompt_tokens] = -100
        if "attention_mask" in full:
            labels[full["attention_mask"] == 0] = -100
        if not bool((labels != -100).any()):
            raise SafetyGateError("assistant-only label mask is empty")
        full["labels"] = labels
        diagnostics["prompt_tokens"] = prompt_tokens
        diagnostics["supervised_tokens"] = int((labels != -100).sum().item())
        return full, diagnostics

    def __call__(self, examples: Sequence[OcrRecord]) -> dict[str, Any]:
        if len(examples) != 1:
            raise SafetyGateError(
                f"Model27 is memory-certified only for batch size 1, got {len(examples)}"
            )
        batch, _ = self.encode_training(examples[0])
        return batch


def _linear_like(module: Any) -> bool:
    return (
        module.__class__.__name__ in {"Linear", "Linear4bit"}
        and isinstance(getattr(module, "in_features", None), int)
        and isinstance(getattr(module, "out_features", None), int)
        and module.in_features > 0
        and module.out_features > 0
    )


def discover_lora_targets(
    model: Any,
    *,
    expected_language_layers: int = EXPECTED_LANGUAGE_LAYERS,
    include_merger: bool = True,
) -> LoraTargetAudit:
    """Find only verified language q/v and optional patch-merger linears."""

    modules = dict(model.named_modules())
    language: dict[tuple[int, str], str] = {}
    for name, module in modules.items():
        match = LANGUAGE_TARGET.fullmatch(name)
        if match is None:
            continue
        if not _linear_like(module):
            raise SafetyGateError(f"LoRA target is not a supported linear: {name}")
        key = (int(match.group("layer")), match.group("projection"))
        if key in language:
            raise SafetyGateError(f"duplicate LoRA target for {key}: {name}")
        language[key] = name

    expected_keys = {
        (layer, projection)
        for layer in range(expected_language_layers)
        for projection in ("q_proj", "v_proj")
    }
    if set(language) != expected_keys:
        missing = sorted(expected_keys - set(language))
        extra = sorted(set(language) - expected_keys)
        raise SafetyGateError(
            f"Qwen2-VL language module audit failed: missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )

    merger: list[str] = []
    if include_merger:
        for name in MERGER_TARGETS:
            module = modules.get(name)
            if module is None or not _linear_like(module):
                raise SafetyGateError(f"verified merger linear is absent/incompatible: {name}")
            merger.append(name)
        observed_merger_linears = {
            name
            for name, module in modules.items()
            if name.startswith("visual.merger.") and _linear_like(module)
        }
        if observed_merger_linears != set(MERGER_TARGETS):
            raise SafetyGateError(
                "unexpected visual merger linear layout: "
                f"{sorted(observed_merger_linears)}"
            )

    ordered_language = [language[key] for key in sorted(language)]
    targets = tuple([*ordered_language, *merger])
    expected_count = expected_language_layers * 2 + (2 if include_merger else 0)
    if len(targets) != expected_count or len(set(targets)) != expected_count:
        raise SafetyGateError("LoRA target module count is not exact")
    return LoraTargetAudit(
        language_root="model.layers",
        language_layers=expected_language_layers,
        q_modules=expected_language_layers,
        v_modules=expected_language_layers,
        merger_modules=len(merger),
        target_modules=targets,
    )


def exact_lora_target_pattern(targets: Sequence[str]) -> str:
    """Build one anchored regex so PEFT cannot condense exact paths to suffixes."""

    if not targets or len(set(targets)) != len(targets):
        raise SafetyGateError("LoRA targets must be nonempty and unique")
    return r"^(?:" + "|".join(re.escape(target) for target in targets) + r")$"


def audit_injected_lora_targets(
    model: Any, targets: Sequence[str]
) -> dict[str, Any]:
    """Verify both PEFT's persisted matcher and the modules it actually wrapped."""

    expected = tuple(targets)
    expected_pattern = exact_lora_target_pattern(expected)
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, Mapping) or "default" not in peft_config:
        raise SafetyGateError("PEFT default adapter configuration is absent")
    configured_pattern = getattr(peft_config["default"], "target_modules", None)
    if configured_pattern != expected_pattern:
        raise SafetyGateError("persisted PEFT target matcher is not the exact-path regex")

    tuner = getattr(model, "base_model", None)
    observed = tuple(getattr(tuner, "targeted_module_names", ()))
    if len(observed) != len(expected) or set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise SafetyGateError(
            f"injected LoRA target mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    return {
        "configured_pattern": configured_pattern,
        "injected_modules": len(observed),
    }


def freeze_vision_backbone(model: Any) -> int:
    visual = getattr(model, "visual", None)
    if visual is None or not hasattr(visual, "parameters"):
        raise SafetyGateError("expected pinned Qwen2-VL model.visual module is absent")
    count = 0
    for parameter in visual.parameters():
        parameter.requires_grad = False
        count += int(parameter.numel())
    if count <= 0:
        raise SafetyGateError("vision backbone contained no parameters")
    return count


def audit_trainable_parameters(
    model: Any,
    targets: Sequence[str],
    *,
    max_trainable_parameters: int,
) -> dict[str, Any]:
    target_set = set(targets)
    matched: set[str] = set()
    trainable_names: list[str] = []
    trainable_count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_names.append(name)
        trainable_count += int(parameter.numel())
        if ".lora_A." not in name and ".lora_B." not in name:
            raise SafetyGateError(f"non-LoRA parameter is unexpectedly trainable: {name}")
        owners = [target for target in target_set if target in name]
        if len(owners) != 1:
            raise SafetyGateError(f"cannot map trainable parameter to one target: {name}")
        matched.add(owners[0])
        if "visual." in name and owners[0] not in MERGER_TARGETS:
            raise SafetyGateError(f"vision backbone parameter is trainable: {name}")
    if matched != target_set:
        raise SafetyGateError(
            f"not every target received trainable LoRA weights: {sorted(target_set - matched)}"
        )
    if trainable_count <= 0 or trainable_count > max_trainable_parameters:
        raise SafetyGateError(
            f"trainable parameter count {trainable_count:,} is outside (0, "
            f"{max_trainable_parameters:,}]"
        )
    return {
        "trainable_parameters": trainable_count,
        "trainable_tensors": len(trainable_names),
        "target_modules": len(matched),
        "sample_names": trainable_names[:8],
    }


def _device_type(device: Any) -> str:
    if isinstance(device, int):
        return "cuda"
    if hasattr(device, "type"):
        return str(device.type)
    value = str(device).casefold()
    if value.isdigit() or value.startswith("cuda"):
        return "cuda"
    return value


def audit_cuda_residency(model: Any) -> dict[str, Any]:
    """Reject CPU, disk, and meta offload for parameters, buffers, or device map."""

    device_map = getattr(model, "hf_device_map", None) or {}
    bad_map = {
        str(name): str(device)
        for name, device in device_map.items()
        if _device_type(device) != "cuda"
    }
    if bad_map:
        raise SafetyGateError(f"CPU/disk/meta entries in hf_device_map: {bad_map}")

    bad_tensors: list[str] = []
    parameter_tensors = 0
    for name, parameter in model.named_parameters():
        parameter_tensors += 1
        if _device_type(parameter.device) != "cuda":
            bad_tensors.append(f"parameter:{name}:{parameter.device}")
    buffer_tensors = 0
    for name, buffer in model.named_buffers():
        buffer_tensors += 1
        if _device_type(buffer.device) != "cuda":
            bad_tensors.append(f"buffer:{name}:{buffer.device}")
    if bad_tensors:
        raise SafetyGateError(f"non-CUDA model tensors: {bad_tensors[:8]}")
    return {
        "hf_device_map": {str(key): str(value) for key, value in device_map.items()},
        "parameter_tensors": parameter_tensors,
        "buffer_tensors": buffer_tensors,
    }


def audit_nf4_modules(model: Any, minimum_modules: int) -> dict[str, Any]:
    if not bool(getattr(model, "is_loaded_in_4bit", False)):
        raise SafetyGateError("model does not report is_loaded_in_4bit=True")
    modules = [
        name
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Linear4bit"
    ]
    if len(modules) < minimum_modules:
        raise SafetyGateError(
            f"only {len(modules)} Linear4bit modules; expected at least {minimum_modules}"
        )
    return {"linear4bit_modules": len(modules), "sample_names": modules[:8]}


def verify_processor_budget(processor: Any, min_pixels: int, max_pixels: int) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise SafetyGateError("Qwen processor has no image_processor")
    actual_min = int(getattr(image_processor, "min_pixels", -1))
    actual_max = int(getattr(image_processor, "max_pixels", -1))
    if (actual_min, actual_max) != (min_pixels, max_pixels):
        raise SafetyGateError(
            "processor did not retain the pixel budget: "
            f"{(actual_min, actual_max)} != {(min_pixels, max_pixels)}"
        )


def _require_single_process_cuda(torch: Any) -> int:
    if not torch.cuda.is_available():
        raise SafetyGateError("Model27 requires CUDA; CPU training is intentionally disabled")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise SafetyGateError("Model27 is certified only for one local CUDA process")
    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    return device_index


def load_processor(config: Mapping[str, Any]) -> Any:
    from transformers import AutoProcessor

    min_pixels = int(_nested(config, "image", "min_pixels"))
    max_pixels = int(_nested(config, "image", "max_pixels"))
    checkpoint = _absolute(str(_nested(config, "model", "local_path")))
    processor = AutoProcessor.from_pretrained(
        str(checkpoint),
        trust_remote_code=False,
        local_files_only=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    verify_processor_budget(processor, min_pixels, max_pixels)
    return processor


def load_training_model(config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Load the pinned base, prepare k-bit gradients, freeze, then inject LoRA."""

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    device_index = _require_single_process_cuda(torch)
    compute_dtype_name = str(_nested(config, "quantization", "compute_dtype"))
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if compute_dtype_name not in dtypes:
        raise ValueError("quantization.compute_dtype must be float16 or bfloat16")
    compute_dtype = dtypes[compute_dtype_name]
    if compute_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SafetyGateError("configured bfloat16 is unsupported on this CUDA device")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    checkpoint = _absolute(str(_nested(config, "model", "local_path")))
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(checkpoint),
        trust_remote_code=False,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": device_index},
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    frozen_vision_parameters = freeze_vision_backbone(model)
    target_audit = discover_lora_targets(
        model,
        expected_language_layers=int(
            _nested(config, "architecture", "expected_language_layers")
        ),
        include_merger=bool(_nested(config, "lora", "include_verified_merger")),
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=int(_nested(config, "lora", "alpha")),
        lora_dropout=float(_nested(config, "lora", "dropout")),
        bias="none",
        task_type="CAUSAL_LM",
        # PEFT 0.14 condenses lists of >=20 targets to suffixes.  A single
        # anchored regex preserves this audited full-path contract in the
        # saved adapter configuration as well as during initial injection.
        target_modules=exact_lora_target_pattern(target_audit.target_modules),
    )
    model = get_peft_model(model, lora)
    injected = audit_injected_lora_targets(model, target_audit.target_modules)
    trainable = audit_trainable_parameters(
        model,
        target_audit.target_modules,
        max_trainable_parameters=int(
            _nested(config, "smoke", "max_trainable_parameters")
        ),
    )
    quantized = audit_nf4_modules(
        model, int(_nested(config, "smoke", "minimum_4bit_modules"))
    )
    residency = audit_cuda_residency(model)
    if len(target_audit.target_modules) != int(
        _nested(config, "smoke", "expected_target_modules")
    ):
        raise SafetyGateError("post-discovery LoRA target count gate failed")
    return model, {
        "device_index": device_index,
        "frozen_vision_parameters": frozen_vision_parameters,
        "lora_targets": asdict(target_audit),
        "injected_lora": injected,
        "trainable": trainable,
        "quantization": quantized,
        "residency": residency,
    }


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _model_device(model: Any) -> Any:
    for parameter in model.parameters():
        if _device_type(parameter.device) == "cuda":
            return parameter.device
    raise SafetyGateError("model has no CUDA parameter")


def estimate_visual_tokens(
    image_path: str | Path,
    *,
    min_pixels: int,
    max_pixels: int,
    merged_patch_size: int = 28,
) -> int:
    """Mirror Qwen2-VL smart-resize token arithmetic without decoding pixels."""

    with Image.open(image_path) as image:
        width, height = image.size
    factor = merged_patch_size
    if min(height, width) < factor:
        raise SafetyGateError(
            f"image {image_path} has a dimension below Qwen's {factor}-pixel minimum"
        )
    if max(height, width) / min(height, width) > 200:
        raise SafetyGateError(f"image {image_path} exceeds Qwen's 200:1 aspect limit")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = math.floor(height / scale / factor) * factor
        resized_width = math.floor(width / scale / factor) * factor
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    if min(resized_height, resized_width) < factor:
        raise SafetyGateError(f"smart resize would collapse a dimension for {image_path}")
    return resized_height * resized_width // factor**2


def select_smoke_record(
    records: Sequence[OcrRecord],
    *,
    min_pixels: int,
    max_pixels: int,
    merged_patch_size: int = 28,
) -> OcrRecord:
    labelled = [record for record in records if record.label is not None]
    if not labelled:
        raise ValueError("no labelled record is available for the smoke step")

    def estimated_sequence(record: OcrRecord) -> tuple[int, int, str]:
        image_tokens = estimate_visual_tokens(
            record.image_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            merged_patch_size=merged_patch_size,
        )
        # Character count is a conservative tokenizer-independent label proxy.
        return image_tokens + len(record.label or ""), image_tokens, record.identifier

    return max(labelled, key=estimated_sequence)


def run_memory_smoke(
    model: Any,
    collator: QwenOcrCollator,
    record: OcrRecord,
    *,
    max_allocated_gib: float,
    max_reserved_gib: float,
) -> dict[str, Any]:
    """Run a real batch-one forward/backward and enforce total allocated VRAM."""

    import torch

    device = _model_device(model)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        cpu_batch, tokens = collator.encode_training(record)
        batch = _to_device(cpu_batch, device)
        result = model(**batch)
        loss = result.loss
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise SafetyGateError("smoke forward returned a missing/non-finite loss")
        loss.backward()
        torch.cuda.synchronize(device)
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
        loss_value = float(loss.detach().cpu().item())
    except torch.OutOfMemoryError as exc:
        raise SafetyGateError("Model27 representative smoke batch ran out of VRAM") from exc
    finally:
        model.zero_grad(set_to_none=True)
    report = {
        "record_id": record.identifier,
        **tokens,
        "loss": loss_value,
        "peak_allocated_bytes": allocated,
        "peak_allocated_gib": allocated / GIB,
        "peak_reserved_bytes": reserved,
        "peak_reserved_gib": reserved / GIB,
        "allocated_limit_gib": max_allocated_gib,
        "reserved_limit_gib": max_reserved_gib,
    }
    enforce_memory_limits(
        allocated,
        reserved,
        max_allocated_gib=max_allocated_gib,
        max_reserved_gib=max_reserved_gib,
    )
    return report


def enforce_memory_limits(
    allocated_bytes: int,
    reserved_bytes: int,
    *,
    max_allocated_gib: float,
    max_reserved_gib: float,
) -> None:
    """Apply strict allocated and allocator-reserved CUDA memory ceilings."""

    allocated_threshold = int(max_allocated_gib * GIB)
    reserved_threshold = int(max_reserved_gib * GIB)
    if allocated_bytes >= allocated_threshold:
        raise SafetyGateError(
            f"smoke allocated peak {allocated_bytes / GIB:.3f} GiB is not below "
            f"{max_allocated_gib:.3f} GiB"
        )
    if reserved_bytes >= reserved_threshold:
        raise SafetyGateError(
            f"smoke reserved peak {reserved_bytes / GIB:.3f} GiB is not below "
            f"{max_reserved_gib:.3f} GiB"
        )


def _decode_generated(processor: Any, generated_ids: Any) -> str:
    value = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    # Apply the same documented chat-role cleanup used by the challenge scorer
    # before persisting predictions.  Otherwise local validation could forgive
    # a generated ``assistant:`` prefix that the submitted CSV still contains.
    return strip_conversation_tokens(trim_label_edges(value))


def greedy_predict(
    model: Any,
    processor: Any,
    collator: QwenOcrCollator,
    records: Sequence[OcrRecord],
    *,
    max_new_tokens: int,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> list[str]:
    import torch

    if not records:
        raise ValueError("at least one record is required for prediction")
    device = _model_device(model)
    was_training = bool(model.training)
    old_cache = bool(getattr(model.config, "use_cache", False))
    model.eval()
    model.config.use_cache = True
    predictions: list[str] = []
    try:
        for position, record in enumerate(records, start=1):
            batch, _ = collator.encode_prompt(record)
            batch = _to_device(batch, device)
            prompt_tokens = int(batch["input_ids"].shape[-1])
            with torch.inference_mode():
                output = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    use_cache=True,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
            generated = output[:, prompt_tokens:]
            predictions.append(_decode_generated(processor, generated))
            if position % 50 == 0 or position == len(records):
                print(f"generated {position}/{len(records)}", flush=True)
    finally:
        model.config.use_cache = old_cache
        model.train(was_training)
    return predictions


def exact_metrics(records: Sequence[OcrRecord], predictions: Sequence[str]) -> dict[str, float]:
    if len(records) != len(predictions):
        raise ValueError("record/prediction count mismatch")
    references: list[str] = []
    for record in records:
        if record.label is None:
            raise ValueError(f"record {record.identifier!r} has no reference")
        references.append(record.label)
    return competition_error(references, list(predictions))


def write_predictions(
    path: str | Path,
    records: Sequence[OcrRecord],
    predictions: Sequence[str],
) -> Path:
    if len(records) != len(predictions):
        raise ValueError("record/prediction count mismatch")
    output = reject_model22_destination(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Held-out/OOF records carry a fold and retain it for provenance.  Test
    # records do not: omitting an otherwise empty fold column also keeps the
    # candidate directly compatible with Model26's one-prediction-column
    # deployment-medoid input contract.
    include_fold = any(record.fold is not None for record in records)
    fieldnames = ["ID", "prediction"]
    if include_fold:
        fieldnames.append("fold")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, prediction in zip(records, predictions, strict=True):
            row: dict[str, str | int] = {
                "ID": record.identifier,
                "prediction": trim_label_edges(prediction),
            }
            if include_fold:
                row["fold"] = "" if record.fold is None else record.fold
            writer.writerow(row)
    return output


def _json_write(path: Path, payload: Any) -> None:
    reject_model22_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _runtime_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "config": config,
    }


def _make_validation_callback(
    processor: Any,
    collator: QwenOcrCollator,
    validation_records: Sequence[OcrRecord],
    output_dir: Path,
    max_new_tokens: int,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> Any:
    from transformers import TrainerCallback

    class ExactMetricCallback(TrainerCallback):
        def __init__(self) -> None:
            self.best_score = -math.inf
            self.best_epoch: int | None = None

        def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any, **_: Any) -> Any:
            if not state.is_world_process_zero:
                return control
            epoch = int(round(float(state.epoch or 0)))
            predictions = greedy_predict(
                model,
                processor,
                collator,
                validation_records,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            metrics = exact_metrics(validation_records, predictions)
            prediction_path = output_dir / "validation" / f"epoch_{epoch:03d}.csv"
            write_predictions(prediction_path, validation_records, predictions)
            payload = {
                "epoch": epoch,
                "rows": len(validation_records),
                "generation": {
                    "do_sample": False,
                    "num_beams": 1,
                    "num_return_sequences": 1,
                    "max_new_tokens": max_new_tokens,
                    "repetition_penalty": repetition_penalty,
                    "no_repeat_ngram_size": no_repeat_ngram_size,
                },
                "metrics": metrics,
                "predictions": str(prediction_path),
            }
            _json_write(output_dir / "validation" / f"epoch_{epoch:03d}.json", payload)
            score = float(metrics["multi_score"])
            if score > self.best_score + 1e-12:
                self.best_score = score
                self.best_epoch = epoch
                best_dir = output_dir / "best_adapter"
                model.save_pretrained(best_dir, safe_serialization=True)
                processor.save_pretrained(best_dir)
                _json_write(
                    output_dir / "best.json",
                    {
                        **payload,
                        "adapter": str(best_dir),
                        "selection": "maximum full-fold greedy exact challenge multi_score",
                    },
                )
            print(
                f"fold validation epoch={epoch} multi_score={score:.9f} "
                f"best_epoch={self.best_epoch}",
                flush=True,
            )
            return control

    return ExactMetricCallback()


def train(config_path: str | Path, fold: int, *, smoke_only: bool = False) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    enable_offline_mode()
    config, resolved_config = load_config(config_path)
    if fold not in range(5):
        raise ValueError("fold must be one of 0,1,2,3,4")
    provenance = verify_starter_provenance(config)
    checkpoint = verify_local_checkpoint(config)
    seed = int(_nested(config, "seed"))
    set_seed(seed)

    csv_path = _absolute(str(_nested(config, "train_csv")))
    image_dir = _absolute(str(_nested(config, "image_dir")))
    records = load_records(
        csv_path, image_dir, require_target=True, require_five_folds=True
    )
    training_records = tuple(record for record in records if record.fold != fold)
    validation_records = tuple(record for record in records if record.fold == fold)
    output_dir = require_new_output_dir(str(config["output_dir"]).format(fold=fold))

    processor = load_processor(config)
    model, model_audit = load_training_model(config)
    collator = QwenOcrCollator(
        processor,
        str(_nested(config, "prompt")),
        int(_nested(config, "image", "max_image_tokens")),
        int(_nested(config, "image", "max_sequence_tokens")),
    )
    smoke_record = select_smoke_record(
        training_records,
        min_pixels=int(_nested(config, "image", "min_pixels")),
        max_pixels=int(_nested(config, "image", "max_pixels")),
        merged_patch_size=int(_nested(config, "image", "merged_patch_size")),
    )
    smoke = run_memory_smoke(
        model,
        collator,
        smoke_record,
        max_allocated_gib=float(_nested(config, "smoke", "max_allocated_gib")),
        max_reserved_gib=float(_nested(config, "smoke", "max_reserved_gib")),
    )
    audit = {
        "fold": fold,
        "rows": {
            "all": len(records),
            "train": len(training_records),
            "validation": len(validation_records),
        },
        "provenance": provenance,
        "local_checkpoint": checkpoint,
        "model": model_audit,
        "smoke": smoke,
    }
    if smoke_only:
        smoke_dir = reject_model22_destination(str(_nested(config, "smoke", "report_dir")))
        _json_write(smoke_dir / f"fold{fold}.json", audit)
        return audit

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **_runtime_manifest(config),
        **audit,
        "resolved_config": str(resolved_config),
        "train_csv": str(csv_path),
        "train_csv_sha256": _sha256(csv_path),
    }
    _json_write(output_dir / "run_manifest.json", manifest)

    import torch
    from transformers import Trainer, TrainingArguments

    training = _nested(config, "training")
    callback = _make_validation_callback(
        processor,
        collator,
        validation_records,
        output_dir,
        int(_nested(config, "generation", "max_new_tokens")),
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
        full_determinism=True,
        num_train_epochs=float(_nested(training, "epochs")),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(_nested(training, "gradient_accumulation")),
        learning_rate=float(_nested(training, "learning_rate")),
        weight_decay=float(_nested(training, "weight_decay")),
        warmup_ratio=float(_nested(training, "warmup_ratio")),
        lr_scheduler_type=str(_nested(training, "scheduler")),
        max_grad_norm=float(_nested(training, "max_grad_norm")),
        logging_steps=int(_nested(training, "logging_steps")),
        eval_strategy="no",
        save_strategy="no",
        fp16=str(_nested(config, "quantization", "compute_dtype")) == "float16",
        bf16=str(_nested(config, "quantization", "compute_dtype")) == "bfloat16",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        report_to="none",
    )
    set_seed(seed)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=RecordDataset(training_records),
        data_collator=collator,
        callbacks=[callback],
    )
    result = trainer.train()
    if callback.best_epoch is None:
        raise SafetyGateError("training ended without full-fold generated validation")
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)
    complete = {
        "fold": fold,
        "best_epoch": callback.best_epoch,
        "best_multi_score": callback.best_score,
        "training_metrics": result.metrics,
        "final_adapter": str(final_dir),
        "best_adapter": str(output_dir / "best_adapter"),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
    }
    _json_write(output_dir / "complete.json", complete)
    return complete


def _load_inference_model(config: Mapping[str, Any], adapter_path: Path) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    device_index = _require_single_process_cuda(torch)
    compute_dtype = (
        torch.float16
        if str(_nested(config, "quantization", "compute_dtype")) == "float16"
        else torch.bfloat16
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    checkpoint = _absolute(str(_nested(config, "model", "local_path")))
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        str(checkpoint),
        trust_remote_code=False,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": device_index},
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    targets = discover_lora_targets(base).target_modules
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    audit_injected_lora_targets(model, targets)
    audit_cuda_residency(model)
    audit_nf4_modules(model, int(_nested(config, "smoke", "minimum_4bit_modules")))
    model.eval()
    return model, targets


def infer(
    config_path: str | Path,
    input_csv: str | Path,
    adapter_path: str | Path,
    output_csv: str | Path,
    *,
    fold: int | None = None,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    enable_offline_mode()
    config, _ = load_config(config_path)
    verify_starter_provenance(config)
    checkpoint_audit = verify_local_checkpoint(config)
    if fold is not None and fold not in range(5):
        raise ValueError("fold must be one of 0,1,2,3,4")
    seed = int(_nested(config, "seed"))
    set_seed(seed)
    records = load_records(
        _absolute(input_csv),
        _absolute(str(_nested(config, "image_dir"))),
        require_target=False,
        require_five_folds=False,
    )
    if fold is not None:
        if not records or records[0].fold is None:
            raise SchemaError("--fold requires a fold column")
        records = tuple(record for record in records if record.fold == fold)
    if not records:
        raise ValueError("inference selection is empty")
    adapter = _absolute(adapter_path)
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    output = reject_model22_destination(output_csv)
    processor = load_processor(config)
    model, targets = _load_inference_model(config, adapter)
    collator = QwenOcrCollator(
        processor,
        str(_nested(config, "prompt")),
        int(_nested(config, "image", "max_image_tokens")),
        int(_nested(config, "image", "max_sequence_tokens")),
    )
    predictions = greedy_predict(
        model,
        processor,
        collator,
        records,
        max_new_tokens=int(_nested(config, "generation", "max_new_tokens")),
    )
    write_predictions(output, records, predictions)
    report: dict[str, Any] = {
        "rows": len(records),
        "fold": fold,
        "adapter": str(adapter),
        "local_checkpoint": checkpoint_audit,
        "output": str(output),
        "target_modules": list(targets),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "max_new_tokens": int(_nested(config, "generation", "max_new_tokens")),
        },
    }
    if all(record.label is not None for record in records):
        report["metrics"] = exact_metrics(records, predictions)
    _json_write(output.with_suffix(".metrics.json"), report)
    if submission_path is not None:
        if sample_path is None:
            raise ValueError("--sample is required with --submission")
        by_id = {
            record.identifier: prediction
            for record, prediction in zip(records, predictions, strict=True)
        }
        write_submission(
            _absolute(sample_path),
            reject_model22_destination(submission_path),
            by_id,
            cleaner=clean_prediction_preserving_internal_spaces,
        )
    return report


def train_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Guarded Model27 Qwen2-VL QLoRA training")
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    result = train(args.config, args.fold, smoke_only=args.smoke_only)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def infer_main(default_config: str | Path | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Greedy Model27 Qwen2-VL inference")
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    result = infer(
        args.config,
        args.input,
        args.adapter,
        args.output,
        fold=args.fold,
        sample_path=args.sample,
        submission_path=args.submission,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
