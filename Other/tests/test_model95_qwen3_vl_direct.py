from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from road_htr.qwen3_vl_direct import (
    load_config,
    parse_generation,
    prepare_image_manifest,
    read_records,
    render_prompt,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_config(tmp_path: Path, module_hash: str) -> Path:
    config = {
        "schema_version": 1,
        "model": {
            "id": "owner/model",
            "revision": "a" * 40,
            "license": "Apache-2.0",
            "dtype": "bfloat16",
        },
        "data": {
            "truth": str(tmp_path / "folds.csv"),
            "test": str(tmp_path / "test.csv"),
            "image_dir": str(tmp_path / "images"),
        },
        "prompt": "read this image and convert to text, it's old english. Do not change to normal english",
        "generation": {"temperature": 0.0},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_parser_only_removes_response_wrappers() -> None:
    assert parse_generation("Transcription: publique w^ch") == "publique w^ch"
    assert parse_generation('```text\npublique\nAct\n```') == "publique Act"
    assert parse_generation('"sayde pnts"') == "sayde pnts"


def test_render_prompt_uses_one_image_and_exact_text() -> None:
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            assert messages[0]["content"] == [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "read this image and convert to text, it's old english. Do not change to normal english",
                },
            ]
            return "rendered"

    prompt = "read this image and convert to text, it's old english. Do not change to normal english"
    assert render_prompt(Processor(), prompt) == "rendered"


def test_prepare_manifest_contains_no_labels(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "abc.jpg").write_bytes(b"fake-jpeg")
    write_csv(
        tmp_path / "folds.csv",
        ["ID", "Target", "fold"],
        [{"ID": "abc", "Target": "SECRET LABEL", "fold": 0}],
    )
    write_csv(tmp_path / "test.csv", ["ID"], [{"ID": "abc"}])
    config = make_config(tmp_path, "unused")
    output = tmp_path / "records.jsonl"
    report = prepare_image_manifest(config, "oof", output, fold=0)
    assert report["labels_in_records"] is False
    assert "SECRET LABEL" not in output.read_text(encoding="utf-8")
    records = read_records(output)
    assert records == [
        {
            "fold": 0,
            "id": "abc",
            "image_name": "abc.jpg",
            "image_sha256": file_hash(images / "abc.jpg"),
            "split": "oof",
        }
    ]


def test_oof_requires_fold(tmp_path: Path) -> None:
    config = make_config(tmp_path, "unused")
    with pytest.raises(ValueError, match="fold is required"):
        prepare_image_manifest(config, "oof", tmp_path / "records.jsonl")


def test_config_rejects_mutable_revision(tmp_path: Path) -> None:
    path = make_config(tmp_path, "unused")
    config = json.loads(path.read_text(encoding="utf-8"))
    config["model"]["revision"] = "main"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        load_config(path)
