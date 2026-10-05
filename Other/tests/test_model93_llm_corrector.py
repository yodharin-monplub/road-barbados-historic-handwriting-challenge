from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from road_htr.llm_corrector import (
    parse_generation,
    prepare_prompts,
    sha256,
    verify_model_audit,
)


class Model93LlmCorrectorTests(unittest.TestCase):
    def test_model_audit_is_bound_to_config_and_snapshot_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / ("a" * 40)
            snapshot.mkdir()
            metadata_path = snapshot / "config.json"
            metadata_path.write_text("{}", encoding="utf-8")
            shard_path = snapshot / "model-00001-of-00001.safetensors"
            shard_path.write_bytes(b"weights")
            config = {
                "model": {
                    "id": "Qwen/test",
                    "revision": "a" * 40,
                    "expected_shards": 1,
                    "expected_weight_bytes": len(b"weights"),
                    "expected_weight_file_bytes": len(b"weights"),
                    "expected_metadata_sha256": {"config.json": sha256(metadata_path)},
                }
            }
            config_path = root / "model93.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            audit = {
                "complete": True,
                "config_sha256": sha256(config_path),
                "model_id": "Qwen/test",
                "revision": "a" * 40,
                "snapshot": str(snapshot),
                "metadata": {
                    "config.json": {
                        "bytes": metadata_path.stat().st_size,
                        "sha256": sha256(metadata_path),
                    }
                },
                "shards": [
                    {
                        "name": shard_path.name,
                        "bytes": shard_path.stat().st_size,
                        "sha256": sha256(shard_path),
                    }
                ],
            }
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            verified, verified_snapshot = verify_model_audit(
                audit_path, config, config_path
            )
            self.assertTrue(verified["complete"])
            self.assertEqual(verified_snapshot, snapshot)

            shard_path.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "shard changed"):
                verify_model_audit(audit_path, config, config_path)

    def test_parser_accepts_exact_tag_and_single_line(self) -> None:
        options = {
            "maximum_characters": 80,
            "maximum_edit_ratio": 0.75,
            "minimum_word_ratio": 0.4,
            "maximum_word_ratio": 1.6,
        }
        self.assertEqual(
            parse_generation(
                "<transcription>Know yee that I</transcription>",
                "Know ye that I",
                **options,
            ),
            ("Know yee that I", None),
        )
        self.assertEqual(
            parse_generation("Transcription: publique Act", "public Act", **options),
            ("publique Act", None),
        )

    def test_parser_falls_back_on_unsafe_generation(self) -> None:
        options = {
            "maximum_characters": 80,
            "maximum_edit_ratio": 0.4,
            "minimum_word_ratio": 0.5,
            "maximum_word_ratio": 1.5,
        }
        primary = "the sayde parcel"
        cases = (
            ("", "blank_output"),
            ("one line\nanother line", "malformed_output"),
            ("<think>guess</think>\nanswer", "malformed_output"),
            ("thé sayde parcel", "non_ascii_output"),
            ("completely unrelated invented sentence", "edit_distance_outlier"),
        )
        for value, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(
                    parse_generation(value, primary, **options), (primary, reason)
                )

    def test_oof_prompt_retrieval_excludes_evaluated_fold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "truth.csv"
            predictions = root / "predictions.csv"
            test = root / "test.csv"
            sample = root / "sample.csv"
            rows = [
                ("f0a", "secret fold zero alpha", 0, "secret fold zero alpho"),
                ("f0b", "secret fold zero beta", 0, "secret fold zero beto"),
                ("f1c", "publique correction one", 1, "publique correctiom one"),
                ("f1i", "publique identity one", 1, "publique identity one"),
                ("f2c", "sayde correction two", 2, "sayde correctiom two"),
                ("f2i", "sayde identity two", 2, "sayde identity two"),
                ("f3c", "parcel correction three", 3, "parcel correctiom three"),
                ("f3i", "parcel identity three", 3, "parcel identity three"),
                ("f4c", "heires correction four", 4, "heires correctiom four"),
                ("f4i", "heires identity four", 4, "heires identity four"),
            ]
            with truth.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "Target", "fold"])
                writer.writeheader()
                for identifier, target, fold, _ in rows:
                    writer.writerow({"ID": identifier, "Target": target, "fold": fold})
            with predictions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
                writer.writeheader()
                for identifier, _, _, prediction in rows:
                    writer.writerow({"ID": identifier, "prediction": prediction})
            with test.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID"])
                writer.writeheader()
                writer.writerow({"ID": "test"})
            with sample.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "Target"])
                writer.writeheader()
                writer.writerow({"ID": "test", "Target": ""})
            test_predictions = root / "test_predictions.csv"
            with test_predictions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
                writer.writeheader()
                writer.writerow({"ID": "test", "prediction": "test prediction"})

            config = {
                "schema_version": 1,
                "model": {
                    "id": "Qwen/Qwen3-30B-A3B",
                    "revision": "a" * 40,
                    "license": "Apache-2.0",
                    "dtype": "bfloat16",
                    "max_model_len": 4096,
                },
                "data": {"truth": str(truth), "test": str(test), "sample_submission": str(sample)},
                "sources": {
                    "oof": {name: str(predictions) for name in ("primary", "full", "enhanced", "center", "center_enhanced", "kraken")},
                    "test": {name: str(test_predictions) for name in ("primary", "full", "enhanced", "center", "center_enhanced", "kraken")},
                },
                "retrieval": {
                    "correction_examples": 3,
                    "identity_examples": 1,
                    "maximum_example_edit_ratio": 0.3,
                    "character_similarity_weight": 0.65,
                    "word_similarity_weight": 0.35,
                },
                "generation": {"thinking": False, "temperature": 0.0},
                "validity": {"ascii_only": True},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "fold0.jsonl"
            manifest = prepare_prompts(config_path, "oof", output, fold=0)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual({row["id"] for row in records}, {"f0a", "f0b"})
            for row in records:
                self.assertEqual(len(row["retrieved_examples"]), 4)
                self.assertTrue(all(item["fold"] != 0 for item in row["retrieved_examples"]))
                self.assertNotIn("secret fold zero alpha", row["messages"][1]["content"])

            test_output = root / "test_prompts.jsonl"
            test_manifest = prepare_prompts(config_path, "test", test_output)
            test_record = json.loads(test_output.read_text(encoding="utf-8").strip())
            self.assertEqual(test_manifest["rows"], 1)
            self.assertIn("PRIMARY: test prediction", test_record["messages"][1]["content"])
            self.assertEqual(len(test_record["retrieved_examples"]), 4)


if __name__ == "__main__":
    unittest.main()
