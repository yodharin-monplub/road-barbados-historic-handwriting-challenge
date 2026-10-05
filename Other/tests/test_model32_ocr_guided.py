from __future__ import annotations

import copy
import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from road_htr import qwen2_vl_challenger as model27
from road_htr.qwen2_vl_ocr_guided import (
    CANONICAL_TRAIN_SHA256,
    DRAFT_RECEIPT_SCHEMA_VERSION,
    DRAFT_KIND,
    EMPTY_DRAFT_POLICY,
    MODEL27_CONFIG_SHA256,
    MODEL27_IMPLEMENTATION_SHA256,
    MODEL32_DRAFT_ROOT,
    GuidedOcrRecord,
    QwenGuidedCollator,
    _draft_receipt_payload,
    _read_draft_csv,
    build_guided_messages,
    build_guided_prompt,
    compare_fold_to_draft,
    infer,
    load_config,
    prepare_drafts,
    require_model32_output,
    select_guided_smoke_record,
    stable_sha256,
    train,
    verify_draft_artifact,
)
from road_htr.schema import SchemaError


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model32" / "config.yaml"


class Model32ConfigTests(unittest.TestCase):
    def test_reuses_exact_model27_nf4_lora_and_memory_contract(self) -> None:
        config = load_config(CONFIG_PATH)
        effective = config.effective_model27

        generator_config = yaml.safe_load(
            (ROOT / "model32" / "model8_draft_config.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("train_csv", generator_config)

        self.assertEqual(stable_sha256(config.base_config_path), MODEL27_CONFIG_SHA256)
        self.assertEqual(
            stable_sha256(config.base_implementation_path),
            MODEL27_IMPLEMENTATION_SHA256,
        )
        self.assertEqual(effective["model"]["local_files_only"], True)
        self.assertEqual(effective["model"]["trust_remote_code"], False)
        self.assertEqual(effective["quantization"]["type"], "nf4")
        self.assertEqual(effective["quantization"]["double_quant"], True)
        self.assertEqual(effective["lora"]["language_projections"], ["q_proj", "v_proj"])
        self.assertEqual(effective["architecture"]["expected_target_modules"], 58)
        self.assertEqual(effective["smoke"]["max_allocated_gib"], 5.3)
        self.assertEqual(effective["smoke"]["max_reserved_gib"], 5.3)
        self.assertEqual(effective["output_dir"], "outputs/model32/fold{fold}")

    def test_safety_assertions_and_prompt_placeholder_fail_closed(self) -> None:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = []
        weakened = copy.deepcopy(raw)
        weakened["safety_assertions"]["max_reserved_gib_exclusive"] = 5.4
        mutations.append(weakened)
        target_prompt = copy.deepcopy(raw)
        target_prompt["prompt_template"] += "\n{target}"
        mutations.append(target_prompt)
        wrong_output = copy.deepcopy(raw)
        wrong_output["output_dir"] = "outputs/model27/fold{fold}"
        mutations.append(wrong_output)
        model25_kind = copy.deepcopy(raw)
        model25_kind["draft_source"]["kind"] = "model25_crossfit"
        mutations.append(model25_kind)
        model25_path = copy.deepcopy(raw)
        model25_path["draft_source"]["train_csv"] = "outputs/model25/allfold_oof.csv"
        mutations.append(model25_path)
        old_receipt_contract = copy.deepcopy(raw)
        old_receipt_contract["draft_source"]["receipt_schema_version"] = 1
        mutations.append(old_receipt_contract)
        dropped_blanks = copy.deepcopy(raw)
        dropped_blanks["draft_source"]["empty_draft_policy"] = "drop_empty_rows"
        mutations.append(dropped_blanks)

        for changed in mutations:
            with self.subTest(change=changed):
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", encoding="utf-8"
                ) as handle:
                    yaml.safe_dump(changed, handle, sort_keys=False)
                    handle.flush()
                    with self.assertRaises((ValueError, model27.SafetyGateError)):
                        load_config(handle.name)

    def test_runtime_source_reuses_all_model27_hard_gates(self) -> None:
        source = inspect.getsource(train)
        self.assertIn("model27.load_training_model", source)
        self.assertIn("model27.run_memory_smoke", source)
        self.assertIn("model27.enforce_memory_limits", source)
        self.assertIn("QwenGuidedCollator", source)
        self.assertNotIn("device_map=\"auto\"", source)


class Model32PromptAndComparisonTests(unittest.TestCase):
    def test_prompt_preserves_internal_spaces_and_treats_braces_as_draft_text(self) -> None:
        template = "Image first. Untrusted draft:\n{draft}\nReturn only text."
        draft = "  A  {not_a_field}  B  "
        prompt = build_guided_prompt(template, draft)

        self.assertIn("A  {not_a_field}  B", prompt)
        self.assertNotIn("  A", prompt)
        messages = build_guided_messages(template, draft, "  True  label  ")
        self.assertEqual(messages[0]["content"][0], {"type": "image"})
        self.assertEqual(messages[1]["content"][0]["text"], "True  label")
        self.assertTrue(issubclass(QwenGuidedCollator, model27.QwenOcrCollator))

    def test_present_empty_draft_stays_blank_between_markers(self) -> None:
        template = "before\n--- DRAFT START ---\n{draft}\n--- DRAFT END ---\nafter"

        self.assertEqual(
            build_guided_prompt(template, ""),
            "before\n--- DRAFT START ---\n\n--- DRAFT END ---\nafter",
        )
        with self.assertRaisesRegex(ValueError, "present draft value"):
            build_guided_prompt(template, None)

    def test_whole_fold_comparison_never_builds_per_row_oracle(self) -> None:
        records = (
            GuidedOcrRecord("a", Path("a.png"), "right", 0, "wrong"),
            GuidedOcrRecord("b", Path("b.png"), "draft wins", 0, "draft wins"),
        )
        # Challenger wins row a and loses row b. The result reports two whole
        # vectors and never returns a hybrid chosen from each row's truth.
        report = compare_fold_to_draft(records, ["right", "wrong"])

        self.assertEqual(report["scope"], "whole_fold_report_only")
        self.assertFalse(report["per_row_selection_performed"])
        self.assertEqual(report["emitted_predictions"], "challenger")
        self.assertNotIn("predictions", report)

    def test_exact_metric_observes_genuine_internal_double_spaces(self) -> None:
        records = (
            GuidedOcrRecord("a", Path("a.png"), "A  B", 0, "A B"),
        )
        exact = compare_fold_to_draft(records, ["A  B"])
        collapsed = compare_fold_to_draft(records, ["A B"])

        self.assertEqual(exact["challenger"]["multi_score"], 1.0)
        self.assertLess(collapsed["challenger"]["multi_score"], 1.0)

    def test_smoke_proxy_counts_added_draft_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            Image.new("RGB", (56, 28), "white").save(first)
            Image.new("RGB", (56, 28), "white").save(second)
            records = (
                GuidedOcrRecord("short", first, "x", 0, "d"),
                GuidedOcrRecord("long", second, "x", 0, "d" * 200),
            )

            selected = select_guided_smoke_record(
                records,
                min_pixels=28 * 28,
                max_pixels=320 * 28 * 28,
            )

            self.assertEqual(selected.identifier, "long")


class Model32DraftLineageTests(unittest.TestCase):
    IDS = ("uGI8F9Er0c5XwdnX", "Dqt2gc3V5RJqmLE5")

    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _artifact(
        self,
        targets: tuple[str, str] = ("truth one", "truth two"),
        draft_predictions: tuple[str, str] = ("draft  one", "draft two"),
    ):
        input_csv = self.root / "input.csv"
        with input_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Target", "fold"])
            writer.writeheader()
            writer.writerow({"ID": self.IDS[0], "Target": targets[0], "fold": 0})
            writer.writerow({"ID": self.IDS[1], "Target": targets[1], "fold": 1})
        draft_csv = self.root / "draft.csv"
        with draft_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
            writer.writeheader()
            writer.writerow({"ID": self.IDS[0], "prediction": draft_predictions[0]})
            writer.writerow({"ID": self.IDS[1], "prediction": draft_predictions[1]})
        draft_csv.with_suffix(".jsonl").write_text(
            "".join(
                json.dumps(
                    {"id": identifier, "candidates": [prediction]}
                )
                + "\n"
                for identifier, prediction in zip(
                    self.IDS, draft_predictions, strict=True
                )
            ),
            encoding="utf-8",
        )
        id_projection = draft_csv.with_suffix(".ids.csv")
        with id_projection.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID"])
            writer.writeheader()
            writer.writerows({"ID": identifier} for identifier in self.IDS)
        generator = yaml.safe_load(
            self.config.generator_config_path.read_text(encoding="utf-8")
        )
        checkpoint = (ROOT / "checkpoints" / "model8" / "medium.safetensors").resolve()
        source_manifest = draft_csv.with_suffix(".manifest.json")
        source_manifest.write_text(
            json.dumps(
                {
                    "config": generator,
                    "resolved_config": str(self.config.generator_config_path),
                    "config_sha256": stable_sha256(self.config.generator_config_path),
                    "input_csv": str(id_projection.resolve()),
                    "input_sha256": stable_sha256(id_projection),
                    "model": str(checkpoint),
                    "model_sha256": stable_sha256(checkpoint),
                    "fold": None,
                    "image_variant": "full",
                    "constrain_charset": False,
                    "batch_size": 1,
                    "preserve_internal_spaces": True,
                    "rows": 2,
                    "seconds": 1.0,
                    "python": "test",
                }
            ),
            encoding="utf-8",
        )
        receipt = draft_csv.with_suffix(".receipt.json")
        receipt.write_text(
            json.dumps(
                _draft_receipt_payload(
                    self.config,
                    input_csv.resolve(),
                    draft_csv.resolve(),
                    source_manifest.resolve(),
                )
            ),
            encoding="utf-8",
        )
        return input_csv, draft_csv, receipt

    def test_valid_zero_shot_full_unconstrained_artifact_is_accepted(self) -> None:
        input_csv, drafts, receipt = self._artifact()

        artifact = verify_draft_artifact(self.config, input_csv, drafts, receipt)

        self.assertEqual(artifact.audit["kind"], DRAFT_KIND)
        self.assertTrue(artifact.audit["label_independent"])
        self.assertEqual(artifact.audit["challenge_target_access"], "none")
        self.assertEqual(artifact.predictions[self.IDS[0]], "draft  one")
        self.assertEqual(artifact.audit["receipt_schema_version"], 2)
        self.assertEqual(artifact.audit["empty_draft_count"], 0)
        self.assertEqual(artifact.audit["empty_draft_ids"], [])

    def test_present_empty_zero_shot_draft_is_accepted_and_receipted(self) -> None:
        input_csv, drafts, receipt = self._artifact(
            draft_predictions=("", "draft two")
        )

        artifact = verify_draft_artifact(self.config, input_csv, drafts, receipt)
        payload = json.loads(receipt.read_text(encoding="utf-8"))

        self.assertEqual(artifact.predictions[self.IDS[0]], "")
        self.assertEqual(payload["schema_version"], DRAFT_RECEIPT_SCHEMA_VERSION)
        self.assertEqual(payload["empty_draft_policy"], EMPTY_DRAFT_POLICY)
        self.assertEqual(payload["empty_draft_count"], 1)
        self.assertEqual(payload["empty_draft_ids"], [self.IDS[0]])
        self.assertEqual(
            artifact.audit["empty_draft_ids_sha256"],
            payload["empty_draft_ids_sha256"],
        )

    def test_missing_draft_row_is_rejected_even_when_checksum_is_updated(self) -> None:
        input_csv, drafts, receipt = self._artifact()
        rows = drafts.read_text(encoding="utf-8").splitlines()
        drafts.write_text("\n".join(rows[:2]) + "\n", encoding="utf-8")
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        payload["draft_csv_sha256"] = stable_sha256(drafts)
        receipt.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(SchemaError, "IDs/order differ"):
            verify_draft_artifact(self.config, input_csv, drafts, receipt)

        with self.assertRaisesRegex(SchemaError, "IDs/order differ"):
            _read_draft_csv(drafts, "ID", self.IDS)

    def test_absent_csv_value_is_distinct_from_present_empty_string(self) -> None:
        _, drafts, _ = self._artifact()
        drafts.write_text(
            "ID,prediction\n"
            f"{self.IDS[0]}\n"
            f"{self.IDS[1]},draft two\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(SchemaError, "missing zero-shot draft value"):
            _read_draft_csv(drafts, "ID", self.IDS)

    def test_empty_draft_receipt_ledger_mismatch_is_rejected(self) -> None:
        input_csv, drafts, receipt = self._artifact(
            draft_predictions=("", "draft two")
        )
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        mutations = {
            "schema_version": 1,
            "empty_draft_count": 0,
            "empty_draft_ids": [self.IDS[1]],
            "empty_draft_ids_sha256": "0" * 64,
        }
        for key, value in mutations.items():
            changed = copy.deepcopy(payload)
            changed[key] = value
            receipt.write_text(json.dumps(changed), encoding="utf-8")
            with self.subTest(key=key):
                with self.assertRaisesRegex(model27.SafetyGateError, "draft receipt"):
                    verify_draft_artifact(self.config, input_csv, drafts, receipt)
        receipt.write_text(json.dumps(payload), encoding="utf-8")

    def test_model25_or_charset_constrained_receipts_fail_closed(self) -> None:
        input_csv, drafts, receipt = self._artifact()
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        for key, value in (
            ("kind", "model25_crossfit"),
            ("constrain_charset", True),
            ("challenge_target_access", "training_labels"),
        ):
            changed = copy.deepcopy(payload)
            changed[key] = value
            receipt.write_text(json.dumps(changed), encoding="utf-8")
            with self.subTest(key=key):
                with self.assertRaisesRegex(model27.SafetyGateError, "draft receipt"):
                    verify_draft_artifact(self.config, input_csv, drafts, receipt)
        receipt.write_text(json.dumps(payload), encoding="utf-8")

    def test_source_manifest_constraint_tampering_fails(self) -> None:
        input_csv, drafts, receipt = self._artifact()
        manifest = drafts.with_suffix(".manifest.json")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["constrain_charset"] = True
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        receipt_payload["source_manifest_sha256"] = stable_sha256(manifest)
        receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")

        with self.assertRaisesRegex(model27.SafetyGateError, "constrain_charset"):
            verify_draft_artifact(self.config, input_csv, drafts, receipt)

    def test_confidence_rows_must_reproduce_the_prediction_csv(self) -> None:
        input_csv, drafts, receipt = self._artifact()
        details = drafts.with_suffix(".jsonl")
        lines = details.read_text(encoding="utf-8").splitlines()
        payload = json.loads(lines[0])
        payload["candidates"] = ["forged label"]
        lines[0] = json.dumps(payload)
        details.write_text("\n".join(lines) + "\n", encoding="utf-8")
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        receipt_payload["confidence_jsonl_sha256"] = stable_sha256(details)
        receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")

        with self.assertRaisesRegex(model27.SafetyGateError, "confidence/prediction"):
            verify_draft_artifact(self.config, input_csv, drafts, receipt)

    def test_drafts_and_prompts_are_invariant_to_heldout_label_content(self) -> None:
        input_csv, drafts, receipt = self._artifact(("outer label A", "train"))
        first = verify_draft_artifact(self.config, input_csv, drafts, receipt)
        first_prompt = build_guided_prompt(
            "draft={draft}", first.predictions[self.IDS[0]]
        )

        # Rebuild only the receipt/manifest integrity envelope after changing a
        # Target. The static predictions themselves remain byte-identical.
        input_csv.write_text(
            "ID,Target,fold\n"
            f"{self.IDS[0]},COMPLETELY DIFFERENT OUTER LABEL,0\n"
            f"{self.IDS[1]},train,1\n",
            encoding="utf-8",
        )
        manifest = drafts.with_suffix(".manifest.json")
        receipt.write_text(
            json.dumps(
                _draft_receipt_payload(
                    self.config,
                    input_csv.resolve(),
                    drafts.resolve(),
                    manifest.resolve(),
                )
            ),
            encoding="utf-8",
        )
        second = verify_draft_artifact(self.config, input_csv, drafts, receipt)
        second_prompt = build_guided_prompt(
            "draft={draft}", second.predictions[self.IDS[0]]
        )

        self.assertEqual(first.predictions, second.predictions)
        self.assertEqual(first_prompt, second_prompt)

    def test_draft_id_order_and_edge_whitespace_are_strict(self) -> None:
        input_csv, drafts, receipt = self._artifact()
        rows = drafts.read_text(encoding="utf-8").splitlines()
        drafts.write_text("\n".join([rows[0], rows[2], rows[1]]) + "\n", encoding="utf-8")

        with self.assertRaises((SchemaError, model27.SafetyGateError)):
            verify_draft_artifact(self.config, input_csv, drafts, receipt)

    def test_cpu_draft_dry_run_reads_no_labels_and_creates_nothing(self) -> None:
        prospective = MODEL32_DRAFT_ROOT / "train_model8_raw.csv"
        related = (
            prospective,
            prospective.with_suffix(".manifest.json"),
            prospective.with_suffix(".receipt.json"),
            prospective.with_suffix(".jsonl"),
            prospective.with_suffix(".ids.csv"),
        )

        # The canonical artifact may legitimately exist after the one-time
        # producer has run. In that state prepare_drafts must fail before any
        # write, and the existing receipt envelope must remain byte-identical.
        if any(path.exists() for path in related):
            before = {
                path: stable_sha256(path) if path.is_file() else None
                for path in related
            }
            with self.assertRaises(FileExistsError):
                prepare_drafts(
                    CONFIG_PATH,
                    ROOT / "data" / "processed" / "train_folds.csv",
                    prospective,
                    dry_run=True,
                )
            after = {
                path: stable_sha256(path) if path.is_file() else None
                for path in related
            }
            self.assertEqual(after, before)
            return

        plan = prepare_drafts(
            CONFIG_PATH,
            ROOT / "data" / "processed" / "train_folds.csv",
            prospective,
            dry_run=True,
        )

        self.assertEqual(plan["input_sha256"], CANONICAL_TRAIN_SHA256)
        self.assertEqual(
            plan["receipt_schema_version"], DRAFT_RECEIPT_SCHEMA_VERSION
        )
        self.assertEqual(plan["runtime"]["accelerator"], "cpu")
        self.assertFalse(plan["runtime"]["constrain_charset"])
        self.assertEqual(plan["runtime"]["challenge_target_access"], "none")
        self.assertTrue(plan["inference_input_csv"].endswith(".ids.csv"))
        self.assertFalse(prospective.exists())

    def test_oof_runtime_rejects_nonconfigured_drafts_before_cuda(self) -> None:
        with self.assertRaisesRegex(
            model27.SafetyGateError, "exact configured draft"
        ):
            infer(
                CONFIG_PATH,
                ROOT / "data" / "processed" / "train_folds.csv",
                self.root / "model25.csv",
                self.root / "model25.receipt.json",
                self.root / "adapter",
                ROOT / "outputs" / "model32" / "oof" / "never.csv",
                fold=0,
            )


class Model32OutputIsolationTests(unittest.TestCase):
    def test_writes_are_confined_to_model32(self) -> None:
        accepted = require_model32_output("outputs/model32/oof/fold0.csv")
        self.assertIn("outputs/model32", str(accepted))
        for path in (
            "outputs/model27/fold0/new.csv",
            "outputs/model22/new.csv",
            "submissions/model32.csv",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(model27.SafetyGateError, "must remain"):
                    require_model32_output(path)


if __name__ == "__main__":
    unittest.main()
