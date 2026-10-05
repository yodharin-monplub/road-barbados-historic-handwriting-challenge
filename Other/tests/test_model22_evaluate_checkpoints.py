from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "model22"
    / "evaluate_checkpoints.py"
)
SPEC = importlib.util.spec_from_file_location("model22_evaluate_checkpoints", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> None:
        self.commands.append(command)
        if len(command) > 1 and command[1] == "convert":
            sources = [Path(value) for value in command if value.endswith(".ckpt")]
            if len(sources) != 1:
                raise AssertionError(f"conversion received {len(sources)} checkpoints")
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(sources[0].name.encode("utf-8"))
            return

        if len(command) > 1 and Path(command[1]).name == "infer.py":
            model = Path(command[command.index("--checkpoint") + 1])
            config = Path(command[command.index("--config") + 1])
            config_values = json.loads(config.read_text(encoding="utf-8"))
            input_csv = Path(command[command.index("--input") + 1])
            output = Path(command[command.index("--output") + 1])
            fold = int(command[command.index("--fold") + 1])
            batch_size = int(command[command.index("--batch-size") + 1])

            with input_csv.open(encoding="utf-8", newline="") as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle)
                    if int(row["fold"]) == fold
                ]
            # Epoch 10 is perfect despite a lower Kraken filename metric.
            perfect = b"checkpoint_10" in model.read_bytes()
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            "ID": row["ID"],
                            "prediction": row["Target"] if perfect else "wrong",
                        }
                    )
            output.with_suffix(".jsonl").write_text("{}\n", encoding="utf-8")
            output.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "input_sha256": _sha256(input_csv),
                        "config_sha256": _sha256(config),
                        "model_sha256": _sha256(model),
                        "fold": fold,
                        "constrain_charset": False,
                        "batch_size": batch_size,
                        "preserve_internal_spaces": config_values["inference"].get(
                            "preserve_internal_spaces", False
                        ),
                        "rows": len(rows),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return

        raise AssertionError(f"unexpected command: {command}")


class EvaluateCheckpointsTests(unittest.TestCase):
    def _fixture(self, root: Path, *, dry_run: bool = False):
        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir()
        # Write out of order to ensure discovery is numeric, not lexicographic.
        (checkpoint_dir / "checkpoint_10-0.8000.ckpt").write_bytes(b"epoch 10")
        (checkpoint_dir / "checkpoint_abort.ckpt").write_bytes(b"ignore me")
        (checkpoint_dir / "checkpoint_02-0.9500.ckpt").write_bytes(b"epoch 2")
        # Kraken writes this only after the final scored checkpoint.
        (checkpoint_dir / "best_0.9500.safetensors").write_bytes(b"best")

        input_csv = root / "train_folds.csv"
        input_csv.write_text(
            "ID,Target,fold\na,alpha beta,0\nb,gamma,0\nc,other,1\n",
            encoding="utf-8",
        )
        image_dir = root / "images"
        image_dir.mkdir()
        # Training excludes ``a``, but external checkpoint evaluation must
        # still infer and score it because it is part of the canonical fold.
        exclude_ids = root / "exclude_ids.txt"
        exclude_ids.write_text("a\n", encoding="utf-8")
        config = root / "config.yaml"
        inference = {
            "accelerator": "gpu",
            "device": "auto",
            "precision": "bf16-mixed",
            "threads": 4,
            "line_workers": 0,
            "padding": 16,
        }
        current_config = {
            "train_csv": str(input_csv),
            "image_dir": str(image_dir),
            "image_variant": "full",
            "exclude_ids": str(exclude_ids),
            "output_dir": str(checkpoint_dir),
            "inference": {
                **inference,
                "constrain_charset": False,
                "preserve_internal_spaces": True,
            },
            "training": {"ctc_subsampling": 8, "strip_target_edges": True},
        }
        config.write_text(json.dumps(current_config), encoding="utf-8")
        training_config = {
            **current_config,
            "exclude_ids": "old-training-exclusions.txt",
            "inference": {**inference, "constrain_charset": True},
            "training": {},
        }
        (checkpoint_dir / "run_manifest.json").write_text(
            json.dumps(
                {"config": training_config, "preparation": {"fold": 0}},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        # Kraken writes the best weights only after training finishes.
        (checkpoint_dir / "best_0.9500.safetensors").write_bytes(b"best")
        options = MODULE.EvaluationOptions(
            checkpoint_dir=checkpoint_dir,
            input_csv=input_csv,
            output_dir=root / "evaluation",
            config=config,
            fold=0,
            ketos=root / "fake-ketos",
            python=Path(sys.executable),
            batch_size=1,
            expected_input_sha256=_sha256(input_csv),
            expected_total_rows=3,
            expected_fold_rows=2,
            dry_run=dry_run,
        )
        return options

    def test_each_checkpoint_is_converted_alone_and_exact_score_selects_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._fixture(Path(directory))
            runner = FakeRunner()
            leaderboard = MODULE.evaluate_checkpoints(options, runner=runner)

            conversions = [
                command
                for command in runner.commands
                if len(command) > 1 and command[1] == "convert"
            ]
            self.assertEqual(
                [Path(command[-1]).name for command in conversions],
                ["checkpoint_02-0.9500.ckpt", "checkpoint_10-0.8000.ckpt"],
            )
            for command in conversions:
                self.assertEqual(sum(value.endswith(".ckpt") for value in command), 1)
                self.assertNotIn("*", " ".join(command))

            inference = [
                command
                for command in runner.commands
                if len(command) > 1 and Path(command[1]).name == "infer.py"
            ]
            self.assertEqual(len(inference), 2)
            self.assertTrue(all("--no-constrain-charset" in command for command in inference))
            self.assertTrue(all("--id" not in command for command in inference))
            self.assertTrue(all("--input" in command for command in inference))
            self.assertEqual(
                [row["epoch"] for row in leaderboard["results"]], [2, 10]
            )
            self.assertEqual(leaderboard["best"]["epoch"], 10)
            self.assertEqual(leaderboard["best"]["multi_score"], 1.0)
            self.assertEqual(
                set(leaderboard["training_current_config_differences"]),
                {
                    "exclude_ids",
                    "inference.constrain_charset",
                    "inference.preserve_internal_spaces",
                    "training.ctc_subsampling",
                    "training.strip_target_edges",
                },
            )
            with Path(leaderboard["best"]["predictions"]).open(
                encoding="utf-8", newline=""
            ) as handle:
                prediction_ids = {row["ID"] for row in csv.DictReader(handle)}
            self.assertEqual(prediction_ids, {"a", "b"})
            metrics_path = MODULE._artifact_paths(
                options.output_dir,
                MODULE.discover_checkpoints(options.checkpoint_dir)[-1],
            ).metrics
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertIs(
                metrics_payload["signature"]["preserve_internal_spaces"], True
            )
            self.assertTrue((options.output_dir / "leaderboard.json").is_file())
            self.assertTrue((options.output_dir / "leaderboard.csv").is_file())

            def unexpected(_command: list[str]) -> None:
                raise AssertionError("completed artifacts should have been resumed")

            resumed = MODULE.evaluate_checkpoints(options, runner=unexpected)
            self.assertEqual(resumed["best"]["epoch"], 10)

    def test_corrupt_inference_sidecar_is_regenerated_without_reconversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._fixture(Path(directory))
            MODULE.evaluate_checkpoints(options, runner=FakeRunner())
            checkpoint = MODULE.discover_checkpoints(options.checkpoint_dir)[0]
            artifacts = MODULE._artifact_paths(options.output_dir, checkpoint)
            artifacts.prediction_details.write_text("corrupt\n", encoding="utf-8")

            runner = FakeRunner()
            MODULE.evaluate_checkpoints(options, runner=runner)

            conversions = [
                command for command in runner.commands if command[1] == "convert"
            ]
            inference = [
                command
                for command in runner.commands
                if Path(command[1]).name == "infer.py"
            ]
            self.assertEqual(conversions, [])
            self.assertEqual(len(inference), 1)
            self.assertEqual(
                artifacts.prediction_details.read_text(encoding="utf-8"), "{}\n"
            )

    def test_dry_run_is_ordered_and_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._fixture(Path(directory), dry_run=True)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = MODULE.evaluate_checkpoints(
                    options,
                    runner=lambda _command: self.fail("dry-run executed a command"),
                )
            self.assertTrue(result["dry_run"])
            self.assertFalse(options.output_dir.exists())
            output = stream.getvalue()
            self.assertLess(output.index("[epoch 02]"), output.index("[epoch 10]"))
            self.assertEqual(output.count(" convert "), 2)
            self.assertEqual(output.count("--no-constrain-charset"), 2)

    def test_duplicate_epoch_versions_are_rejected_as_mixed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            (checkpoint_dir / "checkpoint_02-0.9000.ckpt").write_bytes(b"first")
            (checkpoint_dir / "checkpoint_02-0.9100-v1.ckpt").write_bytes(b"second")

            with self.assertRaisesRegex(ValueError, "mix training runs"):
                MODULE.discover_checkpoints(checkpoint_dir)

    def test_manifest_row_mismatch_fails_before_prediction_is_committed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._fixture(Path(directory))
            underlying = FakeRunner()

            def wrong_row_count(command: list[str]) -> None:
                underlying(command)
                if len(command) > 1 and Path(command[1]).name == "infer.py":
                    output = Path(command[command.index("--output") + 1])
                    manifest_path = output.with_suffix(".manifest.json")
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["rows"] -= 1
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "inference manifest mismatch"):
                MODULE.evaluate_checkpoints(options, runner=wrong_row_count)

            checkpoint = MODULE.discover_checkpoints(options.checkpoint_dir)[0]
            artifacts = MODULE._artifact_paths(options.output_dir, checkpoint)
            self.assertFalse(artifacts.predictions.exists())
            self.assertFalse(artifacts.inference_receipt.exists())

    def test_wrong_ids_are_not_cached_and_plain_retry_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._fixture(Path(directory))
            underlying = FakeRunner()

            def wrong_id(command: list[str]) -> None:
                underlying(command)
                if len(command) > 1 and Path(command[1]).name == "infer.py":
                    output = Path(command[command.index("--output") + 1])
                    with output.open(encoding="utf-8", newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    rows[0]["ID"] = "not-in-fold"
                    with output.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(
                            handle, fieldnames=["ID", "prediction"]
                        )
                        writer.writeheader()
                        writer.writerows(rows)

            with self.assertRaisesRegex(MODULE.SchemaError, "ID mismatch"):
                MODULE.evaluate_checkpoints(options, runner=wrong_id)

            first_checkpoint = MODULE.discover_checkpoints(options.checkpoint_dir)[0]
            first_artifacts = MODULE._artifact_paths(
                options.output_dir, first_checkpoint
            )
            self.assertFalse(first_artifacts.predictions.exists())
            self.assertFalse(first_artifacts.inference_receipt.exists())

            retry = FakeRunner()
            leaderboard = MODULE.evaluate_checkpoints(options, runner=retry)
            self.assertEqual(leaderboard["best"]["multi_score"], 1.0)
            self.assertTrue(first_artifacts.inference_receipt.is_file())

    def test_exact_score_rejects_missing_training_excluded_oof_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._fixture(Path(directory))
            predictions = Path(directory) / "incomplete.csv"
            predictions.write_text("ID,prediction\nb,gamma\n", encoding="utf-8")

            with self.assertRaisesRegex(MODULE.SchemaError, "1 missing"):
                MODULE.score_predictions(options.input_csv, predictions, fold=0)

    def test_score_predictions_uses_exact_competition_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._fixture(Path(directory))
            predictions = Path(directory) / "predictions.csv"
            predictions.write_text(
                "ID,prediction\nb,gamma\na,alpha\n", encoding="utf-8"
            )

            actual = MODULE.score_predictions(options.input_csv, predictions, fold=0)
            expected = MODULE.competition_error(
                ["alpha beta", "gamma"], ["alpha", "gamma"]
            )
            self.assertEqual(actual, expected)

    def test_inference_critical_config_change_is_rejected_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._fixture(Path(directory))
            config = json.loads(options.config.read_text(encoding="utf-8"))
            config["image_variant"] = "center"
            options.config.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "inference-critical"):
                MODULE.evaluate_checkpoints(
                    options,
                    runner=lambda _command: self.fail("runner must not be called"),
                )
            self.assertFalse(options.output_dir.exists())

    def test_input_content_change_is_rejected_by_pinned_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._fixture(Path(directory))
            options.input_csv.write_text(
                "ID,Target,fold\na,changed,0\nb,gamma,0\nc,other,1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(MODULE.SchemaError, "SHA-256 mismatch"):
                MODULE.evaluate_checkpoints(
                    options,
                    runner=lambda _command: self.fail("runner must not be called"),
                )
            self.assertFalse(options.output_dir.exists())

    def test_evaluation_output_must_be_outside_checkpoint_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._fixture(Path(directory))
            unsafe = MODULE.EvaluationOptions(
                **{
                    **options.__dict__,
                    "output_dir": options.checkpoint_dir / "evaluation",
                }
            )

            with self.assertRaisesRegex(ValueError, "disjoint sibling trees"):
                MODULE.evaluate_checkpoints(
                    unsafe,
                    runner=lambda _command: self.fail("runner must not be called"),
                )

    def test_stale_best_marker_older_than_run_manifest_is_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._fixture(Path(directory))
            run_manifest = options.checkpoint_dir / "run_manifest.json"
            run_manifest.write_text(
                run_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "training may still be active"):
                MODULE.evaluate_checkpoints(
                    options,
                    runner=lambda _command: self.fail("runner must not be called"),
                )

    def test_completed_fixed_run_requires_every_expected_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._fixture(Path(directory))
            run_manifest_path = options.checkpoint_dir / "run_manifest.json"
            run_manifest = json.loads(
                run_manifest_path.read_text(encoding="utf-8")
            )
            run_manifest["config"]["training"] = {
                "epochs": 3,
                "validation_frequency": 1,
                "stop": "fixed",
            }
            run_manifest_path.write_text(
                json.dumps(run_manifest, sort_keys=True), encoding="utf-8"
            )
            (options.checkpoint_dir / "best_0.9500.safetensors").write_bytes(
                b"new best"
            )

            with self.assertRaisesRegex(RuntimeError, "incomplete checkpoint set"):
                MODULE.evaluate_checkpoints(
                    options,
                    runner=lambda _command: self.fail("runner must not be called"),
                )

    def test_find_python_preserves_virtualenv_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary_dir = Path(directory) / "venv" / "bin"
            binary_dir.mkdir(parents=True)
            base_python = Path(directory) / "base-python"
            base_python.write_bytes(b"executable")
            ketos = binary_dir / "ketos"
            ketos.write_bytes(b"launcher")
            venv_python = binary_dir / "python"
            venv_python.symlink_to(base_python)

            found = MODULE._find_python(None, ketos)

            self.assertEqual(found, venv_python.absolute())
            self.assertNotEqual(found, base_python.resolve())


if __name__ == "__main__":
    unittest.main()
