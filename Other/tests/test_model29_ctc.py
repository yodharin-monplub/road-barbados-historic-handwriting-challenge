from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from road_htr.challenge_ctc import (
    CharNGramLanguageModel,
    DecoderCandidate,
    WordBoundaryPrior,
    greedy_decode_log_probs,
    log_softmax_rows,
    prefix_beam_search,
)
from road_htr.kraken_probability_export import (
    CapturingGreedyDecoder,
    ExportOptions,
    _checkpoint_provenance,
    _codec_labels,
    _ctc_greedy_text,
    stable_sha256,
)
from road_htr.model29_crossfit import LabeledRow, crossfit_decode


class Model29CtcSemanticsTests(unittest.TestCase):
    @staticmethod
    def _logits(tokens: list[int], classes: int = 3) -> list[list[float]]:
        return [
            [8.0 if index == token else -8.0 for index in range(classes)]
            for token in tokens
        ]

    def test_repeated_class_collapses_without_blank_for_greedy_and_beam(self) -> None:
        log_probs = log_softmax_rows(self._logits([1, 1, 1]))

        self.assertEqual(greedy_decode_log_probs(log_probs, ["a", "b"]), "a")
        self.assertEqual(
            prefix_beam_search(
                log_probs,
                ["a", "b"],
                beam_width=8,
                token_prune=3,
            )[0][0],
            "a",
        )

    def test_blank_separates_repeated_class_for_greedy_and_beam(self) -> None:
        log_probs = log_softmax_rows(self._logits([1, 1, 0, 1]))

        self.assertEqual(greedy_decode_log_probs(log_probs, ["a", "b"]), "aa")
        self.assertEqual(
            prefix_beam_search(
                log_probs,
                ["a", "b"],
                beam_width=8,
                token_prune=3,
            )[0][0],
            "aa",
        )

    def test_beam_uses_numeric_codec_mapping_not_ascii_order(self) -> None:
        log_probs = log_softmax_rows(self._logits([2, 0, 1]))

        self.assertEqual(
            prefix_beam_search(
                log_probs,
                ["z", "&"],
                beam_width=8,
                token_prune=3,
            )[0][0],
            "&z",
        )

    def test_lm_and_word_prior_are_fitted_only_from_explicit_texts(self) -> None:
        lm = CharNGramLanguageModel(order=2, smoothing=0.1).fit(["ab", "ab", "aa"])
        prior = WordBoundaryPrior(smoothing=1.0).fit(["old road", "old house"])

        self.assertGreater(lm.log_probability("a", "b"), lm.log_probability("a", "z"))
        self.assertGreater(
            prior.completed_word_score("the old"),
            prior.completed_word_score("the unseen"),
        )


class Model29ProbabilityCaptureTests(unittest.TestCase):
    def test_collector_trims_padding_transposes_and_delegates_original_objects(self) -> None:
        outputs = [
            [
                [0.8, 0.1, 0.9],
                [0.1, 0.7, 0.05],
                [0.1, 0.2, 0.05],
            ]
        ]
        lengths = [2]
        calls: list[tuple[object, object]] = []

        def delegate(original_outputs, original_lengths):
            calls.append((original_outputs, original_lengths))
            return [[("unchanged", 0, 1, 0.7)]]

        collector = CapturingGreedyDecoder(delegate)
        decoded = collector(outputs, lengths)

        self.assertEqual(decoded, [[("unchanged", 0, 1, 0.7)]])
        self.assertIs(calls[0][0], outputs)
        self.assertIs(calls[0][1], lengths)
        self.assertEqual(
            collector.take(),
            [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]],
        )
        with self.assertRaisesRegex(RuntimeError, "did not invoke"):
            collector.take()

    def test_collector_rejects_ambiguous_multi_line_batch(self) -> None:
        collector = CapturingGreedyDecoder(lambda outputs, lengths: [])
        outputs = [
            [[0.8], [0.2]],
            [[0.7], [0.3]],
        ]

        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            collector(outputs, [1, 1])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("kraken"),
        "Kraken/PyTorch integration environment is optional",
    )
    def test_collector_delegates_real_torch_tensor_and_kraken_lengths(self) -> None:
        import torch
        from kraken.lib.ctc_decoder import greedy_decoder

        outputs = torch.tensor(
            [
                [
                    [0.05, 0.05, 0.90, 0.05],
                    [0.90, 0.90, 0.05, 0.90],
                    [0.05, 0.05, 0.05, 0.05],
                ]
            ],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3])
        expected = greedy_decoder(outputs, lengths)
        collector = CapturingGreedyDecoder(greedy_decoder)

        self.assertEqual(collector(outputs, lengths), expected)
        capture = collector.take()
        self.assertEqual(tuple(capture.shape), (3, 3))
        self.assertTrue(torch.equal(capture, outputs[0, :, :3].transpose(0, 1)))

    def test_codec_mapping_requires_contiguous_single_class_codes(self) -> None:
        codec = type("Codec", (), {"c2l": {"z": [2], "&": [1]}})()
        self.assertEqual(_codec_labels(codec, 3), [None, "&", "z"])

        compound = type("Codec", (), {"c2l": {"x": [1, 2]}})()
        with self.assertRaisesRegex(ValueError, "one CTC class"):
            _codec_labels(compound, 3)

    def test_exporter_independent_greedy_uses_blank_and_repeat_semantics(self) -> None:
        probabilities = [
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.98, 0.01, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
        ]
        self.assertEqual(_ctc_greedy_text(probabilities, [None, "a", "z"]), "aaz")

    def test_fold_export_requires_matching_training_and_conversion_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            truth = root / "train.csv"
            truth.write_text("ID,Target,fold\na,text,0\n", encoding="utf-8")
            run_dir = root / "runs" / "fold0"
            run_dir.mkdir(parents=True)
            source_checkpoint = run_dir / "checkpoint_00-0.5.ckpt"
            source_checkpoint.write_bytes(b"source checkpoint")
            training_manifest = run_dir / "run_manifest.json"
            training_manifest.write_text(
                json.dumps(
                    {
                        "config": {
                            "train_csv": str(truth),
                            "output_dir": str(root / "runs" / "fold{fold}"),
                        },
                        "preparation": {"fold": 0},
                    }
                ),
                encoding="utf-8",
            )
            weights = root / "evaluation" / "converted" / "selected.safetensors"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"converted weights")
            conversion_receipt = weights.with_suffix(".conversion.json")
            conversion_receipt.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "weights": str(weights),
                        "weights_sha256": stable_sha256(weights),
                        "checkpoint": str(source_checkpoint),
                        "checkpoint_sha256": stable_sha256(source_checkpoint),
                    }
                ),
                encoding="utf-8",
            )
            options = ExportOptions(
                config=root / "config.yaml",
                input_csv=truth,
                checkpoint=weights,
                output_dir=root / "export",
                fold=0,
                training_manifest=training_manifest,
                conversion_receipt=conversion_receipt,
                training_excluded_folds=(0,),
            )

            provenance = _checkpoint_provenance(
                options,
                truth,
                stable_sha256(truth),
                weights,
                stable_sha256(weights),
            )

            self.assertEqual(provenance["training_excluded_folds"], [0])
            wrong_fold = ExportOptions(
                **{
                    **options.__dict__,
                    "training_excluded_folds": (1,),
                }
            )
            with self.assertRaisesRegex(ValueError, "exclusion mismatch"):
                _checkpoint_provenance(
                    wrong_fold,
                    truth,
                    stable_sha256(truth),
                    weights,
                    stable_sha256(weights),
                )


class _MemoryProbabilityStore:
    def __init__(
        self,
        identifiers: list[str],
        matrices: dict[str, list[list[float]]],
        training_excluded_folds: tuple[int, ...],
    ):
        self.identifiers = tuple(identifiers)
        self.label_texts = ("a", "b", " ")
        self.matrices = matrices
        self.training_excluded_folds = training_excluded_folds

    def load_log_probabilities(self, identifier: str):
        return self.matrices[identifier]


class Model29CrossfitLeakageTests(unittest.TestCase):
    @staticmethod
    def _matrix(tokens: list[int]) -> list[list[float]]:
        logits = [
            [8.0 if index == token else -8.0 for index in range(4)]
            for token in tokens
        ]
        return log_softmax_rows(logits)

    def _inputs(self, fold_zero_target: str):
        rows = [
            LabeledRow("f0a", fold_zero_target, 0),
            LabeledRow("f0b", "ab", 0),
            LabeledRow("f1a", "ab", 1),
            LabeledRow("f1b", "aa", 1),
            LabeledRow("f2a", "ab", 2),
            LabeledRow("f2b", "aa", 2),
        ]
        matrices = {
            "f0a": self._matrix([1, 0, 1]),
            "f0b": self._matrix([1, 2]),
            "f1a": self._matrix([1, 2]),
            "f1b": self._matrix([1, 0, 1]),
            "f2a": self._matrix([1, 2]),
            "f2b": self._matrix([1, 0, 1]),
        }
        stores = {
            fold: _MemoryProbabilityStore(
                [row.identifier for row in rows if row.fold == fold],
                matrices,
                (fold,),
            )
            for fold in range(3)
        }
        tuning_stores = {
            (outer_fold, tuning_fold): _MemoryProbabilityStore(
                [row.identifier for row in rows if row.fold == tuning_fold],
                matrices,
                tuple(sorted((outer_fold, tuning_fold))),
            )
            for outer_fold in range(3)
            for tuning_fold in range(3)
            if tuning_fold != outer_fold
        }
        candidates = [
            DecoderCandidate(
                name="greedy",
                mode="greedy",
                beam_width=1,
                token_prune=1,
                lm_order=1,
                lm_alpha=0.0,
            ),
            DecoderCandidate(
                name="char",
                mode="beam",
                beam_width=6,
                token_prune=4,
                lm_order=2,
                lm_smoothing=0.2,
                lm_alpha=0.03,
            ),
        ]
        return rows, stores, tuning_stores, candidates

    def test_outer_predictions_and_selection_ignore_that_folds_labels(self) -> None:
        rows_a, stores, tuning_stores, candidates = self._inputs("aa")
        rows_b, _, _, _ = self._inputs("held out words must never be read")

        result_a = crossfit_decode(
            rows_a, stores, candidates, tuning_stores=tuning_stores
        )
        result_b = crossfit_decode(
            rows_b, stores, candidates, tuning_stores=tuning_stores
        )
        fold_zero_ids = [row.identifier for row in rows_a if row.fold == 0]

        self.assertEqual(
            {identifier: result_a.predictions[identifier] for identifier in fold_zero_ids},
            {identifier: result_b.predictions[identifier] for identifier in fold_zero_ids},
        )
        self.assertEqual(
            result_a.fold_audits[0]["selected_candidate"],
            result_b.fold_audits[0]["selected_candidate"],
        )
        self.assertFalse(result_a.fold_audits[0]["heldout_labels_used_for_selection_or_fit"])
        for nested in result_a.fold_audits[0]["selection"]["nested_fits"]:
            self.assertNotIn(0, nested["fit_folds"])
            self.assertNotIn(nested["tuning_fold"], nested["fit_folds"])

    def test_fixed_candidate_outer_prediction_ignores_that_folds_labels(self) -> None:
        rows_a, stores, _, candidates = self._inputs("aa")
        rows_b, _, _, _ = self._inputs("labels changed after recipe freeze")

        result_a = crossfit_decode(
            rows_a, stores, candidates, fixed_candidate="char"
        )
        result_b = crossfit_decode(
            rows_b, stores, candidates, fixed_candidate="char"
        )
        fold_zero_ids = [row.identifier for row in rows_a if row.fold == 0]

        self.assertEqual(
            {identifier: result_a.predictions[identifier] for identifier in fold_zero_ids},
            {identifier: result_b.predictions[identifier] for identifier in fold_zero_ids},
        )
        self.assertEqual(result_a.fold_audits[0]["selection"]["mode"], "fixed_candidate")

    def test_tuned_mode_rejects_ordinary_single_exclusion_oof_stores(self) -> None:
        rows, stores, _, candidates = self._inputs("aa")
        wrong_inner = {
            (outer_fold, tuning_fold): stores[tuning_fold]
            for outer_fold in range(3)
            for tuning_fold in range(3)
            if tuning_fold != outer_fold
        }

        with self.assertRaisesRegex(ValueError, "did not exclude both folds"):
            crossfit_decode(rows, stores, candidates, tuning_stores=wrong_inner)


if __name__ == "__main__":
    unittest.main()
