from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from road_htr.ctc_decode import CharNGramLM
from road_htr.vlm_ocr_guarded_selector import (
    EXPECTED_POLICY,
    Model52GateError,
    _prediction_map,
    select_line,
)


class Model52GuardedSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lm = CharNGramLM(order=5, smoothing=0.2).fit(
            ["Sealed and delivered in the presence of"] * 20
            + ["By this public act and instrument of protest"] * 20
        )

    def test_blank_fallback_fails_closed(self) -> None:
        with self.assertRaises(Model52GateError):
            select_line("", "Sealed and delivered", self.lm)

    def test_blank_vlm_uses_fallback(self) -> None:
        result = select_line("Sealed and delivered", "", self.lm)
        self.assertEqual(result.selected_source, "fallback")
        self.assertEqual(result.reason, "blank_vlm")

    def test_role_marker_uses_fallback(self) -> None:
        result = select_line(
            "Sealed and delivered",
            "assistant: Sealed and delivered",
            self.lm,
        )
        self.assertEqual(result.reason, "role_marker")

    def test_repeated_four_word_ngram_uses_fallback(self) -> None:
        result = select_line(
            "By this public act and instrument of protest",
            "Sealed and delivered in the presence of Sealed and delivered in the presence of",
            self.lm,
        )
        self.assertEqual(result.reason, "repeated_word_ngram")

    def test_extreme_length_uses_fallback(self) -> None:
        result = select_line(
            "By this public act and instrument of protest",
            "Sealed",
            self.lm,
        )
        self.assertIn(result.reason, {"character_length_ratio", "token_length_ratio"})

    def test_large_character_divergence_uses_fallback(self) -> None:
        relaxed_lm = replace(EXPECTED_POLICY, minimum_language_advantage=-100.0)
        result = select_line(
            "zzzz zzzz zzzz zzzz zzzz zzzz",
            "Sealed and delivered in the presence",
            self.lm,
            relaxed_lm,
        )
        self.assertEqual(result.reason, "char_distance")

    def test_close_language_improvement_can_select_vlm(self) -> None:
        permissive = replace(
            EXPECTED_POLICY,
            minimum_language_advantage=-0.01,
            edit_distance_penalty=0.0,
        )
        result = select_line(
            "sealed and delivered in the presence of",
            "Sealed and delivered in the presence of",
            self.lm,
            permissive,
        )
        self.assertEqual(result.selected_source, "vlm")
        self.assertEqual(result.reason, "guarded_language_advantage")

    def test_insufficient_language_advantage_uses_fallback(self) -> None:
        strict = replace(EXPECTED_POLICY, minimum_language_advantage=100.0)
        result = select_line(
            "sealed and delivered in the presence of",
            "Sealed and delivered in the presence of",
            self.lm,
            strict,
        )
        self.assertEqual(result.reason, "insufficient_language_advantage")

    def test_identical_line_is_stable(self) -> None:
        result = select_line("Sealed and delivered", "Sealed and delivered", self.lm)
        self.assertEqual(result.selected, "Sealed and delivered")
        self.assertEqual(result.reason, "identical")

    def test_full_oof_fallback_can_be_safely_subset(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text("ID,prediction\na,first\nb,second\n", encoding="utf-8")
            values, receipt = _prediction_map(path, ["b"], allow_superset=True)
            self.assertEqual(values, {"b": "second"})
            self.assertEqual(receipt["artifact_rows"], 2)
            self.assertEqual(receipt["used_rows"], 1)
            self.assertTrue(receipt["accepted_superset"])

    def test_vlm_vector_rejects_extra_ids(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text("ID,prediction\na,first\nb,second\n", encoding="utf-8")
            with self.assertRaises(Model52GateError):
                _prediction_map(path, ["b"])


if __name__ == "__main__":
    unittest.main()
