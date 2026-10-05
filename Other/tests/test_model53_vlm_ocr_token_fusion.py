from __future__ import annotations

import unittest
from dataclasses import replace

from road_htr.vlm_ocr_token_fusion import (
    EXPECTED_POLICY,
    Model53GateError,
    TokenLanguageModel,
    fuse_line,
)


class Model53TokenFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TokenLanguageModel(
            ["sealed and delivered in the presence of"] * 20
            + ["by this public act and instrument"] * 10
            + ["a rare word"]
        )
        self.permissive = replace(
            EXPECTED_POLICY,
            minimum_candidate_frequency=1,
            minimum_candidate_source_ratio=1.0,
            minimum_context_score_advantage=-100.0,
        )

    def test_blank_fallback_fails_closed(self) -> None:
        with self.assertRaises(Model53GateError):
            fuse_line("", "sealed and delivered", self.model)

    def test_blank_vlm_is_noop(self) -> None:
        output, changes = fuse_line("sealed and delivered", "", self.model)
        self.assertEqual(output, "sealed and delivered")
        self.assertEqual(changes, [])

    def test_single_locally_anchored_edit_can_be_copied(self) -> None:
        output, changes = fuse_line(
            "sealed and delivored in the presence of",
            "sealed and delivered in the presence of",
            self.model,
            self.permissive,
        )
        self.assertEqual(output, "sealed and delivered in the presence of")
        self.assertEqual(len(changes), 1)

    def test_insertion_is_never_copied(self) -> None:
        fallback = "sealed delivered"
        output, changes = fuse_line(fallback, "sealed and delivered", self.model, self.permissive)
        self.assertEqual(output, fallback)
        self.assertEqual(changes, [])

    def test_multiword_rewrite_is_never_copied(self) -> None:
        fallback = "sealed badly wrong presence"
        output, changes = fuse_line(
            fallback,
            "sealed and delivered presence",
            self.model,
            self.permissive,
        )
        self.assertEqual(output, fallback)
        self.assertEqual(changes, [])

    def test_unattested_candidate_is_rejected(self) -> None:
        fallback = "sealed and delivored in the presence of"
        output, changes = fuse_line(
            fallback,
            "sealed and delivxred in the presence of",
            self.model,
        )
        self.assertEqual(output, fallback)
        self.assertEqual(changes, [])

    def test_edge_punctuation_must_match(self) -> None:
        fallback = "sealed and delivored, in the presence"
        output, changes = fuse_line(
            fallback,
            "sealed and delivered. in the presence",
            self.model,
            self.permissive,
        )
        self.assertEqual(output, fallback)
        self.assertEqual(changes, [])

    def test_unanchored_replacement_is_rejected(self) -> None:
        fallback = "xxx delivored yyy"
        output, changes = fuse_line(
            fallback,
            "aaa delivered bbb",
            self.model,
            self.permissive,
        )
        self.assertEqual(output, fallback)
        self.assertEqual(changes, [])

    def test_original_spacing_is_preserved_around_change(self) -> None:
        output, changes = fuse_line(
            "sealed  and delivored\tin the presence of",
            "sealed and delivered in the presence of",
            self.model,
            self.permissive,
        )
        self.assertEqual(output, "sealed  and delivered\tin the presence of")
        self.assertEqual(len(changes), 1)


if __name__ == "__main__":
    unittest.main()
