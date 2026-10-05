from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from road_htr.metrics import competition_error
from road_htr.model34_token_fusion import (
    FineCandidate,
    Rule,
    _row_cost,
    _score_costs,
    align_tokens,
    apply_rule_files,
    paired_bootstrap,
    prepare_row,
    fuse_prepared,
)
from road_htr.schema import SchemaError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Model34TokenFusionTests(unittest.TestCase):
    def _rule(self, **updates: object) -> Rule:
        values = {
            "kind": "fusion",
            "granularity": "token",
            "quality_kind": "min",
            "confidence_threshold": 0.9,
            "maximum_token_distance": 0.5,
            "maximum_row_distance": 1.0,
            "context": "none",
            "token_mode": "any",
        }
        values.update(updates)
        return Rule(**values)

    def test_alignment_exposes_donor_insertion(self) -> None:
        row = prepare_row(
            "the house",
            FineCandidate("the old house", (0.99,) * len("the old house")),
        )
        alignment = align_tokens(row.base_tokens, row.fine_tokens)
        self.assertEqual(alignment, ((0, 0), (None, 1), (1, 2)))

    def test_high_confidence_close_token_is_donated(self) -> None:
        text = "the hous"
        fine = "the house"
        row = prepare_row(text, FineCandidate(fine, (0.99,) * len(fine)))
        self.assertEqual(fuse_prepared(row, self._rule()), "the house")

    def test_case_and_fold_safe_specials_stay_owned_by_model25(self) -> None:
        base = "S^t James"
        fine = "st James"
        row = prepare_row(base, FineCandidate(fine, (0.999,) * len(fine)))
        permissive = self._rule(maximum_token_distance=1.0)
        self.assertEqual(fuse_prepared(row, permissive), base)

        case_row = prepare_row(
            "Lord James", FineCandidate("lord James", (0.999,) * len("lord James"))
        )
        self.assertEqual(fuse_prepared(case_row, permissive), "Lord James")

    def test_character_mode_rejects_a_low_confidence_changed_character(self) -> None:
        base = "cot"
        candidate = FineCandidate("cat", (0.99, 0.1, 0.99))
        row = prepare_row(base, candidate)
        rule = self._rule(
            granularity="character",
            quality_kind="mean",
            confidence_threshold=0.5,
            maximum_token_distance=1.0,
        )
        self.assertEqual(fuse_prepared(row, rule), base)

    def test_cost_aggregation_matches_public_metric_implementation(self) -> None:
        references = ["alpha beta", "gamma"]
        hypotheses = ["alpha zeta", "gamma"]
        direct = competition_error(references, hypotheses)
        from_costs = _score_costs(
            _row_cost(reference, hypothesis)
            for reference, hypothesis in zip(references, hypotheses, strict=True)
        )
        self.assertAlmostEqual(
            float(from_costs["multi_score"]), float(direct["multi_score"]), places=14
        )

    def test_bootstrap_is_deterministic(self) -> None:
        baseline = [_row_cost("abc", "axc"), _row_cost("def", "dxf")]
        challenger = [_row_cost("abc", "abc"), _row_cost("def", "def")]
        left = paired_bootstrap(baseline, challenger, samples=50, seed=34)
        right = paired_bootstrap(baseline, challenger, samples=50, seed=34)
        self.assertEqual(left, right)
        self.assertEqual(left["positive_fraction"], 1.0)

    def test_apply_fails_closed_on_id_mismatch_and_bad_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.csv"
            with baseline.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"ID": "a", "prediction": "alpha"},
                        {"ID": "b", "prediction": "beta"},
                    ]
                )
            fine = root / "fine.jsonl"
            fine.write_text(
                json.dumps(
                    {
                        "id": "a",
                        "candidates": ["alpha"],
                        "character_confidences": [0.9] * 5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            deployment = root / "deployment.json"
            deployment.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "rule": asdict(Rule(kind="noop")),
                        "protected_characters": "^&",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SchemaError, "IDs differ"):
                apply_rule_files(
                    baseline,
                    fine,
                    deployment,
                    root / "out.csv",
                    _sha256(baseline),
                    _sha256(fine),
                )
            with self.assertRaisesRegex(SchemaError, "baseline SHA-256"):
                apply_rule_files(
                    baseline,
                    fine,
                    deployment,
                    root / "out.csv",
                    "0" * 64,
                    _sha256(fine),
                )


if __name__ == "__main__":
    unittest.main()
