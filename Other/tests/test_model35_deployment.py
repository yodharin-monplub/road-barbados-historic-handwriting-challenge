from __future__ import annotations

import json
from pathlib import Path
import unittest

from road_htr.crossfit_postprocess import StageDefinition, StageFit
from road_htr.crossfit_postprocess import STAGES
from road_htr.fold_model_postprocess import FoldModelData, fit_apply_fold_pipeline
from road_htr.model35_deployment import FIXED_STAGE_NAMES, fit_apply_deployment


class Model35DeploymentTests(unittest.TestCase):
    def stages(self, seen, broken=None):
        stages = []
        for name in FIXED_STAGE_NAMES:
            def fit(source, references, name=name):
                seen.append((name, tuple(source), tuple(references)))
                values = [value + "/" + name for value in source]
                if broken == "train" and name == "context":
                    values = []
                return StageFit({}, name, values, [[] for _ in values], {})

            def transform(source, state, config, name=name):
                values = [value + "/" + state for value in source]
                if broken == "test" and name == "context":
                    values = []
                return values, [[] for _ in values]

            stages.append(StageDefinition(name, name, fit, transform))
        return stages

    def test_fixed_sequence_never_fits_test_and_preserves_spaces(self):
        seen = []
        training, references, test = ["train  raw"], ["train truth"], ["test  raw"]
        result = fit_apply_deployment(training, references, test, stages=self.stages(seen))
        self.assertEqual(seen, [
            ("context", ("train  raw",), ("train truth",)),
            ("boundaries", ("train  raw/context",), ("train truth",)),
            ("confusions", ("train  raw/context/boundaries",), ("train truth",)),
        ])
        self.assertEqual(result.predictions, ("test  raw/context/boundaries/confusions",))
        self.assertEqual(result.stage_predictions["raw"], ("test  raw",))
        self.assertEqual(training, ["train  raw"])
        self.assertEqual(test, ["test  raw"])
        for report in result.fits.values():
            self.assertEqual(report["diagnostic_status"], "TRAIN_SIDE_IN_SAMPLE_NOT_OOF_DO_NOT_CLAIM")

    def test_rejects_stage_selection_and_bad_row_counts(self):
        with self.assertRaises(ValueError):
            fit_apply_deployment(["a"], ["b"], ["c"], stages=self.stages([])[:2])
        for broken in ("train", "test"):
            with self.subTest(broken=broken), self.assertRaises(RuntimeError):
                fit_apply_deployment(["a"], ["b"], ["c"], stages=self.stages([], broken))

    def test_rejects_empty_misaligned_or_nontext_inputs(self):
        for args in (([], [], ["c"]), (["a"], [], ["c"]), (["a"], ["b"], []), ([1], ["b"], ["c"])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                fit_apply_deployment(*args, stages=self.stages([]))

    def test_terminal_matches_frozen_policy(self):
        policy_path = Path(__file__).resolve().parents[1] / "model35/terminal_policy.json"
        policy = json.loads(policy_path.read_text())
        self.assertEqual(policy["terminal_stage"], FIXED_STAGE_NAMES[-1])
        self.assertEqual(policy["terminal_stage_index"], len(FIXED_STAGE_NAMES))

    def test_real_correctors_match_fold_pipeline_stage_by_stage(self):
        training = ["the parisb of Saint James"] * 12
        references = ["the parish of Saint James"] * 12
        test = ["the parisb of Saint James", "Saint  James"]
        data = FoldModelData(
            "ID", tuple(f"id{index}" for index in range(14)),
            tuple(references + ["unused heldout label A", "unused heldout label B"]),
            tuple([1] * 12 + [0] * 2), tuple(training + test),
        )
        validated = fit_apply_fold_pipeline(data, 0, STAGES[:3])
        deployed = fit_apply_deployment(training, references, test)
        self.assertEqual(deployed.stage_predictions, validated.stage_predictions)
        self.assertNotEqual(deployed.predictions[0], test[0])
        for stage in FIXED_STAGE_NAMES:
            self.assertEqual(deployed.fits[stage]["selected_config"],
                             validated.stage_fits[stage]["selected_config"])


if __name__ == "__main__":
    unittest.main()
