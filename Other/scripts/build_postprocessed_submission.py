#!/usr/bin/env python3
"""Rebuild a frozen postprocessed candidate from raw OCR predictions."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.approximate_boundary_corrector import apply as apply_approximate  # noqa: E402
from road_htr.boundary_corrector import apply as apply_boundaries  # noqa: E402
from road_htr.confusion_corrector import apply as apply_confusions  # noqa: E402
from road_htr.contextual_corrector import apply as apply_context  # noqa: E402
from road_htr.supervised_error_corrector import apply as apply_supervised  # noqa: E402
from road_htr.submission import write_submission  # noqa: E402
from road_htr.view_selector import read_prediction_map  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("full", "two-view", "lm", "three-view"), required=True
    )
    parser.add_argument(
        "--config-set", choices=("fold0", "allfold"), default="fold0"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train_folds.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/Test.csv"))
    parser.add_argument(
        "--sample", type=Path, default=Path("data/raw/SampleSubmission.csv")
    )
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/pipeline"))
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--context-config", type=Path)
    parser.add_argument("--boundary-config", type=Path)
    parser.add_argument("--confusion-config", type=Path)
    parser.add_argument("--approximate-config", type=Path)
    parser.add_argument("--supervised-config", type=Path)
    parser.add_argument("--training-predictions", type=Path)
    args = parser.parse_args()

    if (args.supervised_config is None) != (args.training_predictions is None):
        parser.error(
            "--supervised-config and --training-predictions must be supplied together"
        )
    if args.config_set == "fold0":
        if args.profile in {"lm", "three-view"}:
            parser.error(f"the {args.profile} profile has no fold-0 default config chain")
        suffix = "full_view" if args.profile == "full" else "two_view"
        defaults = {
            "context": ROOT / "model11" / f"fold0_{suffix}_config.json",
            "boundary": ROOT / "model12" / "fold0_config.json",
            "confusion": ROOT / "model13" / f"fold0_{suffix}_config.json",
            "approximate": ROOT / "model14" / f"fold0_{suffix}_config.json",
        }
    else:
        suffix = args.profile.replace("-", "_")
        defaults = {
            "context": ROOT / "model11" / f"allfold_{suffix}_config.json",
            "boundary": ROOT / "model12" / f"allfold_{suffix}_config.json",
            "confusion": ROOT / "model13" / f"allfold_{suffix}_config.json",
            "approximate": ROOT / "model14" / f"allfold_{suffix}_config.json",
        }
    context_config = args.context_config or defaults["context"]
    boundary_config = args.boundary_config or defaults["boundary"]
    confusion_config = args.confusion_config or defaults["confusion"]
    approximate_config = args.approximate_config or defaults["approximate"]
    for config in (
        context_config,
        boundary_config,
        confusion_config,
        approximate_config,
    ):
        resolved = config if config.is_absolute() else ROOT / config
        if not resolved.is_file():
            parser.error(f"missing frozen config: {resolved}")
    work_dir = args.work_dir if args.work_dir.is_absolute() else ROOT / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    stage11 = work_dir / f"{args.profile}_model11.csv"
    stage12 = work_dir / f"{args.profile}_model12.csv"
    stage13 = work_dir / f"{args.profile}_model13.csv"
    stage14 = work_dir / f"{args.profile}_model14.csv"
    stage15 = work_dir / f"{args.profile}_model15.csv"
    apply_context(args.train, args.test, args.predictions, context_config, stage11)
    apply_boundaries(args.train, args.test, stage11, boundary_config, stage12)
    apply_confusions(args.train, args.test, stage12, confusion_config, stage13)
    apply_approximate(
        args.train,
        args.test,
        stage13,
        approximate_config,
        stage14,
    )
    final_predictions = stage14
    if args.supervised_config is not None and args.training_predictions is not None:
        apply_supervised(
            args.train,
            args.training_predictions,
            args.test,
            stage14,
            args.supervised_config,
            stage15,
        )
        final_predictions = stage15
    _, prediction_map = read_prediction_map(final_predictions)
    submission = args.submission if args.submission.is_absolute() else ROOT / args.submission
    write_submission(args.sample, submission, prediction_map)
    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    print(f"submission={submission}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
