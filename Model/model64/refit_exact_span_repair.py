#!/usr/bin/env python3
"""Model64: refit Model61 exact-span repair on gated Model63 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from model54.restore_final_tokens import resolve  # noqa: E402
from model61.exact_span_repair import build_outer_fold_maps, fit_mapping, repair_exact_spans  # noqa: E402
from road_htr.crossfold_pair_stage import apply_pair_stage, evaluate_pair_stage  # noqa: E402

OUTPUT_ROOT = (ROOT / "outputs/model64").resolve()


class Model64GateError(RuntimeError): pass


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve(path); config = json.loads(resolved.read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "model_number": 64, "source_model": "model63", "fit_policy": "outer_fold_excluded_exact_replacement_spans_refit", "maximum_source_tokens": 2, "maximum_target_tokens": 2, "minimum_support": 3, "minimum_precision": 0.5, "operation": "difflib_replace_blocks_only", "application_order": "longest_source_first_left_to_right", "output_root": "outputs/model64"}
    if config != expected: raise Model64GateError("Model64 config is immutable")
    return config, resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Model64 refit exact-span repair"); subs = parser.add_subparsers(dest="command", required=True)
    oof=subs.add_parser("oof"); oof.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json")); oof.add_argument("--truth",type=Path,default=ROOT/"data/processed/train_folds.csv"); oof.add_argument("--predictions",type=Path,required=True); oof.add_argument("--output-dir",type=Path,required=True)
    app=subs.add_parser("apply"); app.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json")); app.add_argument("--truth",type=Path,default=ROOT/"data/processed/train_folds.csv"); app.add_argument("--training-predictions",type=Path,required=True); app.add_argument("--test",type=Path,default=ROOT/"data/raw/Test.csv"); app.add_argument("--predictions",type=Path,required=True); app.add_argument("--output-dir",type=Path,required=True); app.add_argument("--sample",type=Path); app.add_argument("--submission",type=Path)
    args=parser.parse_args(); config,path=load_config(args.config)
    repair=lambda value,mapping: repair_exact_spans(value,mapping,int(config["maximum_source_tokens"]))
    common={"model_number":64,"config":config,"config_path":path,"implementation_path":Path(__file__).resolve(),"output_root":OUTPUT_ROOT,"truth_path":args.truth}
    if args.command=="oof": report=evaluate_pair_stage(**common,predictions_path=args.predictions,output_dir=args.output_dir,build_maps=build_outer_fold_maps,repair=repair)
    else: report=apply_pair_stage(**common,training_predictions_path=args.training_predictions,test_path=args.test,predictions_path=args.predictions,output_dir=args.output_dir,fit_mapping=fit_mapping,repair=repair,sample_path=args.sample,submission_path=args.submission)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
