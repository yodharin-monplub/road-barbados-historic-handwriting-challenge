#!/usr/bin/env python3
"""Model66: exact left/current/right-context substitution repair."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from model54.restore_final_tokens import resolve  # noqa: E402
from road_htr.crossfold_pair_stage import apply_pair_stage, evaluate_pair_stage  # noqa: E402

OUTPUT_ROOT = (ROOT / "outputs/model66").resolve()
Context = tuple[str, str, str]


class Model66GateError(RuntimeError): pass


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved=resolve(path); config=json.loads(resolved.read_text(encoding="utf-8"))
    expected={"schema_version":1,"model_number":66,"source_model":"model65","fit_policy":"outer_fold_excluded_exact_three_token_context_substitution","minimum_support":2,"minimum_precision":0.8,"operation":"difflib_single_token_replace_with_exact_left_right_context","output_root":"outputs/model66"}
    if config != expected: raise Model66GateError("Model66 config is immutable")
    return config,resolved


def fit_mapping(predictions: Sequence[str], references: Sequence[str], config: Mapping[str, Any]) -> dict[Context, dict[str, Any]]:
    if len(predictions) != len(references): raise Model66GateError("prediction/reference length mismatch")
    occurrences: Counter[Context] = Counter(); outcomes: dict[Context, Counter[str]] = defaultdict(Counter)
    for prediction,reference in zip(predictions,references,strict=True):
        source,target=prediction.split(),reference.split(); bounded=["<s>",*source,"</s>"]
        occurrences.update(tuple(bounded[index:index+3]) for index in range(len(bounded)-2))
        matcher=difflib.SequenceMatcher(None,source,target,autojunk=False)
        for tag,source_start,source_end,target_start,target_end in matcher.get_opcodes():
            if tag=="replace" and source_end-source_start==1 and target_end-target_start==1:
                context=(source[source_start-1] if source_start else "<s>",source[source_start],source[source_start+1] if source_start+1<len(source) else "</s>")
                outcomes[context][target[target_start]] += 1
    mapping: dict[Context, dict[str, Any]] = {}
    for context,counts in outcomes.items():
        target,support=max(counts.items(),key=lambda item:(item[1],item[0])); observations=occurrences[context]; precision=support/observations
        if target != context[1] and support>=int(config["minimum_support"]) and precision>=float(config["minimum_precision"]):
            mapping[context]={"target":target,"support":support,"observations":observations,"precision":precision}
    return mapping


def build_outer_fold_maps(predictions: Sequence[str], references: Sequence[str], folds: Sequence[int], config: Mapping[str, Any]) -> dict[int, dict[Context, dict[str, Any]]]:
    return {heldout:fit_mapping([value for value,fold in zip(predictions,folds,strict=True) if fold!=heldout],[value for value,fold in zip(references,folds,strict=True) if fold!=heldout],config) for heldout in sorted(set(folds))}


def repair_contextual_substitutions(value: str, mapping: Mapping[Context, Mapping[str, Any]]) -> tuple[str,list[dict[str,Any]]]:
    tokens=value.split(); bounded=["<s>",*tokens,"</s>"]; corrected=[]; changes=[]
    for index in range(1,len(bounded)-1):
        context=tuple(bounded[index-1:index+2]); rule=mapping.get(context)
        if rule is None: corrected.append(bounded[index]); continue
        target=str(rule["target"]); corrected.append(target)
        changes.append({"token_index":index-1,"left":context[0],"before":context[1],"right":context[2],"after":target,"support":int(rule["support"]),"observations":int(rule["observations"]),"precision":float(rule["precision"])})
    return " ".join(corrected),changes


def main() -> None:
    parser=argparse.ArgumentParser(description="Model66 exact-context substitutions"); subs=parser.add_subparsers(dest="command",required=True)
    oof=subs.add_parser("oof"); oof.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json")); oof.add_argument("--truth",type=Path,default=ROOT/"data/processed/train_folds.csv"); oof.add_argument("--predictions",type=Path,required=True); oof.add_argument("--output-dir",type=Path,required=True)
    app=subs.add_parser("apply"); app.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json")); app.add_argument("--truth",type=Path,default=ROOT/"data/processed/train_folds.csv"); app.add_argument("--training-predictions",type=Path,required=True); app.add_argument("--test",type=Path,default=ROOT/"data/raw/Test.csv"); app.add_argument("--predictions",type=Path,required=True); app.add_argument("--output-dir",type=Path,required=True); app.add_argument("--sample",type=Path); app.add_argument("--submission",type=Path)
    args=parser.parse_args(); config,path=load_config(args.config)
    common={"model_number":66,"config":config,"config_path":path,"implementation_path":Path(__file__).resolve(),"output_root":OUTPUT_ROOT,"truth_path":args.truth}
    if args.command=="oof": report=evaluate_pair_stage(**common,predictions_path=args.predictions,output_dir=args.output_dir,build_maps=build_outer_fold_maps,repair=repair_contextual_substitutions)
    else: report=apply_pair_stage(**common,training_predictions_path=args.training_predictions,test_path=args.test,predictions_path=args.predictions,output_dir=args.output_dir,fit_mapping=fit_mapping,repair=repair_contextual_substitutions,sample_path=args.sample,submission_path=args.submission)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
