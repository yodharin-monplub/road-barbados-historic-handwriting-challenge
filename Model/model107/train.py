"""Materialize an isolated Model92-derived config, then train Model107."""
from pathlib import Path
import argparse
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from road_htr.vlm_cli import train
from road_htr.stackmix import eligible_rows, validate_synthetic, absolute
from road_htr.vlm_data import read_id_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold",type=int,default=0)
    parser.add_argument("--all-data",action="store_true")
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    fold=None if args.all_data else args.fold
    label="all" if args.all_data else str(fold)
    config=yaml.safe_load((ROOT/"model92/config.yaml").read_text())
    config["output_dir"]="outputs/model107/train/fold{fold}"
    config["validation_fold"]=fold
    config["synthetic_manifest"]=str(args.manifest or f"outputs/model107/fold{label}/training/manifest.json")
    config["training"]["max_steps"]=500 if args.all_data else 400
    config["training"]["save_steps"]=config["training"]["max_steps"]
    if args.all_data:
        config["output_dir"]="outputs/model107/train/foldall"
    truth=absolute(config["train_csv"])
    table,eligible=eligible_rows(truth,fold,read_id_file(absolute(config["exclude_ids"])))
    rows=validate_synthetic(config["synthetic_manifest"],truth,fold,eligible,table.target_column)
    if len(rows)!=len(eligible)//4:
        raise ValueError("Expected 80/20 real/synthetic dataset mix")
    output=ROOT/f"outputs/model107/config_fold{label}.yaml"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(yaml.safe_dump(config,sort_keys=False))
    print(f"Config ready: {output}; real={len(eligible)}, synthetic={len(rows)}",flush=True)
    if not args.prepare_only:
        if args.all_data:
            import json
            decision=json.loads((ROOT/"outputs/model107/promotion_decision.json").read_text())
            if not decision["passes_submission_gate"]:
                raise ValueError("All-data training requires a passing full-OOF promotion gate")
        train(output,fold)


if __name__=="__main__":
    main()
