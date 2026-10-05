"""Compare a held-fold candidate with the fixed Model92 baseline."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from road_htr.metrics import competition_error
from road_htr.schema import read_table, predictions_by_id
from road_htr.stackmix import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold",type=int,default=0)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--baseline",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    table=read_table(ROOT/"data/processed/train_folds.csv",require_target=True)
    truth={r[table.id_column]:r[table.target_column] for r in table.rows if int(r["fold"])==args.fold}
    scores={}
    for name,path in (("candidate",args.predictions),("baseline",args.baseline or ROOT/f"outputs/model92/fold{args.fold}_oof_step400.csv")):
        predictions=predictions_by_id(read_table(path),"prediction")
        if set(predictions)!=set(truth):
            raise ValueError("Prediction IDs must match the complete heldout fold")
        scores[name]=competition_error(list(truth.values()),[predictions[i] for i in truth])
    delta=scores["candidate"]["multi_score"]-scores["baseline"]["multi_score"]
    report={"fold":args.fold,"rows":len(truth),"scores":scores,"delta":delta,
            "development_gate_passed":delta>=0.001,
            "note":"Development screen only; full OOF confirmation required before submission."}
    save_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
