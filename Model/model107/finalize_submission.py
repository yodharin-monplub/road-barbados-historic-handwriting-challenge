"""Fit the frozen selector and validate a submission after the OOF gate passes."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from road_htr.crossfit_candidate_selector import fit_deployment,apply_deployment
from road_htr.submission import validate_submission
from road_htr.stackmix import save_json,sha256
from road_htr.schema import read_table,predictions_by_id


def main():
    out=ROOT/"outputs/model107"
    decision=json.loads((out/"promotion_decision.json").read_text())
    if not decision["passes_submission_gate"]:
        raise ValueError("Full-OOF promotion gate failed")
    if (ROOT/"submissions/model107_stackmix_selector.csv").exists():
        raise ValueError("Refusing to overwrite an existing submission")
    fit_deployment(ROOT/"data/processed/train_folds.csv",ROOT/"outputs/model104/full_oof/crossfit_predictions.csv",
                   out/"full_oof.csv",out/"deployment_selector.json",ROOT/"model104/config.json")
    primary=predictions_by_id(read_table(ROOT/"outputs/model104/test.csv"),"prediction")
    raw=predictions_by_id(read_table(out/"test_raw.csv"),"prediction")
    if set(primary)!=set(raw):
        raise ValueError("Test candidate IDs differ")
    blanks=[i for i,t in raw.items() if not t.strip()]
    challenger=out/"test_with_blank_fallback.csv"
    with challenger.open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["ID","prediction"])
        writer.writerows((i,t if t.strip() else primary[i]) for i,t in raw.items())
    submission=ROOT/"submissions/model107_stackmix_selector.csv"
    result=apply_deployment(ROOT/"data/raw/Test.csv",ROOT/"outputs/model104/test.csv",challenger,
        out/"deployment_selector.json",out/"test_selected.csv",sample_path=ROOT/"data/raw/SampleSubmission.csv",submission_path=submission)
    validation=validate_submission(submission,ROOT/"data/raw/SampleSubmission.csv")
    save_json(out/"submission_report.json",{"selection":result,"validation":validation,"blank_fallback_ids":blanks,
        "file":str(submission.relative_to(ROOT)),"sha256":sha256(submission),"public_score":None,
        "note":"Prepared only; not uploaded to Zindi. OOF evidence is not a public/private score guarantee."})
    print(f"Submission ready: {submission}")


if __name__=="__main__": main()
