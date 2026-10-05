"""Validate and replay the existing character recipe as a separate candidate."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"model112"))
from fusion import fuse
from road_htr.stackmix import sha256,save_json
from road_htr.schema import read_table,predictions_by_id
from road_htr.metrics import competition_error
from road_htr.submission import write_submission,validate_submission,clean_prediction_preserving_internal_spaces

OUT=ROOT/"outputs/model113"


def read(path):
    table=read_table(path)
    return dict(predictions_by_id(table,"prediction" if "prediction" in table.fieldnames else table.target_column))


def main():
    output=ROOT/"submissions/model113_family_character_ensemble.csv"
    if output.exists(): raise ValueError("Refusing to overwrite an existing submission")
    protocol=json.loads((ROOT/"outputs/model112/protocol.json").read_text())
    for name,digest in protocol["code_sha256"].items():
        if sha256(ROOT/name)!=digest: raise ValueError("Previously evaluated ensemble implementation changed")
    truth_path=ROOT/"data/processed/train_folds.csv"
    if sha256(truth_path)!=protocol["truth_sha256"]: raise ValueError("Canonical training data changed")
    experiment=json.loads((ROOT/"outputs/model112/report.json").read_text())
    measured=experiment["results"]["diagnostic_family_char"]
    truth=read_table(truth_path,require_target=True)
    oof_path=ROOT/"outputs/model112/diagnostic_family_char_oof.csv"
    oof=read(oof_path)
    ids=[r[truth.id_column] for r in truth.rows]
    if set(oof)!=set(ids): raise ValueError("Incomplete OOF candidate")
    score=competition_error([r[truth.target_column] for r in truth.rows],[oof[i] for i in ids])["multi_score"]
    if abs(score-measured["score"])>1e-12 or any(r["gain"]<=0 for r in measured["per_fold"].values()):
        raise ValueError("Character candidate validation score/gains not reproduced")
    previous=json.loads((ROOT/"outputs/model112/submission_report.json").read_text())
    word_path=ROOT/previous["file"]
    if sha256(word_path)!=previous["sha256"]: raise ValueError("Model112 submission changed")
    words=read(word_path)
    baseline_path=ROOT/"submissions/model104_model105_selector.csv"
    if sha256(baseline_path)!="2fc2dea27a0b9741200d845fc929fb9af3670105424f346bda4239d0dabab735":
        raise ValueError("Validated anchor changed")
    baseline=read(baseline_path)
    test=ROOT/"data/raw/Test.csv";sample=ROOT/"data/raw/SampleSubmission.csv"
    table=read_table(test);ids=[r[table.id_column] for r in table.rows]
    sources={}
    for name,witness in previous["sources"].items():
        path=ROOT/witness["path"]
        if sha256(path)!=witness["sha256"]: raise ValueError("Previously verified test source changed")
        if "receipt_sha256" in witness:
            receipt_path=path.with_suffix(".manifest.json")
            if sha256(receipt_path)!=witness["receipt_sha256"]: raise ValueError("Inference receipt changed")
            receipt=json.loads(receipt_path.read_text())
            if receipt["fold"] is not None or receipt["rows"]!=len(ids) or receipt["input_sha256"]!=sha256(test) or receipt["output_sha256"]!=witness["sha256"]:
                raise ValueError("Source is not the verified all-data test prediction")
        sources[name]=read(path)
    if set(baseline)!=set(ids) or set(words)!=set(ids) or any(set(v)!=set(ids) for v in sources.values()):
        raise ValueError("Test IDs differ")
    predictions={};changes=[];abstentions=[]
    for identifier in ids:
        candidates={name:values[identifier] for name,values in sources.items()}
        if any(not text.strip() for text in candidates.values()):
            predictions[identifier]=baseline[identifier];abstentions.append(identifier)
        else:
            predictions[identifier]=fuse(baseline[identifier],candidates,family=True,character=True)
        if predictions[identifier]!=baseline[identifier] or predictions[identifier]!=words[identifier]:
            changes.append({"ID":identifier,"original_anchor":baseline[identifier],"model112_word":words[identifier],
                            "model113_character":predictions[identifier]})
    OUT.mkdir(parents=True,exist_ok=True)
    write_submission(sample,output,predictions,cleaner=clean_prediction_preserving_internal_spaces)
    validation=validate_submission(output,sample)
    # Check the serialized file preserves the exact outputs, not only CSV schema.
    if read(output)!=predictions: raise ValueError("Serialization changed prediction text")
    with (OUT/"test_changes.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["ID","original_anchor","model112_word","model113_character"])
        writer.writeheader();writer.writerows(changes)
    report={"file":str(output.relative_to(ROOT)),"sha256":sha256(output),"validation":validation,
            "recipe":"diagnostic_family_char (unchanged)","anchor":"model104_model105_selector",
            "changed_vs_original_anchor":sum(predictions[i]!=baseline[i] for i in ids),
            "changed_vs_model112_word":sum(predictions[i]!=words[i] for i in ids),
            "blank_source_abstentions":abstentions,"development_oof_score":score,
            "development_delta_vs_word":score-experiment["results"]["diagnostic_family_word"]["score"],
            "input_model112_report_sha256":sha256(ROOT/"outputs/model112/submission_report.json"),
            "oof_sha256":sha256(oof_path),"sources":previous["sources"],
            "model112_public_score_user_reported":0.911129378,"public_score":None,"uploaded":False,
            "note":"Secondary recipe selected after development evaluation. No unbiased performance or leaderboard improvement guarantee."}
    save_json(OUT/"submission_report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="sources"},indent=2))


if __name__=="__main__":main()
