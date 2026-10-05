"""Replay primary family-word rule on saved test outputs; never upload to Zindi."""
import csv
import json
from run import ROOT, OUT, TRUTH, read, sha256, save_json
from fusion import fuse
from road_htr.schema import read_table
from road_htr.submission import write_submission,validate_submission,clean_prediction_preserving_internal_spaces


def main():
    experiment=json.loads((OUT/"report.json").read_text())
    result=experiment["results"]["diagnostic_family_word"]
    if result["gain"]<.0005 or result["conditional_bootstrap_95"][0]<=0 or any(r["gain"]<=0 for r in result["per_fold"].values()):
        raise ValueError("Primary word rule did not show sufficiently consistent development gains")
    protocol=json.loads((OUT/"protocol.json").read_text())
    for name,digest in protocol["code_sha256"].items():
        if sha256(ROOT/name)!=digest: raise ValueError("Validated ensemble code changed")
    test=ROOT/"data/raw/Test.csv";sample=ROOT/"data/raw/SampleSubmission.csv"
    table=read_table(test);ids=[r[table.id_column] for r in table.rows]
    baseline_path=ROOT/"submissions/model104_model105_selector.csv"
    if sha256(baseline_path)!="2fc2dea27a0b9741200d845fc929fb9af3670105424f346bda4239d0dabab735":
        raise ValueError("Best submitted baseline changed")
    baseline=read(baseline_path)
    configs={
        "model73":("outputs/model83/model77_test/full.csv","outputs/model77/all/best_adapter"),
        "model92":("outputs/model105/test.csv","outputs/model105/all/best_adapter"),
        "model107":("outputs/model107/test_raw.csv","outputs/model107/train/foldall/best_adapter")}
    source={};ledger={}
    for name,(path,adapter) in configs.items():
        path=ROOT/path;receipt_path=path.with_suffix(".manifest.json")
        receipt=json.loads(receipt_path.read_text())
        if receipt["fold"] is not None or receipt["rows"]!=len(ids) or receipt["input_sha256"]!=sha256(test) or receipt["output_sha256"]!=sha256(path):
            raise ValueError("Test prediction provenance mismatch")
        if receipt["image_variant"]!="full": raise ValueError("Expected full-image view")
        for filename,digest in receipt["adapter_files_sha256"].items():
            if filename in ("adapter_model.safetensors","adapter_config.json","processor_config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja"):
                if sha256(ROOT/adapter/filename)!=digest: raise ValueError("All-data adapter checksum mismatch")
        source[name]=read(path)
        ledger[name]={"path":str(path.relative_to(ROOT)),"sha256":sha256(path),"receipt_sha256":sha256(receipt_path)}
    kraken=ROOT/"outputs/model35/deployment_epoch07/confusions_test.csv"
    previous=json.loads((ROOT/"outputs/model87/report.json").read_text())
    if sha256(kraken)!=previous["sources"]["kraken"]["sha256"]: raise ValueError("Kraken test witness changed")
    source["kraken"]=read(kraken);ledger["kraken"]={"path":str(kraken.relative_to(ROOT)),"sha256":sha256(kraken)}
    if set(baseline)!=set(ids) or any(set(v)!=set(ids) for v in source.values()): raise ValueError("Test IDs differ")
    predictions={};changes=[];abstained=[]
    for identifier in ids:
        candidates={name:values[identifier] for name,values in source.items()}
        if any(not text.strip() for text in candidates.values()):
            # Do not turn a missing source into a duplicated baseline vote.
            predictions[identifier]=baseline[identifier];abstained.append(identifier);continue
        predictions[identifier]=fuse(baseline[identifier],candidates,family=True,character=False)
        if predictions[identifier]!=baseline[identifier]:
            changes.append({"ID":identifier,"before":baseline[identifier],"after":predictions[identifier]})
    output=ROOT/"submissions/model112_family_word_ensemble.csv"
    if output.exists(): raise ValueError("Refusing to overwrite a submission")
    write_submission(sample,output,predictions,cleaner=clean_prediction_preserving_internal_spaces)
    validation=validate_submission(output,sample)
    with (OUT/"test_changes.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["ID","before","after"]);writer.writeheader();writer.writerows(changes)
    report={"file":str(output.relative_to(ROOT)),"sha256":sha256(output),"validation":validation,
            "changed_lines":len(changes),"blank_source_abstentions":abstained,"sources":ledger,
            "development_oof_score":result["score"],"public_score":None,"uploaded":False,
            "note":"Experimental candidate. Development gains inherit upstream selection and model-selection risk; public/private improvement is not guaranteed."}
    save_json(OUT/"submission_report.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
