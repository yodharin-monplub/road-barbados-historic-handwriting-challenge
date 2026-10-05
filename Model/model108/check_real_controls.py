"""Diagnose checker behavior on real crops of the exact 200 requested words."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from road_htr.stackmix import absolute,sha256,save_json
from road_htr.schema import read_table,predictions_by_id
from prepare import validate_requests

OUT=ROOT/"outputs/model108/fold0/real_crop_control"


def prepare():
    request_path=ROOT/"outputs/model108/fold0/pilot/requests.json"
    manifest=json.loads(request_path.read_text())
    rows=validate_requests(manifest)
    bank=json.loads(absolute(manifest["bank_path"]).read_text())
    words={(w["source_id"],tuple(w["text_span"])):w for w in bank["words"]}
    images=OUT/"images"
    images.mkdir(parents=True,exist_ok=True)
    controls=[]
    for row in rows:
        word=words[row["target_source_id"],tuple(row["target_span"])]
        if word["text"]!=row["target"] or sha256(absolute(word["image"]))!=word["sha256"]:
            raise ValueError("Original crop text/hash mismatch")
        link=images/f"{row['id']}.png"
        if link.exists():
            if link.resolve()!=absolute(word["image"]).resolve():
                raise ValueError("Control ID points to a different crop")
        else:
            link.symlink_to(absolute(word["image"]))
        controls.append({"ID":row["id"],"Target":row["target"],"source":word})
    with (OUT/"labels.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["ID","Target"])
        writer.writeheader()
        writer.writerows({k:r[k] for k in ["ID","Target"]} for r in controls)
    save_json(OUT/"control_manifest.json",{"method":"real_training_crop_control","requests_sha256":sha256(request_path),
        "rows":controls,"note":"Original source lines were used to train the OCR checkers and crops selected by Kraken confidence. Optimistic diagnostic control, NOT unseen CV or proof of perfect crop labels."})
    return bank


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker",choices=["kraken","lighton"])
    args=parser.parse_args()
    bank=prepare()
    if args.checker:
        checker=args.checker
        config=yaml.safe_load((ROOT/f"outputs/model108/fold0/pilot/{checker}_config.yaml").read_text())
        config["image_dir"]=str(OUT/"images")
        config_path=OUT/f"{checker}_config.yaml"
        config_path.write_text(yaml.safe_dump(config,sort_keys=False))
        if checker=="kraken":
            from road_htr.kraken_cli import infer
            infer(config_path,OUT/"labels.csv",OUT/"kraken.csv",checkpoint=absolute(bank["signature"]["config"]["checkpoint"]))
        else:
            from road_htr.vlm_cli import infer
            infer(config_path,OUT/"labels.csv",OUT/"lighton.csv",adapter_path=ROOT/"outputs/model107/train/fold0/best_adapter")
        return
    for checker,environment in [("kraken",".venv-kraken"),("lighton",".venv-lighton")]:
        subprocess.run([str(ROOT/environment/"bin/python"),"model108/check_real_controls.py","--checker",checker],cwd=ROOT,check=True)
    truth=predictions_by_id(read_table(OUT/"labels.csv",require_target=True),"Target")
    scores={}
    for name,directory in [("real_controls",OUT),("model108",ROOT/"outputs/model108/fold0/pilot"),("model109",ROOT/"outputs/model109/fold0/pilot")]:
        predictions={c:predictions_by_id(read_table(directory/f"{c}.csv"),"prediction") for c in ["kraken","lighton"]}
        if any(set(p)!=set(truth) for p in predictions.values()): raise ValueError("Mismatched control IDs")
        scores[name]={c:sum(p[i]==t for i,t in truth.items()) for c,p in predictions.items()}
        scores[name]["both"]=sum(all(p[i]==t for p in predictions.values()) for i,t in truth.items())
    report={"rows":len(truth),"exact_counts":scores,"scope":"Diagnostic: the same requested historical words, real automatic crops versus generated images. Controls are in-training and aligner-selected, not unbiased validation."}
    save_json(OUT/"report.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
