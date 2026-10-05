"""Resume only missing Model108/109 quality checks; never repeat generator training."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from road_htr.stackmix import save_json, sha256
from road_htr.schema import read_table, predictions_by_id
from run_model109_pilot import stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-checkpoint",action="store_true")
    args=parser.parse_args()
    if args.wait_checkpoint:
        import yaml
        from road_htr.vlm_cli import _verify_local_checkpoint
        config=yaml.safe_load((ROOT/"model92/config.yaml").read_text())
        for attempt in range(24):
            try:
                _verify_local_checkpoint(config,ROOT/config["local_path"])
                save_json(ROOT/"outputs/model108/base_checkpoint_verified.json",{"verified":True,"files":len(config["expected_files_sha256"])})
                break
            except (OSError,ValueError,RuntimeError):
                save_json(ROOT/"outputs/model109/status.json",{"stage":"waiting_checkpoint_transfer","attempt":attempt+1})
                time.sleep(600)
        else:
            raise RuntimeError("Checkpoint transfer did not pass verification within four hours")
    for model in [108,109]:
        directory=ROOT/f"outputs/model{model}/fold0/pilot"
        generated=json.loads((directory/"generation_manifest.json").read_text())
        if not generated["complete"] or len(generated["rows"])!=200:
            raise ValueError("Expected the existing complete 200-word generation")
        for checker in ["kraken","lighton"]:
            csv=directory/f"{checker}.csv"
            receipt=directory/f"{checker}_audit_receipt.json"
            valid=False
            if csv.exists() and receipt.exists():
                data=json.loads(receipt.read_text())
                valid=(data.get("output_sha256")==sha256(csv)
                       and data.get("generation_manifest_sha256")==sha256(directory/"generation_manifest.json")
                       and data.get("labels_sha256")==sha256(directory/"labels.csv"))
                if valid:
                    valid=set(predictions_by_id(read_table(csv),"prediction"))=={r["id"] for r in generated["rows"]}
            if not valid:
                stage(f"model{model}_{checker}_audit",["model108/audit.py","--directory",str(directory),"--checker",checker],
                      ".venv-kraken" if checker=="kraken" else ".venv-lighton")
        stage(f"model{model}_score",["model108/audit.py","--directory",str(directory),"--checker","score"])
    reports={str(m):json.loads((ROOT/f"outputs/model{m}/fold0/pilot/quality_report.json").read_text()) for m in [108,109]}
    before=set(reports["108"]["accepted_ids"])
    after=set(reports["109"]["accepted_ids"])
    comparison={"reports":reports,"acceptance_delta":reports["109"]["acceptance_rate"]-reports["108"]["acceptance_rate"],
        "newly_correct_with_finetuning":len(after-before),"lost_after_finetuning":len(before-after),
        "both_correct":len(before & after),"submission_ready":False,
        "note":"Matched requests/seeds; source lines excluded from generator fitting. Text-fidelity diagnostic, not OCR competition CV."}
    save_json(ROOT/"outputs/model109/comparison.json",comparison)
    save_json(ROOT/"outputs/model109/status.json",{"stage":"pilot_complete","submission_ready":False})
    print(json.dumps({k:v for k,v in comparison.items() if k!="reports"},indent=2))


if __name__=="__main__":
    try: main()
    except Exception as error:
        save_json(ROOT/"outputs/model109/status.json",{"stage":"audit_failed","error":str(error)})
        raise
