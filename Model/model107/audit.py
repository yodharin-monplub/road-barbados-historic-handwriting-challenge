"""Automatic round-trip audit of the pilot; never manually alter labels."""
import argparse
import csv
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from road_htr.kraken_cli import infer
from road_htr.schema import read_table,predictions_by_id
from road_htr.metrics import competition_error
from road_htr.stackmix import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=ROOT/"outputs/model107/fold0/pilot")
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"outputs/model22/checkpoint_evaluation/fold0/converted/checkpoint_07-0.9209.safetensors")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=parser.parse_args()
    directory=args.directory.resolve()
    config=yaml.safe_load((ROOT/"model36/config.yaml").read_text())
    config["image_dir"]=str(directory/"images")
    config["inference"].update(accelerator="gpu" if args.device=="cuda" else "cpu",precision="32-true",threads=2)
    config_path=directory/"audit_config.yaml"
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    output=directory/"roundtrip.csv"
    # The human-readable labels file also has an image-path column. Keep the
    # recognizer input unambiguous: its schema treats 'image' as an ID alias.
    input_path=directory/"audit_input.csv"
    with (directory/"labels.csv").open(encoding="utf-8-sig") as handle:
        rows=list(csv.DictReader(handle))
    with input_path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["ID","Target"])
        writer.writeheader()
        writer.writerows({"ID":r["ID"],"Target":r["Target"]} for r in rows)
    infer(config_path,input_path,output,checkpoint=args.checkpoint)
    table=read_table(input_path,require_target=True)
    predictions=predictions_by_id(read_table(output),"prediction")
    targets=[r[table.target_column] for r in table.rows]
    predicted=[predictions[r[table.id_column]] for r in table.rows]
    report={"rows":len(targets),"exact_lines":sum(a==b for a,b in zip(targets,predicted)),
        "blank_lines":sum(not p.strip() for p in predicted),
        "metrics":competition_error(targets,predicted),
        "note":"Same-aligner consistency diagnostic, NOT an independent CV score or proof of correct labels."}
    save_json(directory/"roundtrip_report.json",report)
    print(report)


if __name__=="__main__":
    main()
