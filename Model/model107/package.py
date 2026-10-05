"""Package only the explicit Model107 code and derived training artifacts."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from road_htr.stackmix import sha256,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds",nargs="+",default=["0"],choices=["0","1","2","3","4","all"])
    parser.add_argument("--name",default="transfer")
    args=parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("Package name must be a filename stem")
    files=list((ROOT/"model107").glob("*.py"))+list((ROOT/"model107").glob("*.json"))
    files += [ROOT/"model107/readme.txt",ROOT/"model107/run_gpu.sh"]
    files += [ROOT/"src/road_htr"/name for name in ("stackmix.py","vlm_data.py","vlm_cli.py")]
    files += [ROOT/"model107/run_confirmation.sh"]
    if "all" in args.folds:
        files += [ROOT/"model107/run_deployment.sh", ROOT/"outputs/model107/promotion_decision.json"]
    for fold in args.folds:
        files += [ROOT/f"outputs/model107/config_fold{fold}.yaml",ROOT/f"outputs/model107/fold{fold}/bank/bank.json"]
        training=ROOT/f"outputs/model107/fold{fold}/training"
        files += [training/"manifest.json",training/"labels.csv"]
        manifest=json.loads((training/"manifest.json").read_text())
        files += [ROOT/r["image"] for r in manifest["rows"]]
    files=sorted(set(files))
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in files}
    ledger=ROOT/f"outputs/model107/{args.name}_manifest.json"
    save_json(ledger,{"files":hashes})
    archive=ROOT/f"outputs/model107/{args.name}.tar.gz"
    with tarfile.open(archive,"w:gz") as tar:
        for path in files+[ledger]:
            tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    print(json.dumps({"archive":str(archive),"sha256":sha256(archive),"files":len(files),"bytes":archive.stat().st_size}))


if __name__=="__main__":
    main()
