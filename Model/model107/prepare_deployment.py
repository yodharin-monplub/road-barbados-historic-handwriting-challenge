"""Prepare all-data StackMix assets, kept separate from every CV fold."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from road_htr.stackmix import build_bank,synthesize,save_json


def main():
    config=json.loads((ROOT/"model107/config.json").read_text())
    config.update(fold=None,
        checkpoint="outputs/model36/deployment_epoch07/epoch07.safetensors",
        training_manifest="outputs/model36/foldall/run_manifest.json",
        conversion_receipt="outputs/model36/deployment_epoch07/epoch07.conversion.json")
    out=ROOT/"outputs/model107/foldall"
    save_json(out/"prepare_config.json",config)
    build_bank(config,out/"bank",device="cuda")
    if not (out/"training/manifest.json").exists():
        synthesize(config,out/"bank/bank.json",out/"training",count=1023)
    subprocess.run([sys.executable,"model107/train.py","--all-data","--prepare-only"],cwd=ROOT,check=True)
    save_json(out/"preparation_complete.json",{"purpose":"deployment_only_not_CV","ready":True,
        "note":"Training and submission remain gated on full-OOF confirmation."})


if __name__=="__main__":
    main()
