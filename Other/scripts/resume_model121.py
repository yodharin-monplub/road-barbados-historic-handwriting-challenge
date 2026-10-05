"""Recover Model121 after the user reboots into the updated NVIDIA driver."""
import argparse
import fcntl
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_model121 as supervisor

OUT=ROOT/"outputs/model121"
ARCHIVE=OUT/"pre_driver_restart"
REPEAT="fold1_model107_greedy"
KEEP=("fold1_model73_greedy","fold1_model73_beam")


def preflight_files():
    api=runpy.run_path(str(ROOT/"model121/experiment.py"),run_name="model121_recovery_api")
    api["check"]()
    for job in (*KEEP,REPEAT):
        fold,name,mode=api["parse_job"](job)
        api["verify_receipt"](OUT/f"{job}.csv",fold,api["adapter"](fold,name),ROOT/f"model120/{name}_{mode}.json")
    for job in api["JOBS"]:
        if job in (*KEEP,REPEAT):continue
        for suffix in (".csv",".jsonl",".manifest.json",".log"):
            if (OUT/f"{job}{suffix}").exists():raise RuntimeError(f"Unexpected later-stage output: {job}{suffix}")
    if ARCHIVE.exists() or (OUT/"driver_recovery_receipt.json").exists() or (OUT/"report.json").exists():
        raise RuntimeError("Recovery already started or experiment completed; inspect before retrying")
    for suffix in (".csv",".jsonl",".manifest.json",".log"):
        if not (OUT/f"{REPEAT}{suffix}").is_file():raise RuntimeError("Missing pre-restart artifact")
    return api


def gpu_ready():
    unit=subprocess.run(["systemctl","--user","show","road-model121.service","--property=LoadState","--property=ActiveState"],text=True,capture_output=True)
    fields=dict(line.split("=",1) for line in unit.stdout.splitlines() if "=" in line)
    if unit.returncode and fields.get("LoadState")!="not-found":
        raise RuntimeError("Cannot verify original supervisor state")
    state=fields.get("ActiveState")
    if state is None and fields.get("LoadState")!="not-found":
        raise RuntimeError("Missing original supervisor state")
    if state in ("active","activating","deactivating","reloading"):
        raise RuntimeError("Original supervisor is still live; do not start a duplicate")
    result=subprocess.run(["nvidia-smi","--query-gpu=driver_version,memory.free","--format=csv,noheader,nounits"],text=True,capture_output=True)
    if result.returncode:raise RuntimeError("NVIDIA driver mismatch remains; user reboot required. No files changed.")
    driver,free=result.stdout.strip().splitlines()[0].split(",")
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
    if apps or int(free.strip())<5000:raise RuntimeError("GPU is occupied; no other process will be interrupted")
    subprocess.run([str(ROOT/".venv-lighton/bin/python"),"-c",
                    "import torch; assert torch.cuda.is_available(); x=torch.ones(1,device='cuda'); assert x.item()==1; print('CUDA allocation/readback passed')"],check=True)
    return driver.strip()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    api=preflight_files()
    if not args.run:
        print("Recovery file/protocol preflight passed; no files changed and no jobs started")
        return
    with (OUT/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        driver=gpu_ready()
        receipt={"old_kernel_driver":"610.43.02","installed_update":"610.57.04","new_driver":driver,
                 "previous_status":json.loads((OUT/"status.json").read_text()),
                 "recovery_code_sha256":api["pilot"].sha256(Path(__file__)),
                 "protocol_sha256":api["pilot"].sha256(OUT/"protocol.json"),
                 "kept_completed_pair":list(KEEP),"repeated_for_matched_driver":REPEAT,
                 "archived_hashes":{suffix:api["pilot"].sha256(OUT/f"{REPEAT}{suffix}") for suffix in (".csv",".jsonl",".manifest.json",".log")}}
        api["pilot"].save_json(OUT/"driver_recovery_receipt.json",receipt)
        ARCHIVE.mkdir()
        for suffix in receipt["archived_hashes"]:
            (OUT/f"{REPEAT}{suffix}").rename(ARCHIVE/f"{REPEAT}{suffix}")
        try:
            for job in api["JOBS"]:
                if job not in KEEP:supervisor.run(job)
            supervisor.run("evaluate")
            report=json.loads((OUT/"report.json").read_text())
            supervisor.record("pilot_complete",scores=report["scores"],gains=report["gains"],passes_screen=report["confirmation_passed"])
        except Exception as error:
            supervisor.record("needs_attention",error=str(error))
            raise


if __name__=="__main__":main()
