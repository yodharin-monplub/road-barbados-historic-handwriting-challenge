"""Local confirmation supervisor: wait for free GPU, check at 600-second intervals."""
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/model111"
PYTHON = str(ROOT/".venv-lighton/bin/python")


def record(stage,**fields):
    value={"stage":stage,"time":time.strftime("%Y-%m-%dT%H:%M:%S%z"),**fields}
    p=OUT/"confirmation_status.tmp"
    p.write_text(json.dumps(value,indent=2)+"\n");p.replace(OUT/"confirmation_status.json")
    print(json.dumps(value),flush=True)


def gpu_state():
    p=subprocess.run(["nvidia-smi","--query-gpu=memory.free,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
    line=p.stdout.strip().splitlines()[0]
    free,util=[int(v.strip()) for v in line.split(",")]
    jobs=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True).stdout.strip()
    return {"free_mib":free,"utilization":util,"compute_active":bool(jobs)}


def main():
    pilot=json.loads((OUT/"fold0/report.json").read_text())
    if not pilot["passes_development_screen"]: raise RuntimeError("Pilot gate failed")
    for fold in range(1,5):
        record("verifying_fold_inputs",fold=fold)
        subprocess.run([PYTHON,"model111/confirm.py","prepare","--fold",str(fold)],cwd=ROOT,check=True)
    for fold in range(1,5):
        for judge in ("model73","model107"):
            receipt=OUT/f"fold{fold}/{judge}_scores.manifest.json"
            complete=receipt.exists() and json.loads(receipt.read_text()).get("complete")
            if not complete:
                while True:
                    state=gpu_state()
                    if state["free_mib"]>=5000 and state["utilization"]<20 and not state["compute_active"]: break
                    record("waiting_for_local_gpu",fold=fold,judge=judge,**state)
                    time.sleep(600)
            record("scoring",fold=fold,judge=judge)
            with (OUT/f"fold{fold}/{judge}.log").open("a") as log:
                p=subprocess.Popen([PYTHON,"model111/confirm.py","score","--fold",str(fold),"--judge",judge],cwd=ROOT,
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                while True:
                    try:
                        code=p.wait(timeout=600);break
                    except subprocess.TimeoutExpired:
                        path=OUT/f"fold{fold}/{judge}_scores.jsonl"
                        count=0
                        if path.exists():
                            with path.open() as handle: count=sum(1 for _ in handle)
                        record("scoring",fold=fold,judge=judge,completed_rows=count,total_rows=819)
                if code: raise RuntimeError(f"fold{fold} {judge} scoring failed; inspect saved log")
        record("evaluating_fold",fold=fold)
        with (OUT/f"fold{fold}/evaluation.log").open("a") as log:
            subprocess.run([PYTHON,"model111/evaluate.py","--fold",str(fold)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((OUT/f"fold{fold}/report.json").read_text())
        record("fold_complete",fold=fold,raw_gain=report["clean_primary_gain"])
    record("combining_full_oof")
    with (OUT/"full_evaluation.log").open("a") as log:
        subprocess.run([PYTHON,"model111/evaluate_full.py"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((OUT/"full_oof_report.json").read_text())
    record("confirmation_complete",passes_raw_confirmation=report["passes_raw_confirmation"],
           full_oof_score=report["results"]["all_five_folds"]["scores"]["clean_real_mean_nll"],submission_ready=False,cloud_used=False)


if __name__=="__main__":
    try:
        with (OUT/"confirmation.lock").open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main()
    except Exception as error:
        record("needs_attention",error=str(error));raise
