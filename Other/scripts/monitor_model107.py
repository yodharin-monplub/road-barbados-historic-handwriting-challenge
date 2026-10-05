"""Check Model107 every ten minutes, preserve its artifacts, score and stop Vast."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/model107"
REMOTE="/workspace/road-htr"
CLI="/home/msi/.local/share/vastai-cli/bin/vastai"
INSTANCE="50076619"


def run(args,timeout=180):
    p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"{args[0]} failed: {p.stderr[-1200:]}")
    return p.stdout


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def record(stage,**kwargs):
    value={"time":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"stage":stage,**kwargs}
    temp=OUT/"supervisor_status.tmp"
    temp.write_text(json.dumps(value,indent=2)+"\n")
    temp.replace(OUT/"supervisor_status.json")
    print(json.dumps(value),flush=True)


def api_state():
    data=json.loads(run([CLI,"show","instance",INSTANCE,"--raw"]))
    if str(data.get("id"))!=INSTANCE:
        raise RuntimeError("Vast instance identity mismatch")
    return data


def is_stopped(data):
    return data.get("actual_status")=="stopped" or (
        data.get("actual_status")=="exited" and data.get("cur_state")=="stopped"
        and data.get("intended_status")=="stopped")


def snapshot(ssh):
    code='''
import pathlib, json, subprocess
root=pathlib.Path('/workspace/road-htr/outputs/model107')
jobs=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        args=(p/'cmdline').read_bytes().decode().split('\\0')
        if any(a in ('model107/train.py','model107/infer.py') for a in args):
            jobs.append({'pid':p.name,'command':args[:3]})
    except (OSError,UnicodeError): pass
logs={}
for name in ('gpu_train.log','gpu_infer.log'):
    p=root/name
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-1000))
            logs[name]=f.read().decode(errors='replace')
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps({'jobs':jobs,'gpu_jobs':gpu,'logs':logs,'complete':(root/'fold0_oof.manifest.json').exists()}))
'''
    return json.loads(run(ssh+["python3 -c "+shlex.quote(code)]))


def verify_downloads(fold=0):
    manifest=json.loads((OUT/f"fold{fold}_oof.manifest.json").read_text())
    for name,key in ((f"fold{fold}_oof.csv","output_sha256"),(f"fold{fold}_oof.jsonl","detail_sha256")):
        if digest(OUT/name)!=manifest[key]:
            raise RuntimeError(f"Hash mismatch: {name}")
    if manifest["rows"]!=819 or manifest["fold"]!=fold:
        raise RuntimeError("Wrong validation partition")
    if manifest["input_sha256"]!=digest(ROOT/"data/processed/train_folds.csv"):
        raise RuntimeError("Wrong canonical labels")
    if manifest["config_sha256"]!=digest(OUT/f"config_fold{fold}.yaml"):
        raise RuntimeError("Model config mismatch")
    for name,sha in manifest["adapter_files_sha256"].items():
        if digest(OUT/f"train/fold{fold}/best_adapter"/name)!=sha:
            raise RuntimeError(f"Adapter hash mismatch: {name}")
    training=json.loads((OUT/f"train/fold{fold}/run_manifest.json").read_text())
    if (training["real_train_rows"],training["synthetic_rows"],training["validation_rows"])!=(3276,819,819):
        raise RuntimeError("Unexpected training mix")
    if training["synthetic_manifest_sha256"]!=digest(OUT/f"fold{fold}/training/manifest.json"):
        raise RuntimeError("Training synthetic provenance mismatch")


def stop(ssh):
    if is_stopped(api_state()): return
    state=snapshot(ssh)
    if state["jobs"] or state["gpu_jobs"]:
        raise RuntimeError("Refusing to stop while model/GPU jobs remain")
    run([CLI,"stop","instance",INSTANCE,"--raw"])
    for attempt in range(24):
        if is_stopped(api_state()): return
        time.sleep(10)
    raise RuntimeError("Stop requested; completion not yet confirmed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",default="145.236.167.14")
    parser.add_argument("--port",required=True)
    args=parser.parse_args()
    host="root@"+args.host
    ssh=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=20","-p",args.port,host]
    scp=["scp","-q","-o","BatchMode=yes","-P",args.port]
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"supervisor.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if api_state().get("public_ipaddr")!=args.host:
            raise RuntimeError("SSH host does not match the authorized instance")
        failures=0
        while True:
            try:
                state=snapshot(ssh)
                record("monitoring",**state)
                if state["complete"] and not state["jobs"]: break
                if not state["jobs"]:
                    time.sleep(10)
                    state=snapshot(ssh)
                    if not state["jobs"]:
                        if state["complete"]: break
                        # No successful run to promote. Preserve logs then stop the
                        # idle instance; its persistent disk retains partial work.
                        run(scp+[f"{host}:{REMOTE}/outputs/model107/gpu_train.log",str(OUT)])
                        stop(ssh)
                        record("training_failed_instance_stopped",logs=state["logs"])
                        return
                failures=0
            except Exception as error:
                failures+=1
                record("monitor_check_failed",attempt=failures,error=str(error))
                if failures>=3: raise
            time.sleep(600)
        record("collecting")
        names=["fold0_oof.csv","fold0_oof.jsonl","fold0_oof.manifest.json","gpu_train.log","gpu_infer.log"]
        run(scp+[f"{host}:{REMOTE}/outputs/model107/{n}" for n in names]+[str(OUT)],timeout=1200)
        target=OUT/"train/fold0"
        target.mkdir(parents=True,exist_ok=True)
        run(scp+["-r",f"{host}:{REMOTE}/outputs/model107/train/fold0/best_adapter",str(target)],timeout=1800)
        run(scp+[f"{host}:{REMOTE}/outputs/model107/train/fold0/run_manifest.json",str(target)])
        verify_downloads()
        record("artifacts_verified")
        stop(ssh)
        record("instance_stopped")
        run([str(ROOT/".venv-kraken/bin/python"),"model107/evaluate.py",
             "--predictions","outputs/model107/fold0_oof.csv","--output","outputs/model107/fold0_report.json"])
        report=json.loads((OUT/"fold0_report.json").read_text())
        record("complete",instance_status="stopped",report=report)


if __name__=="__main__":
    try: main()
    except Exception as error:
        record("needs_attention",error=str(error))
        raise
