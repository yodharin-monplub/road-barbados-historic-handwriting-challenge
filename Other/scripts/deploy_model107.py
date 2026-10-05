"""Gated final refit on the authorized Vast instance; verified download, stop, submission."""
import argparse
import fcntl
import json
import shlex
import time
from monitor_model107 import ROOT,OUT,REMOTE,CLI,INSTANCE,run,digest,api_state,is_stopped
from continue_model107 import snapshot,stop_idle
from monitor_model105 import verify_prediction


def record(stage,**fields):
    value={"stage":stage,"time":time.strftime("%Y-%m-%dT%H:%M:%S%z"),**fields}
    temp=OUT/"deployment_status.tmp"
    temp.write_text(json.dumps(value,indent=2)+"\n")
    temp.replace(OUT/"deployment_status.json")
    print(json.dumps(value),flush=True)


def main():
    decision=json.loads((OUT/"promotion_decision.json").read_text())
    if not decision["passes_submission_gate"]:
        record("not_promoted",reason="Full-OOF gate failed; no cloud refit")
        return
    if not (OUT/"foldall/preparation_complete.json").exists():
        raise RuntimeError("Prepare deployment artifacts locally before renting")
    package=json.loads(run([str(ROOT/".venv-kraken/bin/python"),"model107/package.py","--folds","all","--name","deployment_transfer"],timeout=1200))
    if not is_stopped(api_state()):
        raise RuntimeError("Expected stopped instance; refusing to interfere with another run")
    run([CLI,"start","instance",INSTANCE,"--raw"])
    record("starting_instance")
    ssh=None
    try:
        for _ in range(90):
            state=api_state()
            ports=(state.get("ports") or {}).get("22/tcp",[])
            if state.get("actual_status")=="running" and ports:
                host=state["public_ipaddr"]
                if host!="145.236.167.14": raise RuntimeError("Unexpected instance address")
                port=str(ports[0]["HostPort"])
                ssh=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=20","-p",port,"root@"+host]
                try:
                    run(ssh+["test -x /workspace/road-htr/.venv-lighton/bin/python"])
                    break
                except RuntimeError: pass
            time.sleep(10)
        else: raise RuntimeError("SSH startup timed out")
        if snapshot(ssh)["gpu_jobs"]: raise RuntimeError("GPU unexpectedly busy")
        scp=["scp","-q","-o","BatchMode=yes","-P",port]
        run(scp+[package["archive"],f"root@{host}:{REMOTE}/outputs/model107/deployment_transfer.tar.gz"],timeout=1800)
        code=f"""
import pathlib,tarfile,json,hashlib
root=pathlib.Path({REMOTE!r})
p=root/'outputs/model107/deployment_transfer.tar.gz'
with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()=={package['sha256']!r}
with tarfile.open(p) as tar: tar.extractall(root,filter='data')
ledger=json.loads((root/'outputs/model107/deployment_transfer_manifest.json').read_text())
for name,sha in ledger['files'].items():
 with (root/name).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==sha,name
print('Upload verified')
"""
        run(ssh+["python3 -c "+shlex.quote(code)],timeout=600)
        # subprocess.Popen fully detaches all three streams, unlike the previous
        # shell AND-list launch which accidentally retained the SSH pipes.
        launch=f"""
import subprocess,pathlib
root=pathlib.Path({REMOTE!r})
with (root/'outputs/model107/deployment_launcher.log').open('xb') as log:
 p=subprocess.Popen(['bash','model107/run_deployment.sh'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
"""
        run(ssh+["python3 -c "+shlex.quote(launch)])
        time.sleep(10)
        while True:
            state=snapshot(ssh)
            record("running_refit_or_inference",**state)
            if not state["jobs"]:
                break
            time.sleep(600)
        names=["test_raw.csv","test_raw.jsonl","test_raw.manifest.json","deployment_train.log","deployment_infer.log"]
        run(scp+[f"root@{host}:{REMOTE}/outputs/model107/{name}" for name in names]+[str(OUT)],timeout=1800)
        target=OUT/"train/foldall"
        target.mkdir(parents=True,exist_ok=True)
        run(scp+["-r",f"root@{host}:{REMOTE}/outputs/model107/train/foldall/best_adapter",str(target)],timeout=1800)
        run(scp+[f"root@{host}:{REMOTE}/outputs/model107/train/foldall/run_manifest.json",str(target)])
        receipt=verify_prediction(OUT/"test_raw.csv")
        if receipt["config_sha256"]!=digest(OUT/"config_foldall.yaml") or receipt["fold"] is not None:
            raise RuntimeError("Deployment config/partition mismatch")
        for name,sha in receipt["adapter_files_sha256"].items():
            if digest(target/"best_adapter"/name)!=sha: raise RuntimeError("Adapter download hash mismatch")
        training=json.loads((target/"run_manifest.json").read_text())
        if (training["real_train_rows"],training["synthetic_rows"],training["validation_rows"])!=(4095,1023,0):
            raise RuntimeError("Wrong final training mix")
        if training["synthetic_manifest_sha256"]!=digest(OUT/"foldall/training/manifest.json"):
            raise RuntimeError("Final training provenance mismatch")
        stop_idle(ssh)
        record("artifacts_verified_instance_stopped")
    except Exception:
        if ssh:
            try: stop_idle(ssh)
            except Exception as error: record("cleanup_needs_attention",error=str(error))
        else: run([CLI,"stop","instance",INSTANCE,"--raw"])
        raise
    run([str(ROOT/".venv-kraken/bin/python"),"model107/finalize_submission.py"],timeout=3600)
    record("submission_ready",report=json.loads((OUT/"submission_report.json").read_text()))


if __name__=="__main__":
    try:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--wait",action="store_true")
        args=parser.parse_args()
        with (OUT/"deployment.lock").open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.wait:
                while not (OUT/"promotion_decision.json").exists():
                    status=json.loads((OUT/"confirmation_status.json").read_text())
                    if status.get("stage")=="needs_attention":
                        raise RuntimeError("Confirmation needs attention; refusing to start a final refit")
                    record("waiting_for_full_oof",confirmation_stage=status.get("stage"))
                    time.sleep(600)
            main()
    except Exception as error:
        record("needs_attention",error=str(error))
        raise
