"""Wait for local preparation, run four Vast folds, collect/verify/stop and score."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time
from monitor_model107 import ROOT, OUT, REMOTE, CLI, INSTANCE, run, digest, api_state, is_stopped, verify_downloads


def record(stage, **kwargs):
    value = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": stage, **kwargs}
    path = OUT / "confirmation_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(value), flush=True)


def snapshot(ssh):
    code = """
import json,pathlib,subprocess
root=pathlib.Path('/workspace/road-htr/outputs/model107')
jobs=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  args=(p/'cmdline').read_bytes().decode().split('\\0')
  if any(a in ('model107/train.py','model107/infer.py','model107/run_confirmation.sh') for a in args): jobs.append(p.name)
 except (OSError,UnicodeError): pass
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps({'jobs':jobs,'gpu_jobs':gpu,'completed':[f for f in range(1,5) if (root/f'fold{f}_oof.manifest.json').exists()]}))
"""
    return json.loads(run(ssh + ["python3 -c " + shlex.quote(code)]))


def stop_idle(ssh):
    state = snapshot(ssh)
    if state["jobs"] or state["gpu_jobs"]:
        raise RuntimeError("Refusing to stop an active job")
    run([CLI, "stop", "instance", INSTANCE, "--raw"])
    for _ in range(30):
        if is_stopped(api_state()):
            return
        time.sleep(10)
    raise RuntimeError("Stop not yet confirmed")


def main():
    with (OUT / "confirmation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            path = OUT / "preparation_status.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            record("waiting_local_preparation", preparation=state)
            if state.get("stage") == "failed":
                raise RuntimeError("Local preparation failed; cloud not started")
            if state.get("stage") == "complete":
                break
            active = subprocess.run(["systemctl", "--user", "is-active", "road-model107-prepare-confirmation.service"], capture_output=True, text=True)
            if active.stdout.strip() != "active":
                raise RuntimeError("Preparation service stopped without success; cloud not started")
            time.sleep(600)
        package = json.loads(run([str(ROOT / ".venv-kraken/bin/python"), "model107/package.py",
                                  "--folds", "1", "2", "3", "4", "--name", "confirmation_transfer"], timeout=1200))
        state = api_state()
        if not is_stopped(state):
            raise RuntimeError("Expected stopped instance; refusing to interfere with another run")
        record("starting_instance", instance=INSTANCE)
        run([CLI, "start", "instance", INSTANCE, "--raw"])
        ssh = None
        try:
            for _ in range(60):
                state = api_state()
                ports = (state.get("ports") or {}).get("22/tcp", [])
                if state.get("actual_status") == "running" and ports:
                    host = state["public_ipaddr"]
                    if host != "145.236.167.14":
                        raise RuntimeError("Instance address changed; new identity verification required")
                    port = str(ports[0]["HostPort"])
                    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-p", port, "root@" + host]
                    try:
                        run(ssh + ["test -x /workspace/road-htr/.venv-lighton/bin/python"])
                        break
                    except RuntimeError:
                        pass
                time.sleep(10)
            else:
                raise RuntimeError("Instance did not become SSH-ready")
            if snapshot(ssh)["gpu_jobs"]:
                raise RuntimeError("Instance GPU unexpectedly busy")
            scp = ["scp", "-q", "-o", "BatchMode=yes", "-P", port]
            record("uploading", bytes=package["bytes"], files=package["files"])
            run(scp + [package["archive"], f"root@{host}:{REMOTE}/outputs/model107/confirmation_transfer.tar.gz"], timeout=1800)
            code = f"""
import hashlib,json,pathlib,tarfile
root=pathlib.Path({REMOTE!r})
p=root/'outputs/model107/confirmation_transfer.tar.gz'
with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()=={package['sha256']!r}
with tarfile.open(p) as tar: tar.extractall(root,filter='data')
ledger=json.loads((root/'outputs/model107/confirmation_transfer_manifest.json').read_text())
for name,expected in ledger['files'].items():
 with (root/name).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name
print('Verified',len(ledger['files']),'files')
"""
            run(ssh + ["python3 -c " + shlex.quote(code)], timeout=600)
            run(ssh + [f"cd {REMOTE} && (nohup bash model107/run_confirmation.sh > outputs/model107/confirmation_launcher.log 2>&1 < /dev/null &)"], timeout=60)
            time.sleep(10)
            while True:
                state = snapshot(ssh)
                record("training_confirmation", **state)
                if not state["jobs"]:
                    break
                time.sleep(600)
            completed = state["completed"]
            for fold in completed:
                names = [f"fold{fold}_oof.csv", f"fold{fold}_oof.jsonl", f"fold{fold}_oof.manifest.json",
                         f"fold{fold}_train.log", f"fold{fold}_infer.log"]
                run(scp + [f"root@{host}:{REMOTE}/outputs/model107/{name}" for name in names] + [str(OUT)], timeout=1800)
                target = OUT / f"train/fold{fold}"
                target.mkdir(parents=True, exist_ok=True)
                run(scp + ["-r", f"root@{host}:{REMOTE}/outputs/model107/train/fold{fold}/best_adapter", str(target)], timeout=1800)
                run(scp + [f"root@{host}:{REMOTE}/outputs/model107/train/fold{fold}/run_manifest.json", str(target)])
                verify_downloads(fold)
            # Copy available error logs too; partial checkpoints remain on persistent disk.
            run(scp + [f"root@{host}:{REMOTE}/outputs/model107/confirmation_launcher.log", str(OUT)])
            stop_idle(ssh)
            record("artifacts_verified_instance_stopped", completed=completed)
            if completed != [1, 2, 3, 4]:
                raise RuntimeError("Confirmation ended early; preserved disk and stopped billing for compute")
        except Exception:
            # Fail-safe stop only when no training/GPU work remains. Never destroy.
            if ssh is not None:
                try:
                    stop_idle(ssh)
                except Exception as error:
                    record("cleanup_needs_attention", error=str(error))
            elif not is_stopped(api_state()):
                run([CLI, "stop", "instance", INSTANCE, "--raw"])
            raise
        for fold in range(1, 5):
            run([str(ROOT / ".venv-kraken/bin/python"), "model107/evaluate.py", "--fold", str(fold),
                 "--predictions", f"outputs/model107/fold{fold}_oof.csv", "--output", f"outputs/model107/fold{fold}_report.json"])
        run([str(ROOT / ".venv-kraken/bin/python"), "model107/evaluate_full.py"], timeout=7200)
        record("confirmation_complete", decision=json.loads((OUT / "promotion_decision.json").read_text()))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        record("needs_attention", error=str(error))
        raise
