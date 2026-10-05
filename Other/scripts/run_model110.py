"""Queue the fold-zero OCR ablation after Model107; verify downloads and stop Vast."""
import fcntl
import json
from pathlib import Path
import shlex
import time
from monitor_model107 import ROOT, REMOTE, CLI, INSTANCE, run, digest, api_state, is_stopped

OUT = ROOT / "outputs/model110"


def record(stage, **fields):
    value = {"stage": stage, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
    temp = OUT / "run_status.tmp"
    temp.write_text(json.dumps(value, indent=2)+"\n")
    temp.replace(OUT / "run_status.json")
    print(json.dumps(value), flush=True)


def snapshot(ssh):
    code = """
import json,pathlib,subprocess
jobs=[]
other=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  args=(p/'cmdline').read_bytes().decode().split('\\0')
  if any(a in ('model110/train.py','model110/infer.py','model110/run_gpu.sh') for a in args): jobs.append(p.name)
  if any(a in ('model107/train.py','model107/infer.py','model107/run_confirmation.sh','model107/run_deployment.sh') for a in args): other.append(p.name)
 except (OSError,UnicodeError): pass
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
root=pathlib.Path('/workspace/road-htr/outputs/model110')
print(json.dumps({'jobs':jobs,'other_jobs':other,'gpu_jobs':gpu,'complete':(root/'fold0_oof.manifest.json').exists()}))
"""
    return json.loads(run(ssh+["python3 -c "+shlex.quote(code)]))


def stop_idle(ssh):
    state = snapshot(ssh)
    if state["jobs"] or state["other_jobs"] or state["gpu_jobs"]:
        raise RuntimeError("Cannot stop: active compute/job remains")
    run([CLI, "stop", "instance", INSTANCE, "--raw"])
    for _ in range(30):
        if is_stopped(api_state()): return
        time.sleep(10)
    raise RuntimeError("Instance stop not confirmed")


def verify_downloads():
    manifest = json.loads((OUT / "fold0_oof.manifest.json").read_text())
    for name, key in (("fold0_oof.csv", "output_sha256"), ("fold0_oof.jsonl", "detail_sha256")):
        if digest(OUT / name) != manifest[key]: raise RuntimeError("Prediction download checksum mismatch")
    if manifest["rows"] != 819 or manifest["fold"] != 0:
        raise RuntimeError("Incomplete/wrong validation partition")
    if manifest["input_sha256"] != digest(ROOT / "data/processed/train_folds.csv") or manifest["config_sha256"] != digest(OUT / "config_fold0.yaml"):
        raise RuntimeError("Prediction provenance mismatch")
    adapter = OUT / "train/fold0/best_adapter"
    for name, sha in manifest["adapter_files_sha256"].items():
        if digest(adapter / name) != sha: raise RuntimeError("Adapter download checksum mismatch")
    training = json.loads((adapter.parent / "run_manifest.json").read_text())
    if (training["real_train_rows"], training["synthetic_rows"], training["validation_rows"]) != (3276, 819, 819):
        raise RuntimeError("Unexpected training mix")
    if training["synthetic_manifest_sha256"] != digest(OUT / "fold0/training/manifest.json"):
        raise RuntimeError("Synthetic training provenance mismatch")


def main():
    while True:
        prep_path = OUT / "preparation_status.json"
        prep = json.loads(prep_path.read_text()) if prep_path.exists() else {}
        deployment_path = ROOT / "outputs/model107/deployment_status.json"
        deployment = json.loads(deployment_path.read_text()) if deployment_path.exists() else {}
        if prep.get("stage") == "needs_attention":
            raise RuntimeError("Local preparation needs attention; cloud not started")
        if deployment.get("stage") in ("needs_attention", "cleanup_needs_attention"):
            raise RuntimeError("Model107 needs attention; refusing to take its instance")
        ready = prep.get("stage") == "ready_for_ocr_ablation"
        released = deployment.get("stage") in ("not_promoted", "submission_ready")
        record("waiting_for_preparation_and_model107", preparation=prep.get("stage"), model107=deployment.get("stage"))
        if ready and released: break
        time.sleep(600)
    package = json.loads(run([str(ROOT / ".venv-kraken/bin/python"), "model110/package.py"], timeout=1200))
    if not is_stopped(api_state()):
        raise RuntimeError("Expected stopped instance; refusing to interfere with another job")
    run([CLI, "start", "instance", INSTANCE, "--raw"])
    record("starting_instance", instance=INSTANCE)
    ssh = None
    try:
        for _ in range(90):
            state = api_state()
            ports = (state.get("ports") or {}).get("22/tcp", [])
            if state.get("actual_status") == "running" and ports:
                host = state["public_ipaddr"]
                if host != "145.236.167.14": raise RuntimeError("Unexpected instance address")
                port = str(ports[0]["HostPort"])
                ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-p", port, "root@"+host]
                try:
                    run(ssh+["test -x /workspace/road-htr/.venv-lighton/bin/python"])
                    break
                except RuntimeError: pass
            time.sleep(10)
        else: raise RuntimeError("Instance SSH startup timed out")
        state = snapshot(ssh)
        if state["jobs"] or state["other_jobs"] or state["gpu_jobs"]:
            raise RuntimeError("GPU instance unexpectedly busy")
        scp = ["scp", "-q", "-o", "BatchMode=yes", "-P", port]
        run(ssh+["mkdir -p /workspace/road-htr/outputs/model110"])
        record("uploading", files=package["files"], bytes=package["bytes"])
        run(scp+[package["archive"], f"root@{host}:{REMOTE}/outputs/model110/transfer.tar.gz"], timeout=1800)
        code = f"""
import pathlib,tarfile,json,hashlib
root=pathlib.Path({REMOTE!r})
p=root/'outputs/model110/transfer.tar.gz'
with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()=={package['sha256']!r}
with tarfile.open(p) as tar: tar.extractall(root,filter='data')
ledger=json.loads((root/'outputs/model110/transfer_manifest.json').read_text())
for name,sha in ledger['files'].items():
 with (root/name).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==sha,name
print('Upload verified')
"""
        run(ssh+["python3 -c "+shlex.quote(code)], timeout=600)
        launch = f"""
import subprocess,pathlib
root=pathlib.Path({REMOTE!r})
with (root/'outputs/model110/launcher.log').open('xb') as log:
 p=subprocess.Popen(['bash','model110/run_gpu.sh'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
"""
        run(ssh+["python3 -c "+shlex.quote(launch)])
        time.sleep(10)
        failures = 0
        while True:
            try:
                state = snapshot(ssh)
                failures = 0
            except Exception as error:
                failures += 1
                record("monitor_connection_retry", attempts=failures, error=str(error))
                if failures >= 6: raise
                time.sleep(600)
                continue
            record("training_or_inference", **state)
            if not state["jobs"]: break
            time.sleep(600)
        if not state["complete"]:
            # Keep remote disk/checkpoints recoverable, stop only when idle.
            for name in ("launcher.log", "fold0_train.log", "fold0_infer.log"):
                try: run(scp+[f"root@{host}:{REMOTE}/outputs/model110/{name}", str(OUT)])
                except RuntimeError: pass
            raise RuntimeError("OCR experiment ended before predictions were complete")
        record("downloading_results")
        names = ["fold0_oof.csv", "fold0_oof.jsonl", "fold0_oof.manifest.json", "fold0_train.log", "fold0_infer.log", "launcher.log"]
        run(scp+[f"root@{host}:{REMOTE}/outputs/model110/{name}" for name in names]+[str(OUT)], timeout=1800)
        target = OUT / "train/fold0"
        target.mkdir(parents=True, exist_ok=True)
        run(scp+["-r", f"root@{host}:{REMOTE}/outputs/model110/train/fold0/best_adapter", str(target)], timeout=1800)
        run(scp+[f"root@{host}:{REMOTE}/outputs/model110/train/fold0/run_manifest.json", str(target)])
        verify_downloads()
        stop_idle(ssh)
        record("artifacts_verified_instance_stopped")
    except Exception:
        if ssh:
            try: stop_idle(ssh)
            except Exception as error: record("cleanup_needs_attention", error=str(error))
        else:
            run([CLI, "stop", "instance", INSTANCE, "--raw"])
        raise
    run([str(ROOT / ".venv-kraken/bin/python"), "model107/evaluate.py", "--fold", "0", "--predictions",
         "outputs/model110/fold0_oof.csv", "--output", "outputs/model110/fold0_report.json"])
    record("fold0_ablation_complete", report=json.loads((OUT / "fold0_report.json").read_text()),
           note="Development result only. No submission without independent/full-OOF confirmation.")


if __name__ == "__main__":
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "cloud.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main()
    except Exception as error:
        record("needs_attention", error=str(error))
        raise
