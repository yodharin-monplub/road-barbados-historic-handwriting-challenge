"""Attach to the already-running confirmation job without relaunching training."""
import fcntl
import json
from pathlib import Path
import time
from continue_model107 import snapshot, stop_idle, record
from monitor_model107 import ROOT, OUT, REMOTE, run, api_state, verify_downloads


def collect(fold, scp, host):
    target = OUT / f"train/fold{fold}"
    if (OUT / f"fold{fold}_oof.manifest.json").exists() and (target / "run_manifest.json").exists():
        try:
            verify_downloads(fold)
            return
        except (OSError, KeyError, ValueError, RuntimeError):
            pass
    names = [f"fold{fold}_oof.csv", f"fold{fold}_oof.jsonl", f"fold{fold}_oof.manifest.json",
             f"fold{fold}_train.log", f"fold{fold}_infer.log"]
    run(scp + [f"root@{host}:{REMOTE}/outputs/model107/{name}" for name in names] + [str(OUT)], timeout=1800)
    target.mkdir(parents=True, exist_ok=True)
    run(scp + ["-r", f"root@{host}:{REMOTE}/outputs/model107/train/fold{fold}/best_adapter", str(target)], timeout=1800)
    run(scp + [f"root@{host}:{REMOTE}/outputs/model107/train/fold{fold}/run_manifest.json", str(target)])
    verify_downloads(fold)
    run([str(ROOT / ".venv-kraken/bin/python"), "model107/evaluate.py", "--fold", str(fold),
         "--predictions", f"outputs/model107/fold{fold}_oof.csv", "--output", f"outputs/model107/fold{fold}_report.json"])


def main():
    with (OUT / "confirmation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = api_state()
        host = state.get("public_ipaddr")
        if host != "145.236.167.14" or state.get("actual_status") != "running":
            raise RuntimeError("Expected the authorized running instance")
        port = str(state["ports"]["22/tcp"][0]["HostPort"])
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3", "-p", port, "root@" + host]
        scp = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-P", port]
        collected = set()
        failures = 0
        while True:
            try:
                state = snapshot(ssh)
                record("monitoring_resumed", **state, verified_folds=sorted(collected))
                for fold in state["completed"]:
                    if fold not in collected:
                        collect(fold, scp, host)
                        collected.add(fold)
                        record("fold_verified", fold=fold, verified_folds=sorted(collected))
                if not state["jobs"]:
                    run(scp + [f"root@{host}:{REMOTE}/outputs/model107/confirmation_launcher.log", str(OUT)])
                    stop_idle(ssh)
                    record("instance_stopped", verified_folds=sorted(collected))
                    break
                failures = 0
            except Exception as error:
                failures += 1
                # Keep retrying transient monitoring/download errors, preserving
                # running training and persistent instance storage.
                record("monitor_retry", failures=failures, error=str(error))
            time.sleep(600)
        if collected != {1, 2, 3, 4}:
            raise RuntimeError("Training ended early; instance stopped with partial artifacts retained")
        record("evaluating_full_oof", instance_status="stopped")
        run([str(ROOT / ".venv-kraken/bin/python"), "model107/evaluate_full.py"], timeout=7200)
        record("confirmation_complete", instance_status="stopped", decision=json.loads((OUT / "promotion_decision.json").read_text()))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        record("needs_attention", error=str(error))
        raise
