"""Recover after Model114 training without changing the frozen ML protocol."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model114"))
sys.path.insert(0, str(ROOT / "scripts"))
import pilot
import run_seed_pilots as supervisor


def wait_for_free_gpu():
    """Allow CUDA teardown time; wait for other jobs without interrupting them."""
    attempts = 0
    while True:
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        memory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
        ).strip()
        free = int(memory.splitlines()[0])
        if not apps and free >= 5000:
            return
        # Brief grace period for a just-exited CUDA process, followed by the
        # user's requested ten-minute monitoring interval for an occupied GPU.
        delay = 10 if attempts < 6 else 600
        supervisor.record("waiting_for_gpu", free_mib=free,
                          compute_process_present=bool(apps), next_check_seconds=delay)
        time.sleep(delay)
        attempts += 1


def preflight():
    pilot.check()
    base = ROOT / "outputs/model114/fold0"
    state = json.loads((base / "checkpoint-400/trainer_state.json").read_text())
    if state["global_step"] != 400:
        raise RuntimeError("Model114 training is not complete")
    adapter = base / "best_adapter/adapter_model.safetensors"
    checkpoint = base / "checkpoint-400/adapter_model.safetensors"
    if pilot.sha256(adapter) != pilot.sha256(checkpoint):
        raise RuntimeError("Model114 saved adapter differs from checkpoint400")
    for path in ("outputs/model114/infer.log", "outputs/model114/evaluate.log",
                 "outputs/model114/fold0_oof_step400.csv", "outputs/model114/report.json",
                 "outputs/model115/fold0", "outputs/model115/train.log",
                 "outputs/model115/infer.log", "outputs/model115/evaluate.log",
                 "outputs/model115/fold0_oof_step400.csv", "outputs/model115/report.json",
                 "outputs/model114/recovery_receipt.json"):
        if (ROOT / path).exists():
            raise RuntimeError(f"Recovery would overwrite an existing artifact: {path}")
    return adapter


def main():
    with (supervisor.OUT / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        adapter = preflight()
        pilot.save_json(supervisor.OUT / "recovery_receipt.json", {
            "previous_status": json.loads((supervisor.OUT / "seed_pair_status.json").read_text()),
            "recovery_code_sha256": pilot.sha256(Path(__file__)),
            "protocol_sha256": pilot.sha256(pilot.PROTOCOL),
            "model114_adapter_sha256": pilot.sha256(adapter),
            "change": "Resume at Model114 inference; replace fail-fast GPU check with non-destructive waiting. Training/evaluation code and configurations unchanged."
        })
        supervisor.free_gpu = wait_for_free_gpu
        try:
            supervisor.run("infer", 114)
            supervisor.run("evaluate", 114)
            for action in ("train", "infer", "evaluate"):
                supervisor.run(action, 115)
            reports = {str(n): json.loads((ROOT / f"outputs/model{n}/report.json").read_text())
                       for n in (114, 115)}
            supervisor.record("pilots_complete", results={
                n: {k: r[k] for k in ("ensemble_gain", "advance_to_confirmation_screen")}
                for n, r in reports.items()})
        except Exception as error:
            supervisor.record("needs_attention", error=str(error))
            raise


if __name__ == "__main__":
    main()
