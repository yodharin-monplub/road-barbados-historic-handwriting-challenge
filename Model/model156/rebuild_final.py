#!/usr/bin/env python3
"""Rebuild the final submission from the members' saved predictions and verify it.

    python Model/model156/rebuild_final.py

Needs only Python 3.10+ (no GPU, no third-party packages): it runs the Model124
ensemble on the twelve members' saved test predictions in Other/outputs, writes
Other/submissions/model156_final_rebuilt.csv, and checks it byte-for-byte against the
submission that scored public 0.919377358 (Model152). It also recomputes the
complete out-of-fold CV of the same ensemble.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent

# Order matters: the first member is the vote anchor (Model155 confirmed this choice).
MEMBERS = [
    "model134_pseudo3", "model140_pseudo4", "model122_prod_lora", "model130_kraken",
    "model136_qwen25vl7b", "model143_qwen3vl4b_vision", "model144_qwen25vl7b_vision",
    "model145_qwen3vl4b_vision_pseudo4", "model146_qwen3vl8b_vision",
    "model147_qwen3vl8b_pseudo5", "model149_kraken_pseudo5", "model150_kraken_pseudo4",
]
EXPECTED_SHA256 = "7cdaada34149b920f58755f707fff2bb736740cbd780022c5d504c94389a7389"
EXPECTED_CV = 0.9224986529610719
ENSEMBLE = ROOT / "Model" / "model124" / "ensemble.py"
BUILD = ROOT / "Model" / "model124" / "build_submission.py"
OUTPUT = ROOT / "Other" / "submissions" / "model156_final_rebuilt.csv"


def run(args: list[str]) -> str:
    result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"FAILED: {' '.join(args)}\n{result.stderr}")
    return result.stdout


def main() -> None:
    missing = [m for m in MEMBERS if not (ROOT / "Other" / "outputs" / m).is_dir()]
    if missing:
        sys.exit(f"missing member outputs: {missing}")

    oof = run([str(ENSEMBLE), "oof", "--members", *MEMBERS, "--min-votes", "2"])
    cv = json.loads(oof[oof.index("{"):])["ensemble"]
    print(f"complete out-of-fold CV: {cv:.6f}  (expected {EXPECTED_CV:.6f})")

    run([str(ENSEMBLE), "test", "--members", *MEMBERS, "--min-votes", "2", "--fold-min-votes", "3"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    run([str(BUILD), "--anchor", str(ROOT / "Other/outputs/model124/test_ensemble.csv"),
         "--output", str(OUTPUT)])
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"rebuilt: {OUTPUT.relative_to(ROOT)}")
    print(f"sha256:  {digest}")
    ok = digest == EXPECTED_SHA256 and abs(cv - EXPECTED_CV) < 1e-9
    print("REPRODUCED" if ok else f"MISMATCH (expected sha256 {EXPECTED_SHA256})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
