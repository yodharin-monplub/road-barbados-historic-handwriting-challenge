"""Explicit generated-data/code allowlist for the already-authorized Vast instance."""
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import sha256, save_json, absolute


def main():
    out = ROOT / "outputs/model110"
    files = list((ROOT / "model110").glob("*.py"))
    files += [ROOT / "model110/config.json", ROOT / "model110/readme.txt", ROOT / "model110/run_gpu.sh",
              ROOT / "model92/config.yaml", ROOT / "data/processed/train_folds.csv", ROOT / "config/known_bad_train_ids.txt",
              ROOT / "checkpoints/model108/asset_manifest.json", out / "config_fold0.yaml",
              out / "fold0/training/manifest.json", out / "fold0/training/labels.csv", out / "fold0/training/passages.txt"]
    files += [ROOT / "src/road_htr" / name for name in ("stackmix.py", "paragraph_synthetic.py", "vlm_data.py", "vlm_cli.py")]
    words_dir = out / "fold0/words"
    files += [words_dir / name for name in ("generation_manifest.json", "requests.json", "labels.csv", "kraken.csv", "lighton.csv",
                                           "kraken_audit_receipt.json", "lighton_audit_receipt.json")]
    requests = json.loads((words_dir / "requests.json").read_text())
    files.append(absolute(requests["bank_path"]))
    for row in requests["rows"]:
        files += [absolute(ref["image"]) for ref in row["style_references"]]
    for manifest in (words_dir / "generation_manifest.json", out / "fold0/training/manifest.json"):
        files += [absolute(row["image"]) for row in json.loads(manifest.read_text())["rows"]]
    files = sorted(set(files))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    ledger = out / "transfer_manifest.json"
    save_json(ledger, {"files": hashes})
    archive = out / "transfer.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in files+[ledger]:
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    print(json.dumps({"archive": str(archive), "sha256": sha256(archive), "files": len(files), "bytes": archive.stat().st_size}))


if __name__ == "__main__":
    main()
