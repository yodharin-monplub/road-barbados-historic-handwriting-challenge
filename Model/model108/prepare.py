"""Deterministic train-only word requests with five same-line style references."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import absolute, sha256, save_json, eligible_rows
from road_htr.vlm_data import read_id_file

SUPPORTED = set('_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')


def validate_requests(manifest):
    truth = ROOT / "data/processed/train_folds.csv"
    fold = manifest["fold"]
    table, eligible = eligible_rows(truth, fold, read_id_file(ROOT / "config/known_bad_train_ids.txt"))
    bank_path = absolute(manifest["bank_path"])
    if manifest["truth_sha256"] != sha256(truth) or manifest["bank_sha256"] != sha256(bank_path):
        raise ValueError("Request truth/bank hash mismatch")
    bank = json.loads(bank_path.read_text())
    if bank["fold"] != fold or bank["lineage"]["training_excluded_folds"] != [fold]:
        raise ValueError("Style aligner saw the heldout fold")
    words = {json.dumps(w, sort_keys=True) for w in bank["words"]}
    ids = set()
    for row in manifest["rows"]:
        if row["id"] in ids:
            raise ValueError("Duplicate request ID")
        ids.add(row["id"])
        if not row["target"] or len(row["target"]) > 18 or not set(row["target"]) <= SUPPORTED:
            raise ValueError("Unsupported requested word; no normalization allowed")
        source = row["target_source_id"]
        a, b = row["target_span"]
        if source not in eligible or eligible[source][table.target_column][a:b] != row["target"]:
            raise ValueError("Requested word is not an exact eligible training substring")
        refs = row["style_references"]
        if len(refs) != 5 or len({r["source_id"] for r in refs}) != 1 or len({r["image"] for r in refs}) != 5:
            raise ValueError("Expected five distinct crops from one eligible source line")
        for ref in refs:
            if ref["source_id"] not in eligible or json.dumps(ref, sort_keys=True) not in words:
                raise ValueError("Unknown or heldout style donor")
            if sha256(absolute(ref["image"])) != ref["sha256"]:
                raise ValueError("Style crop hash mismatch")
    return manifest["rows"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    bank_path = ROOT / f"outputs/model107/fold{args.fold}/bank/bank.json"
    bank = json.loads(bank_path.read_text())
    config = json.loads((ROOT / "model108/config.json").read_text())
    groups = defaultdict(list)
    vocabulary = {}
    for word in bank["words"]:
        groups[word["source_id"]].append(word)
        if 1 <= len(word["text"]) <= 18 and set(word["text"]) <= SUPPORTED:
            vocabulary.setdefault(word["text"], word)
    groups = {k: v for k, v in groups.items() if len(v) >= 5}
    if not groups or len(vocabulary) < args.count:
        raise ValueError("Insufficient eligible requests/style references")
    rng = random.Random(config["seed"])
    rows = []
    for position, text in enumerate(rng.sample(sorted(vocabulary), args.count)):
        target = vocabulary[text]
        source = rng.choice(sorted(groups))
        rows.append({"id": f"diffusionpen_f{args.fold}_{position:06d}", "target": text,
                     "target_source_id": target["source_id"], "target_span": target["text_span"],
                     "style_references": rng.sample(groups[source], 5), "seed": config["seed"] + position})
    manifest = {"method": "diffusionpen", "fold": args.fold, "truth_sha256": sha256(ROOT / "data/processed/train_folds.csv"),
                "bank_path": str(bank_path.relative_to(ROOT)), "bank_sha256": sha256(bank_path), "rows": rows}
    validate_requests(manifest)
    output = ROOT / f"outputs/model108/fold{args.fold}/pilot/requests.json"
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise ValueError("Refusing to replace an existing different pilot")
    else:
        save_json(output, manifest)
    print(f"Validated {len(rows)} fold-isolated word requests: {output}")


if __name__ == "__main__":
    main()
