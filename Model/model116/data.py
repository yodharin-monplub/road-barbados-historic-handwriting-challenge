"""Fold-isolated random3/4-word composition and strict provenance validation."""
import csv
import json
from pathlib import Path
import random
import statistics
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.schema import ImageIndex, read_table
from road_htr.stackmix import absolute, sha256, save_json
from road_htr.vlm_data import read_id_file

TRUTH = ROOT / "data/processed/train_folds.csv"
BANK = ROOT / "outputs/model107/fold0/bank/bank.json"
OUT = ROOT / "outputs/model116/synthetic"
SEED = 20260916


def eligible_rows():
    table = read_table(TRUTH, require_target=True)
    excluded = read_id_file(ROOT / "config/known_bad_train_ids.txt")
    eligible = {r[table.id_column]: r for r in table.rows
                if int(r["fold"]) != 0 and r[table.id_column] not in excluded}
    if len(eligible) != 3276:
        raise ValueError("Unexpected eligible training set")
    return table, eligible


def verify_bank():
    table, eligible = eligible_rows()
    bank = json.loads(BANK.read_text())
    if bank["fold"] != 0 or bank["signature"]["truth_sha256"] != sha256(TRUTH):
        raise ValueError("Crop bank fold/truth mismatch")
    lineage = bank["lineage"]
    if lineage["training_excluded_folds"] != [0]:
        raise ValueError("Aligner trained on validation fold")
    for field in ("training_manifest", "conversion_receipt", "source_checkpoint"):
        if sha256(absolute(lineage[field])) != lineage[field + "_sha256"]:
            raise ValueError(f"Changed aligner provenance: {field}")
    training = json.loads(absolute(lineage["training_manifest"]).read_text())
    if training["preparation"]["fold"] != 0 or training["preparation"]["train_rows"] != 3276:
        raise ValueError("Aligner manifest does not confirm fold0 exclusion")
    if sha256(absolute(bank["signature"]["config"]["checkpoint"])) != bank["signature"]["checkpoint_sha256"]:
        raise ValueError("Changed converted aligner checkpoint")
    index = ImageIndex(ROOT / "data/raw/images")
    hashes = {}
    for donor in bank["words"]:
        identifier = donor["source_id"]
        if identifier not in eligible or donor["source_fold"] != int(eligible[identifier]["fold"]):
            raise ValueError("Heldout/excluded donor or wrong source fold")
        a, b = donor["text_span"]
        label = eligible[identifier][table.target_column]
        if not 0 <= a < b <= len(label) or label[a:b] != donor["text"] or len(donor["text"].split()) != 1:
            raise ValueError("Donor label/span mismatch")
        if donor["confidence"] < 0.95:
            raise ValueError("Low-confidence donor")
        if identifier not in hashes:
            hashes[identifier] = sha256(index.resolve(identifier))
        if hashes[identifier] != donor["source_sha256"] or sha256(absolute(donor["image"])) != donor["sha256"]:
            raise ValueError("Changed source image or word crop")
    if len(bank["words"]) != 24840:
        raise ValueError("Unexpected crop bank size")
    return bank


def compose(donors):
    height = int(statistics.median(d["box"][3] for d in donors))
    pieces = []
    for donor in donors:
        with Image.open(absolute(donor["image"])) as image:
            width = max(1, round(image.width * height / image.height))
            pieces.append(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS))
    gap = max(1, round(height * 0.15))
    width = sum(p.width for p in pieces) + gap * (len(pieces) - 1)
    if width / height > 36:
        return None
    border = sorted((pieces[0].getpixel((x, 0)) for x in range(pieces[0].width)), key=sum)
    result = Image.new("RGB", (width, height), border[len(border) // 2])
    x = 0
    for piece in pieces:
        result.paste(piece, (x, 0))
        x += piece.width + gap
    return result


def generate():
    if OUT.exists():
        raise RuntimeError("Refusing to overwrite synthetic dataset")
    bank = verify_bank()
    words = sorted(bank["words"], key=lambda d: d["image"])
    rng = random.Random(SEED)
    rows, seen = [], set()
    (OUT / "images").mkdir(parents=True)
    for _ in range(81900):
        if len(rows) == 819:
            break
        donors = rng.sample(words, rng.choice((3, 4)))
        key = tuple(d["image"] for d in donors)
        if key in seen or len({d["source_id"] for d in donors}) < 2:
            continue
        image = compose(donors)
        if image is None:
            continue
        identifier = f"shuffle3to4_f0_{len(rows):06d}"
        path = OUT / "images" / f"{identifier}.png"
        image.save(path)
        seen.add(key)
        rows.append({"id": identifier, "target": " ".join(d["text"] for d in donors),
                     "image": str(path.relative_to(ROOT)), "sha256": sha256(path),
                     "source_ids": sorted({d["source_id"] for d in donors}), "donors": donors})
    if len(rows) != 819:
        raise RuntimeError("Could not generate819 unique combinations")
    manifest = {"schema_version": 1, "method": "shuffled_short_words", "fold": 0,
                "truth_sha256": sha256(TRUTH), "bank_path": str(BANK.relative_to(ROOT)),
                "bank_sha256": sha256(BANK), "seed": SEED, "rows": rows}
    save_json(OUT / "manifest.json", manifest)
    with (OUT / "labels.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Target", "image"])
        writer.writerows((r["id"], r["target"], r["image"]) for r in rows)
    save_json(OUT / "summary.json", {"rows": len(rows), "crop_bank_size": len(words),
              "three_word_rows": sum(len(r["donors"]) == 3 for r in rows),
              "four_word_rows": sum(len(r["donors"]) == 4 for r in rows),
              "unique_crops_used": len({d["image"] for r in rows for d in r["donors"]}),
              "original_rows": 3276, "validation_rows": 819})


def validate_rows(manifest, bank, eligible, target_column):
    if manifest["method"] != "shuffled_short_words" or manifest["fold"] != 0:
        raise ValueError("Wrong synthetic method/fold")
    if bank["fold"] != 0 or bank["lineage"]["training_excluded_folds"] != [0]:
        raise ValueError("Leaky crop bank")
    allowed = {json.dumps(d, sort_keys=True) for d in bank["words"]}
    seen, combinations = set(), set()
    for row in manifest["rows"]:
        if row["id"] in seen or not row["id"].startswith("shuffle3to4_f0_"):
            raise ValueError("Duplicate/invalid synthetic ID")
        seen.add(row["id"])
        donors = row["donors"]
        if len(donors) not in (3, 4):
            raise ValueError("Expected3 or4 words")
        if row["target"] != " ".join(d["text"] for d in donors):
            raise ValueError("Synthetic label differs from exact donor sequence")
        source_ids = {d["source_id"] for d in donors}
        if len(source_ids) < 2 or set(row["source_ids"]) != source_ids or not source_ids <= set(eligible):
            raise ValueError("Heldout/excluded/single-source donors")
        combination = tuple(d["image"] for d in donors)
        if len(set(combination)) != len(donors) or combination in combinations:
            raise ValueError("Repeated crop or duplicate combination")
        combinations.add(combination)
        for donor in donors:
            if json.dumps(donor, sort_keys=True) not in allowed:
                raise ValueError("Unknown or modified donor")
            a, b = donor["text_span"]
            if eligible[donor["source_id"]][target_column][a:b] != donor["text"]:
                raise ValueError("Donor label mismatch")
        path = absolute(row["image"]).resolve()
        if not path.is_relative_to((OUT / "images").resolve()) or sha256(path) != row["sha256"]:
            raise ValueError("Synthetic image path/hash mismatch")
    if len(seen) != len(eligible) // 4:
        raise ValueError("Expected80/20 real/synthetic row mix")
    return manifest["rows"]


def validate_synthetic(manifest_path, truth, fold, eligible, target_column):
    if fold != 0 or sha256(truth) != sha256(TRUTH):
        raise ValueError("Only canonical fold0 training is authorized")
    expected_table, expected = eligible_rows()
    if eligible != expected or target_column != expected_table.target_column:
        raise ValueError("Training selection differs from canonical eligible rows")
    manifest = json.loads(absolute(manifest_path).read_text())
    if manifest["truth_sha256"] != sha256(TRUTH) or manifest["bank_sha256"] != sha256(BANK) or absolute(manifest["bank_path"]).resolve() != BANK.resolve():
        raise ValueError("Synthetic provenance mismatch")
    return validate_rows(manifest, verify_bank(), eligible, target_column)
