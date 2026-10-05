"""Fold-isolated handwriting recombination using automatic CTC alignment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image

from .schema import ImageIndex, read_table
from .vlm_data import read_id_file

ROOT = Path(__file__).resolve().parents[2]


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def eligible_rows(truth, fold, excluded):
    table = read_table(truth, require_target=True)
    if "fold" not in table.fieldnames:
        raise ValueError("Canonical folds are required")
    rows = {r[table.id_column]: r for r in table.rows
            if int(r["fold"]) != fold and r[table.id_column] not in excluded}
    if not rows:
        raise ValueError("No eligible training rows")
    return table, rows


def ctc_align(probabilities, labels):
    """Exact CTC Viterbi alignment, including the blank between repeated letters."""
    p = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2 or not len(labels) or not np.isfinite(p).all():
        raise ValueError("Invalid CTC inputs")
    if np.any(labels <= 0) or np.any(labels >= p.shape[1]):
        raise ValueError("Unsupported codec labels")
    if np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=2e-4):
        raise ValueError("Expected normalized CTC probabilities")
    states = np.zeros(2 * len(labels) + 1, dtype=np.int64)
    states[1::2] = labels
    skip = (states != 0) & (states != np.roll(states, 2))
    skip[:2] = False
    emissions = np.log(np.maximum(p[:, states], 1e-30))
    previous = np.full(len(states), -np.inf)
    previous[:2] = emissions[0, :2]
    back = np.zeros((len(p), len(states)), dtype=np.uint8)
    for t in range(1, len(p)):
        one = np.concatenate(([-np.inf], previous[:-1]))
        two = np.concatenate(([-np.inf, -np.inf], previous[:-2]))
        two[~skip] = -np.inf
        options = np.stack((previous, one, two))
        back[t] = np.argmax(options, axis=0)
        previous = np.max(options, axis=0) + emissions[t]
    state = len(states) - 2 + int(previous[-1] > previous[-2])
    if not np.isfinite(previous[state]):
        raise ValueError("Impossible CTC alignment")
    route = np.zeros(len(p), dtype=np.int64)
    for t in range(len(p) - 1, -1, -1):
        route[t] = state
        state -= int(back[t, state])
    spans = []
    for index, label in enumerate(labels):
        frames = np.flatnonzero(route == 2 * index + 1)
        if not len(frames):
            raise ValueError("Unaligned character")
        spans.append((int(frames[0]), int(frames[-1]) + 1, float(p[frames, label].max())))
    return spans


def build_bank(config, output, limit=None, device="cpu"):
    from kraken.configs import RecognitionInferenceConfig
    from kraken.lib.ctc_decoder import greedy_decoder
    from kraken.tasks import RecognitionTaskModel
    from .kraken_cli import _align_kraken_sequence_lengths_with_network, _make_batch_segmentation
    from .kraken_probability_export import CapturingGreedyDecoder, ExportOptions, _checkpoint_provenance

    output = absolute(output)
    output.mkdir(parents=True, exist_ok=True)
    truth = absolute(config["truth"])
    checkpoint = absolute(config["checkpoint"])
    fold = None if config["fold"] is None else int(config["fold"])
    excluded = read_id_file(absolute(config["exclude_ids"]))
    table, eligible = eligible_rows(truth, fold, excluded)
    options = ExportOptions(config=absolute("model29/config.yaml"), input_csv=truth,
        checkpoint=checkpoint, output_dir=output, fold=fold,
        training_manifest=absolute(config["training_manifest"]),
        conversion_receipt=absolute(config["conversion_receipt"]))
    lineage = _checkpoint_provenance(options, truth, sha256(truth), checkpoint, sha256(checkpoint))
    signature = {"config": config, "truth_sha256": sha256(truth),
                 "checkpoint_sha256": sha256(checkpoint), "code_sha256": sha256(__file__)}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    image_index = ImageIndex(absolute(config["image_dir"]))
    ids = sorted(eligible, key=lambda x: hashlib.sha256(f'{config["seed"]}:{x}'.encode()).hexdigest())
    if limit is not None:
        ids = ids[:limit]
    capture = CapturingGreedyDecoder(greedy_decoder)
    inference = RecognitionInferenceConfig(accelerator="gpu" if device == "cuda" else "cpu",
        device="auto", precision="32-true", batch_size=1, num_threads=2,
        num_line_workers=0, padding=16, bidi_reordering=False,
        raise_on_error=True, decoder=capture)
    model = RecognitionTaskModel.load_model(checkpoint)
    model.net.prepare_for_inference(inference)
    _align_kraken_sequence_lengths_with_network(model)
    words, rejections = [], Counter()
    for position, identifier in enumerate(ids, 1):
        row = eligible[identifier]
        source = image_index.resolve(identifier)
        source_hash = sha256(source)
        cache = output / "cache" / f"{identifier}.json"
        saved = json.loads(cache.read_text()) if cache.exists() else None
        if saved and saved.get("fingerprint") == fingerprint and saved.get("image_sha256") == source_hash:
            words.extend(saved["words"])
            rejections.update(saved["rejections"])
            continue
        target = row[table.target_column]
        result, rejected = [], Counter()
        # No normalization: unknown/compound codec characters must not silently vanish.
        codec = model.net.codec.c2l
        if any(len(codec.get(c, [])) != 1 for c in target):
            rejected["unsupported_character"] += 1
        elif not target.strip() or any(c in target for c in "\n\r\t"):
            rejected["unsupported_layout"] += 1
        else:
            with Image.open(source) as handle:
                image = handle.convert("RGB")
            canvas, segmentation = _make_batch_segmentation([(identifier, source, image)])
            records = list(model.net.predict(canvas, segmentation))
            if len(records) != 1 or records[0].id != identifier:
                raise RuntimeError("Recognition ID mismatch")
            probabilities = capture.take().numpy()
            try:
                spans = ctc_align(probabilities, [codec[c][0] for c in target])
            except ValueError:
                spans = []
                rejected["failed_alignment"] += 1
            for token in re.finditer(r"\S+", target) if spans else []:
                a, b = token.span()
                scores = [s[2] for s in spans[a:b]]
                if min(scores) < config["min_character_confidence"] or np.mean(scores) < config["min_word_confidence"]:
                    rejected["low_confidence_word"] += 1
                    continue
                # Boundaries lie halfway into adjoining whitespace, not through ink.
                left = (spans[a - 1][0] + spans[a - 1][1]) / 2 if a else spans[a][0] - 1
                right = (spans[b][0] + spans[b][1]) / 2 if b < len(spans) else spans[b - 1][1] + 1
                x0 = model.net._scale_val(left, 0, image.width)
                x1 = model.net._scale_val(right, 0, image.width)
                if x1 - x0 < 3:
                    rejected["invalid_boundary"] += 1
                    continue
                crop_path = output / "crops" / f"{identifier}_{a}_{b}.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                image.crop((x0, 0, x1, image.height)).save(crop_path)
                result.append({"source_id": identifier, "source_fold": int(row["fold"]),
                    "source_sha256": source_hash, "text": token.group(), "text_span": [a,b],
                    "box": [x0,0,x1,image.height], "confidence": float(np.mean(scores)),
                    "image": str(crop_path.relative_to(ROOT)), "sha256": sha256(crop_path)})
        save_json(cache, {"fingerprint": fingerprint, "image_sha256": source_hash,
                          "words": result, "rejections": dict(rejected)})
        words.extend(result)
        rejections.update(rejected)
        if position % 25 == 0 or position == len(ids):
            print(f"aligned {position}/{len(ids)}, accepted words={len(words)}", flush=True)
    save_json(output / "bank.json", {"schema_version":1, "fold":fold, "signature":signature,
        "lineage":lineage, "source_ids":ids, "words":words, "rejections":dict(rejections)})
    return {"aligned_rows":len(ids), "words":len(words), "rejections":dict(rejections)}


def synthesize(config, bank_path, output, count=200):
    bank_path, output = absolute(bank_path), absolute(output)
    bank = json.loads(bank_path.read_text())
    fold = None if config["fold"] is None else int(config["fold"])
    truth = absolute(config["truth"])
    table, eligible = eligible_rows(truth, fold, read_id_file(absolute(config["exclude_ids"])))
    if bank["fold"] != fold or bank["signature"]["truth_sha256"] != sha256(truth):
        raise ValueError("Crop bank fold or truth mismatch")
    vocabulary = defaultdict(list)
    for word in bank["words"]:
        if word["source_id"] not in eligible:
            raise ValueError("Heldout or excluded crop source")
        target = eligible[word["source_id"]][table.target_column]
        a,b = word["text_span"]
        if target[a:b] != word["text"] or sha256(absolute(word["image"])) != word["sha256"]:
            raise ValueError("Crop text/hash mismatch")
        vocabulary[word["text"]].append(word)
    targets = [r for r in eligible.values() if r[table.target_column].strip()
               and all(w in vocabulary for w in r[table.target_column].split())
               and not any(c in r[table.target_column] for c in "\t\n\r")]
    if not targets:
        raise ValueError("No fully covered target lines; expand the crop bank")
    rng = random.Random(config["seed"])
    targets.sort(key=lambda r:r[table.id_column])
    output.mkdir(parents=True, exist_ok=True)
    generated, seen = [], set()
    failures = Counter()
    for attempt in range(count * 100):
        if len(generated) == count:
            break
        row = rng.choice(targets)
        tokens = re.findall(r"\S+|\s+", row[table.target_column])
        donors = [rng.choice(vocabulary[t]) for t in tokens if not t.isspace()]
        if len({w["source_id"] for w in donors}) < 2:
            failures["single_source"] += 1
            continue
        key = tuple(w["image"] for w in donors)
        if key in seen:
            failures["duplicate_donors"] += 1
            continue
        height = int(np.median([w["box"][3] for w in donors]))
        pieces, cursor = [], 0
        for token in tokens:
            if token.isspace():
                pieces.append(max(1, round(height * config["space_height_fraction"] * len(token))))
            else:
                with Image.open(absolute(donors[cursor]["image"])) as image:
                    width = max(1, round(image.width * height / image.height))
                    pieces.append(image.convert("RGB").resize((width,height), Image.Resampling.LANCZOS))
                cursor += 1
        width = sum(p if isinstance(p,int) else p.width for p in pieces)
        if width / height > config["max_aspect_ratio"]:
            failures["too_wide"] += 1
            continue
        # Use a bright border-paper pixel from the supplied crop for gap padding.
        first = next(p for p in pieces if not isinstance(p,int))
        border = [first.getpixel((x,0)) for x in range(first.width)]
        paper = sorted(border, key=sum)[len(border)//2]
        composed = Image.new("RGB", (width,height), paper)
        offset = 0
        for piece in pieces:
            if isinstance(piece,int):
                offset += piece
            else:
                composed.paste(piece,(offset,0))
                offset += piece.width
        identifier = f"stackmix_f{fold if fold is not None else 'all'}_{len(generated):06d}"
        path = output / "images" / f"{identifier}.png"
        path.parent.mkdir(exist_ok=True)
        composed.save(path)
        seen.add(key)
        generated.append({"id":identifier, "target":row[table.target_column],
            "target_source_id":row[table.id_column], "image":str(path.relative_to(ROOT)),
            "sha256":sha256(path), "source_ids":sorted({w["source_id"] for w in donors}),
            "donors":donors})
    if len(generated) != count:
        raise RuntimeError(f"Only {len(generated)}/{count} diverse lines available; expand bank")
    manifest = {"schema_version":1, "method":"stackmix", "fold":fold,
        "truth_sha256":sha256(truth), "bank_sha256":sha256(bank_path),
        "bank_path":str(bank_path.relative_to(ROOT)),
        "seed":config["seed"], "rows":generated, "rejections":dict(failures)}
    save_json(output / "manifest.json",manifest)
    with (output / "labels.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["ID","Target","image"])
        writer.writeheader()
        writer.writerows({"ID":r["id"],"Target":r["target"],"image":r["image"]} for r in generated)
    return {"rows":len(generated),"covered_target_lines":len(targets),"rejections":dict(failures)}


def validate_synthetic(manifest_path, truth, fold, eligible, target_column):
    """Validate every donor and image before opt-in augmentation enters training."""
    manifest = json.loads(absolute(manifest_path).read_text())
    if manifest["fold"] != fold or manifest["truth_sha256"] != sha256(truth):
        raise ValueError("Synthetic manifest fold/truth mismatch")
    if manifest["method"] == "diffusionpen_paragraph":
        from .paragraph_synthetic import validate_paragraph_synthetic
        return validate_paragraph_synthetic(manifest, truth, fold, eligible, target_column)
    if manifest["method"] != "stackmix":
        raise ValueError("Only the approved StackMix method is currently enabled")
    bank_path=absolute(manifest["bank_path"])
    if sha256(bank_path)!=manifest["bank_sha256"]:
        raise ValueError("Crop-bank provenance hash mismatch")
    bank=json.loads(bank_path.read_text())
    expected_excluded = [] if fold is None else [fold]
    if bank["fold"]!=fold or bank["lineage"]["training_excluded_folds"]!=expected_excluded:
        raise ValueError("Aligner was not trained with the validation fold excluded")
    bank_words={json.dumps(d,sort_keys=True) for d in bank["words"]}
    ids = set()
    for row in manifest["rows"]:
        if row["id"] in ids or row["id"] in eligible:
            raise ValueError("Duplicate synthetic ID")
        ids.add(row["id"])
        if row["target_source_id"] not in eligible or any(s not in eligible for s in row["source_ids"]):
            raise ValueError("Synthetic sample uses heldout/excluded data")
        if row["target"] != eligible[row["target_source_id"]][target_column]:
            raise ValueError("Synthetic target is not the original training transcription")
        if row["target"].split() != [d["text"] for d in row["donors"]]:
            raise ValueError("Donor text does not match synthetic label")
        if set(row["source_ids"]) != {d["source_id"] for d in row["donors"]}:
            raise ValueError("Missing donor provenance")
        for donor in row["donors"]:
            if json.dumps(donor,sort_keys=True) not in bank_words:
                raise ValueError("Unknown donor crop")
            a,b=donor["text_span"]
            if eligible[donor["source_id"]][target_column][a:b] != donor["text"]:
                raise ValueError("Donor transcript mismatch")
        path=absolute(row["image"]).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != row["sha256"]:
            raise ValueError("Synthetic image path/hash mismatch")
    if not ids:
        raise ValueError("Empty synthetic dataset")
    return manifest["rows"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["bank","synthesize"])
    parser.add_argument("--config",type=Path,default=ROOT/"model107/config.json")
    parser.add_argument("--output",required=True)
    parser.add_argument("--bank")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--count",type=int,default=200)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if args.command=="bank":
        report=build_bank(config,args.output,args.limit,args.device)
    else:
        if not args.bank:
            parser.error("synthesize requires --bank")
        report=synthesize(config,args.bank,args.output,args.count)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
