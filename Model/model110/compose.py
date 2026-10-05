"""Compose new train-derived passages; render lines using generated word pixels."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import absolute, sha256, save_json, eligible_rows, validate_synthetic
from road_htr.vlm_data import read_id_file
from road_htr.paragraph_synthetic import validate_word_bank


def main():
    import numpy as np
    from PIL import Image
    config = json.loads((ROOT / "model110/config.json").read_text())
    root = absolute(config["output_dir"])
    out = root / "training"
    if (out / "manifest.json").exists():
        raise ValueError("Refusing to overwrite a composed experiment")
    truth = absolute(config["truth"])
    table, eligible = eligible_rows(truth, config["fold"], read_id_file(ROOT / "config/known_bad_train_ids.txt"))
    word_dir = root / "words"
    generation = word_dir / "generation_manifest.json"
    manifest = {"schema_version": 1, "method": "diffusionpen_paragraph", "fold": config["fold"],
                "truth_sha256": sha256(truth), "generation_path": str(generation.relative_to(ROOT)),
                "generation_sha256": sha256(generation), "word_filter": config["word_filter"],
                "seed": config["seed"], "config_sha256": sha256(ROOT / "model110/config.json"), "checkers": {}}
    for checker in ("kraken", "lighton"):
        receipt = word_dir / f"{checker}_audit_receipt.json"
        manifest["checkers"][checker] = {"receipt_path": str(receipt.relative_to(ROOT)), "receipt_sha256": sha256(receipt),
                                             "prediction_path": str((word_dir / f"{checker}.csv").relative_to(ROOT))}
    words, accepted = validate_word_bank(manifest, truth, config["fold"], eligible, table.target_column)
    rendered, by_style = {}, defaultdict(dict)
    for identifier in sorted(accepted):
        row = words[identifier]
        with Image.open(absolute(row["image"])) as source:
            image = source.convert("RGB")
        ink = np.asarray(image.convert("L")) < 180
        xs = np.where(ink.any(axis=0))[0]
        if len(xs) < 2 or ink.sum() < 15:
            continue
        image = image.crop((max(0, int(xs[0])-3), 0, min(image.width, int(xs[-1])+4), image.height))
        rendered[identifier] = image
        by_style[row["style"]][row["target"]] = identifier
    styles = sorted(s for s, vocabulary in by_style.items() if len(vocabulary) >= config["min_style_vocabulary"])
    if not styles:
        raise ValueError("Insufficient accepted vocabulary in every style; no OCR training launched")
    bigrams, trigrams = defaultdict(list), defaultdict(list)
    originals = set()
    for identifier, row in sorted(eligible.items()):
        tokens = row[table.target_column].split()
        originals.add(" ".join(tokens))
        for index in range(len(tokens)-1):
            edge = {"source_id": identifier, "token_index": index}
            bigrams[tokens[index]].append((tokens[index+1], edge))
            if index:
                trigrams[tuple(tokens[index-1:index+1])].append((tokens[index+1], edge))
    rng = random.Random(config["seed"])
    rows, paragraphs, seen = [], [], set()
    out.mkdir(parents=True, exist_ok=True)
    images = out / "images"
    images.mkdir(exist_ok=True)
    attempts = 0
    while len(rows) < config["synthetic_lines"] and attempts < 50000:
        attempts += 1
        style = rng.choice(styles)
        vocabulary = by_style[style]
        starts = sorted(w for w in vocabulary if w[0].isupper() and w in bigrams)
        if not starts:
            continue
        tokens, edges = [rng.choice(starts)], []
        nlines = min(config["lines_per_paragraph"], config["synthetic_lines"] - len(rows))
        for _ in range(nlines*config["words_per_line"]-1):
            choices = [p for p in trigrams.get(tuple(tokens[-2:]), []) if p[0] in vocabulary]
            if not choices:
                choices = [p for p in bigrams[tokens[-1]] if p[0] in vocabulary]
            choices = [p for p in choices if tokens[-4:].count(p[0]) < 2]
            if not choices:
                break
            word, edge = rng.choice(choices)
            tokens.append(word); edges.append(edge)
        if len(tokens) != nlines*config["words_per_line"]:
            continue
        chunks = [tokens[i:i+config["words_per_line"]] for i in range(0, len(tokens), config["words_per_line"])]
        texts = [" ".join(chunk) for chunk in chunks]
        if len(set(texts)) != len(texts) or any(text in seen or text in originals for text in texts):
            continue
        if any(max(Counter(chunk).values()) > 4 for chunk in chunks):
            continue
        paragraph_id = f"m110_f0_p{len(paragraphs):05d}"
        for index, (chunk, target) in enumerate(zip(chunks, texts)):
            identifier = f"{paragraph_id}_l{index}"
            donors = [vocabulary[token] for token in chunk]
            parts = [rendered[d] for d in donors]
            image = Image.new("RGB", (16+sum(p.width for p in parts)+10*(len(parts)-1), 80), "white")
            left = 8
            for part in parts:
                image.paste(part, (left, 8)); left += part.width+10
            path = images / f"{identifier}.png"
            image.save(path)
            start = index*config["words_per_line"]
            rows.append({"id": identifier, "paragraph_id": paragraph_id, "line_index": index, "style": style,
                         "target": target, "word_ids": donors, "transitions": edges[start:start+len(chunk)-1],
                         "previous_line_transition": edges[start-1] if index else None,
                         "image": str(path.relative_to(ROOT)), "sha256": sha256(path)})
            seen.add(target)
        paragraphs.append("\n".join(texts))
    if len(rows) != config["synthetic_lines"]:
        raise ValueError(f"Only composed {len(rows)} lines; require {config['synthetic_lines']} before training")
    manifest["rows"] = rows
    save_json(out / "manifest.json", manifest)
    validate_synthetic(out / "manifest.json", truth, config["fold"], eligible, table.target_column)
    with (out / "labels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["ID", "Target", "image"])
        writer.writerows((r["id"], r["target"], r["image"]) for r in rows)
    (out / "passages.txt").write_text("\n\n".join(paragraphs)+"\n")
    report = {"stage": "synthetic_training_ready", "rows": len(rows), "paragraphs": len(paragraphs),
              "generated_words": len(words), "union_exact_words": len(accepted), "usable_word_images": len(rendered),
              "vocabulary_by_style": {s: len(v) for s, v in by_style.items()}, "attempts": attempts,
              "manifest_sha256": sha256(out / "manifest.json"), "competition_cv": None,
              "note": "Training-only n-gram recombinations; not guaranteed coherent prose. Intended labels remain noisy despite OCR filtering."}
    save_json(root / "preparation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
