"""Validate train-only, generated-word passage augmentation (Model110)."""
from collections import defaultdict
import json

from .stackmix import ROOT, absolute, sha256
from .schema import read_table, predictions_by_id


def checked_json(path, digest):
    path = absolute(path).resolve()
    if not path.is_relative_to(ROOT) or sha256(path) != digest:
        raise ValueError("Paragraph artifact path/hash mismatch")
    return json.loads(path.read_text())


def validate_word_bank(manifest, truth, fold, eligible, target_column):
    if fold != 0 or manifest["fold"] != fold or manifest["truth_sha256"] != sha256(truth):
        raise ValueError("Paragraph experiment/checkers are restricted to held-out fold zero")
    generated = checked_json(manifest["generation_path"], manifest["generation_sha256"])
    if manifest["config_sha256"] != sha256(ROOT / "model110/config.json") or generated["config_sha256"] != manifest["config_sha256"]:
        raise ValueError("Paragraph generation configuration changed")
    if generated["code_sha256"] != sha256(ROOT / "model110/generate_words.py") or generated["assets_sha256"] != sha256(ROOT / "checkpoints/model108/asset_manifest.json"):
        raise ValueError("Generator code or pretrained asset ledger changed")
    requests = checked_json(generated["requests_path"], generated["requests_sha256"])
    if not generated["complete"] or generated["fold"] != fold or requests["fold"] != fold:
        raise ValueError("Incomplete generation or wrong fold")
    if requests["truth_sha256"] != manifest["truth_sha256"] or generated.get("adapter_sha256") is not None:
        raise ValueError("Expected train-only requests and the unadapted pretrained generator")
    bank = checked_json(requests["bank_path"], requests["bank_sha256"])
    if bank["fold"] != fold or bank["lineage"]["training_excluded_folds"] != [fold]:
        raise ValueError("Word aligner saw held-out data")
    known = {json.dumps(r, sort_keys=True) for r in bank["words"]}
    requested = {}
    for row in requests["rows"]:
        if row["id"] in requested:
            raise ValueError("Duplicate word request")
        source = row["target_source_id"]
        a, b = row["target_span"]
        if source not in eligible or eligible[source][target_column][a:b] != row["target"]:
            raise ValueError("Word request uses held-out data or changes its label")
        refs = row["style_references"]
        if len(refs) != 5 or len({r["image"] for r in refs}) != 5 or len({r["source_id"] for r in refs}) != 1:
            raise ValueError("Expected five same-line style references")
        for ref in refs:
            if ref["source_id"] not in eligible or json.dumps(ref, sort_keys=True) not in known:
                raise ValueError("Unknown or held-out style reference")
            if sha256(absolute(ref["image"])) != ref["sha256"]:
                raise ValueError("Style reference hash mismatch")
        requested[row["id"]] = row
    words = {}
    for row in generated["rows"]:
        identifier = row["id"]
        if identifier in words or identifier not in requested:
            raise ValueError("Duplicate/unknown generated word")
        if any(row.get(key) != value for key, value in requested[identifier].items()):
            raise ValueError("Generated word differs from its request")
        path = absolute(row["image"]).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != row["sha256"]:
            raise ValueError("Generated word image hash mismatch")
        words[identifier] = row
    if set(words) != set(requested):
        raise ValueError("Missing generated words")
    predictions = {}
    for checker in ("kraken", "lighton"):
        item = manifest["checkers"][checker]
        receipt = checked_json(item["receipt_path"], item["receipt_sha256"])
        if receipt["fold"] != fold or receipt["checker"] != checker or receipt["generation_manifest_sha256"] != manifest["generation_sha256"]:
            raise ValueError("OCR filter receipt mismatch")
        path = absolute(item["prediction_path"])
        if sha256(path) != receipt["output_sha256"]:
            raise ValueError("OCR filter predictions changed")
        predictions[checker] = predictions_by_id(read_table(path), "prediction")
        if set(predictions[checker]) != set(words):
            raise ValueError("OCR filter IDs differ")
    if manifest["word_filter"] != "at_least_one_frozen_OCR_exact_match":
        raise ValueError("Unexpected word filtering rule")
    accepted = {identifier for identifier, row in words.items()
                if any(values[identifier] == row["target"] for values in predictions.values())}
    return words, accepted


def validate_lines(rows, words, accepted, eligible, target_column):
    ids, texts = set(), set()
    originals = {r[target_column] for r in eligible.values()}
    paragraphs = defaultdict(list)
    for row in rows:
        if row["id"] in ids or row["id"] in eligible:
            raise ValueError("Duplicate synthetic line ID")
        ids.add(row["id"])
        donors = row["word_ids"]
        if not donors or any(identifier not in accepted for identifier in donors):
            raise ValueError("Line contains an unaccepted generated word")
        tokens = [words[identifier]["target"] for identifier in donors]
        if row["target"] != " ".join(tokens) or row["target"] in texts or row["target"] in originals:
            raise ValueError("Line target differs, is duplicated, or is not newly composed")
        texts.add(row["target"])
        if {words[identifier]["style"] for identifier in donors} != {row["style"]}:
            raise ValueError("Inconsistent handwriting style")
        edges = row["transitions"]
        if len(edges) != len(tokens) - 1:
            raise ValueError("Missing within-line text provenance")
        for index, edge in enumerate(edges):
            source = edge["source_id"]
            offset = edge["token_index"]
            if source not in eligible or eligible[source][target_column].split()[offset:offset+2] != tokens[index:index+2]:
                raise ValueError("Text transition uses held-out data or wrong words")
        path = absolute(row["image"]).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != row["sha256"]:
            raise ValueError("Synthetic line image path/hash mismatch")
        paragraphs[row["paragraph_id"]].append(row)
    if not ids:
        raise ValueError("Empty synthetic dataset")
    for group in paragraphs.values():
        group.sort(key=lambda r: r["line_index"])
        if [r["line_index"] for r in group] != list(range(len(group))) or len({r["style"] for r in group}) != 1:
            raise ValueError("Invalid paragraph grouping")
        for previous, current in zip(group, group[1:]):
            edge = current["previous_line_transition"]
            source, offset = edge["source_id"], edge["token_index"]
            pair = [previous["target"].split()[-1], current["target"].split()[0]]
            if source not in eligible or eligible[source][target_column].split()[offset:offset+2] != pair:
                raise ValueError("Cross-line transition provenance mismatch")
    return rows


def validate_paragraph_synthetic(manifest, truth, fold, eligible, target_column):
    words, accepted = validate_word_bank(manifest, truth, fold, eligible, target_column)
    return validate_lines(manifest["rows"], words, accepted, eligible, target_column)
