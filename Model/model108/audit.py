"""Require exact generated-word agreement from two fold-excluded OCR architectures."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import absolute, sha256, save_json
from road_htr.schema import read_table, predictions_by_id
from prepare import validate_requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--checker", choices=["kraken", "lighton", "score"], required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    generated = json.loads((directory / "generation_manifest.json").read_text())
    request_path = absolute(generated["requests_path"])
    if sha256(request_path) != generated["requests_sha256"] or not generated["complete"]:
        raise ValueError("Need a complete pilot with matching request provenance")
    manifest = json.loads(request_path.read_text())
    requests = {r["id"]: r for r in validate_requests(manifest)}
    rows = generated["rows"]
    if len(rows) != len(requests) or {r["id"] for r in rows} != set(requests):
        raise ValueError("Pilot IDs differ")
    for row in rows:
        if row["target"] != requests[row["id"]]["target"] or sha256(absolute(row["image"])) != row["sha256"]:
            raise ValueError("Generated image or target changed")
    fold = generated["fold"]
    if fold != 0:
        raise ValueError("Initial pilot checkers are pinned to fold zero only")
    input_path = directory / "labels.csv"
    labels = predictions_by_id(read_table(input_path, require_target=True), "Target")
    if labels != {row["id"]: row["target"] for row in rows}:
        raise ValueError("Pilot labels CSV differs from the immutable requested targets")
    if args.checker == "kraken":
        from road_htr.kraken_cli import infer
        from road_htr.kraken_probability_export import ExportOptions, _checkpoint_provenance
        bank = json.loads(absolute(manifest["bank_path"]).read_text())
        preparation = bank["signature"]["config"]
        checkpoint = absolute(preparation["checkpoint"])
        truth = absolute(preparation["truth"])
        options = ExportOptions(config=ROOT / "model29/config.yaml", input_csv=truth, checkpoint=checkpoint,
            output_dir=directory, fold=fold, training_manifest=absolute(preparation["training_manifest"]),
            conversion_receipt=absolute(preparation["conversion_receipt"]))
        lineage = _checkpoint_provenance(options, truth, sha256(truth), checkpoint, sha256(checkpoint))
        config = yaml.safe_load((ROOT / "model36/config.yaml").read_text())
        config["image_dir"] = str(directory / "images")
        config["inference"].update(accelerator="gpu", precision="32-true", threads=2)
        path = directory / "kraken_config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        infer(path, input_path, directory / "kraken.csv", checkpoint=checkpoint)
        save_json(directory / "kraken_lineage.json", lineage)
    elif args.checker == "lighton":
        from road_htr.vlm_cli import infer
        adapter = ROOT / "outputs/model107/train/fold0/best_adapter"
        training = json.loads((adapter.parent / "run_manifest.json").read_text())
        receipt = json.loads((ROOT / "outputs/model107/fold0_oof.manifest.json").read_text())
        if training["config"]["validation_fold"] != fold or training["validation_rows"] != 819 or receipt["fold"] != fold:
            raise ValueError("LightOn checker fold mismatch")
        for name, digest in receipt["adapter_files_sha256"].items():
            if sha256(adapter / name) != digest:
                raise ValueError("LightOn checker adapter mismatch")
        config = yaml.safe_load((ROOT / "outputs/model107/config_fold0.yaml").read_text())
        config["image_dir"] = str(directory / "images")
        config["generation"]["max_new_tokens"] = 32
        path = directory / "lighton_config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        infer(path, input_path, directory / "lighton.csv", adapter_path=adapter)
    else:
        predictions = {checker: predictions_by_id(read_table(directory / f"{checker}.csv"), "prediction")
                       for checker in ["kraken", "lighton"]}
        for values in predictions.values():
            if set(values) != set(requests):
                raise ValueError("Checker prediction IDs differ")
        comparisons = []
        for row in rows:
            identifier, text = row["id"], row["target"]
            comparisons.append({"ID": identifier, "requested": text,
                "kraken": predictions["kraken"][identifier], "lighton": predictions["lighton"][identifier],
                "accepted": predictions["kraken"][identifier] == text == predictions["lighton"][identifier]})
        with (directory / "word_comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        accepted = [r["ID"] for r in comparisons if r["accepted"]]
        save_json(directory / "quality_report.json", {"rows": len(rows), "accepted_words": len(accepted),
            "acceptance_rate": len(accepted) / len(rows), "accepted_ids": accepted,
            "exact_by_checker": {checker: sum(p["requested"] == p[checker] for p in comparisons) for checker in predictions},
            "generation_manifest_sha256": sha256(directory / "generation_manifest.json"),
            "competition_cv": None, "note": "Automatic text-fidelity screen, not proof of true labels or competition performance."})
        print(f"Exact agreement accepted {len(accepted)}/{len(rows)} words")
    if args.checker != "score":
        save_json(directory / f"{args.checker}_audit_receipt.json", {
            "checker": args.checker,
            "output_sha256": sha256(directory / f"{args.checker}.csv"),
            "generation_manifest_sha256": sha256(directory / "generation_manifest.json"),
            "labels_sha256": sha256(input_path), "fold": fold,
            "audit_code_sha256": sha256(__file__)})


if __name__ == "__main__":
    main()
