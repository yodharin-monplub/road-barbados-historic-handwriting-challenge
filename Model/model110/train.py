"""Fixed Model92 OCR recipe with an 80/20 real/generated-line training mix."""
import argparse
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_htr.stackmix import absolute, eligible_rows, validate_synthetic
from road_htr.vlm_data import read_id_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "model92/config.yaml").read_text())
    config["output_dir"] = "outputs/model110/train/fold{fold}"
    config["validation_fold"] = 0
    config["synthetic_manifest"] = "outputs/model110/fold0/training/manifest.json"
    config["training"]["max_steps"] = 400
    config["training"]["save_steps"] = 400
    truth = absolute(config["train_csv"])
    table, eligible = eligible_rows(truth, 0, read_id_file(absolute(config["exclude_ids"])))
    rows = validate_synthetic(config["synthetic_manifest"], truth, 0, eligible, table.target_column)
    if len(rows) != len(eligible)//4:
        raise ValueError("Require the preregistered 80/20 training mix")
    output = ROOT / "outputs/model110/config_fold0.yaml"
    text = yaml.safe_dump(config, sort_keys=False)
    if output.exists() and output.read_text() != text:
        raise ValueError("Refusing to change an existing experiment config")
    output.write_text(text)
    print(f"Config ready: real={len(eligible)} synthetic={len(rows)}", flush=True)
    if not args.prepare_only:
        from road_htr.vlm_cli import train
        train(output, 0)


if __name__ == "__main__":
    main()
