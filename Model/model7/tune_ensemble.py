#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from road_htr.ensemble import (  # noqa: E402
    apply_config,
    parse_sources,
    prepare_arrays,
    tune,
    write_prediction_csv,
)
from road_htr.metrics import competition_error  # noqa: E402
from road_htr.schema import read_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--nbest", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--regularization", type=float, default=0.0002)
    args = parser.parse_args()

    truth = read_table(args.truth, require_target=True)
    assert truth.target_column is not None
    ids = [row[truth.id_column].strip() for row in truth.rows]
    references = [row[truth.target_column] for row in truth.rows]
    sources = parse_sources(args.source)
    arrays = prepare_arrays(sources, ids, references, args.nbest)
    config = tune(arrays, sources, args.trials, args.seed, args.regularization)
    config["nbest_per_source"] = args.nbest
    selected, _, _, _ = apply_config(arrays, config)
    config["verified_score"] = competition_error(references, selected)

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_prediction_csv(args.output_predictions, truth.id_column, ids, selected)
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

