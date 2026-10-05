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
    agreement_rows,
    apply_config,
    parse_sources,
    prepare_arrays,
    save_submission_if_requested,
    write_prediction_csv,
)
from road_htr.schema import read_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agreement-output", type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()

    table = read_table(args.test)
    ids = [row[table.id_column].strip() for row in table.rows]
    sources = parse_sources(args.source)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    arrays = prepare_arrays(
        sources, ids, nbest_per_source=int(config.get("nbest_per_source", 5))
    )
    selected, _, _, weights = apply_config(arrays, config)
    write_prediction_csv(args.output, table.id_column, ids, selected)
    save_submission_if_requested(args.sample, args.submission, ids, selected)

    if args.agreement_output:
        rows = agreement_rows(arrays, selected, weights, float(config["char_mix"]))
        args.agreement_output.parent.mkdir(parents=True, exist_ok=True)
        with args.agreement_output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(ids)} consensus predictions to {args.output}")


if __name__ == "__main__":
    main()

