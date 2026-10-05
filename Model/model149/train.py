#!/usr/bin/env python3
"""Model149: the Model22/130 Kraken recipe on round-5 pseudo-labels.

Kraken reads one labelled CSV, so the pseudo rows are added per fold:
Data/processed/train_folds_pseudo5_fold{fold}.csv holds the 4,095 real rows plus
the 1,373 round-5 pseudo rows, the pseudo rows tagged with a fold other than the
validation fold. The held-out 819 rows of fold k stay exactly as they were, so
the fold score remains comparable with Model130's.

config.yaml therefore carries {fold} in train_csv/prepared_dir/output_dir; this
wrapper renders it before handing the file to the shared trainer.
"""
from pathlib import Path
import argparse
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))

from road_htr.kraken_cli import train_main  # noqa: E402


def main() -> None:
    known, _ = argparse.ArgumentParser(add_help=False).parse_known_args()
    fold = None
    for index, value in enumerate(sys.argv):
        if value == "--fold":
            fold = int(sys.argv[index + 1])
    if fold is None:
        raise SystemExit("--fold is required")
    text = (HERE / "config.yaml").read_text(encoding="utf-8").replace("{fold}", str(fold))
    handle = tempfile.NamedTemporaryFile("w", suffix=f"_fold{fold}.yaml",
                                         delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    print(f"rendered config: {handle.name}", flush=True)
    train_main(handle.name)


if __name__ == "__main__":
    main()
