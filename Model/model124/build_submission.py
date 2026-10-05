#!/usr/bin/env python3
"""Vote several test-prediction files into one validated Zindi submission.

  python model124/build_submission.py --anchor A.csv --sources B.csv C.csv \
      --min-votes 3 --output submissions/name.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from road_htr.submission import validate_submission, write_submission  # noqa: E402
from vote import read_predictions, vote_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--sources", nargs="*", default=[])
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", default="Data/raw/SampleSubmission.csv")
    args = parser.parse_args()

    anchor = read_predictions(args.anchor)
    sources = [read_predictions(path) for path in args.sources]
    voted = vote_all(anchor, sources, args.min_votes) if sources else dict(anchor)
    # Zindi rejects empty predictions: fall back to the anchor, then to the
    # first source that produced any text for that image.
    blank = [key for key, value in voted.items() if not value.strip()]
    for key in blank:
        candidates = [anchor.get(key, "")] + [source.get(key, "") for source in sources]
        voted[key] = next((c for c in candidates if c.strip()), "")
    unresolved = [key for key, value in voted.items() if not value.strip()]
    if unresolved:
        raise SystemExit(f"every source is blank for {unresolved}; inspect those images")

    output = ROOT / args.output
    write_submission(ROOT / args.sample, output, voted)
    report = validate_submission(output, ROOT / args.sample)
    report.update(
        anchor=args.anchor, sources=args.sources, min_votes=args.min_votes,
        changed_vs_anchor=sum(1 for k in anchor if anchor[k] != voted[k]),
        recovered_blank=len(blank),
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        output=str(output),
    )
    (output.with_suffix(".report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
