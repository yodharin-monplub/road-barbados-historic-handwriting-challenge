#!/usr/bin/env python3
"""Summarize every Model122 run found under outputs/ as one table."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent


def main() -> None:
    rows = []
    for history in sorted(ROOT.glob("Other/outputs/model122_*/fold*/history.json")):
        run = history.parent.parent.name.replace("model122_", "")
        fold = history.parent.name
        try:
            entries = json.loads(history.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for entry in entries:
            rows.append((run, fold, entry["epoch"], entry.get("beams", 1),
                         entry["multi_score"], entry["weighted_wer"], entry["weighted_cer"],
                         entry.get("seconds", 0) / 60))
    if not rows:
        print("no runs yet")
        return
    print(f"{'run':<22}{'fold':<7}{'ep':>3}{'beams':>6}{'score':>12}{'wWER':>9}{'wCER':>9}{'min':>7}")
    for run, fold, epoch, beams, score, wer, cer, minutes in rows:
        print(f"{run:<22}{fold:<7}{epoch:>3}{beams:>6}{score:>12.6f}{wer:>9.4f}{cer:>9.4f}{minutes:>7.1f}")
    best = max(rows, key=lambda r: r[4])
    print(f"\nbest: {best[0]} {best[1]} epoch {best[2]} beams {best[3]} score {best[4]:.6f}")


if __name__ == "__main__":
    main()
