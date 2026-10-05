#!/usr/bin/env python3
"""Second cut of Model157: split the loss by crop height AND by error severity.

Severity = character edits / target length: gross errors (>= 0.4) are wrong-line or
missing-line transcriptions - the fixable kind; small ones (< 0.1) are character slips.
Reads Model/model157/row_errors.json written by analyze_errors.py.
"""
import json
from pathlib import Path

rows = json.loads((Path(__file__).parent / "row_errors.json").read_text())
heights = sorted(r["size"][1] for r in rows if r["size"])
median_h = heights[len(heights) // 2]
total = sum(r["loss"] for r in rows)


def band(r):
    h = r["size"][1] if r["size"] else median_h
    return "tall (>=1.8x)" if h >= 1.8 * median_h else ("mid (1.3-1.8x)" if h >= 1.3 * median_h else "normal")


def severity(r):
    rel = r["ec"] / max(r["nref"], 1)
    if rel == 0:
        return "0 exact"
    return "1 small <0.1" if rel < 0.1 else ("2 mid 0.1-0.4" if rel < 0.4 else "3 GROSS >=0.4")


cells = {}
for r in rows:
    key = (band(r), severity(r))
    c = cells.setdefault(key, [0, 0.0, 0, 0])
    c[0] += 1
    c[1] += r["loss"]
    c[2] += r["nhyp"] > r["nref"] * 1.3   # model wrote clearly more than the label
    c[3] += r["nhyp"] < r["nref"] * 0.7   # model wrote clearly less than the label
print(f"median height {median_h}px   total loss {total:.6f}")
print(f"{'height band':<16}{'severity':<16}{'rows':>6}{'loss %':>9}{'  longer':>9}{'  shorter':>9}")
for key in sorted(cells):
    n, loss, longer, shorter = cells[key]
    print(f"{key[0]:<16}{key[1]:<16}{n:>6}{100 * loss / total:>8.2f}%{longer:>9}{shorter:>9}")
for b in ("tall (>=1.8x)", "mid (1.3-1.8x)", "normal"):
    n = sum(c[0] for k, c in cells.items() if k[0] == b)
    loss = sum(c[1] for k, c in cells.items() if k[0] == b)
    print(f"  {b:<16} {n:>5} rows  {100 * loss / total:6.2f}% of loss  "
          f"per-row loss x{(loss / n) / (total / len(rows)):.2f} of average")
