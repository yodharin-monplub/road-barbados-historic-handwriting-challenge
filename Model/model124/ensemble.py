#!/usr/bin/env python3
"""Model124: multi-member ensemble, validated out-of-fold and applied to test.

OOF: for every fold, vote the members' held-out predictions for that fold
(anchor = the first member), merge the five folds and score the result. Each
row is predicted only by models that never saw it, so the number is a genuine
complete cross-validation of the ensemble rule.

TEST: each member first collapses its five fold models into one vector by a
strict majority vote, then the members vote with the same rule as the OOF
validation, so the deployed rule is the validated one.

  python model124/ensemble.py oof  --members model122_prod model122_prod_lora model123_qwen3vl4b_lora
  python model124/ensemble.py test --members ... --output submissions/name.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "AGENTS.md").exists():  # layout-independent project root
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "Other" / "src"))
sys.path.insert(0, str(ROOT / "Model" / "model124"))

from road_htr.metrics import competition_error  # noqa: E402
from vote import read_predictions, vote_all  # noqa: E402


def truth() -> dict[str, str]:
    with (ROOT / "Data/processed/train_folds.csv").open(encoding="utf-8") as handle:
        return {row["ID"]: row["Target"] for row in csv.DictReader(handle)}


def score(predictions: dict[str, str]) -> dict[str, float]:
    reference = truth()
    ids = [i for i in predictions if i in reference]
    return competition_error([reference[i] for i in ids], [predictions[i] for i in ids])


def find(member: str, fold: int, kind: str) -> Path | None:
    """Locate a member's file for one fold, preferring beam over greedy."""
    base = ROOT / "Other" / "outputs" / member
    patterns = ([f"fold{fold}*/oof_final_beam*.csv", f"fold{fold}*/oof_ep*_greedy.csv"]
                if kind == "oof" else [f"fold{fold}*/test_beam*.csv"])
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[-1]
    return None


def member_fold_vectors(member: str, kind: str) -> dict[int, dict[str, str]]:
    vectors = {}
    for fold in range(5):
        path = find(member, fold, kind)
        if path is not None:
            vectors[fold] = read_predictions(path)
    return vectors


def command_oof(args) -> None:
    members = {m: member_fold_vectors(m, "oof") for m in args.members}
    for name, folds in members.items():
        print(f"{name}: OOF folds {sorted(folds)}")
    anchor_name = args.members[0]
    merged_anchor: dict[str, str] = {}
    merged_voted: dict[str, str] = {}
    for fold in range(5):
        if fold not in members[anchor_name]:
            continue
        anchor = members[anchor_name][fold]
        sources = [members[m][fold] for m in args.members[1:] if fold in members[m]]
        merged_anchor.update(anchor)
        merged_voted.update(vote_all(anchor, sources, args.min_votes))
    result = {
        "members": args.members,
        "min_votes": args.min_votes,
        "rows": len(merged_voted),
        "anchor_only": score(merged_anchor)["multi_score"],
        "ensemble": score(merged_voted)["multi_score"],
    }
    result["gain"] = result["ensemble"] - result["anchor_only"]
    result["detail"] = score(merged_voted)
    print(json.dumps(result, indent=2))
    out = ROOT / "Other" / "outputs" / "model124"
    out.mkdir(parents=True, exist_ok=True)
    (out / "oof_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "oof_ensemble.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "prediction"])
        for key, value in merged_voted.items():
            writer.writerow([key, value])


def member_test_consensus(member: str, min_votes: int) -> dict[str, str] | None:
    vectors = member_fold_vectors(member, "test")
    if not vectors:
        return None
    folds = sorted(vectors)
    anchor = vectors[folds[0]]
    sources = [vectors[f] for f in folds[1:]]
    print(f"{member}: test folds {folds}")
    return vote_all(anchor, sources, min_votes) if sources else dict(anchor)


def command_test(args) -> None:
    consensus = {}
    for member in args.members:
        vector = member_test_consensus(member, args.fold_min_votes)
        if vector is not None:
            consensus[member] = vector
    names = [m for m in args.members if m in consensus]
    if not names:
        raise SystemExit("no member produced test predictions")
    anchor = consensus[names[0]]
    voted = vote_all(anchor, [consensus[m] for m in names[1:]], args.min_votes)
    for key, value in list(voted.items()):
        if not value.strip():
            candidates = [anchor.get(key, "")] + [consensus[m].get(key, "") for m in names[1:]]
            voted[key] = next((c for c in candidates if c.strip()), "")
    tmp = ROOT / "Other" / "outputs" / "model124" / "test_ensemble.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "prediction"])
        for key, value in voted.items():
            writer.writerow([key, value])
    print(json.dumps({"members": names, "rows": len(voted),
                      "changed_vs_anchor": sum(1 for k in anchor if anchor[k] != voted[k]),
                      "written": str(tmp)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("oof", "test"):
        p = sub.add_parser(name)
        p.add_argument("--members", nargs="+", required=True)
        p.add_argument("--min-votes", type=int, default=2)
        if name == "test":
            p.add_argument("--fold-min-votes", type=int, default=3)
            p.add_argument("--output")
        p.set_defaults(func=command_oof if name == "oof" else command_test)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
