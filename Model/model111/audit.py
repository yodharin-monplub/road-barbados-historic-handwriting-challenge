"""Audit saved OOF changes; test changes have no correctness labels."""
from collections import Counter
import csv
import json
import math
import re
from common import ROOT, OUT, TRUTH, predictions, read_table, sha256, save_json
from road_htr.metrics import competition_error, levenshtein, strip_conversation_tokens


def category(a, b):
    if a.casefold() == b.casefold(): return "case_only"
    if re.sub(r"[^\w\s]", "", a) == re.sub(r"[^\w\s]", "", b): return "punctuation_only"
    if re.findall(r"\d+", a) != re.findall(r"\d+", b): return "includes_numeric_change"
    return "word_or_mixed_change"


def csv_output(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = read_table(TRUTH, require_target=True)
    truth = {r[table.id_column]: strip_conversation_tokens(r[table.target_column]) for r in table.rows}
    before = predictions("outputs/model104/full_oof/crossfit_predictions.csv")
    after = predictions("outputs/model107/selector_oof/crossfit_predictions.csv")
    if set(before) != set(after) or set(before) != set(truth): raise ValueError("OOF IDs differ")
    wd = sum(math.sqrt(max(1, len(t.split()))) for t in truth.values())
    cd = sum(math.sqrt(max(1, len(t))) for t in truth.values())
    def cost(t, p):
        p = strip_conversation_tokens(p)
        return .5*(levenshtein(t.split(), p.split())*math.sqrt(max(1,len(t.split())))/wd/12 +
                   levenshtein(t, p)*math.sqrt(max(1,len(t)))/cd/55)
    changes = []
    folds = {r[table.id_column]: int(r["fold"]) for r in table.rows}
    for identifier in before:
        a, b, t = before[identifier], after[identifier], truth[identifier]
        if a == b: continue
        gain = cost(t,a)-cost(t,b)
        changes.append({"ID": identifier, "fold": folds[identifier], "before": a, "after": b, "truth": t,
                        "category": category(a,b), "score_contribution_gain": gain,
                        "outcome": "improved" if gain > 1e-15 else "worsened" if gain < -1e-15 else "tied",
                        "broke_exact_match": a == t and b != t, "fixed_exact_match": b == t and a != t})
    csv_output(OUT / "validation_changes.csv", changes)
    ta = predictions("submissions/model104_model105_selector.csv")
    tb = predictions("submissions/model107_stackmix_selector.csv")
    if set(ta) != set(tb): raise ValueError("Submission IDs differ")
    test = [{"ID": i, "before": ta[i], "after": tb[i], "category": category(ta[i],tb[i])}
            for i in ta if ta[i] != tb[i]]
    csv_output(OUT / "test_changes.csv", test)
    report = {"oof_changed_lines": len(changes), "outcomes": dict(Counter(r["outcome"] for r in changes)),
              "broke_exact_matches": sum(r["broke_exact_match"] for r in changes),
              "fixed_exact_matches": sum(r["fixed_exact_match"] for r in changes),
              "oof_gain": sum(r["score_contribution_gain"] for r in changes),
              "by_category": {c: {"rows": sum(r["category"] == c for r in changes),
                  "gain": sum(r["score_contribution_gain"] for r in changes if r["category"] == c)}
                  for c in sorted({r["category"] for r in changes})},
              "test_changed_lines": len(test), "test_change_categories": dict(Counter(r["category"] for r in test)),
              "public_before": .909699454, "public_after": .907264706,
              "notes": ["Test labels are unavailable; no test correctness claims.",
                        "Reused upstream selector hierarchy is not fully nested; OOF audit is diagnostic.",
                        "OOF uses fold-specific models; deployment uses all-data refits, another possible source of mismatch."]}
    save_json(OUT / "audit_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
