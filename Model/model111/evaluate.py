"""Score fixed image-likelihood rules; never fit rules on the evaluated labels."""
import csv
import argparse
import json
import math
from common import ROOT, OUT, TRUTH, choose, read_table, sha256, save_json
from road_htr.metrics import competition_error, levenshtein, strip_conversation_tokens


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    fold = args.fold
    root = OUT / f"fold{fold}"
    requests = json.loads((root / "requests.json").read_text())
    if requests["truth_sha256"] != sha256(TRUTH) or requests["fold"] != fold: raise ValueError("Truth/fold changed")
    rows = requests["rows"]
    ids = [r["id"] for r in rows]
    judges = {}
    for name in ("model73", "model107"):
        path = root / f"{name}_scores.jsonl"
        receipt = json.loads((root / f"{name}_scores.manifest.json").read_text())
        if not receipt["complete"] or receipt["limit"] is not None or receipt["rows"] != 819:
            raise ValueError("Need both complete full-fold scoring runs")
        if receipt["requests_sha256"] != sha256(root / "requests.json") or receipt["output_sha256"] != sha256(path):
            raise ValueError("Scoring artifact checksum mismatch")
        if receipt["judge"] != requests["judges"][name]: raise ValueError("Wrong scoring judge")
        if receipt["score_code_sha256"] != sha256(ROOT/"model111/score.py") or receipt["common_code_sha256"] != sha256(ROOT/"model111/common.py"):
            raise ValueError("Scoring/selection rule changed from the pilot")
        loaded = [json.loads(line) for line in path.read_text().splitlines()]
        if [r["id"] for r in loaded] != ids: raise ValueError("Scoring IDs differ")
        judges[name] = {r["id"]: r for r in loaded}
    table = read_table(TRUTH, require_target=True)
    references = {r[table.id_column]: r[table.target_column] for r in table.rows if int(r["fold"]) == fold}
    if set(references) != set(ids): raise ValueError("Canonical heldout IDs differ")
    truth = [references[i] for i in ids]
    results = {}
    for name in ("model73", "model107", "model104"):
        results[name] = [r["candidates"][name] for r in rows]
    choices = []
    for pair, primary, challenger in (("clean", "model107", "model73"), ("diagnostic", "model104", "model107")):
        for condition in ("real", "shuffled"):
            for conservative in (False, True):
                key = f"{pair}_{condition}_{'both_judges' if conservative else 'mean_nll'}"
                output = []
                for row in rows:
                    texts = row["candidates"]
                    if texts[primary] == texts[challenger]:
                        selected = primary
                    else:
                        nlls = [(j[row["id"]]["scores"][condition][primary]["mean_nll"],
                                 j[row["id"]]["scores"][condition][challenger]["mean_nll"]) for j in judges.values()]
                        selected = choose(primary, challenger, nlls, conservative)
                    output.append(texts[selected])
                    choices.append({"ID": row["id"], "rule": key, "selected": selected, "prediction": texts[selected]})
                results[key] = output
    scores = {key: competition_error(truth, values) for key, values in results.items()}
    clean_gain = scores["clean_real_mean_nll"]["multi_score"]-scores["model107"]["multi_score"]
    visual_gain = scores["clean_real_mean_nll"]["multi_score"]-scores["clean_shuffled_mean_nll"]["multi_score"]
    # Recompute the true metric on paired row bootstrap samples. This interval is
    # conditional on existing predictions; it does not account for model selection.
    normalized = [strip_conversation_tokens(t) for t in truth]
    weights_w = np.array([math.sqrt(max(1,len(t.split()))) for t in normalized])
    weights_c = np.array([math.sqrt(max(1,len(t))) for t in normalized])
    dw, dc = [], []
    for t, base, candidate in zip(normalized, results["model107"], results["clean_real_mean_nll"]):
        base, candidate = strip_conversation_tokens(base), strip_conversation_tokens(candidate)
        dw.append(levenshtein(t.split(), base.split())-levenshtein(t.split(), candidate.split()))
        dc.append(levenshtein(t, base)-levenshtein(t, candidate))
    dw, dc = np.array(dw)*weights_w, np.array(dc)*weights_c
    rng = np.random.default_rng(20260908)
    boot = []
    for _ in range(5000):
        sample = rng.integers(0,len(rows),len(rows))
        boot.append(.5*(dw[sample].sum()/weights_w[sample].sum()/12 + dc[sample].sum()/weights_c[sample].sum()/55))
    sensitivity = {}
    for name, judge in judges.items():
        gaps = [row["scores"]["shuffled"][candidate]["mean_nll"]-row["scores"]["real"][candidate]["mean_nll"]
                for row in judge.values() if row["scores"] for candidate in ("model73", "model107")]
        sensitivity[name] = float(np.mean(gaps))
    report = {"rows": len(rows), "fold": fold, "scores": scores, "clean_primary_gain": clean_gain,
              "clean_real_vs_shuffled_gain": visual_gain, "conditional_paired_bootstrap_95": np.quantile(boot,[.025,.975]).tolist(),
              "mean_nll_reduction_with_correct_image": sensitivity,
              "passes_development_screen": clean_gain >= .001 and visual_gain > 0,
              "diagnostic_gain_vs_model104": scores["diagnostic_real_mean_nll"]["multi_score"]-scores["model104"]["multi_score"],
              "submission_ready": False,
              "notes": ["Mean-NLL rule and equal judge weights were fixed before scoring; both-judge rule is a registered secondary diagnostic.",
                        "Model104 comparisons reuse a non-fully-nested upstream ensemble and are diagnostic only.",
                        "Fold0 was used in historical model development; no untouched-holdout claim.",
                        "Shuffled images measure sensitivity, not a text-only baseline. Full-fold confirmation is required before deployment."]}
    with (root / "choices.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(choices[0])); writer.writeheader(); writer.writerows(choices)
    for key in ("clean_real_mean_nll", "diagnostic_real_mean_nll"):
        with (root / f"{key}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["ID", "prediction"]); writer.writerows(zip(ids,results[key]))
    save_json(root / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
