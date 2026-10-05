"""Verify and combine fixed-rule Model111 results; no fitting or deployment."""
import csv
import json
import math
from common import ROOT, OUT, TRUTH, read_table, sha256, save_json
from confirm import fixed_protocol
from road_htr.metrics import competition_error, levenshtein, strip_conversation_tokens


def bootstrap(truth, baseline, candidate):
    import numpy as np
    refs = [strip_conversation_tokens(t) for t in truth]
    ww = np.array([math.sqrt(max(1,len(t.split()))) for t in refs])
    cw = np.array([math.sqrt(max(1,len(t))) for t in refs])
    dw,dc = [],[]
    for t,a,b in zip(refs,baseline,candidate):
        a,b = strip_conversation_tokens(a),strip_conversation_tokens(b)
        dw.append(levenshtein(t.split(),a.split())-levenshtein(t.split(),b.split()))
        dc.append(levenshtein(t,a)-levenshtein(t,b))
    dw,dc = np.array(dw)*ww,np.array(dc)*cw
    rng = np.random.default_rng(20260908)
    samples = []
    for _ in range(5000):
        i = rng.integers(0,len(refs),len(refs))
        samples.append(.5*(dw[i].sum()/ww[i].sum()/12+dc[i].sum()/cw[i].sum()/55))
    return {"percentile_95":np.quantile(samples,[.025,.975]).tolist(),
            "positive_fraction":float(np.mean(np.array(samples)>0)),
            "note":"Conditional paired-row bootstrap; not independent of historical model development or record correlation."}


def main():
    protocol = fixed_protocol()
    if json.loads((OUT/"confirmation_protocol.json").read_text()) != protocol: raise ValueError("Protocol changed")
    truth_table = read_table(TRUTH,require_target=True)
    canonical = {r[truth_table.id_column]:r for r in truth_table.rows}
    values,folds,sources = {},{},{}
    for fold in range(5):
        root = OUT/f"fold{fold}"
        request = json.loads((root/"requests.json").read_text())
        ids = {r["id"] for r in request["rows"]}
        expected = {i for i,r in canonical.items() if int(r["fold"])==fold}
        if ids != expected or len(request["rows"])!=819 or request["fold"]!=fold or request["truth_sha256"]!=sha256(TRUTH):
            raise ValueError("Full-OOF request partition/provenance mismatch")
        sources[str(fold)] = {"requests_sha256":sha256(root/"requests.json"),"judges":{}}
        for judge in ("model73","model107"):
            path = root/f"{judge}_scores.jsonl"
            receipt = json.loads((root/f"{judge}_scores.manifest.json").read_text())
            if not receipt["complete"] or receipt["limit"] is not None or receipt["rows"]!=819:
                raise ValueError("Incomplete fold scores")
            if receipt["output_sha256"]!=sha256(path) or receipt["requests_sha256"]!=sources[str(fold)]["requests_sha256"]:
                raise ValueError("Score receipt checksum mismatch")
            if receipt["score_code_sha256"]!=protocol["score_code_sha256"] or receipt["common_code_sha256"]!=protocol["common_code_sha256"]:
                raise ValueError("Confirmation scoring/selection implementation differs from pilot")
            if receipt["judge"]!=request["judges"][judge]: raise ValueError("Scoring judge differs")
            sources[str(fold)]["judges"][judge] = sha256(root/f"{judge}_scores.manifest.json")
        for row in request["rows"]:
            for name,text in row["candidates"].items(): values.setdefault(name,{})[row["id"]]=text
        with (root/"choices.csv").open() as handle:
            seen = set()
            for row in csv.DictReader(handle):
                if row["ID"] not in ids or (row["rule"],row["ID"]) in seen: raise ValueError("Choice IDs duplicated/out of fold")
                seen.add((row["rule"],row["ID"]))
                if row["prediction"] != values[row["selected"]][row["ID"]]: raise ValueError("Choice transcript was changed")
                values.setdefault(row["rule"],{})[row["ID"]] = row["prediction"]
        report = json.loads((root/"report.json").read_text())
        if report["fold"]!=fold: raise ValueError("Wrong fold report")
        folds[str(fold)] = {"raw_model107":report["scores"]["model107"]["multi_score"],
                            "model111":report["scores"]["clean_real_mean_nll"]["multi_score"],
                            "gain":report["clean_primary_gain"],
                            "real_vs_shuffled_gain":report["clean_real_vs_shuffled_gain"],
                            "diagnostic_gain_vs_model104":report["diagnostic_gain_vs_model104"]}
    if any(set(v)!=set(canonical) for v in values.values()): raise ValueError("Incomplete combined predictions")
    results = {}
    for label,ids in (("all_five_folds",sorted(canonical)),
                       ("confirmation_folds_1_to_4",sorted(i for i,r in canonical.items() if int(r["fold"])!=0))):
        truth = [canonical[i][truth_table.target_column] for i in ids]
        scores = {name:competition_error(truth,[v[i] for i in ids])["multi_score"] for name,v in values.items()}
        results[label] = {"rows":len(ids),"scores":scores,"raw_gain":scores["clean_real_mean_nll"]-scores["model107"],
                          "real_vs_shuffled_gain":scores["clean_real_mean_nll"]-scores["clean_shuffled_mean_nll"],
                          "diagnostic_gain_vs_model104":scores["diagnostic_real_mean_nll"]-scores["model104"],
                          "paired_bootstrap":bootstrap(truth,[values["model107"][i] for i in ids],
                                                       [values["clean_real_mean_nll"][i] for i in ids])}
    confirmation = results["confirmation_folds_1_to_4"]
    positive = sum(folds[str(f)]["gain"]>0 for f in range(1,5))
    gate = protocol["gate"]
    passed = (confirmation["raw_gain"]>=gate["minimum_confirmation_gain"] and positive>=gate["minimum_positive_folds"]
              and confirmation["paired_bootstrap"]["percentile_95"][0]>0 and confirmation["real_vs_shuffled_gain"]>0)
    report = {"results":results,"by_fold":folds,"positive_confirmation_folds":positive,"passes_raw_confirmation":passed,
              "submission_ready":False,"cloud_used":False,"sources":sources,
              "notes":["No rule/threshold/weight fitting during confirmation; secondary rules remain diagnostic.",
                       "Model104 candidate has a reused non-fully-nested upstream selector hierarchy; its comparisons are diagnostic only.",
                       "Confirmation against raw Model107 does not establish improvement over the best submitted ensemble.",
                       "No test predictions, cloud rental, or Zindi submission were triggered."]}
    for rule in ("clean_real_mean_nll","diagnostic_real_mean_nll"):
        with (OUT/f"{rule}_full_oof.csv").open("w",newline="") as handle:
            writer=csv.writer(handle);writer.writerow(["ID","prediction"]);writer.writerows((i,values[rule][i]) for i in sorted(canonical))
    save_json(OUT/"full_oof_report.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
