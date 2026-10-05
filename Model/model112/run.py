"""Frozen four-model ensemble comparison and error-diversity audit."""
import csv
import itertools
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fusion import fuse, NAMES
from road_htr.stackmix import sha256,save_json
from road_htr.schema import read_table,predictions_by_id
from road_htr.metrics import competition_error,levenshtein,strip_conversation_tokens

OUT=ROOT/"outputs/model112"
TRUTH=ROOT/"data/processed/train_folds.csv"
RECIPES={"family_word":(True,False),"majority_word":(False,False),"family_char":(True,True),"majority_char":(False,True)}


def read(path):
    table=read_table(path)
    return dict(predictions_by_id(table,"prediction" if "prediction" in table.fieldnames else table.target_column))


def sources(truth):
    result={name:{} for name in NAMES}; ledger={}
    assembly=json.loads((ROOT/"outputs/model104/model92_step400_oof.report.json").read_text())
    if assembly["truth_sha256"]!=sha256(TRUTH): raise ValueError("Model92 assembly truth changed")
    kr=json.loads((ROOT/"outputs/model35/allfold_epoch07_verified/report.json").read_text())
    if kr["truth_sha256"]!=sha256(TRUTH): raise ValueError("Kraken truth changed")
    for fold in range(5):
        ids={i for i,r in truth.items() if int(r["fold"])==fold}
        for name in NAMES:
            if name=="kraken":
                path=ROOT/f"outputs/model35/fold{fold}_epoch07_verified/predictions/03_confusions_fold{fold}.csv"
                report_path=ROOT/f"outputs/model35/fold{fold}_epoch07_verified/report.json"
                if sha256(report_path)!=kr["fold_reports"][str(fold)]["report_sha256"]: raise ValueError("Kraken fold report changed")
                report=json.loads(report_path.read_text())
                # The fold report binds every stage's output checksum.
                if sha256(path) not in json.dumps(report): raise ValueError("Kraken output not bound to fold receipt")
            else:
                path=ROOT/(f"outputs/model107/fold{fold}_oof.csv" if name=="model107" else f"outputs/{name}/fold{fold}_oof_step400.csv")
                receipt=path.with_suffix(".manifest.json")
                if name=="model92" and fold==0:
                    entry=next(r for r in assembly["folds"] if r["fold"]==fold)
                    if sha256(path)!=entry["sha256"]: raise ValueError("Legacy Model92 fold0 hash changed")
                else:
                    report=json.loads(receipt.read_text())
                    if report["fold"]!=fold or report["input_sha256"]!=sha256(TRUTH) or report["output_sha256"]!=sha256(path):
                        raise ValueError("Source fold/CSV/labels checksum mismatch")
            values=read(path)
            if set(values)!=ids or any(not v.strip() for v in values.values()): raise ValueError("Source IDs/blank prediction mismatch")
            result[name].update(values)
            ledger[f"{name}_fold{fold}"]={"path":str(path.relative_to(ROOT)),"sha256":sha256(path)}
    return result,ledger


def write_predictions(path,ids,values):
    with path.open("w",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["ID","prediction"]);writer.writerows((i,values[i]) for i in ids)


def main():
    import numpy as np
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"report.json").exists(): raise ValueError("Experiment complete; do not overwrite")
    table=read_table(TRUTH,require_target=True)
    truth={r[table.id_column]:r for r in table.rows};ids=sorted(truth)
    source,ledger=sources(truth)
    primary=read(ROOT/"outputs/model104/full_oof/crossfit_predictions.csv")
    if set(primary)!=set(ids): raise ValueError("Best-ensemble IDs differ")
    protocol={"recipes":list(RECIPES),"primary":"raw_family_word","rules":"three of four; family requires Kraken plus two LightOn models",
              "source_ledger":ledger,"best_ensemble_sha256":sha256(ROOT/"outputs/model104/full_oof/crossfit_predictions.csv"),
              "code_sha256":{name:sha256(ROOT/name) for name in ("model112/run.py","model112/fusion.py","model99/evaluate.py","src/road_htr/word_rover.py")},
              "truth_sha256":sha256(TRUTH)}
    frozen=OUT/"protocol.json"
    if frozen.exists() and json.loads(frozen.read_text())!=protocol: raise ValueError("Cannot change frozen experiment")
    save_json(frozen,protocol)
    all_predictions={**source,"best_ensemble":primary}
    for anchor_name,anchor in (("raw",source["model107"]),("diagnostic",primary)):
        for recipe,(family,character) in RECIPES.items():
            name=f"{anchor_name}_{recipe}"
            values={i:fuse(anchor[i],{n:source[n][i] for n in NAMES},family,character) for i in ids}
            all_predictions[name]=values
            write_predictions(OUT/f"{name}_oof.csv",ids,values)
            print(f"Generated {name}; changed {sum(values[i]!=anchor[i] for i in ids)} lines",flush=True)
    refs=[truth[i][table.target_column] for i in ids]
    normalized=[strip_conversation_tokens(t) for t in refs]
    ww=np.array([math.sqrt(max(1,len(t.split()))) for t in normalized]);cw=np.array([math.sqrt(max(1,len(t))) for t in normalized])
    costs={};errors={}
    for name,values in all_predictions.items():
        w=np.array([levenshtein(t.split(),strip_conversation_tokens(values[i]).split()) for t,i in zip(normalized,ids)])*ww
        c=np.array([levenshtein(t,strip_conversation_tokens(values[i])) for t,i in zip(normalized,ids)])*cw
        errors[name]=(w,c);costs[name]=.5*(w/ww.sum()/12+c/cw.sum()/55)
    pairs=[]
    for a,b in itertools.combinations(NAMES,2):
        delta=costs[a]-costs[b]
        pairs.append({"a":a,"b":b,"different_transcripts":sum(source[a][i]!=source[b][i] for i in ids),
                      "b_better_rows":int((delta>1e-15).sum()),"a_better_rows":int((delta< -1e-15).sum()),
                      "error_cost_correlation":float(np.corrcoef(costs[a],costs[b])[0,1])})
    oracle={i:source[min(NAMES,key=lambda n:costs[n][index])][i] for index,i in enumerate(ids)}
    reports={}
    for name,values in all_predictions.items():
        baseline="model107" if name.startswith("raw_") else "best_ensemble"
        w,c=errors[name];bw,bc=errors[baseline]
        per_fold={}
        for fold in range(5):
            mask=np.array([int(truth[i]["fold"])==fold for i in ids])
            score=1-.5*(w[mask].sum()/ww[mask].sum()/12+c[mask].sum()/cw[mask].sum()/55)
            base=1-.5*(bw[mask].sum()/ww[mask].sum()/12+bc[mask].sum()/cw[mask].sum()/55)
            per_fold[str(fold)]={"score":float(score),"gain":float(score-base)}
        report={"score":float(1-costs[name].sum()),"comparison_baseline":baseline,
                "gain":float((costs[baseline]-costs[name]).sum()),"per_fold":per_fold}
        if name.startswith(("raw_","diagnostic_")):
            mask=np.array([int(truth[i]["fold"])!=0 for i in ids]);indices=np.flatnonzero(mask)
            rng=np.random.default_rng(20260910);boot=[]
            for _ in range(3000):
                ix=rng.choice(indices,size=len(indices),replace=True)
                boot.append(.5*((bw[ix]-w[ix]).sum()/ww[ix].sum()/12+(bc[ix]-c[ix]).sum()/cw[ix].sum()/55))
            report["folds1_to4_gain"]=float(.5*((bw[mask]-w[mask]).sum()/ww[mask].sum()/12+(bc[mask]-c[mask]).sum()/cw[mask].sum()/55))
            report["conditional_bootstrap_95"]=np.quantile(boot,[.025,.975]).tolist()
            report["changed_lines"]=sum(values[i]!=all_predictions[baseline][i] for i in ids)
        reports[name]=report
    result={"rows":len(ids),"results":reports,"pairwise_diversity":pairs,
            "label_oracle_score_NOT_achievable_validation":competition_error(refs,[oracle[i] for i in ids])["multi_score"],
            "submission_ready":False,"gpu_used":False,
            "limitations":["All results are development evidence; source recipes were previously selected using these folds.",
                            "Model92 fold0 has only the assembly hash witness locally, not its original inference receipt.",
                            "Best-ensemble-anchored results inherit its non-fully-nested upstream selection history.",
                            "Oracle uses labels and is an upper-bound diagnostic only; it is not a deployable method."]}
    save_json(OUT/"report.json",result)
    print(json.dumps({name:{k:v for k,v in r.items() if k!="per_fold"} for name,r in reports.items()},indent=2))


if __name__=="__main__": main()
