"""Fold1-4 confirmation, fixed two-member beam policy and paired local control."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"model120"))
import experiment as original
pilot=original.pilot
OUT=ROOT/"outputs/model121"
PROTOCOL=OUT/"protocol.json"
JOBS=tuple(f"fold{f}_{n}_{m}" for f in range(1,5) for n in ("model73","model107") for m in ("greedy","beam"))


def adapter(fold,name):
    return ROOT/(f"outputs/model73/fold{fold}/checkpoint-400" if name=="model73" else f"outputs/model107/train/fold{fold}/best_adapter")


def old_prediction(fold,name):
    return ROOT/(f"outputs/model73/fold{fold}_oof_step400.csv" if name=="model73" else f"outputs/model107/fold{fold}_oof.csv")


def verify_receipt(path,fold,ad,config=None):
    r=json.loads(path.with_suffix(".manifest.json").read_text())
    if r["fold"]!=fold or r["rows"]!=819 or r["input_sha256"]!=pilot.sha256(pilot.TRUTH) or r["output_sha256"]!=pilot.sha256(path):
        raise ValueError("Prediction input/fold/output receipt mismatch")
    if config is not None and (r["config_sha256"]!=pilot.sha256(config) or Path(r["adapter"]).resolve()!=ad.resolve()):
        raise ValueError("Inference config/adapter receipt mismatch")
    for file,expected in r["adapter_files_sha256"].items():
        if pilot.sha256(ad/file)!=expected: raise ValueError(f"Changed adapter file: {ad/file}")
    return r


def prepare():
    p=original.check()
    if PROTOCOL.exists():raise RuntimeError("Protocol already exists")
    files=["model121/experiment.py","scripts/run_model121.py","model120/experiment.py",
           "outputs/model120/protocol.json","outputs/model120/report.json"]
    for fold in range(1,5):
        for name in ("model73","model107"):
            path=old_prediction(fold,name);ad=adapter(fold,name)
            r=verify_receipt(path,fold,ad)
            for mode in ("greedy","beam"):
                config=ROOT/f"model120/{name}_{mode}.json"
                c=json.loads(config.read_text());g=dict(r["generation"]);g["num_beams"]=3 if mode=="beam" else 1
                if c["generation"]!=g or c["prompt"]!=r["prompt"] or c["seed"]!=r["seed"] or c["expected_files_sha256"]!=r["base_files_sha256"]:
                    raise ValueError("Confirmation recipe differs from original adapter inference settings")
            files.append(str(path.with_suffix(".manifest.json").relative_to(ROOT)))
            files += [str(q.relative_to(ROOT)) for q in ad.rglob("*") if q.is_file()]
    p["hashes"].update({f:pilot.sha256(ROOT/f) for f in files})
    p.update({"models":{"121":"Unchanged Model120 policy, paired confirmation folds1-4"},
              "confirmation_folds":[1,2,3,4],"minimum_gain_each_baseline":0.0005,
              "minimum_nonnegative_folds":3,"bootstrap_samples":3000,"bootstrap_seed":20260921,
              "gate_exception":"Model120 missed its original paired-control screen; this is a separately documented research follow-up, not a retroactive pass."})
    pilot.save_json(PROTOCOL,p)
    print("Eight fold-excluded adapters verified;16 confirmation inference passes frozen",flush=True)


def check():
    pilot.PROTOCOL=PROTOCOL
    return pilot.check()


def parse_job(job):
    f,n,m=job.split("_")
    return int(f[4:]),n,m


def evaluate():
    import numpy as np
    from road_htr.metrics import strip_conversation_tokens,levenshtein
    p=check();table,truth=pilot.canonical();source,_=pilot.sources(truth)
    ids=sorted(i for i,r in truth.items() if int(r["fold"]) in p["confirmation_folds"])
    if len(ids)!=3276:raise ValueError("Wrong confirmation rows")
    anchor=pilot.read_predictions(ROOT/"outputs/model104/full_oof/crossfit_predictions.csv")
    old=pilot.read_predictions(ROOT/"outputs/model112/diagnostic_family_word_oof.csv")
    ensembles={"original_model112":{i:old[i] for i in ids},"local_greedy_pair":{},"both_members_beam3":{}}
    members={};empty={}
    for fold in p["confirmation_folds"]:
        fold_ids=[i for i in ids if int(truth[i]["fold"])==fold]
        predictions={}
        for name in ("model73","model107"):
            for mode in ("greedy","beam"):
                job=f"fold{fold}_{name}_{mode}";path=OUT/f"{job}.csv"
                verify_receipt(path,fold,adapter(fold,name),ROOT/f"model120/{name}_{mode}.json")
                values=pilot.read_predictions(path)
                if set(values)!=set(fold_ids):raise ValueError("Wrong fold prediction coverage")
                predictions[name,mode]=values;empty[job]=sum(not v.strip() for v in values.values())
                members[job]=pilot.competition_error([truth[i][table.target_column] for i in fold_ids],[values[i] for i in fold_ids])["multi_score"]
        for i in fold_ids:
            donors={n:source[n][i] for n in pilot.NAMES}
            if pilot.fuse(anchor[i],donors,True,False)!=old[i]:raise ValueError("Original Model112 replay failed")
            for mode,label in (("greedy","local_greedy_pair"),("beam","both_members_beam3")):
                changed=dict(donors)
                for name in ("model73","model107"):changed[name]=predictions[name,mode][i]
                ensembles[label][i]=pilot.fuse(anchor[i],changed,True,False) if all(v.strip() for v in changed.values()) else anchor[i]
    refs=[strip_conversation_tokens(truth[i][table.target_column]) for i in ids]
    weights=np.array([[max(len(r.split()),1)**.5,max(len(r),1)**.5] for r in refs])
    errors={}
    for name,values in ensembles.items():
        errors[name]=np.array([[levenshtein(r.split(),strip_conversation_tokens(values[i]).split()),
                               levenshtein(r,strip_conversation_tokens(values[i]))] for i,r in zip(ids,refs)])*weights
    def score(name,ix):
        return float(1-.5*np.sum(errors[name][ix].sum(axis=0)/weights[ix].sum(axis=0)/[12.,55.]))
    all_ix=np.arange(len(ids));scores={n:score(n,all_ix) for n in ensembles}
    # Independently cross-check the cached metric arithmetic against the official helper.
    for name,values in ensembles.items():
        direct=pilot.competition_error([truth[i][table.target_column] for i in ids],[values[i] for i in ids])["multi_score"]
        if abs(direct-scores[name])>1e-12:raise ValueError("Bootstrap metric implementation mismatch")
    gains={b:scores["both_members_beam3"]-scores[b] for b in p["baselines"]}
    per_fold={}
    for fold in p["confirmation_folds"]:
        ix=np.array([k for k,i in enumerate(ids) if int(truth[i]["fold"])==fold])
        fs={n:score(n,ix) for n in ensembles}
        per_fold[str(fold)]={"scores":fs,"gains":{b:fs["both_members_beam3"]-fs[b] for b in p["baselines"]}}
    rng=np.random.default_rng(p["bootstrap_seed"]);boot={b:[] for b in p["baselines"]}
    for _ in range(p["bootstrap_samples"]):
        ix=rng.integers(0,len(ids),size=len(ids));denom=weights[ix].sum(axis=0)
        for b in boot:boot[b].append(float(.5*np.sum((errors[b][ix]-errors["both_members_beam3"][ix]).sum(axis=0)/denom/[12.,55.])))
    ci={b:np.quantile(v,[.025,.975]).tolist() for b,v in boot.items()}
    passed=all(gains[b]>=p["minimum_gain_each_baseline"] and ci[b][0]>0 and
               sum(per_fold[str(f)]["gains"][b]>=0 for f in p["confirmation_folds"])>=p["minimum_nonnegative_folds"] for b in p["baselines"])
    report={"model":121,"rows":len(ids),"folds":p["confirmation_folds"],"scores":scores,"gains":gains,
            "per_fold":per_fold,"conditional_row_bootstrap_95":ci,"members":members,"empty_predictions":empty,
            "confirmation_passed":passed,"submission_ready":False,"protocol_sha256":pilot.sha256(PROTOCOL),
            "gate_exception":p["gate_exception"],"limitations":p["limitations"]+" Row bootstrap ignores possible within-page dependence."}
    if (OUT/"report.json").exists():raise RuntimeError("Completed report already exists")
    for name,values in ensembles.items():pilot.save_json(OUT/f"{name}.json",values)
    pilot.save_json(OUT/"report.json",report);print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("prepare","check","evaluate",*JOBS))
    action=parser.parse_args().action
    if action=="prepare":prepare()
    check()
    if action in JOBS:
        fold,name,mode=parse_job(action);path=OUT/f"{action}.csv"
        if path.exists():raise RuntimeError("Predictions already exist")
        pilot.infer(ROOT/f"model120/{name}_{mode}.json",pilot.TRUTH,path,adapter_path=adapter(fold,name),fold=fold)
    elif action=="evaluate":evaluate()


if __name__=="__main__":main()
