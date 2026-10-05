"""Paired local greedy/beam inference of two original Model112 voters."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"model114"))
import pilot

OUT=ROOT/"outputs/model120"
PROTOCOL=OUT/"protocol.json"
ADAPTERS={"model73":ROOT/"outputs/model73/fold0/checkpoint-400",
          "model107":ROOT/"outputs/model107/train/fold0/best_adapter"}
ORIGINALS={"model73":ROOT/"outputs/model73/fold0_oof_step400.csv",
           "model107":ROOT/"outputs/model107/fold0_oof.csv"}
JOBS=tuple(f"{name}_{mode}" for name in ADAPTERS for mode in ("greedy","beam"))


def verify_original(name):
    path=ORIGINALS[name]
    receipt=json.loads(path.with_suffix(".manifest.json").read_text())
    if receipt["fold"]!=0 or receipt["rows"]!=819 or receipt["input_sha256"]!=pilot.sha256(pilot.TRUTH) or receipt["output_sha256"]!=pilot.sha256(path):
        raise ValueError("Original source OOF provenance mismatch")
    for file,expected in receipt["adapter_files_sha256"].items():
        if pilot.sha256(ADAPTERS[name]/file)!=expected: raise ValueError(f"Changed adapter {name}/{file}")
    return receipt


def prepare():
    p=pilot.check()
    if PROTOCOL.exists(): raise RuntimeError("Protocol already exists")
    files=["model120/experiment.py","scripts/run_model120.py","model114/pilot.py","outputs/model114/protocol.json"]
    for name in ADAPTERS:
        receipt=verify_original(name)
        for mode in ("greedy","beam"):
            config_path=ROOT/f"model120/{name}_{mode}.json"
            config=json.loads(config_path.read_text())
            generation=dict(receipt["generation"])
            generation["num_beams"]=3 if mode=="beam" else 1
            if config["generation"]!=generation or config["prompt"]!=receipt["prompt"] or config["image_variant"]!=receipt["image_variant"] or config["seed"]!=receipt["seed"]:
                raise ValueError("Unintended inference setting change")
            if config["expected_files_sha256"]!=receipt["base_files_sha256"] or config["revision"]!=receipt["revision"]:
                raise ValueError("Base checkpoint mismatch")
            files.append(str(config_path.relative_to(ROOT)))
        files += [str(f.relative_to(ROOT)) for f in ADAPTERS[name].rglob("*") if f.is_file()]
        files.append(str(ORIGINALS[name].with_suffix(".manifest.json").relative_to(ROOT)))
    p["hashes"].update({f:pilot.sha256(ROOT/f) for f in files})
    p.update({"models":{"120":"Model73 and107 original adapters; paired greedy/beam3"},
              "primary":"both_members_beam3","minimum_gain_each_baseline":0.0005,
              "baselines":["original_model112","local_greedy_pair"],
              "unchanged_members":["model92","kraken"]})
    pilot.save_json(PROTOCOL,p)
    print("Both source adapters verified; paired four-pass protocol frozen",flush=True)


def check():
    pilot.PROTOCOL=PROTOCOL
    return pilot.check()


def read_candidate(job,ids):
    name=job.rsplit("_",1)[0]
    path=OUT/f"{job}.csv"
    receipt=json.loads(path.with_suffix(".manifest.json").read_text())
    config=ROOT/f"model120/{job}.json"
    if (receipt["fold"]!=0 or receipt["rows"]!=819 or receipt["input_sha256"]!=pilot.sha256(pilot.TRUTH)
            or receipt["output_sha256"]!=pilot.sha256(path) or receipt["config_sha256"]!=pilot.sha256(config)
            or Path(receipt["adapter"]).resolve()!=ADAPTERS[name].resolve()): raise ValueError("New inference provenance mismatch")
    for file,expected in receipt["adapter_files_sha256"].items():
        if pilot.sha256(ADAPTERS[name]/file)!=expected: raise ValueError("Inference adapter mismatch")
    values=pilot.read_predictions(path)
    if set(values)!=set(ids): raise ValueError("Heldout coverage mismatch")
    return values


def evaluate():
    p=check()
    table,truth=pilot.canonical()
    ids=p["validation_ids"]
    candidates={j:read_candidate(j,ids) for j in JOBS}
    source,_=pilot.sources(truth)
    anchor=pilot.read_predictions(ROOT/"outputs/model104/full_oof/crossfit_predictions.csv")
    old=pilot.read_predictions(ROOT/"outputs/model112/diagnostic_family_word_oof.csv")
    ensembles={mode:{} for mode in ("greedy","beam")}
    for i in ids:
        original={n:source[n][i] for n in pilot.NAMES}
        if pilot.fuse(anchor[i],original,True,False)!=old[i]: raise ValueError("Model112 replay failed")
        for mode in ensembles:
            donors=dict(original)
            for name in ADAPTERS: donors[name]=candidates[f"{name}_{mode}"][i]
            ensembles[mode][i]=pilot.fuse(anchor[i],donors,True,False) if all(v.strip() for v in donors.values()) else anchor[i]
    refs=[truth[i][table.target_column] for i in ids]
    score=lambda v:pilot.competition_error(refs,[v[i] for i in ids])["multi_score"]
    scores={"original_model112":score(old),"local_greedy_pair":score(ensembles["greedy"]),"both_members_beam3":score(ensembles["beam"])}
    gains={baseline:scores["both_members_beam3"]-scores[baseline] for baseline in p["baselines"]}
    members={n:{"old_greedy_score":score(source[n]),"local_greedy_score":score(candidates[n+"_greedy"]),
                "beam_score":score(candidates[n+"_beam"]),
                "local_vs_old_changed_lines":sum(candidates[n+"_greedy"][i]!=source[n][i] for i in ids),
                "beam_vs_local_changed_lines":sum(candidates[n+"_beam"][i]!=candidates[n+"_greedy"][i] for i in ids)} for n in ADAPTERS}
    report={"model":120,"fold":0,"rows":819,"scores":scores,"gains":gains,"members":members,
            "advance_to_confirmation_screen":all(v>=p["minimum_gain_each_baseline"] for v in gains.values()),
            "empty_predictions":{j:sum(not v.strip() for v in rows.values()) for j,rows in candidates.items()},
            "changed_ensemble_lines":sum(ensembles["beam"][i]!=old[i] for i in ids),
            "protocol_sha256":pilot.sha256(PROTOCOL),"submission_ready":False,"limitations":p["limitations"]}
    if (OUT/"report.json").exists(): raise RuntimeError("Report already exists")
    for mode,values in ensembles.items(): pilot.save_json(OUT/f"{mode}_ensemble.json",values)
    pilot.save_json(OUT/"report.json",report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","check","evaluate",*JOBS))
    action=parser.parse_args().action
    if action=="prepare":prepare()
    check()
    if action in JOBS:
        name=action.rsplit("_",1)[0]
        output=OUT/f"{action}.csv"
        if output.exists():raise RuntimeError("Prediction file already exists")
        pilot.infer(ROOT/f"model120/{action}.json",pilot.TRUTH,output,adapter_path=ADAPTERS[name],fold=0)
    elif action=="evaluate":evaluate()


if __name__=="__main__":main()
