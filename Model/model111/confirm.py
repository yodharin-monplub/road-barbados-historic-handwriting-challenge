"""Fold-parameterized confirmation using the unchanged pilot likelihood code."""
import argparse
import hashlib
import json
import time
from common import ROOT, OUT, TRUTH, predictions, read_table, sha256, save_json
from score import likelihood, make_prefix
from road_htr.schema import ImageIndex
from road_htr.vlm_cli import load_config, _load_inference_model
from road_htr.preprocess import load_rgb


def fold_info(name, fold):
    if name not in ("model73", "model107") or fold not in range(5):
        raise ValueError("Unknown judge/fold")
    if name == "model73":
        return {"adapter": f"outputs/model73/fold{fold}/checkpoint-400", "config": "model73/config.yaml",
                "receipt": f"outputs/model73/fold{fold}_oof_step400.manifest.json",
                "predictions": f"outputs/model73/fold{fold}_oof_step400.csv"}
    return {"adapter": f"outputs/model107/train/fold{fold}/best_adapter", "config": f"outputs/model107/config_fold{fold}.yaml",
            "receipt": f"outputs/model107/fold{fold}_oof.manifest.json", "predictions": f"outputs/model107/fold{fold}_oof.csv"}


def check_fold_judge(name, fold):
    info = fold_info(name, fold)
    receipt = json.loads((ROOT / info["receipt"]).read_text())
    if receipt["fold"] != fold or receipt["rows"] != 819 or receipt["input_sha256"] != sha256(TRUTH):
        raise ValueError("Judge receipt has wrong fold/labels")
    if receipt["output_sha256"] != sha256(ROOT/info["predictions"]) or receipt["config_sha256"] != sha256(ROOT/info["config"]):
        raise ValueError("Prediction/config checksum mismatch")
    config, _ = load_config(ROOT/info["config"])
    if receipt["base_files_sha256"] != config["expected_files_sha256"] or receipt["prompt"] != config.get("prompt", ""):
        raise ValueError("Base model/prompt provenance differs")
    if name == "model107":
        training = json.loads((ROOT/info["adapter"]).parent.joinpath("run_manifest.json").read_text())
        if training["config"]["validation_fold"] != fold or training["validation_rows"] != 819:
            raise ValueError("Model107 training partition differs")
    verified = {}
    for filename, expected in receipt["adapter_files_sha256"].items():
        if filename in ("optimizer.pt", "rng_state.pth", "scheduler.pt", "trainer_state.json", "training_args.bin", "README.md"):
            continue
        actual = sha256(ROOT/info["adapter"]/filename)
        if actual != expected: raise ValueError(f"Wrong fold/checkpoint inference asset: {filename}")
        verified[filename] = actual
    if "adapter_model.safetensors" not in verified: raise ValueError("Missing judge weights")
    return {**info, "adapter_files_sha256": verified, "receipt_sha256": sha256(ROOT/info["receipt"])}


def fixed_protocol():
    pilot = json.loads((OUT/"fold0/model73_scores.manifest.json").read_text())
    for key, path in (("score_code_sha256", "model111/score.py"), ("common_code_sha256", "model111/common.py")):
        if pilot[key] != sha256(ROOT/path): raise ValueError("Pilot scoring/selection implementation changed")
    return {"score_code_sha256": pilot["score_code_sha256"], "common_code_sha256": pilot["common_code_sha256"],
            "primary_rule": "mean_per_token_nll_including_eos_equal_judge_weight", "confirmation_folds": [1,2,3,4],
            "gate": {"minimum_confirmation_gain": .001, "minimum_positive_folds": 3,
                     "require_positive_bootstrap_lower": True, "require_better_than_shuffled": True},
            "submission_ready": False, "note": "Raw-model confirmation does not prove improvement over the best ensemble. No automatic deployment."}


def validate_requests(request, fold):
    table = read_table(TRUTH, require_target=True)
    expected = {r[table.id_column] for r in table.rows if int(r["fold"]) == fold}
    ids = [r["id"] for r in request["rows"]]
    if request["fold"] != fold or request["truth_sha256"] != sha256(TRUTH) or len(ids) != 819 or set(ids) != expected:
        raise ValueError("Request IDs/fold/labels differ from canonical heldout partition")
    maps = {name: predictions(fold_info(name,fold)["predictions"]) for name in ("model73", "model107")}
    maps["model104"] = predictions("outputs/model104/full_oof/crossfit_predictions.csv")
    for row in request["rows"]:
        if row["control_id"] not in expected or row["control_id"] == row["id"]:
            raise ValueError("Control must be a different same-fold image")
        if row["candidates"] != {name: values[row["id"]] for name,values in maps.items()}:
            raise ValueError("Frozen candidate text changed")
        for image_key, hash_key in (("image","image_sha256"), ("control_image","control_sha256")):
            path = (ROOT/row[image_key]).resolve()
            if not path.is_relative_to(ROOT/"data/raw/images") or sha256(path) != row[hash_key]:
                raise ValueError("Image path/hash mismatch")


def prepare(fold):
    if fold == 0: raise ValueError("Preserve completed pilot; prepare only confirmation folds")
    protocol = fixed_protocol()
    protocol_path = OUT/"confirmation_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Frozen confirmation protocol differs")
    save_json(protocol_path, protocol)
    witnesses = {name: check_fold_judge(name,fold) for name in ("model73", "model107")}
    table = read_table(TRUTH, require_target=True)
    ids = sorted(r[table.id_column] for r in table.rows if int(r["fold"]) == fold)
    maps = {name: predictions(info["predictions"]) for name,info in witnesses.items()}
    if any(set(values) != set(ids) for values in maps.values()): raise ValueError("Raw candidate fold mismatch")
    diagnostic = "outputs/model104/full_oof/crossfit_predictions.csv"
    maps["model104"] = predictions(diagnostic)
    order = sorted(ids, key=lambda i: hashlib.sha256(("model111-control-v1:"+i).encode()).hexdigest())
    controls = dict(zip(order,order[1:]+order[:1]))
    images = ImageIndex(ROOT/"data/raw/images")
    rows = []
    for identifier in ids:
        image, control = images.resolve(identifier), images.resolve(controls[identifier])
        candidates = {name:values[identifier] for name,values in maps.items()}
        if any(not text.strip() for text in candidates.values()): raise ValueError("Empty candidate")
        rows.append({"id": identifier, "image": str(image.relative_to(ROOT)), "image_sha256": sha256(image),
                     "control_id": controls[identifier], "control_image": str(control.relative_to(ROOT)),
                     "control_sha256": sha256(control), "candidates": candidates})
    request = {"fold":fold, "truth_sha256":sha256(TRUTH), "judges":witnesses, "rows":rows,
               "rule":protocol["primary_rule"], "protocol_sha256":sha256(protocol_path),
               "candidate_hashes":{name:sha256(ROOT/info["predictions"]) for name,info in witnesses.items()},
               "diagnostic_candidate_sha256":sha256(ROOT/diagnostic)}
    validate_requests(request,fold)
    path = OUT/f"fold{fold}/requests.json"
    if path.exists() and json.loads(path.read_text()) != request: raise ValueError("Refusing to change frozen requests")
    save_json(path,request)
    print(json.dumps({"fold":fold,"rows":len(rows),"rows_needing_scoring":sum(len(set(r["candidates"].values()))>1 for r in rows)}),flush=True)


def score_fold(fold, judge):
    import torch
    if fold == 0: raise ValueError("Do not overwrite completed pilot")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    protocol = fixed_protocol()
    root = OUT/f"fold{fold}"
    path = root/"requests.json"
    request = json.loads(path.read_text())
    validate_requests(request,fold)
    witness = check_fold_judge(judge,fold)
    if witness != request["judges"][judge] or request["protocol_sha256"] != sha256(OUT/"confirmation_protocol.json"):
        raise ValueError("Judge/protocol differs from prepared request")
    signature = {"requests_sha256":sha256(path),"judge":witness,"fold":fold,"limit":None,"torch":torch.__version__,
                 "score_code_sha256":protocol["score_code_sha256"],"common_code_sha256":protocol["common_code_sha256"],
                 "confirmation_code_sha256":sha256(__file__),"protocol_sha256":request["protocol_sha256"]}
    output = root/f"{judge}_scores.jsonl"
    receipt_path = root/f"{judge}_scores.manifest.json"
    complete = []
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if any(old.get(k) != v for k,v in signature.items()): raise ValueError("Cannot resume changed scoring protocol")
        if old.get("complete"):
            if sha256(output) != old["output_sha256"]: raise ValueError("Completed output checksum mismatch")
            print("Already complete",flush=True); return
        if output.exists(): complete = [json.loads(line) for line in output.read_text().splitlines()]
        if len(complete)>len(request["rows"]) or [r["id"] for r in complete] != [r["id"] for r in request["rows"][:len(complete)]]:
            raise ValueError("Resume prefix differs")
    elif output.exists(): raise ValueError("Unreceipted output exists")
    save_json(receipt_path,{**signature,"complete":False})
    config,_ = load_config(ROOT/witness["config"])
    model,processor = _load_inference_model(config, ROOT/witness["adapter"])
    started = time.monotonic()
    with output.open("a") as handle:
        for row in request["rows"][len(complete):]:
            result = {"id":row["id"],"control_id":row["control_id"],"scores":{}}
            if len(set(row["candidates"].values()))>1:
                for condition,key in (("real","image"),("shuffled","control_image")):
                    prefix = make_prefix(model,processor,load_rgb(ROOT/row[key]),config.get("prompt",""))
                    cache = {text:likelihood(model,processor,prefix,text) for text in sorted(set(row["candidates"].values()))}
                    result["scores"][condition] = {name:cache[text] for name,text in row["candidates"].items()}
            handle.write(json.dumps(result)+"\n"); handle.flush(); complete.append(result)
            if len(complete)%10==0:
                print(f"fold{fold} {judge}: {len(complete)}/819 rows, {time.monotonic()-started:.1f}s",flush=True)
    save_json(receipt_path,{**signature,"complete":True,"rows":819,"output_sha256":sha256(output),"seconds_this_run":time.monotonic()-started})
    print(f"Completed fold{fold} {judge}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","score"])
    parser.add_argument("--fold",type=int,choices=[1,2,3,4],required=True)
    parser.add_argument("--judge",choices=["model73","model107"])
    args = parser.parse_args()
    if args.command=="prepare": prepare(args.fold)
    else:
        if not args.judge: parser.error("score requires --judge")
        score_fold(args.fold,args.judge)
