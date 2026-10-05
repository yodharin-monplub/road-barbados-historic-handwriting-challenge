"""Inference-only repetition-penalty ablation using the matched Model114 adapter."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model114"))
import pilot

OUT = ROOT / "outputs/model118"
CONFIG = ROOT / "model118/config.json"
PROTOCOL = OUT / "protocol.json"
ADAPTER = ROOT / "outputs/model114/fold0/best_adapter"


def prepare():
    original = pilot.check()
    if PROTOCOL.exists(): raise RuntimeError("Protocol already frozen")
    baseline = json.loads((ROOT / "model114/config.json").read_text())
    candidate = json.loads(CONFIG.read_text())
    if candidate["generation"]["repetition_penalty"] != 1.0: raise ValueError("Expected neutral penalty")
    candidate["generation"]["repetition_penalty"] = baseline["generation"]["repetition_penalty"]
    candidate["output_dir"] = baseline["output_dir"]
    if candidate != baseline: raise ValueError("Unexpected decoding/config change")
    paths = ["model118/config.json", "model118/experiment.py", "scripts/run_model118.py",
             "model114/pilot.py", "outputs/model114/protocol.json",
             "outputs/model114/fold0_oof_step400.csv", "outputs/model114/fold0_oof_step400.manifest.json",
             "outputs/model114/replacement_ensemble_fold0.csv"]
    paths += [str(p.relative_to(ROOT)) for p in ADAPTER.rglob("*") if p.is_file()]
    original["hashes"].update({p: pilot.sha256(ROOT / p) for p in paths})
    original.update({"models": {"118": "Model114 existing adapter; no training"},
                     "standalone_gain_gate": 0.001, "matched_control": "model114",
                     "intervention": "repetition_penalty1.05->1.0"})
    receipt = json.loads((ROOT / "outputs/model114/fold0_oof_step400.manifest.json").read_text())
    if Path(receipt["adapter"]).resolve() != ADAPTER.resolve(): raise ValueError("Wrong matched adapter")
    for name, expected in receipt["adapter_files_sha256"].items():
        if pilot.sha256(ADAPTER / name) != expected: raise ValueError("Matched adapter changed")
    pilot.save_json(PROTOCOL, original)
    print("Neutral repetition-penalty protocol frozen", flush=True)


def check():
    pilot.PROTOCOL = PROTOCOL
    return pilot.check()


def evaluate():
    p = check()
    path = OUT / "fold0_oof.csv"
    receipt = json.loads(path.with_suffix(".manifest.json").read_text())
    if (receipt["fold"] != 0 or receipt["rows"] != 819 or receipt["input_sha256"] != pilot.sha256(pilot.TRUTH)
            or receipt["output_sha256"] != pilot.sha256(path) or receipt["config_sha256"] != pilot.sha256(CONFIG)
            or Path(receipt["adapter"]).resolve() != ADAPTER.resolve()): raise ValueError("Inference provenance mismatch")
    for name, expected in receipt["adapter_files_sha256"].items():
        if pilot.sha256(ADAPTER / name) != expected: raise ValueError("Inference adapter changed")
    table, truth = pilot.canonical()
    ids = p["validation_ids"]
    candidate = pilot.read_predictions(path)
    matched = pilot.read_predictions(ROOT / "outputs/model114/fold0_oof_step400.csv")
    if set(candidate) != set(ids) or set(matched) != set(ids): raise ValueError("Wrong heldout coverage")
    source, _ = pilot.sources(truth)
    anchor = pilot.read_predictions(ROOT / "outputs/model104/full_oof/crossfit_predictions.csv")
    old = pilot.read_predictions(ROOT / "outputs/model112/diagnostic_family_word_oof.csv")
    matched_ensemble = pilot.read_predictions(ROOT / "outputs/model114/replacement_ensemble_fold0.csv")
    replacement = {}
    for i in ids:
        donors = {n: source[n][i] for n in pilot.NAMES}
        if pilot.fuse(anchor[i], donors, True, False) != old[i]: raise ValueError("Model112 replay mismatch")
        donors["model92"] = candidate[i]
        replacement[i] = pilot.fuse(anchor[i], donors, True, False) if candidate[i].strip() else anchor[i]
    refs = [truth[i][table.target_column] for i in ids]
    scores = {n: pilot.competition_error(refs, [v[i] for i in ids]) for n,v in
              (("candidate",candidate),("matched_model114",matched),("model112",old),
               ("replacement_ensemble",replacement),("model114_replacement_ensemble",matched_ensemble))}
    gain = scores["candidate"]["multi_score"]-scores["matched_model114"]["multi_score"]
    egain = scores["replacement_ensemble"]["multi_score"]-scores["model112"]["multi_score"]
    report = {"model":118,"fold":0,"rows":819,"scores":scores,"standalone_gain_vs_matched_control":gain,
              "ensemble_gain":egain,"advance_to_confirmation_screen":gain>=p["standalone_gain_gate"] and egain>=p["confirmation_screen_gain"],
              "changed_prediction_lines":sum(candidate[i]!=matched[i] for i in ids),
              "empty_predictions":sum(not candidate[i].strip() for i in ids),
              "protocol_sha256":pilot.sha256(PROTOCOL),"prediction_sha256":pilot.sha256(path),
              "submission_ready":False,"limitations":p["limitations"]}
    if (OUT/"report.json").exists(): raise RuntimeError("Report already exists")
    pilot.save_json(OUT/"replacement_predictions.json",replacement)
    pilot.save_json(OUT/"report.json",report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","check","infer","evaluate"))
    action=parser.parse_args().action
    if action=="prepare": prepare()
    check()
    if action=="infer":
        path=OUT/"fold0_oof.csv"
        if path.exists(): raise RuntimeError("Predictions already exist")
        pilot.infer(CONFIG,pilot.TRUTH,path,adapter_path=ADAPTER,fold=0)
    elif action=="evaluate": evaluate()


if __name__=="__main__": main()
