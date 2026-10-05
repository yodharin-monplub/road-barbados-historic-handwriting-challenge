"""Freeze label-blind candidate requests and shuffled-image controls."""
import hashlib
import json
from common import ROOT, OUT, TRUTH, JUDGES, check_judge, predictions, read_table, sha256, save_json
from road_htr.schema import ImageIndex


def main():
    witnesses = {name: check_judge(name) for name in JUDGES}
    table = read_table(TRUTH, require_target=True)
    ids = sorted(r[table.id_column] for r in table.rows if int(r["fold"]) == 0)
    maps = {name: predictions(info["predictions"]) for name, info in JUDGES.items()}
    for values in maps.values():
        if set(values) != set(ids): raise ValueError("Raw candidates do not match heldout fold")
    diagnostic = "outputs/model104/full_oof/crossfit_predictions.csv"
    maps["model104"] = predictions(diagnostic)
    order = sorted(ids, key=lambda identifier: hashlib.sha256(("model111-control-v1:"+identifier).encode()).hexdigest())
    controls = dict(zip(order, order[1:]+order[:1]))
    images = ImageIndex(ROOT / "data/raw/images")
    rows = []
    for identifier in ids:
        candidates = {name: values[identifier] for name, values in maps.items()}
        if any(not text.strip() for text in candidates.values()): raise ValueError("Empty candidate")
        image = images.resolve(identifier)
        control = images.resolve(controls[identifier])
        rows.append({"id": identifier, "image": str(image.relative_to(ROOT)), "image_sha256": sha256(image),
                     "control_id": controls[identifier], "control_image": str(control.relative_to(ROOT)),
                     "control_sha256": sha256(control), "candidates": candidates})
    manifest = {"fold": 0, "truth_sha256": sha256(TRUTH), "judges": witnesses,
                "candidate_hashes": {name: sha256(ROOT/info["predictions"]) for name, info in JUDGES.items()},
                "diagnostic_candidate_sha256": sha256(ROOT/diagnostic),
                "rule": "mean_per_token_nll_including_eos_equal_judge_weight", "rows": rows,
                "prepare_code_sha256": sha256(__file__)}
    destination = OUT / "fold0/requests.json"
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        raise ValueError("Refusing to change frozen pilot requests")
    save_json(destination, manifest)
    print(json.dumps({"rows": len(rows), "rows_needing_scoring": sum(len(set(r["candidates"].values()))>1 for r in rows)}))


if __name__ == "__main__": main()
