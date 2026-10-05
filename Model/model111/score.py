"""Score exact OCR completions under real and label-independently shuffled images."""
import argparse
import json
import math
import time
from common import ROOT, OUT, TRUTH, check_judge, sha256, save_json
from road_htr.vlm_cli import load_config, _load_inference_model
from road_htr.preprocess import load_rgb


def completion_inputs(prefix, target_ids):
    """Feed prefix + all but final target; last N positions predict N target IDs."""
    import torch
    if prefix["input_ids"].shape[0] != 1 or len(target_ids) == 0:
        raise ValueError("Expected one prompt and a nonempty completion")
    inputs = dict(prefix)
    device = prefix["input_ids"].device
    targets = torch.tensor([target_ids], dtype=torch.long, device=device)
    inputs["input_ids"] = torch.cat((prefix["input_ids"], targets[:, :-1]), dim=1)
    inputs["attention_mask"] = torch.cat((prefix["attention_mask"], torch.ones_like(targets[:, :-1])), dim=1)
    if "token_type_ids" in inputs or "position_ids" in inputs:
        raise ValueError("Unexpected position/type IDs require explicit extension")
    return inputs, targets


def likelihood(model, processor, prefix, text):
    import torch
    tokenizer = processor.tokenizer
    target_ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.decode(target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
        raise ValueError("Candidate tokenization does not preserve the exact transcription")
    eos = tokenizer.eos_token_id
    if eos is None or not target_ids or any(i in set(tokenizer.all_special_ids) for i in target_ids):
        raise ValueError("Need ordinary candidate tokens and an explicit EOS")
    inputs, targets = completion_inputs(prefix, target_ids+[eos])
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False, logits_to_keep=targets.shape[1]).logits
        if logits.shape[:2] != targets.shape:
            raise ValueError("Model did not return exactly the completion prediction positions")
        losses = torch.nn.functional.cross_entropy(logits[0].float(), targets[0], reduction="none")
        values = losses.cpu().tolist()
    if any(not math.isfinite(value) for value in values): raise ValueError("Nonfinite candidate NLL")
    return {"mean_nll": sum(values)/len(values), "sum_nll": sum(values), "tokens_with_eos": len(values),
            "eos_nll": values[-1]}


def make_prefix(model, processor, image, prompt):
    import torch
    message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt")
    device = next(p.device for p in model.parameters() if p.device.type != "meta")
    return {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
            if hasattr(value, "to") else value for key, value in inputs.items()}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", choices=["model73", "model107"], required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1: raise ValueError("Invalid smoke limit")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA not available in this execution context")
    torch.set_num_threads(2)
    root = OUT / "fold0"
    request_path = root / "requests.json"
    request = json.loads(request_path.read_text())
    witness = check_judge(args.judge)
    if witness != request["judges"][args.judge] or request["truth_sha256"] != sha256(TRUTH):
        raise ValueError("Frozen request/judge provenance changed")
    rows = request["rows"]
    if args.limit:
        rows = [r for r in rows if len(set(r["candidates"].values())) > 1][:args.limit]
    name = args.judge if args.limit is None else f"smoke{args.limit}_{args.judge}"
    output = root / f"{name}_scores.jsonl"
    receipt_path = root / f"{name}_scores.manifest.json"
    signature = {"requests_sha256": sha256(request_path), "judge": witness,
                 "score_code_sha256": sha256(__file__), "common_code_sha256": sha256(ROOT / "model111/common.py"),
                 "limit": args.limit, "torch": torch.__version__}
    complete = []
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if any(old.get(k) != v for k,v in signature.items()): raise ValueError("Cannot resume a different scoring protocol")
        if old.get("complete"):
            if sha256(output) != old["output_sha256"]: raise ValueError("Completed output checksum mismatch")
            print("Already complete", flush=True); return
        if output.exists():
            complete = [json.loads(line) for line in output.read_text().splitlines()]
        if [r["id"] for r in complete] != [r["id"] for r in rows[:len(complete)]]:
            raise ValueError("Resume IDs differ from frozen prefix")
    elif output.exists():
        raise ValueError("Unreceipted output exists")
    save_json(receipt_path, {**signature, "complete": False})
    config, _ = load_config(ROOT / witness["config"])
    model, processor = _load_inference_model(config, ROOT / witness["adapter"])
    start_time = time.monotonic()
    with output.open("a") as handle:
        for row in rows[len(complete):]:
            if row["id"] == row["control_id"]: raise ValueError("Control image is not shuffled")
            result = {"id": row["id"], "control_id": row["control_id"], "scores": {}}
            if len(set(row["candidates"].values())) > 1:
                for condition, image_key, hash_key in (("real", "image", "image_sha256"),
                                                        ("shuffled", "control_image", "control_sha256")):
                    path = ROOT / row[image_key]
                    if sha256(path) != row[hash_key]: raise ValueError("Image checksum mismatch")
                    prefix = make_prefix(model, processor, load_rgb(path), config.get("prompt", ""))
                    cache = {text: likelihood(model, processor, prefix, text) for text in sorted(set(row["candidates"].values()))}
                    result["scores"][condition] = {name: cache[text] for name,text in row["candidates"].items()}
            handle.write(json.dumps(result)+"\n"); handle.flush()
            complete.append(result)
            if len(complete) % 10 == 0 or args.limit:
                print(f"{args.judge}: {len(complete)}/{len(rows)} rows, {time.monotonic()-start_time:.1f}s", flush=True)
    save_json(receipt_path, {**signature, "complete": True, "rows": len(rows), "output_sha256": sha256(output),
                             "seconds_this_run": time.monotonic()-start_time})
    print(f"Completed {args.judge}: {len(rows)} rows", flush=True)


if __name__ == "__main__": main()
