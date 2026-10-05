"""Build a frequent training-only vocabulary and reproducible style-conditioned requests."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"model108"))
from road_htr.stackmix import absolute,sha256,save_json,eligible_rows
from road_htr.vlm_data import read_id_file
from prepare import SUPPORTED,validate_requests


def main():
    config=json.loads((ROOT/"model110/config.json").read_text())
    bank_path=absolute(config["crop_bank"])
    bank=json.loads(bank_path.read_text())
    truth=absolute(config["truth"])
    table,eligible=eligible_rows(truth,config["fold"],read_id_file(ROOT/"config/known_bad_train_ids.txt"))
    counts=Counter(w for r in eligible.values() for w in r[table.target_column].split())
    groups=defaultdict(list)
    examples={}
    for word in bank["words"]:
        groups[word["source_id"]].append(word)
        text=word["text"]
        if len(text)<=18 and text and set(text)<=SUPPORTED:
            examples.setdefault(text,word)
    vocab=sorted(examples,key=lambda w:(-counts[w],w))[:config["vocabulary_size"]]
    sources=sorted(s for s,words in groups.items() if len(words)>=5)
    rng=random.Random(config["seed"])
    chosen=rng.sample(sources,config["styles"])
    rows=[]
    for style,source in enumerate(chosen):
        refs=rng.sample(groups[source],5)
        for text in vocab:
            word=examples[text]
            rows.append({"id":f"paragraph_f{config['fold']}_s{style}_{len(rows):06d}","target":text,
                "target_source_id":word["source_id"],"target_span":word["text_span"],
                "style_references":refs,"style":style,"seed":config["seed"]+len(rows)})
    manifest={"method":"diffusionpen","fold":config["fold"],"truth_sha256":sha256(truth),
        "bank_path":str(bank_path.relative_to(ROOT)),"bank_sha256":sha256(bank_path),"rows":rows}
    validate_requests(manifest)
    out=absolute(config["output_dir"])/"words"
    path=out/"requests.json"
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError("Refusing to replace different generator requests")
    save_json(path,manifest)
    save_json(out/"vocabulary.json",{"words":vocab,"frequencies":{w:counts[w] for w in vocab},"style_source_ids":chosen})
    print(f"Prepared {len(rows)} words: {len(vocab)} historical tokens, {len(chosen)} styles",flush=True)


if __name__=="__main__": main()
