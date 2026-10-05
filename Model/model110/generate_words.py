"""Batched, seeded DiffusionPen word generation for paragraph augmentation."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"model108"))
from road_htr.stackmix import absolute,sha256,save_json
from prepare import validate_requests
from runtime import load_models,image_tensor


def main():
    import torch
    import numpy as np
    from PIL import Image
    config=json.loads((ROOT/"model110/config.json").read_text())
    out=absolute(config["output_dir"])/"words"
    path=out/"requests.json"
    manifest=json.loads(path.read_text())
    rows=validate_requests(manifest)
    ledger=absolute(config["generator_assets"])
    signature={"requests_sha256":sha256(path),"assets_sha256":sha256(ledger),"config_sha256":sha256(ROOT/"model110/config.json"),
               "code_sha256":sha256(__file__)}
    torch.set_num_threads(2)
    network,style,vae,scheduler,tokenizer=load_models()
    images=out/"images"
    images.mkdir(parents=True,exist_ok=True)
    complete=[]
    receipt_path=out/"generation_manifest.json"
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text())
        if any(old.get(k)!=v for k,v in signature.items()): raise ValueError("Cannot resume a different generation recipe")
        complete=old["rows"]
        for expected,saved in zip(rows,complete):
            if expected["id"]!=saved["id"] or sha256(absolute(saved["image"]))!=saved["sha256"]:
                raise ValueError("Generated prefix hash/ID mismatch")
    features={}
    with torch.inference_mode():
        for row in rows:
            if row["style"] not in features:
                features[row["style"]]=style(torch.stack([image_tensor(r["image"]) for r in row["style_references"]]).cuda())
        for start in range(len(complete),len(rows),config["generation_batch_size"]):
            batch=rows[start:start+config["generation_batch_size"]]
            tokens=tokenizer([r["target"] for r in batch],padding="max_length",truncation=False,max_length=40,return_tensors="pt").to("cuda")
            style_features=torch.cat([features[r["style"]] for r in batch])
            latent=torch.cat([torch.randn((1,4,8,32),device="cuda",generator=torch.Generator(device="cuda").manual_seed(r["seed"])) for r in batch])
            scheduler.set_timesteps(config["diffusion_steps"])
            for timestep in scheduler.timesteps:
                t=torch.full((len(batch),),int(timestep),device="cuda",dtype=torch.long)
                residual=network(latent,timesteps=t,context=tokens,style_extractor=style_features)
                latent=scheduler.step(residual,timestep,latent).prev_sample
            pixels=(vae.decode(latent/0.18215).sample/2+0.5).clamp(0,1)
            for row,pixel in zip(batch,pixels):
                image=Image.fromarray((pixel.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
                dest=images/f"{row['id']}.png"
                image.save(dest)
                complete.append(dict(row,image=str(dest.relative_to(ROOT)),sha256=sha256(dest)))
            save_json(receipt_path,{**signature,"method":"diffusionpen","fold":config["fold"],"requests_path":str(path.relative_to(ROOT)),
                "adapter_sha256":None,"rows":complete,"complete":len(complete)==len(rows)})
            print(f"Generated {len(complete)}/{len(rows)} word images",flush=True)
    with (out/"labels.csv").open("w",newline="") as handle:
        writer=csv.writer(handle); writer.writerow(["ID","Target"])
        writer.writerows((r["id"],r["target"]) for r in complete)


if __name__=="__main__": main()
