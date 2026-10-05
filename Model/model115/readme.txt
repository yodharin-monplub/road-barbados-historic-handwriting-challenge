Model115: second local LightOnOCR seed diversity pilot
====================================================
Seed20260915; Model114 uses20260914. Both start fresh from the same pinned base.
See model114/readme.txt for the shared training and frozen evaluation protocol.
Same real-data folds, augmentation, rank64 LoRA, bf16, 400 steps, effective batch16.
This run is queued after Model114 by scripts/run_seed_pilots.py.
No external/synthetic data, no extra ensemble votes, and no cloud spending.
Results: outputs/model115/report.json. Configuration: model115/config.json.
Independent fold0 development screen only; not yet a submission candidate.
