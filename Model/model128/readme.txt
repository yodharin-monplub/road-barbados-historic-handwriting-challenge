Model128: centre-crop view member - NOT COMPLETED (laptop too slow)
==================================================================
Idea: the old Model9/17/19 selectors fused a "full" and a "centre" view of each
line crop, so a LightOnOCR member trained on the centre band should make
different errors from the full-image members and add a vote.

Implementation is done and working: model122/train.py gained an "image_variant"
setting, and config_center.json trains a rank-64 LoRA on the centre band with
batch 2 / accumulation 8 to fit the laptop's 6GB RTX 4050.

Why it stopped: measured throughput on the laptop is about 102 seconds per
optimizer step, i.e. roughly 14 hours for one fold and about three days for the
five-fold member - far beyond the value of one extra voter. A first attempt also
died during evaluation, most likely out of memory once generation allocated its
KV cache on top of the training footprint; eval_batch_size was reduced to 1, but
the speed remains the blocker, not the memory.

If revisited, run it on a rented GPU (about 25 minutes per fold) and gate it on
fold 0 first, exactly as Model132 was gated - a same-family variant is not
automatically a useful voter (Model132's second Kraken made the vote worse).

Status: implemented, not trained, no score.
