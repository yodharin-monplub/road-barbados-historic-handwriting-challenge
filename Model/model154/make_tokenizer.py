#!/usr/bin/env python3
"""Give the TrOCR checkpoint the tokenizer.json that transformers 5 requires.

microsoft/trocr-large-handwritten ships only the legacy RoBERTa files (vocab.json +
merges.txt). transformers 5 no longer converts those itself and fails with
"Couldn't instantiate the backend tokenizer". This rebuilds the identical byte-level
BPE tokenizer with the `tokenizers` library - same vocabulary, same merges, RoBERTa
<s> ... </s> post-processing - and writes it next to the weights.

usage: python Model/model154/make_tokenizer.py <checkpoint dir>
"""
import json
import sys
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing

path = Path(sys.argv[1])
tok = ByteLevelBPETokenizer(str(path / "vocab.json"), str(path / "merges.txt"),
                            add_prefix_space=False, trim_offsets=True)
ids = {t: tok.token_to_id(t) for t in ("<s>", "<pad>", "</s>", "<unk>")}
assert ids["<s>"] == 0 and ids["<pad>"] == 1 and ids["</s>"] == 2, ids  # RoBERTa layout
tok.add_special_tokens(["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
tok._tokenizer.post_processor = RobertaProcessing(("</s>", ids["</s>"]), ("<s>", ids["<s>"]),
                                                  trim_offsets=True, add_prefix_space=False)
tok.save(str(path / "tokenizer.json"))

# Sanity: the fast tokenizer must load and round-trip period spellings unchanged.
from transformers import AutoTokenizer  # noqa: E402

fast = AutoTokenizer.from_pretrained(str(path))
for text in ("Witness my hand and seal this day", "y^t holy Evangelists & c."):
    enc = fast(text)["input_ids"]
    back = fast.decode(enc, skip_special_tokens=True)
    assert back == text, (text, back)
    assert enc[0] == 0 and enc[-1] == 2, enc
print(json.dumps({"vocab": fast.vocab_size, "special": ids, "class": type(fast).__name__}))
print("TOKENIZER_OK")
