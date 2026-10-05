"""Exact family-aware voting; case/spelling are never dictionary-normalized."""
from collections import Counter
import importlib.util
from pathlib import Path

from road_htr.word_rover import _tokens, _aligned_to_base

# Reuse the prior case-sensitive character alignment, not its tuned threshold.
_path = Path(__file__).resolve().parents[1]/"model99/evaluate.py"
_spec = importlib.util.spec_from_file_location("model112_character_alignment", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
align_characters = _module.align_to_base
NAMES = ("model73", "model92", "model107", "kraken")


def vote(values, fallback, family):
    if set(values) != set(NAMES): raise ValueError("Expected exactly four named sources")
    counts = Counter(values.values())
    winner, count = min(counts.items(), key=lambda p:(-p[1],repr(p[0])))
    if count < 3: return fallback
    if family and (values["kraken"] != winner or sum(values[n] == winner for n in NAMES[:3]) < 2):
        return fallback
    return winner


def fuse(base_text, sources, family=True, character=False):
    if not base_text.strip() or set(sources) != set(NAMES) or any(not s.strip() for s in sources.values()):
        raise ValueError("Need nonempty anchor and all four named sources")
    base = _tokens(base_text)
    aligned = {name:_aligned_to_base(base,_tokens(text)) for name,text in sources.items()}
    if not character:
        out=[]
        for slot in range(len(base)+1):
            out.extend(vote({n:a[1][slot] for n,a in aligned.items()},(),family))
            if slot<len(base):
                token=vote({n:a[0][slot] for n,a in aligned.items()},base[slot].text,family)
                if token is not None: out.append(token)
        original=" ".join(t.text for t in base)
        result=" ".join(out).strip()
        return base_text if not result or result==original else result
    # Keep exact original spaces/boundaries for character-level repair.
    pieces=[]; end=0
    for index,token in enumerate(base):
        donors={n:align_characters(token.text,a[0][index]) if a[0][index] is not None else None for n,a in aligned.items()}
        chars=[]
        for slot in range(len(token.text)+1):
            insertion=vote({n:a[1][slot] if a is not None else () for n,a in donors.items()},(),family)
            chars.extend(insertion)
            if slot<len(token.text):
                char=vote({n:a[0][slot] if a is not None else None for n,a in donors.items()},token.text[slot],family)
                if char is not None: chars.append(char)
        replacement="".join(chars) or token.text
        pieces.extend([base_text[end:token.start],replacement]);end=token.end
    pieces.append(base_text[end:])
    return "".join(pieces)
