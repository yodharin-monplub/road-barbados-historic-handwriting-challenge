"""Greedy and prefix-beam CTC decoding with a train-only character n-gram LM."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


NEG_INF = -float("inf")
BOS = "\u0002"


def logadd(*values: float) -> float:
    finite = [value for value in values if value != NEG_INF]
    if not finite:
        return NEG_INF
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


class CharNGramLM:
    def __init__(self, order: int = 5, smoothing: float = 0.2):
        if order < 1:
            raise ValueError("order must be positive")
        self.order = order
        self.smoothing = smoothing
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.alphabet: set[str] = set()

    def fit(self, texts: Iterable[str]) -> "CharNGramLM":
        for text in texts:
            self.alphabet.update(text)
            history = BOS * (self.order - 1)
            for character in text:
                context = history[-(self.order - 1) :] if self.order > 1 else ""
                self.counts[context][character] += 1
                history += character
        return self

    def log_probability(self, prefix: str, character: str) -> float:
        context = (BOS * (self.order - 1) + prefix)[-(self.order - 1) :] if self.order > 1 else ""
        counts = self.counts.get(context, Counter())
        vocabulary = max(len(self.alphabet), 1)
        numerator = counts.get(character, 0) + self.smoothing
        denominator = sum(counts.values()) + self.smoothing * vocabulary
        return math.log(numerator / denominator)

    def save(self, path: str | Path) -> None:
        payload = {
            "order": self.order,
            "smoothing": self.smoothing,
            "alphabet": sorted(self.alphabet),
            "counts": {context: dict(values) for context, values in self.counts.items()},
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharNGramLM":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(payload["order"], payload["smoothing"])
        model.alphabet = set(payload["alphabet"])
        for context, values in payload["counts"].items():
            model.counts[context].update(values)
        return model


def greedy_decode(indices: Iterable[int], characters: list[str], blank: int = 0) -> str:
    output: list[str] = []
    previous = blank
    for raw_index in indices:
        index = int(raw_index)
        if index != blank and index != previous:
            output.append(characters[index - 1])
        previous = index
    return "".join(output)


def prefix_beam_search(
    log_probs,
    characters: list[str],
    beam_width: int = 32,
    token_prune: int = 24,
    lm: CharNGramLM | None = None,
    lm_alpha: float = 0.35,
    space_bonus: float = 0.0,
    blank: int = 0,
    nbest: int = 5,
) -> list[tuple[str, float]]:
    """Decode one [time, classes] log-probability array."""

    import torch

    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [time, classes]")
    beams: dict[str, tuple[float, float]] = {"": (0.0, NEG_INF)}
    keep_tokens = min(token_prune, log_probs.shape[1])

    for timestep in log_probs:
        values, indices = torch.topk(timestep, keep_tokens)
        token_scores = list(zip(indices.tolist(), values.tolist(), strict=True))
        if blank not in indices.tolist():
            token_scores.append((blank, float(timestep[blank])))
        next_beams: dict[str, tuple[float, float]] = {}

        def update(prefix: str, add_blank: float = NEG_INF, add_nonblank: float = NEG_INF):
            old_blank, old_nonblank = next_beams.get(prefix, (NEG_INF, NEG_INF))
            next_beams[prefix] = (
                logadd(old_blank, add_blank),
                logadd(old_nonblank, add_nonblank),
            )

        for prefix, (prob_blank, prob_nonblank) in beams.items():
            total = logadd(prob_blank, prob_nonblank)
            for token, token_score in token_scores:
                if token == blank:
                    update(prefix, add_blank=total + token_score)
                    continue
                character = characters[token - 1]
                if prefix and character == prefix[-1]:
                    # Same character without an intervening blank leaves prefix unchanged.
                    update(prefix, add_nonblank=prob_nonblank + token_score)
                    base = prob_blank
                else:
                    base = total
                extended = prefix + character
                language_score = 0.0
                if lm is not None:
                    language_score += lm_alpha * lm.log_probability(prefix, character)
                if character == " ":
                    language_score += space_bonus
                update(extended, add_nonblank=base + token_score + language_score)

        ordered = sorted(
            next_beams.items(),
            key=lambda item: logadd(item[1][0], item[1][1]),
            reverse=True,
        )[:beam_width]
        beams = dict(ordered)

    return [
        (prefix, logadd(prob_blank, prob_nonblank))
        for prefix, (prob_blank, prob_nonblank) in sorted(
            beams.items(),
            key=lambda item: logadd(item[1][0], item[1][1]),
            reverse=True,
        )[:nbest]
    ]
