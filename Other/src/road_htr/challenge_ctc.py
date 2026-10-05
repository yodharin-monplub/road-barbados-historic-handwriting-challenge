"""Memory-bounded CTC prefix beam search for the Model29 decoder.

This module deliberately has no Kraken, PyTorch, or NumPy dependency.  The
probability exporter runs inside the Kraken environment, while unit tests and
the cross-fold selection logic can exercise the decoder with ordinary Python
sequences.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


NEG_INF = -float("inf")
BOS = "\u0002"


def logadd(*values: float) -> float:
    """Stable log(sum(exp(values))) with support for all-negative-infinity."""

    finite = [value for value in values if value != NEG_INF]
    if not finite:
        return NEG_INF
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def log_softmax_rows(logits: Sequence[Sequence[float]]) -> list[list[float]]:
    """Convert synthetic or model logits to row-wise log probabilities."""

    result: list[list[float]] = []
    width: int | None = None
    for row in logits:
        values = [float(value) for value in row]
        if not values:
            raise ValueError("every CTC timestep must contain at least one class")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError("all CTC timesteps must contain the same class count")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("CTC logits must be finite")
        maximum = max(values)
        denominator = maximum + math.log(
            sum(math.exp(value - maximum) for value in values)
        )
        result.append([value - denominator for value in values])
    return result


class CharNGramLanguageModel:
    """Additive-smoothed character n-gram LM fitted only on supplied text."""

    def __init__(self, order: int, smoothing: float):
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("character LM order must be a positive integer")
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("character LM smoothing must be positive")
        self.order = order
        self.smoothing = float(smoothing)
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.alphabet: set[str] = set()
        self.fitted_texts = 0

    def fit(self, texts: Iterable[str]) -> "CharNGramLanguageModel":
        for raw_text in texts:
            text = str(raw_text)
            self.fitted_texts += 1
            self.alphabet.update(text)
            history = BOS * (self.order - 1)
            for character in text:
                context = history[-(self.order - 1) :] if self.order > 1 else ""
                self.counts[context][character] += 1
                history += character
        if not self.fitted_texts:
            raise ValueError("character LM needs at least one training text")
        return self

    def log_probability(self, prefix: str, character: str) -> float:
        if len(character) != 1:
            raise ValueError("character LM scores one Unicode code point at a time")
        context = (
            (BOS * (self.order - 1) + prefix)[-(self.order - 1) :]
            if self.order > 1
            else ""
        )
        counts = self.counts.get(context, Counter())
        # Include one unknown bucket.  A validation-only codec character is
        # therefore scoreable without adding it to the fitted alphabet.
        vocabulary = max(len(self.alphabet) + 1, 1)
        numerator = counts.get(character, 0) + self.smoothing
        denominator = sum(counts.values()) + self.smoothing * vocabulary
        return math.log(numerator / denominator)

    def extension_score(self, prefix: str, token_text: str) -> float:
        score = 0.0
        history = prefix
        for character in token_text:
            score += self.log_probability(history, character)
            history += character
        return score


class WordBoundaryPrior:
    """Training-only reward for completing a word observed in the fit folds.

    The returned value is a log-count ratio against an unseen word, so unseen
    spellings receive zero instead of a very large corpus-size penalty.  This
    keeps the optional feature a lexical boundary prior rather than a hidden
    length penalty.
    """

    def __init__(self, smoothing: float = 1.0):
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("word-boundary smoothing must be positive")
        self.smoothing = float(smoothing)
        self.counts: Counter[str] = Counter()
        self.fitted_texts = 0

    def fit(self, texts: Iterable[str]) -> "WordBoundaryPrior":
        for raw_text in texts:
            self.fitted_texts += 1
            self.counts.update(str(raw_text).split())
        if not self.fitted_texts:
            raise ValueError("word-boundary prior needs at least one training text")
        return self

    def completed_word_score(self, prefix: str) -> float:
        word = prefix.rsplit(" ", 1)[-1]
        if not word:
            return 0.0
        return math.log((self.counts.get(word, 0) + self.smoothing) / self.smoothing)

    def extension_score(self, prefix: str, token_text: str) -> float:
        score = 0.0
        history = prefix
        for character in token_text:
            if character == " ":
                score += self.completed_word_score(history)
            history += character
        return score

    def terminal_score(self, text: str) -> float:
        # A trailing explicit space already scored this word in
        # ``extension_score``. Do not reward it a second time at EOS.
        if not text or text[-1].isspace():
            return 0.0
        return self.completed_word_score(text)


@dataclass(frozen=True)
class DecoderCandidate:
    """One deterministic member of Model29's fixed search grid."""

    name: str
    mode: str = "beam"
    beam_width: int = 12
    token_prune: int = 8
    lm_order: int = 5
    lm_smoothing: float = 0.2
    lm_alpha: float = 0.2
    word_boundary_weight: float = 0.0
    word_boundary_smoothing: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DecoderCandidate":
        allowed = {
            "name",
            "mode",
            "beam_width",
            "token_prune",
            "lm_order",
            "lm_smoothing",
            "lm_alpha",
            "word_boundary_weight",
            "word_boundary_smoothing",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown decoder candidate fields: {unknown}")
        try:
            candidate = cls(**value)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(f"invalid decoder candidate: {value}") from error
        candidate.validate()
        return candidate

    def validate(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("decoder candidate name must be nonempty")
        if self.mode not in {"greedy", "beam"}:
            raise ValueError("decoder candidate mode must be greedy or beam")
        for name, value in (
            ("beam_width", self.beam_width),
            ("token_prune", self.token_prune),
            ("lm_order", self.lm_order),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"decoder candidate {name} must be positive")
        for name, value in (
            ("lm_smoothing", self.lm_smoothing),
            ("word_boundary_smoothing", self.word_boundary_smoothing),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"decoder candidate {name} must be positive")
        for name, value in (
            ("lm_alpha", self.lm_alpha),
            ("word_boundary_weight", self.word_boundary_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"decoder candidate {name} must be non-negative")
        if self.mode == "greedy" and (
            self.lm_alpha != 0 or self.word_boundary_weight != 0
        ):
            raise ValueError("greedy candidates cannot apply language-model weights")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mode": self.mode,
            "beam_width": self.beam_width,
            "token_prune": self.token_prune,
            "lm_order": self.lm_order,
            "lm_smoothing": self.lm_smoothing,
            "lm_alpha": self.lm_alpha,
            "word_boundary_weight": self.word_boundary_weight,
            "word_boundary_smoothing": self.word_boundary_smoothing,
        }


def _validated_label_texts(label_texts: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(str(value) for value in label_texts)
    if not labels or any(not value for value in labels):
        raise ValueError("every nonblank CTC class must map to nonempty text")
    if len(set(labels)) != len(labels):
        raise ValueError("nonblank CTC class texts must be unique")
    return labels


def _row_values(row: Sequence[float]) -> list[float]:
    if hasattr(row, "tolist"):
        values = row.tolist()  # type: ignore[union-attr]
    else:
        values = list(row)
    return [float(value) for value in values]


def _validate_log_row(row: Sequence[float], class_count: int) -> list[float]:
    values = _row_values(row)
    if len(values) != class_count:
        raise ValueError(
            f"CTC class mismatch: expected {class_count}, found {len(values)}"
        )
    if any(math.isnan(value) or value == float("inf") for value in values):
        raise ValueError("CTC log probabilities contain NaN or positive infinity")
    return values


def greedy_decode_log_probs(
    log_probabilities: Sequence[Sequence[float]],
    label_texts: Sequence[str],
    blank: int = 0,
) -> str:
    """Best-path CTC decoding with exact repeat/blank semantics."""

    labels = _validated_label_texts(label_texts)
    if blank != 0:
        raise ValueError("Model29 currently requires CTC blank class 0")
    previous = blank
    output: list[str] = []
    for raw_row in log_probabilities:
        row = _validate_log_row(raw_row, len(labels) + 1)
        token = max(range(len(row)), key=row.__getitem__)
        if token != blank and token != previous:
            output.append(labels[token - 1])
        previous = token
    return "".join(output)


def prefix_beam_search(
    log_probabilities: Sequence[Sequence[float]],
    label_texts: Sequence[str],
    *,
    beam_width: int,
    token_prune: int,
    lm: CharNGramLanguageModel | None = None,
    lm_alpha: float = 0.0,
    word_prior: WordBoundaryPrior | None = None,
    word_boundary_weight: float = 0.0,
    blank: int = 0,
    nbest: int = 1,
) -> list[tuple[str, float]]:
    """Prefix-beam decode one ``[time, classes]`` log-probability matrix.

    Beam keys are tuples of CTC label IDs, not text suffixes.  That distinction
    is what makes repeated-label handling correct even if a codec label emits
    more than one Unicode code point.
    """

    labels = _validated_label_texts(label_texts)
    if blank != 0:
        raise ValueError("Model29 currently requires CTC blank class 0")
    if beam_width <= 0 or token_prune <= 0 or nbest <= 0:
        raise ValueError("beam_width, token_prune, and nbest must be positive")
    if lm_alpha < 0 or word_boundary_weight < 0:
        raise ValueError("language-model weights must be non-negative")
    class_count = len(labels) + 1
    keep_tokens = min(token_prune, class_count)

    # prefix -> (ending in blank, ending in nonblank, emitted text)
    beams: dict[tuple[int, ...], tuple[float, float, str]] = {
        (): (0.0, NEG_INF, "")
    }
    extension_cache: dict[tuple[str, int], tuple[str, float]] = {}

    for raw_row in log_probabilities:
        row = _validate_log_row(raw_row, class_count)
        tokens = heapq.nlargest(keep_tokens, range(class_count), key=row.__getitem__)
        if blank not in tokens:
            tokens.append(blank)
        next_beams: dict[tuple[int, ...], tuple[float, float, str]] = {}

        def update(
            prefix: tuple[int, ...],
            text: str,
            *,
            add_blank: float = NEG_INF,
            add_nonblank: float = NEG_INF,
        ) -> None:
            old_blank, old_nonblank, old_text = next_beams.get(
                prefix, (NEG_INF, NEG_INF, text)
            )
            if old_text != text:
                raise RuntimeError("one CTC label prefix decoded to inconsistent text")
            next_beams[prefix] = (
                logadd(old_blank, add_blank),
                logadd(old_nonblank, add_nonblank),
                text,
            )

        for prefix, (prob_blank, prob_nonblank, text) in beams.items():
            total = logadd(prob_blank, prob_nonblank)
            for token in tokens:
                token_score = row[token]
                if token == blank:
                    update(prefix, text, add_blank=total + token_score)
                    continue

                # Continuing the same unseparated label does not emit a new
                # token and therefore receives no language-model score.
                if prefix and token == prefix[-1]:
                    update(
                        prefix,
                        text,
                        add_nonblank=prob_nonblank + token_score,
                    )
                    acoustic_base = prob_blank
                else:
                    acoustic_base = total

                if acoustic_base == NEG_INF:
                    continue
                cache_key = (text, token)
                cached = extension_cache.get(cache_key)
                if cached is None:
                    token_text = labels[token - 1]
                    extension_score = 0.0
                    if lm is not None and lm_alpha:
                        extension_score += lm_alpha * lm.extension_score(text, token_text)
                    if word_prior is not None and word_boundary_weight:
                        extension_score += word_boundary_weight * word_prior.extension_score(
                            text, token_text
                        )
                    cached = (text + token_text, extension_score)
                    extension_cache[cache_key] = cached
                extended_text, extension_score = cached
                update(
                    prefix + (token,),
                    extended_text,
                    add_nonblank=acoustic_base + token_score + extension_score,
                )

        ordered = heapq.nlargest(
            beam_width,
            next_beams.items(),
            key=lambda item: logadd(item[1][0], item[1][1]),
        )
        beams = dict(ordered)

    ranked: list[tuple[str, float, tuple[int, ...]]] = []
    for prefix, (prob_blank, prob_nonblank, text) in beams.items():
        score = logadd(prob_blank, prob_nonblank)
        if word_prior is not None and word_boundary_weight:
            score += word_boundary_weight * word_prior.terminal_score(text)
        ranked.append((text, score, prefix))
    ranked.sort(key=lambda item: (-item[1], item[2]))
    return [(text, score) for text, score, _ in ranked[:nbest]]


def decode_candidate(
    log_probabilities: Sequence[Sequence[float]],
    label_texts: Sequence[str],
    candidate: DecoderCandidate,
    lm: CharNGramLanguageModel | None,
    word_prior: WordBoundaryPrior | None,
) -> str:
    candidate.validate()
    if candidate.mode == "greedy":
        return greedy_decode_log_probs(log_probabilities, label_texts)
    result = prefix_beam_search(
        log_probabilities,
        label_texts,
        beam_width=candidate.beam_width,
        token_prune=candidate.token_prune,
        lm=lm,
        lm_alpha=candidate.lm_alpha,
        word_prior=word_prior,
        word_boundary_weight=candidate.word_boundary_weight,
        nbest=1,
    )
    return result[0][0] if result else ""
