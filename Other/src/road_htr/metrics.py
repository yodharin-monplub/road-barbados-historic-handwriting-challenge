"""Dependency-free implementation of the challenge's weighted multi-metric."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import TypeVar


T = TypeVar("T")
_ROLE_TOKENS = re.compile(
    r"<\|(?:user|assistant|system)\|>|<\|im_(?:start|end)\|>"
    r"|^\s*(?:assistant|user)\s*:?\s*",
    re.IGNORECASE,
)

# Zindi normalizes each error metric against the starter benchmark before
# combining them. These values reproduce the public benchmark exactly:
# 1 - 0.5 * (11.5289665 / 12 + 53.81177457 / 55) = 0.030428445.
WER_NORMALIZER = 12.0
CER_NORMALIZER = 55.0


def strip_conversation_tokens(value: str) -> str:
    """Mirror the documented removal of model-chat role markers."""

    return _ROLE_TOKENS.sub("", str(value)).strip()


def levenshtein(reference: Sequence[T], hypothesis: Sequence[T]) -> int:
    """Return insertion + deletion + substitution distance in O(min(n,m)) memory."""

    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def _weighted_error(
    references: Sequence[str],
    hypotheses: Sequence[str],
    tokenize: Callable[[str], Sequence[str]],
    weight_factor: float = 0.5,
) -> float:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"reference/prediction count mismatch: {len(references)} != {len(hypotheses)}"
        )
    if not references:
        raise ValueError("at least one sample is required")

    weighted_edits = 0.0
    total_weight = 0.0
    for raw_ref, raw_hyp in zip(references, hypotheses, strict=True):
        ref = tokenize(strip_conversation_tokens(raw_ref))
        hyp = tokenize(strip_conversation_tokens(raw_hyp))
        # Empty references are not expected in this challenge. A unit weight makes
        # the behavior defined without allowing a free non-empty prediction.
        length = max(len(ref), 1)
        weight = math.pow(length, weight_factor)
        weighted_edits += levenshtein(ref, hyp) * weight
        total_weight += weight
    return weighted_edits / total_weight


def weighted_wer(
    references: Sequence[str], hypotheses: Sequence[str], weight_factor: float = 0.5
) -> float:
    return _weighted_error(references, hypotheses, lambda value: value.split(), weight_factor)


def weighted_cer(
    references: Sequence[str], hypotheses: Sequence[str], weight_factor: float = 0.5
) -> float:
    return _weighted_error(references, hypotheses, tuple, weight_factor)


def competition_error(
    references: Sequence[str], hypotheses: Sequence[str], weight_factor: float = 0.5
) -> dict[str, float]:
    """Return raw metrics and the normalized multi-metric used for ranking."""

    wer = weighted_wer(references, hypotheses, weight_factor)
    cer = weighted_cer(references, hypotheses, weight_factor)
    normalized_wer = wer / WER_NORMALIZER
    normalized_cer = cer / CER_NORMALIZER
    mean_error = (normalized_wer + normalized_cer) / 2
    return {
        "weighted_wer": wer,
        "weighted_cer": cer,
        "normalized_wer_error": normalized_wer,
        "normalized_cer_error": normalized_cer,
        "mean_error": mean_error,
        "multi_score": 1 - mean_error,
    }
