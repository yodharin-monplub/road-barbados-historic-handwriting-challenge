"""Validation-tuned N-best consensus for heterogeneous OCR models."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .metrics import CER_NORMALIZER, WER_NORMALIZER, competition_error, levenshtein
from .schema import SchemaError, read_table
from .submission import clean_prediction, write_submission


@dataclass(frozen=True)
class CandidateRecord:
    identifier: str
    candidates: tuple[str, ...]
    scores: tuple[float | None, ...]


def _candidate_paths(path: Path) -> list[Path]:
    if path.is_dir():
        jsonl = sorted(path.rglob("*.jsonl"))
        return jsonl if jsonl else sorted(path.rglob("*.csv"))
    return [path]


def read_candidates(path: str | Path) -> dict[str, CandidateRecord]:
    """Read one candidate file or merge non-overlapping files in a directory."""

    merged: dict[str, CandidateRecord] = {}
    for item in _candidate_paths(Path(path)):
        current: dict[str, CandidateRecord] = {}
        if item.suffix.casefold() == ".jsonl":
            with item.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    identifier = str(row["id"]).strip()
                    values = tuple(clean_prediction(value) for value in row["candidates"])
                    raw_scores = row.get("scores", [None] * len(values))
                    scores = tuple(None if value is None else float(value) for value in raw_scores)
                    current[identifier] = CandidateRecord(identifier, values, scores)
        elif item.suffix.casefold() == ".csv":
            table = read_table(item)
            columns = [name for name in table.fieldnames if name != table.id_column]
            if len(columns) != 1:
                raise SchemaError(f"candidate CSV must have one prediction column: {item}")
            for row in table.rows:
                identifier = row[table.id_column].strip()
                value = clean_prediction(row[columns[0]])
                current[identifier] = CandidateRecord(identifier, (value,), (None,))
        else:
            raise ValueError(f"unsupported candidate file: {item}")
        overlap = set(merged).intersection(current)
        if overlap:
            raise SchemaError(
                f"duplicate IDs while merging {item}: {sorted(overlap)[:5]}"
            )
        merged.update(current)
    if not merged:
        raise ValueError(f"no candidate records found under {path}")
    return merged


def parse_sources(values: Iterable[str]) -> dict[str, dict[str, CandidateRecord]]:
    sources: dict[str, dict[str, CandidateRecord]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"source must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in sources:
            raise ValueError(f"blank or duplicate source name {name!r}")
        sources[name] = read_candidates(Path(raw_path))
    if len(sources) < 2:
        raise ValueError("consensus requires at least two model sources")
    return sources


def _normalized_distance(reference: str, hypothesis: str, level: str) -> float:
    if level == "word":
        ref, hyp = reference.split(), hypothesis.split()
    elif level == "char":
        ref, hyp = tuple(reference), tuple(hypothesis)
    else:
        raise ValueError(level)
    return levenshtein(ref, hyp) / max(len(ref), 1)


@dataclass
class ConsensusArrays:
    ids: list[str]
    source_names: list[str]
    candidates: list[list[str]]
    valid: object
    word_cost: object
    char_cost: object
    truth_loss: object | None
    references: list[str] | None


def prepare_arrays(
    sources: dict[str, dict[str, CandidateRecord]],
    ids: list[str],
    references: list[str] | None = None,
    nbest_per_source: int = 5,
) -> ConsensusArrays:
    import numpy as np

    source_names = list(sources)
    expected = set(ids)
    for name, records in sources.items():
        missing = expected - set(records)
        extra = set(records) - expected
        if missing or extra:
            raise SchemaError(
                f"{name} ID mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
            )

    pools: list[list[str]] = []
    for identifier in ids:
        pool: list[str] = []
        for records in sources.values():
            for value in records[identifier].candidates[:nbest_per_source]:
                if value and value not in pool:
                    pool.append(value)
        if not pool:
            raise ValueError(f"every source produced an empty result for {identifier!r}")
        pools.append(pool)
    max_candidates = max(map(len, pools))
    shape = (len(ids), max_candidates, len(source_names))
    word_cost = np.zeros(shape, dtype=np.float32)
    char_cost = np.zeros(shape, dtype=np.float32)
    valid = np.zeros((len(ids), max_candidates), dtype=bool)
    truth_loss = (
        np.full((len(ids), max_candidates), np.inf, dtype=np.float64)
        if references is not None
        else None
    )

    if references is not None:
        wer_denominator = sum(math.sqrt(max(len(value.split()), 1)) for value in references)
        cer_denominator = sum(math.sqrt(max(len(value), 1)) for value in references)

    for row_index, (identifier, pool) in enumerate(zip(ids, pools, strict=True)):
        valid[row_index, : len(pool)] = True
        source_top = [sources[name][identifier].candidates[0] for name in source_names]
        for candidate_index, candidate in enumerate(pool):
            for source_index, reference in enumerate(source_top):
                word_cost[row_index, candidate_index, source_index] = _normalized_distance(
                    reference, candidate, "word"
                )
                char_cost[row_index, candidate_index, source_index] = _normalized_distance(
                    reference, candidate, "char"
                )
            if references is not None and truth_loss is not None:
                reference = references[row_index]
                wer_contribution = (
                    levenshtein(reference.split(), candidate.split())
                    * math.sqrt(max(len(reference.split()), 1))
                    / wer_denominator
                )
                cer_contribution = (
                    levenshtein(tuple(reference), tuple(candidate))
                    * math.sqrt(max(len(reference), 1))
                    / cer_denominator
                )
                truth_loss[row_index, candidate_index] = (
                    wer_contribution / WER_NORMALIZER
                    + cer_contribution / CER_NORMALIZER
                ) / 2
    return ConsensusArrays(
        ids, source_names, pools, valid, word_cost, char_cost, truth_loss, references
    )


def select_indices(arrays: ConsensusArrays, weights, char_mix: float):
    import numpy as np

    support = (1 - char_mix) * arrays.word_cost + char_mix * arrays.char_cost
    costs = np.tensordot(support, weights, axes=([2], [0]))
    costs[~arrays.valid] = np.inf
    return costs.argmin(axis=1), costs


def selected_texts(arrays: ConsensusArrays, indices) -> list[str]:
    return [
        arrays.candidates[position][int(index)]
        for position, index in enumerate(indices)
    ]


def tune(
    arrays: ConsensusArrays,
    sources: dict[str, dict[str, CandidateRecord]],
    trials: int = 600,
    seed: int = 20260903,
    regularization: float = 0.0002,
) -> dict[str, object]:
    import numpy as np

    if arrays.truth_loss is None or arrays.references is None:
        raise ValueError("truth is required for tuning")
    individual: dict[str, dict[str, float]] = {}
    for name in arrays.source_names:
        predictions = [sources[name][identifier].candidates[0] for identifier in arrays.ids]
        individual[name] = competition_error(arrays.references, predictions)

    errors = np.array([individual[name]["mean_error"] for name in arrays.source_names])
    inverse = 1 / np.maximum(errors, 1e-8)
    inverse /= inverse.sum()
    equal = np.ones(len(arrays.source_names), dtype=np.float64) / len(arrays.source_names)
    proposals: list[tuple[np.ndarray, float]] = []
    for mix in (0.25, 0.4, 0.5, 0.6, 0.75):
        proposals.extend([(equal.copy(), mix), (inverse.copy(), mix)])
        for position in range(len(arrays.source_names)):
            one_hot = np.zeros(len(arrays.source_names))
            one_hot[position] = 1
            proposals.append((one_hot, mix))

    rng = random.Random(seed)
    for _ in range(trials):
        raw = np.array([rng.expovariate(1.0) + 0.04 for _ in arrays.source_names])
        weights = raw / raw.sum()
        proposals.append((weights, rng.uniform(0.2, 0.8)))

    best: dict[str, object] | None = None
    equal_value = 1 / len(arrays.source_names)
    row_indices = np.arange(len(arrays.ids))
    for weights, char_mix in proposals:
        indices, _ = select_indices(arrays, weights, char_mix)
        metric = float(arrays.truth_loss[row_indices, indices].sum())
        penalty = regularization * float(((weights - equal_value) ** 2).sum())
        objective = metric + penalty
        if best is None or objective < best["objective"]:
            best = {
                "objective": objective,
                "mean_error": metric,
                "char_mix": float(char_mix),
                "weights": {
                    name: float(weight)
                    for name, weight in zip(arrays.source_names, weights, strict=True)
                },
                "regularization": regularization,
                "seed": seed,
                "trials": trials,
                "individual_scores": individual,
            }
    assert best is not None
    return best


def apply_config(arrays: ConsensusArrays, config: dict[str, object]):
    import numpy as np

    raw_weights = config["weights"]
    if not isinstance(raw_weights, dict):
        raise ValueError("weights must be a mapping")
    if set(raw_weights) != set(arrays.source_names):
        raise ValueError(
            f"source names differ: config={sorted(raw_weights)}, current={sorted(arrays.source_names)}"
        )
    weights = np.array([float(raw_weights[name]) for name in arrays.source_names])
    weights /= weights.sum()
    indices, costs = select_indices(arrays, weights, float(config["char_mix"]))
    return selected_texts(arrays, indices), indices, costs, weights


def write_prediction_csv(path: str | Path, id_column: str, ids: list[str], values: list[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_column, "prediction"])
        writer.writeheader()
        for identifier, value in zip(ids, values, strict=True):
            writer.writerow({id_column: identifier, "prediction": value})


def agreement_rows(
    arrays: ConsensusArrays,
    selected: list[str],
    weights,
    char_mix: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position, (identifier, chosen) in enumerate(zip(arrays.ids, selected, strict=True)):
        distances: list[float] = []
        exact_votes = 0
        for source_index, _ in enumerate(arrays.source_names):
            # Candidate 0 for each source is represented through its distance
            # to the selected string in the precomputed arrays.
            try:
                candidate_index = arrays.candidates[position].index(chosen)
            except ValueError as exc:
                raise AssertionError("selected candidate missing from pool") from exc
            distance = (1 - char_mix) * float(
                arrays.word_cost[position, candidate_index, source_index]
            ) + char_mix * float(arrays.char_cost[position, candidate_index, source_index])
            distances.append(distance)
            source_top_distance = distance
            if source_top_distance == 0:
                exact_votes += 1
        disagreement = sum(
            float(weight) * min(distance, 1.0)
            for weight, distance in zip(weights, distances, strict=True)
        )
        rows.append(
            {
                "id": identifier,
                "prediction": chosen,
                "agreement": max(0.0, 1.0 - disagreement),
                "exact_votes": exact_votes,
                "source_count": len(arrays.source_names),
            }
        )
    return rows


def save_submission_if_requested(
    sample: str | Path | None,
    submission: str | Path | None,
    ids: list[str],
    predictions: list[str],
) -> None:
    if submission:
        if not sample:
            raise ValueError("sample path is required for a submission")
        write_submission(sample, submission, dict(zip(ids, predictions, strict=True)))
