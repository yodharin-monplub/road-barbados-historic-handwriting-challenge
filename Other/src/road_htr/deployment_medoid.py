"""Fixed five-fold text medoid for Model26 deployment inference.

This module deliberately contains no fitting or score-based configuration.  A
five-fold test ensemble has no ordinary OOF analogue: for any training row,
four of the five fold models trained on that row.  The deployment rule is
therefore fixed in code and consumes test predictions only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ensemble import CandidateRecord, _normalized_distance, write_prediction_csv
from .schema import SchemaError, read_table
from .submission import (
    clean_prediction_preserving_internal_spaces,
    validate_submission,
    write_submission,
)


EXPECTED_SOURCE_COUNT = 5
FIXED_CHAR_MIX = 0.55
FIXED_WORD_MIX = 1.0 - FIXED_CHAR_MIX
MAJORITY_VOTES = EXPECTED_SOURCE_COUNT // 2 + 1
TIE_TOLERANCE = 1e-12


def _clean_candidate(value: object) -> str:
    """Remove generation boundaries without erasing scored internal spaces."""

    return clean_prediction_preserving_internal_spaces(value)


@dataclass(frozen=True)
class MedoidResult:
    prediction: str
    diagnostic: dict[str, Any]


def _prediction_column(fieldnames: Iterable[str], id_column: str, path: Path) -> str:
    columns = [name for name in fieldnames if name != id_column]
    if len(columns) != 1:
        raise SchemaError(
            f"candidate CSV must have exactly one prediction column: {path}"
        )
    return columns[0]


def _read_csv_source(path: Path) -> dict[str, CandidateRecord]:
    table = read_table(path)
    if not table.rows:
        raise SchemaError(f"candidate source is empty: {path}")
    column = _prediction_column(table.fieldnames, table.id_column, path)
    records: dict[str, CandidateRecord] = {}
    for line_number, row in enumerate(table.rows, start=2):
        identifier = row[table.id_column].strip()
        prediction = _clean_candidate(row.get(column))
        if not prediction:
            raise SchemaError(
                f"empty prediction for {identifier!r} at {path}:{line_number}"
            )
        records[identifier] = CandidateRecord(
            identifier=identifier,
            candidates=(prediction,),
            scores=(None,),
        )
    return records


def _top_score(raw_scores: Any, path: Path, line_number: int) -> float | None:
    if raw_scores is None:
        return None
    if not isinstance(raw_scores, list):
        raise SchemaError(f"scores must be a JSON list at {path}:{line_number}")
    if not raw_scores or raw_scores[0] is None:
        return None
    try:
        score = float(raw_scores[0])
    except (TypeError, ValueError) as error:
        raise SchemaError(
            f"invalid top-candidate confidence at {path}:{line_number}"
        ) from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise SchemaError(
            f"top-candidate confidence must be finite and in [0, 1] "
            f"at {path}:{line_number}"
        )
    return score


def _read_jsonl_source(path: Path) -> dict[str, CandidateRecord]:
    records: dict[str, CandidateRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SchemaError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise SchemaError(f"JSONL row must be an object at {path}:{line_number}")
            identifier = str(row.get("id", "")).strip()
            if not identifier:
                raise SchemaError(f"blank ID at {path}:{line_number}")
            if identifier in records:
                raise SchemaError(
                    f"duplicate ID {identifier!r} at {path}:{line_number}"
                )
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise SchemaError(
                    f"missing top candidate for {identifier!r} at {path}:{line_number}"
                )
            prediction = _clean_candidate(candidates[0])
            if not prediction:
                raise SchemaError(
                    f"empty prediction for {identifier!r} at {path}:{line_number}"
                )
            score = _top_score(row.get("scores"), path, line_number)
            records[identifier] = CandidateRecord(
                identifier=identifier,
                candidates=(prediction,),
                scores=(score,),
            )
    if not records:
        raise SchemaError(f"candidate source is empty: {path}")
    return records


def read_source(path: str | Path) -> dict[str, CandidateRecord]:
    """Read one complete, direct CSV or JSONL top-candidate source."""

    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    suffix = candidate.suffix.casefold()
    if suffix == ".csv":
        return _read_csv_source(candidate)
    if suffix == ".jsonl":
        return _read_jsonl_source(candidate)
    raise ValueError(f"candidate source must be a CSV or JSONL file: {candidate}")


def parse_five_sources(
    values: Iterable[str],
) -> dict[str, dict[str, CandidateRecord]]:
    """Parse exactly five unique ``NAME=PATH`` source specifications."""

    specifications = list(values)
    if len(specifications) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"Model26 requires exactly {EXPECTED_SOURCE_COUNT} sources; "
            f"received {len(specifications)}"
        )
    sources: dict[str, dict[str, CandidateRecord]] = {}
    resolved_paths: set[Path] = set()
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"source must be NAME=PATH, got {specification!r}")
        raw_name, raw_path = specification.split("=", 1)
        name = raw_name.strip()
        if not name or name in sources:
            raise ValueError(f"blank or duplicate source name {name!r}")
        if not raw_path.strip():
            raise ValueError(f"source path is blank for {name!r}")
        path = Path(raw_path.strip()).expanduser()
        resolved = path.resolve()
        if resolved in resolved_paths:
            raise ValueError(f"candidate source path is repeated: {path}")
        resolved_paths.add(resolved)
        sources[name] = read_source(path)
    return sources


def validate_complete_sources(
    sources: Mapping[str, Mapping[str, CandidateRecord]], ids: list[str]
) -> None:
    if len(sources) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"Model26 requires exactly {EXPECTED_SOURCE_COUNT} sources; "
            f"received {len(sources)}"
        )
    if not ids:
        raise SchemaError("test table contains no rows")
    expected = set(ids)
    for name, records in sources.items():
        missing = expected - set(records)
        extra = set(records) - expected
        if missing or extra:
            raise SchemaError(
                f"{name} ID mismatch: {len(missing)} missing, {len(extra)} extra; "
                f"examples missing={sorted(missing)[:5]}, "
                f"extra={sorted(extra)[:5]}"
            )
        for identifier in ids:
            record = records[identifier]
            if not record.candidates or not _clean_candidate(record.candidates[0]):
                raise SchemaError(
                    f"{name} has an empty top candidate for {identifier!r}"
                )


def _mixed_distance(reference: str, hypothesis: str) -> float:
    return (
        FIXED_WORD_MIX * _normalized_distance(reference, hypothesis, "word")
        + FIXED_CHAR_MIX * _normalized_distance(reference, hypothesis, "char")
    )


def _candidate_cost(candidate: str, source_predictions: Iterable[str]) -> float:
    distances = [_mixed_distance(source, candidate) for source in source_predictions]
    return math.fsum(distances) / EXPECTED_SOURCE_COUNT


def _confidence_support(
    candidate: str,
    source_names: list[str],
    records: Mapping[str, Mapping[str, CandidateRecord]],
    identifier: str,
) -> tuple[float | None, int]:
    values: list[float] = []
    for name in source_names:
        record = records[name][identifier]
        if record.candidates[0] != candidate or not record.scores:
            continue
        score = record.scores[0]
        if score is not None:
            values.append(float(score))
    return (math.fsum(values), len(values)) if values else (None, 0)


def select_medoid(
    identifier: str,
    sources: Mapping[str, Mapping[str, CandidateRecord]],
) -> MedoidResult:
    """Select one source string using the preregistered Model26 hierarchy."""

    if len(sources) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"Model26 requires exactly {EXPECTED_SOURCE_COUNT} sources; "
            f"received {len(sources)}"
        )
    source_names = list(sources)
    source_predictions = [sources[name][identifier].candidates[0] for name in source_names]
    if any(not _clean_candidate(value) for value in source_predictions):
        raise SchemaError(f"empty source prediction for {identifier!r}")

    vote_counts = Counter(source_predictions)
    candidates = sorted(vote_counts)
    costs = {
        candidate: _candidate_cost(candidate, source_predictions)
        for candidate in candidates
    }
    confidences = {
        candidate: _confidence_support(candidate, source_names, sources, identifier)
        for candidate in candidates
    }
    minimum_cost = min(costs.values())
    tied = [
        candidate
        for candidate in candidates
        if math.isclose(
            costs[candidate], minimum_cost, rel_tol=0.0, abs_tol=TIE_TOLERANCE
        )
    ]
    tied_medoid_count = len(tied)

    majority = [
        candidate for candidate in candidates if vote_counts[candidate] >= MAJORITY_VOTES
    ]
    confidence_tiebreak_used = False
    if majority:
        # Five votes permit at most one distinct three-vote majority.
        prediction = majority[0]
        reason = "exact_majority"
    else:
        if len(tied) == 1:
            prediction = tied[0]
            reason = "medoid"
        else:
            available = {
                candidate: confidences[candidate][0]
                for candidate in tied
                if confidences[candidate][0] is not None
            }
            if available:
                best_confidence = max(float(value) for value in available.values())
                finalists = [
                    candidate
                    for candidate in tied
                    if candidate in available
                    and math.isclose(
                        float(available[candidate]),
                        best_confidence,
                        rel_tol=0.0,
                        abs_tol=TIE_TOLERANCE,
                    )
                ]
                prediction = min(finalists)
                confidence_tiebreak_used = len(finalists) < len(tied)
                reason = (
                    "confidence_tiebreak"
                    if confidence_tiebreak_used
                    else "lexical_tiebreak"
                )
            else:
                prediction = min(tied)
                reason = "lexical_tiebreak"

    selected_cost = costs[prediction]
    candidate_diagnostics = [
        {
            "prediction": candidate,
            "exact_votes": vote_counts[candidate],
            "medoid_cost": costs[candidate],
            "confidence_support": confidences[candidate][0],
            "confidence_sources": confidences[candidate][1],
        }
        for candidate in candidates
    ]
    diagnostic: dict[str, Any] = {
        "id": identifier,
        "prediction": prediction,
        "selection_reason": reason,
        "source_count": EXPECTED_SOURCE_COUNT,
        "unique_candidate_count": len(candidates),
        "exact_votes": vote_counts[prediction],
        "exact_majority": vote_counts[prediction] >= MAJORITY_VOTES,
        "tied_medoid_count": tied_medoid_count,
        "confidence_tiebreak_used": confidence_tiebreak_used,
        "medoid_cost": selected_cost,
        "agreement": max(0.0, 1.0 - min(selected_cost, 1.0)),
        "fixed_char_mix": FIXED_CHAR_MIX,
        "source_predictions": dict(zip(source_names, source_predictions, strict=True)),
        "source_confidences": {
            name: (
                sources[name][identifier].scores[0]
                if sources[name][identifier].scores
                else None
            )
            for name in source_names
        },
        "candidates": candidate_diagnostics,
    }
    return MedoidResult(prediction=prediction, diagnostic=diagnostic)


def _ordered_test_ids(
    test_path: str | Path, sample_path: str | Path | None
) -> tuple[str, list[str]]:
    test = read_table(test_path)
    test_ids = [row[test.id_column].strip() for row in test.rows]
    if not test_ids:
        raise SchemaError("test table contains no rows")
    if sample_path is None:
        return test.id_column, test_ids

    sample = read_table(sample_path)
    sample_ids = [row[sample.id_column].strip() for row in sample.rows]
    if not sample_ids:
        raise SchemaError("sample submission contains no rows")
    missing = set(test_ids) - set(sample_ids)
    extra = set(sample_ids) - set(test_ids)
    if missing or extra:
        raise SchemaError(
            f"test/sample ID mismatch: missing from sample={sorted(missing)[:5]}, "
            f"extra in sample={sorted(extra)[:5]}"
        )
    return sample.id_column, sample_ids


def apply(
    test_path: str | Path,
    source_specs: Iterable[str],
    output_path: str | Path,
    agreement_output: str | Path | None = None,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> dict[str, Any]:
    """Combine five complete test sources and optionally write a submission."""

    if submission_path is not None and sample_path is None:
        raise ValueError("--sample is required with --submission")
    id_column, ids = _ordered_test_ids(test_path, sample_path)
    sources = parse_five_sources(source_specs)
    validate_complete_sources(sources, ids)

    selections = [select_medoid(identifier, sources) for identifier in ids]
    predictions = [selection.prediction for selection in selections]
    prediction_map = dict(zip(ids, predictions, strict=True))
    output = Path(output_path)
    diagnostics = (
        Path(agreement_output)
        if agreement_output is not None
        else output.with_suffix(".agreement.jsonl")
    )
    if output.resolve() == diagnostics.resolve():
        raise ValueError("prediction and agreement outputs must be different files")

    write_prediction_csv(output, id_column, ids, predictions)
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics.open("w", encoding="utf-8") as handle:
        for selection in selections:
            handle.write(json.dumps(selection.diagnostic, ensure_ascii=False) + "\n")

    reason_counts = Counter(
        str(selection.diagnostic["selection_reason"]) for selection in selections
    )
    result: dict[str, Any] = {
        "rows": len(ids),
        "sources": list(sources),
        "source_count": len(sources),
        "weights": {name: 1.0 / EXPECTED_SOURCE_COUNT for name in sources},
        "char_mix": FIXED_CHAR_MIX,
        "selection_reasons": dict(sorted(reason_counts.items())),
        "output": str(output),
        "agreement_output": str(diagnostics),
    }
    if submission_path is not None:
        assert sample_path is not None
        submission = write_submission(
            sample_path,
            submission_path,
            prediction_map,
            cleaner=clean_prediction_preserving_internal_spaces,
        )
        result["submission"] = str(submission)
        result["submission_validation"] = validate_submission(submission, sample_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Model26's fixed equal-weight five-fold text medoid"
    )
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="NAME=PATH; supply exactly five unique fold sources",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--agreement-output",
        type=Path,
        help="defaults to OUTPUT with the suffix .agreement.jsonl",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        help="validate IDs and use the sample submission's row order",
    )
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if args.submission is not None and args.sample is None:
        parser.error("--sample is required with --submission")

    result = apply(
        test_path=args.test,
        source_specs=args.source,
        output_path=args.output,
        agreement_output=args.agreement_output,
        sample_path=args.sample,
        submission_path=args.submission,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
