"""Conservative fixed-majority word ROVER for five OCR sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .deployment_medoid import parse_five_sources, read_source, validate_complete_sources
from .ensemble import CandidateRecord, write_prediction_csv
from .model34_token_fusion import Token, align_tokens
from .schema import read_table
from .submission import (
    clean_prediction_preserving_internal_spaces,
    validate_submission,
    write_submission,
)


EXPECTED_SOURCE_COUNT = 5
MAJORITY_VOTES = 3
TOKEN_RE = re.compile(r"\S+")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(text: str) -> tuple[Token, ...]:
    return tuple(Token(match.group(), match.start(), match.end(), ()) for match in TOKEN_RE.finditer(text))


def _aligned_to_base(
    base: tuple[Token, ...], donor: tuple[Token, ...]
) -> tuple[list[str | None], list[tuple[str, ...]]]:
    words: list[str | None] = [None] * len(base)
    insertions: list[list[str]] = [[] for _ in range(len(base) + 1)]
    consumed = 0
    for left, right in align_tokens(base, donor):
        if left is None:
            assert right is not None
            insertions[consumed].append(donor[right].text)
            continue
        if left != consumed:
            raise RuntimeError("non-monotonic base alignment")
        words[left] = donor[right].text if right is not None else None
        consumed = left + 1
    if consumed != len(base):
        raise RuntimeError("incomplete base alignment")
    return words, [tuple(values) for values in insertions]


def consensus_line(
    base_text: str,
    source_predictions: Sequence[str],
) -> tuple[str, dict]:
    """Return a line changed only by an exact three-of-five aligned vote."""

    if len(source_predictions) != EXPECTED_SOURCE_COUNT:
        raise ValueError("exactly five source predictions are required")
    base_text = clean_prediction_preserving_internal_spaces(base_text)
    sources = [clean_prediction_preserving_internal_spaces(value) for value in source_predictions]
    if not base_text or any(not value for value in sources):
        raise ValueError("base and all five source predictions must be nonempty")

    base = _tokens(base_text)
    aligned = [_aligned_to_base(base, _tokens(value)) for value in sources]
    result: list[str] = []
    changes: list[dict] = []

    def majority(values: Iterable[object], fallback: object) -> tuple[object, int]:
        counts = Counter(values)
        value, votes = min(counts.items(), key=lambda item: (-item[1], repr(item[0])))
        return (value, votes) if votes >= MAJORITY_VOTES else (fallback, counts[fallback])

    for slot in range(len(base) + 1):
        insertion, votes = majority(
            (insertions[slot] for _, insertions in aligned), (),
        )
        assert isinstance(insertion, tuple)
        if insertion:
            result.extend(insertion)
            changes.append(
                {
                    "kind": "insertion",
                    "slot": slot,
                    "before": [],
                    "after": list(insertion),
                    "votes": votes,
                }
            )
        if slot == len(base):
            continue

        fallback = base[slot].text
        word, votes = majority((words[slot] for words, _ in aligned), fallback)
        if word is None:
            changes.append(
                {
                    "kind": "deletion",
                    "token_index": slot,
                    "before": fallback,
                    "after": None,
                    "votes": votes,
                }
            )
        else:
            assert isinstance(word, str)
            result.append(word)
            if word != fallback:
                changes.append(
                    {
                        "kind": "substitution",
                        "token_index": slot,
                        "before": fallback,
                        "after": word,
                        "votes": votes,
                    }
                )

    prediction = " ".join(result).strip()
    if not prediction:
        prediction = base_text
        changes = []
    return prediction, {"before": base_text, "after": prediction, "changes": changes}


def apply(
    input_path: str | Path,
    base_path: str | Path,
    source_specs: Iterable[str],
    output_path: str | Path,
    diagnostic_path: str | Path,
    report_path: str | Path,
    sample_path: str | Path | None = None,
    submission_path: str | Path | None = None,
) -> dict:
    if submission_path is not None and sample_path is None:
        raise ValueError("sample_path is required with submission_path")
    source_specs = list(source_specs)
    table = read_table(input_path)
    ids = [row[table.id_column].strip() for row in table.rows]
    sources = parse_five_sources(source_specs)
    validate_complete_sources(sources, ids)

    base = read_source(base_path)
    if set(base) != set(ids):
        raise ValueError("base IDs do not match input IDs")

    predictions: list[str] = []
    diagnostics: list[dict] = []
    for identifier in ids:
        source_values = [records[identifier].candidates[0] for records in sources.values()]
        prediction, diagnostic = consensus_line(base[identifier].candidates[0], source_values)
        diagnostic["id"] = identifier
        diagnostic["source_predictions"] = dict(zip(sources, source_values, strict=True))
        predictions.append(prediction)
        diagnostics.append(diagnostic)

    output = Path(output_path).resolve()
    write_prediction_csv(output, table.id_column, ids, predictions)
    diagnostic_output = Path(diagnostic_path).resolve()
    diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
    with diagnostic_output.open("w", encoding="utf-8") as handle:
        for diagnostic in diagnostics:
            handle.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")

    source_paths = {
        spec.split("=", 1)[0].strip(): Path(spec.split("=", 1)[1].strip()).resolve()
        for spec in source_specs
    }
    change_counts = Counter(
        change["kind"] for diagnostic in diagnostics for change in diagnostic["changes"]
    )
    result = {
        "schema_version": 1,
        "policy": "exact 3-of-5 aligned word majority; Model81 medoid fallback",
        "rows": len(ids),
        "changed_rows": sum(diagnostic["before"] != diagnostic["after"] for diagnostic in diagnostics),
        "change_counts": dict(sorted(change_counts.items())),
        "source_count": len(sources),
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "base": {"path": str(Path(base_path).resolve()), "sha256": _sha256(Path(base_path).resolve())},
        "output": str(output),
        "output_sha256": _sha256(output),
        "diagnostics": str(diagnostic_output),
        "diagnostics_sha256": _sha256(diagnostic_output),
    }
    if submission_path is not None:
        prediction_map = dict(zip(ids, predictions, strict=True))
        submission = write_submission(
            sample_path,
            submission_path,
            prediction_map,
            cleaner=clean_prediction_preserving_internal_spaces,
        )
        result["submission"] = str(submission)
        result["submission_sha256"] = _sha256(Path(submission))
        result["submission_validation"] = validate_submission(submission, sample_path)

    report = Path(report_path).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    apply(
        args.input,
        args.base,
        args.source,
        args.output,
        args.diagnostics,
        args.report,
        args.sample,
        args.submission,
    )


if __name__ == "__main__":
    main()
