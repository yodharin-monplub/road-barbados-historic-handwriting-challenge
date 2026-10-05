"""Leakage-safe confidence fusion for Model22 and Model25 OCR predictions.

The stronger Model25 transcript is always the base.  Model22 may donate a
word (or only high-confidence characters inside a word) after a deterministic
word alignment.  A finite rule grid contains an explicit no-op candidate.
During evaluation each inner split is decoded with a rule selected without
that split's references.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .metrics import (
    CER_NORMALIZER,
    WER_NORMALIZER,
    competition_error,
    levenshtein,
    strip_conversation_tokens,
)
from .schema import SchemaError, predictions_by_id, read_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN_RE = re.compile(r"\S+")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FineCandidate:
    text: str
    confidences: tuple[float, ...]


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    confidences: tuple[float, ...]


@dataclass(frozen=True)
class CharacterStep:
    base: str | None
    fine: str | None
    fine_confidence: float | None


@dataclass(frozen=True)
class Substitution:
    base_index: int
    fine_index: int
    distance_ratio: float
    has_exact_left_context: bool
    has_exact_right_context: bool
    character_steps: tuple[CharacterStep, ...]


@dataclass(frozen=True)
class PreparedRow:
    baseline: str
    fine: FineCandidate
    base_tokens: tuple[Token, ...]
    fine_tokens: tuple[Token, ...]
    substitutions: tuple[Substitution, ...]
    row_distance_ratio: float


@dataclass(frozen=True)
class Rule:
    kind: str
    granularity: str = "none"
    quality_kind: str = "q10"
    confidence_threshold: float = 1.0
    maximum_token_distance: float = 0.0
    maximum_row_distance: float = 0.0
    context: str = "none"
    token_mode: str = "any"
    allow_character_deletions: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Rule":
        fields = cls.__dataclass_fields__
        unknown = sorted(set(value) - set(fields))
        if unknown:
            raise ValueError(f"unknown fusion rule field(s): {unknown}")
        return cls(**value)


@dataclass(frozen=True)
class RowCost:
    weighted_word_edits: float
    weighted_character_edits: float
    word_weight: float
    character_weight: float


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_artifact(name: str, spec: Mapping[str, Any]) -> Path:
    if set(spec) != {"path", "sha256"}:
        raise SchemaError(f"{name} provenance must contain only path and sha256")
    path = _absolute(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(spec["sha256"]).lower()
    actual = _sha256(path)
    if actual != expected:
        raise SchemaError(
            f"{name} SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return path


def _prediction_map(path: Path) -> tuple[str, list[str], dict[str, str]]:
    table = read_table(path)
    columns = [name for name in table.fieldnames if name != table.id_column]
    prediction_column = table.target_column
    if prediction_column is None:
        if len(columns) != 1:
            raise SchemaError(
                f"prediction CSV must have exactly one value column: {path}"
            )
        prediction_column = columns[0]
    values = dict(predictions_by_id(table, prediction_column))
    identifiers = [row[table.id_column].strip() for row in table.rows]
    if any(not values[identifier] for identifier in identifiers):
        raise SchemaError(f"blank prediction in {path}")
    return table.id_column, identifiers, values


def read_fine_candidates(path: str | Path) -> tuple[list[str], dict[str, FineCandidate]]:
    """Read Kraken JSONL while preserving confidence-to-character positions."""

    identifiers: list[str] = []
    records: dict[str, FineCandidate] = {}
    with _absolute(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SchemaError(f"invalid JSON at {path}:{line_number}") from error
            identifier = str(row.get("id", "")).strip()
            candidates = row.get("candidates")
            raw_confidences = row.get("character_confidences")
            if not identifier or not isinstance(candidates, list) or not candidates:
                raise SchemaError(f"missing fine candidate at {path}:{line_number}")
            if identifier in records:
                raise SchemaError(f"duplicate ID {identifier!r} in {path}")
            text = str(candidates[0])
            if not text:
                raise SchemaError(f"blank fine candidate for {identifier!r}")
            if not isinstance(raw_confidences, list):
                raise SchemaError(
                    f"missing character confidences for {identifier!r}"
                )
            confidences = tuple(float(value) for value in raw_confidences)
            if len(confidences) not in {len(text), len(text) + 1}:
                raise SchemaError(
                    f"confidence length mismatch for {identifier!r}: "
                    f"{len(confidences)} not in {{{len(text)}, {len(text)+1}}}"
                )
            if len(confidences) == len(text) + 1:
                # Kraken can emit a final CTC boundary confidence which has no
                # character in the decoded string.
                confidences = confidences[:-1]
            if any(
                not math.isfinite(value) or value < 0.0 or value > 1.0
                for value in confidences
            ):
                raise SchemaError(f"invalid confidence for {identifier!r}")
            identifiers.append(identifier)
            records[identifier] = FineCandidate(text, confidences)
    if not records:
        raise SchemaError(f"no fine candidates in {path}")
    return identifiers, records


def _tokens(candidate: FineCandidate) -> tuple[Token, ...]:
    return tuple(
        Token(
            match.group(),
            match.start(),
            match.end(),
            candidate.confidences[match.start() : match.end()],
        )
        for match in TOKEN_RE.finditer(candidate.text)
    )


def _plain_tokens(text: str) -> tuple[Token, ...]:
    return tuple(
        Token(match.group(), match.start(), match.end(), ())
        for match in TOKEN_RE.finditer(text)
    )


def _normalized_token(value: str) -> str:
    """Apply only the already-validated deterministic Model23 symbol repair."""

    return value.replace("+", "&")


def _token_distance(left: str, right: str) -> float:
    left_key = _normalized_token(left).casefold()
    right_key = _normalized_token(right).casefold()
    return levenshtein(tuple(left_key), tuple(right_key)) / max(
        len(left_key), len(right_key), 1
    )


def align_tokens(
    baseline: Sequence[Token], fine: Sequence[Token]
) -> tuple[tuple[int | None, int | None], ...]:
    """Minimum-cost word alignment with deterministic diagonal tie-breaking."""

    rows = len(baseline) + 1
    columns = len(fine) + 1
    costs = [[0.0] * columns for _ in range(rows)]
    back: list[list[str | None]] = [[None] * columns for _ in range(rows)]
    for index in range(1, rows):
        costs[index][0] = float(index)
        back[index][0] = "delete"
    for index in range(1, columns):
        costs[0][index] = float(index)
        back[0][index] = "insert"
    priority = {"diagonal": 0, "delete": 1, "insert": 2}
    for i in range(1, rows):
        for j in range(1, columns):
            options = (
                (
                    costs[i - 1][j - 1]
                    + _token_distance(baseline[i - 1].text, fine[j - 1].text),
                    "diagonal",
                ),
                (costs[i - 1][j] + 1.0, "delete"),
                (costs[i][j - 1] + 1.0, "insert"),
            )
            costs[i][j], back[i][j] = min(
                options, key=lambda item: (item[0], priority[item[1]])
            )
    result: list[tuple[int | None, int | None]] = []
    i, j = len(baseline), len(fine)
    while i or j:
        action = back[i][j]
        if action == "diagonal":
            result.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == "delete":
            result.append((i - 1, None))
            i -= 1
        elif action == "insert":
            result.append((None, j - 1))
            j -= 1
        else:  # pragma: no cover - only reachable if the DP is corrupted
            raise RuntimeError(f"invalid alignment backpointer at {i}, {j}")
    result.reverse()
    return tuple(result)


def align_characters(base: str, fine: Token) -> tuple[CharacterStep, ...]:
    """Unit-cost character alignment carrying each donor character confidence."""

    donor = _normalized_token(fine.text)
    rows = len(base) + 1
    columns = len(donor) + 1
    costs = [[0] * columns for _ in range(rows)]
    back: list[list[str | None]] = [[None] * columns for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0] = i
        back[i][0] = "delete"
    for j in range(1, columns):
        costs[0][j] = j
        back[0][j] = "insert"
    priority = {"diagonal": 0, "delete": 1, "insert": 2}
    for i in range(1, rows):
        for j in range(1, columns):
            options = (
                (
                    costs[i - 1][j - 1]
                    + (base[i - 1].casefold() != donor[j - 1].casefold()),
                    "diagonal",
                ),
                (costs[i - 1][j] + 1, "delete"),
                (costs[i][j - 1] + 1, "insert"),
            )
            costs[i][j], back[i][j] = min(
                options, key=lambda item: (item[0], priority[item[1]])
            )
    result: list[CharacterStep] = []
    i, j = len(base), len(donor)
    while i or j:
        action = back[i][j]
        if action == "diagonal":
            result.append(
                CharacterStep(base[i - 1], donor[j - 1], fine.confidences[j - 1])
            )
            i -= 1
            j -= 1
        elif action == "delete":
            result.append(CharacterStep(base[i - 1], None, None))
            i -= 1
        elif action == "insert":
            result.append(
                CharacterStep(None, donor[j - 1], fine.confidences[j - 1])
            )
            j -= 1
        else:  # pragma: no cover
            raise RuntimeError(f"invalid character backpointer at {i}, {j}")
    result.reverse()
    return tuple(result)


def prepare_row(baseline: str, fine: FineCandidate) -> PreparedRow:
    base_tokens = _plain_tokens(baseline)
    fine_tokens = _tokens(fine)
    alignment = align_tokens(base_tokens, fine_tokens)
    mapped = [(left, right) for left, right in alignment if left is not None and right is not None]
    exact = {
        (left, right): (
            _normalized_token(base_tokens[left].text).casefold()
            == _normalized_token(fine_tokens[right].text).casefold()
        )
        for left, right in mapped
    }
    substitutions: list[Substitution] = []
    for position, (left, right) in enumerate(alignment):
        if left is None or right is None or exact[(left, right)]:
            continue
        previous = alignment[position - 1] if position else (None, None)
        following = (
            alignment[position + 1]
            if position + 1 < len(alignment)
            else (None, None)
        )
        left_context = (
            previous[0] is not None
            and previous[1] is not None
            and exact.get((previous[0], previous[1]), False)
        )
        right_context = (
            following[0] is not None
            and following[1] is not None
            and exact.get((following[0], following[1]), False)
        )
        substitutions.append(
            Substitution(
                base_index=left,
                fine_index=right,
                distance_ratio=_token_distance(
                    base_tokens[left].text, fine_tokens[right].text
                ),
                has_exact_left_context=left_context,
                has_exact_right_context=right_context,
                character_steps=align_characters(
                    base_tokens[left].text, fine_tokens[right]
                ),
            )
        )
    base_words = [
        _normalized_token(token.text).casefold() for token in base_tokens
    ]
    fine_words = [
        _normalized_token(token.text).casefold() for token in fine_tokens
    ]
    row_distance = levenshtein(base_words, fine_words) / max(
        len(base_words), len(fine_words), 1
    )
    return PreparedRow(
        baseline=baseline,
        fine=fine,
        base_tokens=base_tokens,
        fine_tokens=fine_tokens,
        substitutions=tuple(substitutions),
        row_distance_ratio=row_distance,
    )


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def confidence_quality(values: Sequence[float], kind: str) -> float:
    if not values:
        return -1.0
    if kind == "min":
        return min(values)
    if kind == "q10":
        return _quantile(values, 0.10)
    if kind == "q25":
        return _quantile(values, 0.25)
    if kind == "mean":
        return fmean(values)
    raise ValueError(f"unknown confidence quality {kind!r}")


def _context_allows(item: Substitution, context: str) -> bool:
    if context == "none":
        return True
    if context == "one":
        return item.has_exact_left_context or item.has_exact_right_context
    if context == "both":
        return item.has_exact_left_context and item.has_exact_right_context
    raise ValueError(f"unknown context gate {context!r}")


def _specials_preserved(base: str, fine: str, protected: str) -> bool:
    return all(fine.count(character) >= base.count(character) for character in protected)


def _character_patch(item: Substitution, threshold: float) -> str:
    output: list[str] = []
    for step in item.character_steps:
        if step.base is not None and step.fine is not None:
            if step.base.casefold() == step.fine.casefold():
                # Model25 owns case; Model22 can only donate spelling.
                output.append(step.base)
            elif (step.fine_confidence or -1.0) >= threshold:
                output.append(step.fine)
            else:
                output.append(step.base)
        elif step.fine is not None:
            if (step.fine_confidence or -1.0) >= threshold:
                output.append(step.fine)
        elif step.base is not None:
            output.append(step.base)
    return "".join(output)


def fuse_prepared(
    row: PreparedRow, rule: Rule, *, protected_characters: str = "^&"
) -> str:
    """Apply one inference-observable rule, always starting from Model25."""

    if rule.kind == "noop":
        return row.baseline
    if rule.kind != "fusion":
        raise ValueError(f"unknown rule kind {rule.kind!r}")
    if row.row_distance_ratio > rule.maximum_row_distance:
        return row.baseline

    replacements: dict[int, str] = {}
    for item in row.substitutions:
        base = row.base_tokens[item.base_index].text
        fine_token = row.fine_tokens[item.fine_index]
        donor = _normalized_token(fine_token.text)
        if item.distance_ratio > rule.maximum_token_distance:
            continue
        if not _context_allows(item, rule.context):
            continue
        if rule.token_mode == "alphabetic" and not (
            base.isalpha() and donor.isalpha()
        ):
            continue
        if rule.token_mode not in {"any", "alphabetic"}:
            raise ValueError(f"unknown token mode {rule.token_mode!r}")
        if not _specials_preserved(base, donor, protected_characters):
            continue
        quality = confidence_quality(fine_token.confidences, rule.quality_kind)
        if quality < rule.confidence_threshold:
            continue
        if rule.granularity == "token":
            replacement = donor
        elif rule.granularity == "character":
            replacement = _character_patch(item, rule.confidence_threshold)
        else:
            raise ValueError(f"unknown fusion granularity {rule.granularity!r}")
        if replacement != base:
            replacements[item.base_index] = replacement

    if not replacements:
        return row.baseline
    values = [token.text for token in row.base_tokens]
    for index, value in replacements.items():
        values[index] = value
    return " ".join(values)


def candidate_rules(config: Mapping[str, Any]) -> list[Rule]:
    space = config["candidate_space"]
    rules = [Rule(kind="noop")]
    for granularity in space["granularities"]:
        for quality_kind in space["quality_kinds"]:
            for threshold in space["confidence_thresholds"]:
                for token_distance in space["maximum_token_distances"]:
                    for row_distance in space["maximum_row_distances"]:
                        for context in space["contexts"]:
                            for token_mode in space["token_modes"]:
                                rules.append(
                                    Rule(
                                        kind="fusion",
                                        granularity=str(granularity),
                                        quality_kind=str(quality_kind),
                                        confidence_threshold=float(threshold),
                                        maximum_token_distance=float(token_distance),
                                        maximum_row_distance=float(row_distance),
                                        context=str(context),
                                        token_mode=str(token_mode),
                                    )
                                )
    if len({json.dumps(asdict(rule), sort_keys=True) for rule in rules}) != len(rules):
        raise ValueError("candidate rule grid contains duplicates")
    return rules


def _row_cost(reference: str, hypothesis: str) -> RowCost:
    ref = strip_conversation_tokens(reference)
    hyp = strip_conversation_tokens(hypothesis)
    ref_words = ref.split()
    hyp_words = hyp.split()
    word_weight = math.sqrt(max(len(ref_words), 1))
    character_weight = math.sqrt(max(len(ref), 1))
    return RowCost(
        levenshtein(ref_words, hyp_words) * word_weight,
        levenshtein(tuple(ref), tuple(hyp)) * character_weight,
        word_weight,
        character_weight,
    )


def _score_costs(costs: Iterable[RowCost]) -> dict[str, float]:
    values = tuple(costs)
    if not values:
        raise ValueError("cannot score an empty split")
    wer = sum(item.weighted_word_edits for item in values) / sum(
        item.word_weight for item in values
    )
    cer = sum(item.weighted_character_edits for item in values) / sum(
        item.character_weight for item in values
    )
    mean_error = (wer / WER_NORMALIZER + cer / CER_NORMALIZER) / 2
    return {
        "weighted_wer": wer,
        "weighted_cer": cer,
        "mean_error": mean_error,
        "multi_score": 1 - mean_error,
    }


def paired_bootstrap(
    baseline_costs: Sequence[RowCost],
    challenger_costs: Sequence[RowCost],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Deterministic paired row bootstrap of challenger multi-score gain."""

    if len(baseline_costs) != len(challenger_costs) or not baseline_costs:
        raise ValueError("paired bootstrap requires equally sized non-empty inputs")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    generator = random.Random(seed)
    count = len(baseline_costs)
    gains: list[float] = []
    for _ in range(samples):
        indices = [generator.randrange(count) for _ in range(count)]
        baseline_score = _score_costs(baseline_costs[index] for index in indices)
        challenger_score = _score_costs(challenger_costs[index] for index in indices)
        gains.append(
            float(challenger_score["multi_score"])
            - float(baseline_score["multi_score"])
        )
    gains.sort()
    lower_index = max(0, math.floor(0.025 * samples))
    upper_index = min(samples - 1, math.ceil(0.975 * samples) - 1)
    return {
        "samples": samples,
        "seed": seed,
        "mean_gain": fmean(gains),
        "percentile_2_5": gains[lower_index],
        "percentile_97_5": gains[upper_index],
        "positive_fraction": sum(gain > 0 for gain in gains) / samples,
    }


def _partition(identifier: str, salt: str, count: int) -> int:
    digest = hashlib.sha256(f"{salt}\0{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, id_column: str, ids: Sequence[str], values: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[id_column, "prediction"])
            writer.writeheader()
            for identifier in ids:
                writer.writerow({id_column: identifier, "prediction": values[identifier]})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_verified_fold(
    config: Mapping[str, Any]
) -> tuple[str, list[str], dict[str, str], dict[str, str], dict[str, FineCandidate], dict[str, str]]:
    """Load the canonical fold and verify the complete Model22 lineage."""

    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise SchemaError("unsupported Model34 config schema")
    provenance = config["provenance"]
    truth_path = _verify_artifact("truth", provenance["truth"])
    baseline_path = _verify_artifact("baseline", provenance["baseline"])
    fine_path = _verify_artifact("fine_jsonl", provenance["fine_jsonl"])
    fine_csv_path = _verify_artifact("fine_csv", provenance["fine_csv"])
    manifest_path = _verify_artifact("fine_manifest", provenance["fine_manifest"])
    receipt_path = _verify_artifact("fine_evaluation", provenance["fine_evaluation"])

    truth = read_table(truth_path, require_target=True)
    if "fold" not in truth.fieldnames:
        raise SchemaError("truth must contain a fold column")
    assert truth.target_column is not None
    truth_ids = [row[truth.id_column].strip() for row in truth.rows]
    truth_by_id = {
        row[truth.id_column].strip(): row[truth.target_column] for row in truth.rows
    }
    outer_fold = int(config["outer_fold"])
    selected_ids = [
        row[truth.id_column].strip()
        for row in truth.rows
        if int(row["fold"]) == outer_fold
    ]
    expected_truth_rows = int(provenance["expected_truth_rows"])
    expected_fold_rows = int(provenance["expected_fold_rows"])
    if len(truth_ids) != expected_truth_rows or len(selected_ids) != expected_fold_rows:
        raise SchemaError(
            f"row-count mismatch: truth={len(truth_ids)}, fold={len(selected_ids)}"
        )

    baseline_id_column, baseline_ids, baseline = _prediction_map(baseline_path)
    if baseline_id_column.casefold() != truth.id_column.casefold():
        raise SchemaError("baseline/truth ID column mismatch")
    if set(baseline_ids) != set(truth_ids):
        raise SchemaError("baseline and truth IDs differ")

    fine_ids, fine = read_fine_candidates(fine_path)
    if set(fine_ids) != set(selected_ids):
        raise SchemaError("fine JSONL and requested outer-fold IDs differ")
    _, companion_ids, companion = _prediction_map(fine_csv_path)
    if set(companion_ids) != set(selected_ids):
        raise SchemaError("fine CSV and requested outer-fold IDs differ")
    mismatched_text = [
        identifier
        for identifier in selected_ids
        if fine[identifier].text != companion[identifier]
    ]
    if mismatched_text:
        raise SchemaError(
            f"fine JSONL/CSV text mismatch, examples={mismatched_text[:5]}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    truth_sha = str(provenance["truth"]["sha256"])
    fine_sha = str(provenance["fine_jsonl"]["sha256"])
    fine_csv_sha = str(provenance["fine_csv"]["sha256"])
    manifest_sha = str(provenance["fine_manifest"]["sha256"])
    required_manifest = {
        "input_sha256": truth_sha,
        "fold": outer_fold,
        "rows": expected_fold_rows,
        "preserve_internal_spaces": True,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise SchemaError(
                f"fine manifest {key} mismatch: {manifest.get(key)!r} != {expected!r}"
            )
    if receipt.get("prediction_details_sha256") != fine_sha:
        raise SchemaError("evaluation receipt does not authenticate fine JSONL")
    if receipt.get("predictions_sha256") != fine_csv_sha:
        raise SchemaError("evaluation receipt does not authenticate fine CSV")
    if receipt.get("inference_manifest_sha256") != manifest_sha:
        raise SchemaError("evaluation receipt does not authenticate manifest")
    signature = receipt.get("signature") or {}
    if signature.get("input_sha256") != truth_sha or signature.get("fold") != outer_fold:
        raise SchemaError("evaluation signature has the wrong truth/fold lineage")
    if signature.get("checkpoint") != provenance["checkpoint_name"]:
        raise SchemaError("evaluation signature has the wrong checkpoint")

    return (
        truth.id_column,
        selected_ids,
        truth_by_id,
        baseline,
        fine,
        {
            "truth_sha256": truth_sha,
            "baseline_sha256": str(provenance["baseline"]["sha256"]),
            "fine_jsonl_sha256": fine_sha,
            "fine_csv_sha256": fine_csv_sha,
            "fine_manifest_sha256": manifest_sha,
            "fine_evaluation_sha256": str(provenance["fine_evaluation"]["sha256"]),
        },
    )


def crossfit(config_path: str | Path, output_path: str | Path, report_path: str | Path, deployment_path: str | Path) -> dict[str, Any]:
    config_file = _absolute(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    id_column, ids, truth, baseline, fine, lineage = load_verified_fold(config)
    prepared = {identifier: prepare_row(baseline[identifier], fine[identifier]) for identifier in ids}
    rules = candidate_rules(config)
    inner_folds = int(config["inner_folds"])
    salt = str(config["partition_salt"])
    minimum_gain = float(config["selection"]["minimum_training_gain"])
    partitions = {identifier: _partition(identifier, salt, inner_folds) for identifier in ids}
    counts = {fold: sum(value == fold for value in partitions.values()) for fold in range(inner_folds)}
    if any(count == 0 for count in counts.values()):
        raise SchemaError(f"empty deterministic inner fold: {counts}")

    reference_costs = {identifier: _row_cost(truth[identifier], baseline[identifier]) for identifier in ids}
    cost_cache: dict[tuple[str, str], RowCost] = {
        (identifier, baseline[identifier]): reference_costs[identifier]
        for identifier in ids
    }
    best: dict[
        int | str,
        tuple[tuple[float, float, int, int], Rule, dict[str, float], float],
    ] = {}
    noop_scores: dict[int | str, dict[str, float]] = {}
    scopes: list[int | str] = [*range(inner_folds), "all"]
    baseline_partition_scores = {
        fold: _score_costs(
            reference_costs[identifier]
            for identifier in ids
            if partitions[identifier] == fold
        )
        for fold in range(inner_folds)
    }
    for order, rule in enumerate(rules):
        predictions = {
            identifier: fuse_prepared(
                prepared[identifier],
                rule,
                protected_characters=str(config["protected_characters"]),
            )
            for identifier in ids
        }
        costs: dict[str, RowCost] = {}
        for identifier in ids:
            cache_key = (identifier, predictions[identifier])
            if cache_key not in cost_cache:
                cost_cache[cache_key] = _row_cost(
                    truth[identifier], predictions[identifier]
                )
            costs[identifier] = cost_cache[cache_key]
        partition_scores = {
            fold: _score_costs(
                costs[identifier]
                for identifier in ids
                if partitions[identifier] == fold
            )
            for fold in range(inner_folds)
        }
        for scope in scopes:
            training_partitions = (
                list(range(inner_folds))
                if scope == "all"
                else [fold for fold in range(inner_folds) if fold != scope]
            )
            training_ids = (
                ids
                if scope == "all"
                else [identifier for identifier in ids if partitions[identifier] != scope]
            )
            score = _score_costs(costs[identifier] for identifier in training_ids)
            worst_partition_gain = min(
                float(partition_scores[fold]["multi_score"])
                - float(baseline_partition_scores[fold]["multi_score"])
                for fold in training_partitions
            )
            changed = sum(
                predictions[identifier] != baseline[identifier]
                for identifier in training_ids
            )
            # Conservative maximin selection: prefer a rule whose weakest
            # available training subgroup improves the most, then its exact
            # aggregate challenge error, then fewer edits.  The held subgroup
            # never participates in its own selection.
            key = (
                -worst_partition_gain,
                float(score["mean_error"]),
                changed,
                order,
            )
            if rule.kind == "noop":
                noop_scores[scope] = score
            if scope not in best or key < best[scope][0]:
                best[scope] = (key, rule, score, worst_partition_gain)

    # A learned rule must clear a predeclared training margin.  Otherwise this
    # model fails closed to byte-for-byte Model25 output.
    for scope in scopes:
        key, rule, score, worst_partition_gain = best[scope]
        if worst_partition_gain < minimum_gain:
            best[scope] = (
                (0.0, float(noop_scores[scope]["mean_error"]), 0, 0),
                rules[0],
                noop_scores[scope],
                0.0,
            )

    selected: dict[str, str] = {}
    split_reports: list[dict[str, Any]] = []
    for held_out in range(inner_folds):
        validation_ids = [identifier for identifier in ids if partitions[identifier] == held_out]
        training_ids = [identifier for identifier in ids if partitions[identifier] != held_out]
        _, rule, training_score, worst_partition_gain = best[held_out]
        values = {
            identifier: fuse_prepared(
                prepared[identifier],
                rule,
                protected_characters=str(config["protected_characters"]),
            )
            for identifier in validation_ids
        }
        selected.update(values)
        validation_score = competition_error(
            [truth[identifier] for identifier in validation_ids],
            [values[identifier] for identifier in validation_ids],
        )
        validation_baseline_score = competition_error(
            [truth[identifier] for identifier in validation_ids],
            [baseline[identifier] for identifier in validation_ids],
        )
        split_reports.append(
            {
                "held_out_inner_fold": held_out,
                "training_rows": len(training_ids),
                "validation_rows": len(validation_ids),
                "rule": asdict(rule),
                "training_score": training_score,
                "training_gain_over_noop": float(training_score["multi_score"])
                - float(noop_scores[held_out]["multi_score"]),
                "worst_training_partition_gain": worst_partition_gain,
                "validation_score": validation_score,
                "validation_baseline_score": validation_baseline_score,
                "validation_gain_over_baseline": float(validation_score["multi_score"])
                - float(validation_baseline_score["multi_score"]),
                "changed_validation_rows": sum(
                    values[identifier] != baseline[identifier]
                    for identifier in validation_ids
                ),
            }
        )

    baseline_score = competition_error(
        [truth[identifier] for identifier in ids],
        [baseline[identifier] for identifier in ids],
    )
    crossfit_score = competition_error(
        [truth[identifier] for identifier in ids],
        [selected[identifier] for identifier in ids],
    )
    bootstrap_config = config["bootstrap"]
    crossfit_costs = [
        _row_cost(truth[identifier], selected[identifier]) for identifier in ids
    ]
    bootstrap = paired_bootstrap(
        [reference_costs[identifier] for identifier in ids],
        crossfit_costs,
        samples=int(bootstrap_config["samples"]),
        seed=int(bootstrap_config["seed"]),
    )
    _, deployment_rule, deployment_training_score, deployment_worst_gain = best["all"]
    output = _absolute(output_path)
    report_file = _absolute(report_path)
    deployment_file = _absolute(deployment_path)
    _write_csv(output, id_column, ids, selected)
    output_sha = _sha256(output)
    report = {
        "schema_version": SCHEMA_VERSION,
        "method": "deterministic inner cross-fit; held-out labels excluded from rule selection",
        "outer_fold": int(config["outer_fold"]),
        "rows": len(ids),
        "candidate_rule_count": len(rules),
        "inner_folds": inner_folds,
        "partition_salt": salt,
        "minimum_training_gain": minimum_gain,
        "selection_objective": "maximize worst available training-partition gain",
        "baseline_score": baseline_score,
        "crossfit_score": crossfit_score,
        "crossfit_gain_over_baseline": float(crossfit_score["multi_score"])
        - float(baseline_score["multi_score"]),
        "changed_rows": sum(selected[i] != baseline[i] for i in ids),
        "paired_row_bootstrap": bootstrap,
        "split_reports": split_reports,
        "lineage": lineage,
        "config_sha256": _sha256(config_file),
        "output": str(output),
        "output_sha256": output_sha,
        "deployment_rule_note": (
            "Selected on all fold-0 labels for future test deployment only; "
            "it is not used to compute the cross-fitted fold-0 estimate."
        ),
        "deployment_rule": asdict(deployment_rule),
        "deployment_training_score": deployment_training_score,
        "deployment_worst_partition_gain": deployment_worst_gain,
    }
    _atomic_json(report_file, report)
    deployment = {
        "schema_version": SCHEMA_VERSION,
        "rule": asdict(deployment_rule),
        "protected_characters": str(config["protected_characters"]),
        "minimum_training_gain": minimum_gain,
        "training_lineage": lineage,
        "source_report": str(report_file),
        "source_report_sha256": _sha256(report_file),
    }
    _atomic_json(deployment_file, deployment)
    return report


def apply_rule_files(
    baseline_path: str | Path,
    fine_path: str | Path,
    deployment_path: str | Path,
    output_path: str | Path,
    expected_baseline_sha256: str,
    expected_fine_sha256: str,
) -> dict[str, Any]:
    """Apply a frozen rule, requiring caller-supplied hashes and exact IDs."""

    baseline_file = _absolute(baseline_path)
    fine_file = _absolute(fine_path)
    if _sha256(baseline_file) != expected_baseline_sha256.lower():
        raise SchemaError("deployment baseline SHA-256 mismatch")
    if _sha256(fine_file) != expected_fine_sha256.lower():
        raise SchemaError("deployment fine JSONL SHA-256 mismatch")
    id_column, ids, baseline = _prediction_map(baseline_file)
    fine_ids, fine = read_fine_candidates(fine_file)
    if set(ids) != set(fine_ids):
        raise SchemaError("deployment baseline/fine IDs differ")
    deployment_file = _absolute(deployment_path)
    deployment = json.loads(deployment_file.read_text(encoding="utf-8"))
    if int(deployment.get("schema_version", -1)) != SCHEMA_VERSION:
        raise SchemaError("unsupported deployment schema")
    rule = Rule.from_mapping(deployment["rule"])
    protected = str(deployment["protected_characters"])
    values = {
        identifier: fuse_prepared(
            prepare_row(baseline[identifier], fine[identifier]),
            rule,
            protected_characters=protected,
        )
        for identifier in ids
    }
    output = _absolute(output_path)
    _write_csv(output, id_column, ids, values)
    return {
        "rows": len(ids),
        "changed_rows": sum(values[i] != baseline[i] for i in ids),
        "rule": asdict(rule),
        "output": str(output),
        "output_sha256": _sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    crossfit_parser = subparsers.add_parser("crossfit")
    crossfit_parser.add_argument("--config", type=Path, required=True)
    crossfit_parser.add_argument("--output", type=Path, required=True)
    crossfit_parser.add_argument("--report", type=Path, required=True)
    crossfit_parser.add_argument("--deployment-config", type=Path, required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--baseline", type=Path, required=True)
    apply_parser.add_argument("--fine", type=Path, required=True)
    apply_parser.add_argument("--deployment-config", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.add_argument("--expected-baseline-sha256", required=True)
    apply_parser.add_argument("--expected-fine-sha256", required=True)
    args = parser.parse_args()
    if args.command == "crossfit":
        result = crossfit(
            args.config, args.output, args.report, args.deployment_config
        )
    else:
        result = apply_rule_files(
            args.baseline,
            args.fine,
            args.deployment_config,
            args.output,
            args.expected_baseline_sha256,
            args.expected_fine_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
