#!/usr/bin/env python3
"""Development/confirmation screen for conservative character-level ROVER."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from road_htr.metrics import competition_error
from road_htr.model34_token_fusion import Token, align_tokens
from road_htr.schema import predictions_by_id, read_table
from road_htr.submission import clean_prediction_preserving_internal_spaces


ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / "data/processed/train_folds.csv"
BASE = ROOT / "outputs/model87/word_rover_oof.csv"
OUT = ROOT / "outputs/model99"
TOKEN_RE = re.compile(r"\S+")
THRESHOLDS = (3, 4, 5)


def prediction_map(path: Path) -> dict[str, str]:
    table = read_table(path)
    column = table.target_column
    if column is None:
        columns = [name for name in table.fieldnames if name != table.id_column]
        if len(columns) != 1:
            raise ValueError(f"ambiguous prediction column: {path}")
        column = columns[0]
    return dict(predictions_by_id(table, column))


def tokens(text: str) -> tuple[Token, ...]:
    return tuple(Token(m.group(), m.start(), m.end(), ()) for m in TOKEN_RE.finditer(text))


def align_to_base(base: str, donor: str) -> tuple[list[str | None], list[tuple[str, ...]]]:
    """Map donor characters and insertions onto one base word."""

    mapped: list[str | None] = [None] * len(base)
    insertions: list[list[str]] = [[] for _ in range(len(base) + 1)]
    rows, columns = len(base) + 1, len(donor) + 1
    costs = [[0] * columns for _ in range(rows)]
    back: list[list[str | None]] = [[None] * columns for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], back[i][0] = i, "delete"
    for j in range(1, columns):
        costs[0][j], back[0][j] = j, "insert"
    priority = {"diagonal": 0, "delete": 1, "insert": 2}
    for i in range(1, rows):
        for j in range(1, columns):
            options = (
                (costs[i - 1][j - 1] + (base[i - 1] != donor[j - 1]), "diagonal"),
                (costs[i - 1][j] + 1, "delete"),
                (costs[i][j - 1] + 1, "insert"),
            )
            costs[i][j], back[i][j] = min(options, key=lambda x: (x[0], priority[x[1]]))
    steps: list[tuple[int | None, int | None]] = []
    i, j = len(base), len(donor)
    while i or j:
        action = back[i][j]
        if action == "diagonal":
            steps.append((i - 1, j - 1)); i -= 1; j -= 1
        elif action == "delete":
            steps.append((i - 1, None)); i -= 1
        elif action == "insert":
            steps.append((None, j - 1)); j -= 1
        else:
            raise RuntimeError("corrupt alignment")
    consumed = 0
    for left, right in reversed(steps):
        if left is None:
            assert right is not None
            insertions[consumed].append(donor[right])
        else:
            if left != consumed:
                raise RuntimeError("non-monotonic character alignment")
            mapped[left] = donor[right] if right is not None else None
            consumed += 1
    return mapped, [tuple(value) for value in insertions]


def majority(values: list[object], fallback: object, threshold: int) -> object:
    counts = Counter(values)
    value, votes = min(counts.items(), key=lambda item: (-item[1], repr(item[0])))
    return value if votes >= threshold else fallback


def fuse_word(base: str, donors: list[str | None], threshold: int) -> str:
    aligned = [align_to_base(base, donor) if donor is not None else None for donor in donors]
    output: list[str] = []
    for slot in range(len(base) + 1):
        insertion_values = [item[1][slot] if item is not None else () for item in aligned]
        insertion = majority(insertion_values, (), threshold)
        assert isinstance(insertion, tuple)
        output.extend(insertion)
        if slot == len(base):
            continue
        values = [item[0][slot] if item is not None else None for item in aligned]
        character = majority(values, base[slot], threshold)
        if character is not None:
            output.append(str(character))
    result = "".join(output)
    return result or base


def fuse_line(base_text: str, source_texts: list[str], threshold: int) -> str:
    base_text = clean_prediction_preserving_internal_spaces(base_text)
    base_tokens = tokens(base_text)
    aligned_sources: list[list[str | None]] = []
    for source in source_texts:
        donor_tokens = tokens(clean_prediction_preserving_internal_spaces(source))
        mapping: list[str | None] = [None] * len(base_tokens)
        for left, right in align_tokens(base_tokens, donor_tokens):
            if left is not None:
                mapping[left] = donor_tokens[right].text if right is not None else None
        aligned_sources.append(mapping)
    words = [
        fuse_word(token.text, [source[index] for source in aligned_sources], threshold)
        for index, token in enumerate(base_tokens)
    ]
    return " ".join(words).strip() or base_text


def fold_sources(fold: int) -> list[dict[str, str]]:
    paths = [
        ROOT / f"outputs/model73/fold{fold}_oof_step400.csv",
        ROOT / f"outputs/model83/fold{fold}/enhanced.csv",
        ROOT / f"outputs/model83/fold{fold}/center.csv",
        ROOT / f"outputs/model83/fold{fold}/center_enhanced.csv",
        ROOT / f"outputs/model35/fold{fold}_epoch07_verified/predictions/03_confusions_fold{fold}.csv",
    ]
    return [prediction_map(path) for path in paths]


def score(rows: list[dict[str, str]], predictions: dict[str, str]) -> dict[str, float]:
    return competition_error(
        [row["Target"] for row in rows],
        [predictions[row["ID"]] for row in rows],
    )


def write_csv(path: Path, rows: list[dict[str, str]], predictions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"])
        writer.writeheader()
        writer.writerows({"ID": row["ID"], "prediction": predictions[row["ID"]]} for row in rows)


def main() -> None:
    truth_table = read_table(TRUTH)
    rows = list(truth_table.rows)
    base = prediction_map(BASE)
    if set(base) != {row["ID"] for row in rows}:
        raise ValueError("Model87 OOF IDs do not match truth")

    candidates: dict[int, dict[str, str]] = {threshold: {} for threshold in THRESHOLDS}
    for fold in range(5):
        fold_rows = [row for row in rows if int(row["fold"]) == fold]
        sources = fold_sources(fold)
        ids = {row["ID"] for row in fold_rows}
        if any(set(source) != ids for source in sources):
            raise ValueError(f"source IDs do not match fold {fold}")
        for row in fold_rows:
            identifier = row["ID"]
            source_texts = [source[identifier] for source in sources]
            for threshold in THRESHOLDS:
                candidates[threshold][identifier] = fuse_line(base[identifier], source_texts, threshold)

    development_rows = [row for row in rows if int(row["fold"]) == 0]
    baseline_development = score(development_rows, base)
    development_scores = {str(t): score(development_rows, p) for t, p in candidates.items()}
    chosen = max(THRESHOLDS, key=lambda t: (development_scores[str(t)]["multi_score"], t))
    confirmation_rows = [row for row in rows if int(row["fold"]) != 0]
    baseline_confirmation = score(confirmation_rows, base)
    chosen_confirmation = score(confirmation_rows, candidates[chosen])
    baseline_all = score(rows, base)
    chosen_all = score(rows, candidates[chosen])

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "character_rover_oof.csv", rows, candidates[chosen])
    report = {
        "schema_version": 1,
        "policy": "character ROVER threshold selected only on fold 0; folds 1-4 confirmation",
        "threshold_candidates": list(THRESHOLDS),
        "chosen_threshold": chosen,
        "development_fold": 0,
        "baseline_development": baseline_development,
        "development_scores": development_scores,
        "baseline_confirmation": baseline_confirmation,
        "chosen_confirmation": chosen_confirmation,
        "confirmation_delta": chosen_confirmation["multi_score"] - baseline_confirmation["multi_score"],
        "baseline_all": baseline_all,
        "chosen_all": chosen_all,
        "all_delta": chosen_all["multi_score"] - baseline_all["multi_score"],
        "changed_rows": sum(candidates[chosen][row["ID"]] != base[row["ID"]] for row in rows),
        "promoted": chosen_confirmation["multi_score"] > baseline_confirmation["multi_score"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
