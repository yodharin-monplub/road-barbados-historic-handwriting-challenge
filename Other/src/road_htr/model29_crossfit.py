"""Leakage-safe cross-fold selection for Model29's PP-OCR CTC decoder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .challenge_ctc import (
    CharNGramLanguageModel,
    DecoderCandidate,
    WordBoundaryPrior,
    decode_candidate,
)
from .kraken_probability_export import stable_sha256
from .metrics import competition_error
from .schema import SchemaError, read_table
from .submission import clean_prediction_preserving_internal_spaces
from .vlm_data import read_id_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_MODEL22_OUTPUT = (PROJECT_ROOT / "outputs" / "model22").resolve()
SCHEMA_VERSION = 1
MAX_GRID_CANDIDATES = 8


@dataclass(frozen=True)
class LabeledRow:
    identifier: str
    target: str
    fold: int


@dataclass(frozen=True)
class CrossfitResult:
    predictions: Mapping[str, str]
    greedy_predictions: Mapping[str, str]
    fold_audits: Mapping[int, Mapping[str, Any]]


def _project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return value


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


def _atomic_prediction_csv(
    path: Path, id_column: str, rows: Sequence[LabeledRow], predictions: Mapping[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[id_column, "prediction"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {id_column: row.identifier, "prediction": predictions[row.identifier]}
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "road_htr" / "challenge_ctc.py",
        PROJECT_ROOT / "src" / "road_htr" / "kraken_probability_export.py",
        PROJECT_ROOT / "src" / "road_htr" / "metrics.py",
        PROJECT_ROOT / "src" / "road_htr" / "schema.py",
        PROJECT_ROOT / "src" / "road_htr" / "submission.py",
        PROJECT_ROOT / "model29" / "decode_oof.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(stable_sha256(path)))
    return digest.hexdigest()


class ProbabilityStore:
    """Strict reader for one fold's immutable probability export."""

    def __init__(
        self,
        directory: str | Path,
        expected_fold: int,
        expected_ids: Sequence[str],
        expected_input_sha256: str,
    ):
        self.directory = Path(directory).resolve()
        self.manifest_path = self.directory / "manifest.json"
        self.index_path = self.directory / "index.json"
        self.manifest = _read_json(self.manifest_path)
        self.index = _read_json(self.index_path)
        if (
            self.manifest.get("schema_version") != SCHEMA_VERSION
            or self.manifest.get("complete") is not True
            or self.manifest.get("fold") != expected_fold
            or self.manifest.get("blank_class") != 0
            or self.manifest.get("matrix_layout") != "time,classes"
        ):
            raise ValueError(f"invalid or incomplete probability manifest: {self.manifest_path}")
        signature = self.manifest.get("signature")
        if not isinstance(signature, dict) or signature.get("input_sha256") != expected_input_sha256:
            raise ValueError(
                f"probability store fold {expected_fold} was exported from a different input CSV"
            )
        if self.manifest.get("index_sha256") != stable_sha256(self.index_path):
            raise ValueError(f"probability index checksum mismatch: {self.index_path}")
        labels = self.manifest.get("codec_labels")
        if (
            not isinstance(labels, list)
            or len(labels) < 2
            or labels[0] is not None
            or any(not isinstance(value, str) or not value for value in labels[1:])
        ):
            raise ValueError(f"invalid codec mapping in {self.manifest_path}")
        if signature.get("codec_labels") != labels:
            raise ValueError("probability manifest codec differs from its signed export signature")
        provenance = self.manifest.get("checkpoint_provenance")
        if provenance != signature.get("checkpoint_provenance") or not isinstance(
            provenance, dict
        ):
            raise ValueError("probability manifest has unsigned checkpoint provenance")
        if (
            provenance.get("kind") != "model22_converted_checkpoint"
            or provenance.get("exported_fold") != expected_fold
        ):
            raise ValueError("probability store is not verified OOF Model22 inference")
        raw_training_excluded_folds = provenance.get("training_excluded_folds")
        if not isinstance(raw_training_excluded_folds, list):
            raise ValueError("probability checkpoint provenance lacks excluded folds")
        self.training_excluded_folds = tuple(
            sorted(int(value) for value in raw_training_excluded_folds)
        )
        if expected_fold not in self.training_excluded_folds:
            raise ValueError(
                f"probability checkpoint was trained on exported fold {expected_fold}"
            )
        self.label_texts = tuple(labels[1:])
        entries = self.index.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"probability index has no entries: {self.index_path}")
        actual_ids = [entry.get("id") if isinstance(entry, dict) else None for entry in entries]
        if actual_ids != list(expected_ids):
            missing = sorted(set(expected_ids) - {value for value in actual_ids if isinstance(value, str)})
            extra = sorted({value for value in actual_ids if isinstance(value, str)} - set(expected_ids))
            raise SchemaError(
                f"probability IDs for fold {expected_fold} do not match truth order: "
                f"{len(missing)} missing, {len(extra)} extra"
            )
        if self.manifest.get("rows") != len(entries):
            raise ValueError("probability manifest row count differs from its index")
        self._entries: dict[str, dict[str, Any]] = {}
        for index, (identifier, raw_entry) in enumerate(
            zip(expected_ids, entries, strict=True)
        ):
            if not isinstance(raw_entry, dict):
                raise ValueError(f"invalid probability entry {index}")
            expected_file = f"probabilities/{index:06d}.npz"
            expected_receipt = self.directory / "receipts" / f"{index:06d}.json"
            if (
                raw_entry.get("index") != index
                or raw_entry.get("file") != expected_file
                or raw_entry.get("dtype") != "float32"
                or raw_entry.get("receipt_sha256") != stable_sha256(expected_receipt)
            ):
                raise ValueError(f"invalid probability index entry for {identifier!r}")
            receipt = _read_json(expected_receipt)
            if {key: value for key, value in raw_entry.items() if key != "receipt_sha256"} != receipt:
                raise ValueError(f"probability receipt mismatch for {identifier!r}")
            shape = raw_entry.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or not isinstance(shape[0], int)
                or shape[0] <= 0
                or shape[1] != len(labels)
            ):
                raise ValueError(f"invalid probability shape for {identifier!r}: {shape}")
            self._entries[identifier] = raw_entry
        self.fold = expected_fold
        self.identifiers = tuple(expected_ids)
        self.manifest_sha256 = stable_sha256(self.manifest_path)
        self.index_sha256 = stable_sha256(self.index_path)

    def load_log_probabilities(self, identifier: str):
        """Load and checksum one full matrix; memory is released by the caller."""

        import numpy as np

        try:
            entry = self._entries[identifier]
        except KeyError as error:
            raise SchemaError(f"ID {identifier!r} is absent from fold {self.fold} emissions") from error
        shard = self.directory / str(entry["file"])
        if shard.resolve().parent != (self.directory / "probabilities").resolve():
            raise ValueError(f"unsafe probability path for {identifier!r}")
        if stable_sha256(shard) != entry.get("sha256"):
            raise ValueError(f"probability checksum mismatch for {identifier!r}")
        try:
            with np.load(shard, allow_pickle=False) as archive:
                if archive.files != ["probabilities"]:
                    raise ValueError("NPZ must contain only 'probabilities'")
                probabilities = archive["probabilities"]
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot load probabilities for {identifier!r}: {error}") from error
        if list(probabilities.shape) != entry["shape"] or str(probabilities.dtype) != "float32":
            raise ValueError(f"probability shape/dtype mismatch for {identifier!r}")
        if not np.isfinite(probabilities).all() or probabilities.min() < 0 or probabilities.max() > 1:
            raise ValueError(f"invalid probability values for {identifier!r}")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=2e-4):
            raise ValueError(f"class probabilities do not sum to one for {identifier!r}")
        with np.errstate(divide="ignore"):
            return np.log(probabilities.astype(np.float64, copy=False))

    def provenance(self) -> dict[str, Any]:
        signature = self.manifest["signature"]
        return {
            "directory": str(self.directory),
            "fold": self.fold,
            "manifest_sha256": self.manifest_sha256,
            "index_sha256": self.index_sha256,
            "checkpoint_sha256": signature["checkpoint_sha256"],
            "config_sha256": signature["config_sha256"],
            "implementation_sha256": signature["implementation_sha256"],
            "rows": len(self.identifiers),
        }


def _normalized_text(value: str, normalization: str) -> str:
    return unicodedata.normalize(normalization, value.strip())


def _fit_decoder_models(
    candidate: DecoderCandidate,
    texts: Sequence[str],
) -> tuple[CharNGramLanguageModel | None, WordBoundaryPrior | None]:
    if candidate.mode == "greedy":
        return None, None
    lm = None
    if candidate.lm_alpha:
        lm = CharNGramLanguageModel(
            candidate.lm_order, candidate.lm_smoothing
        ).fit(texts)
    word_prior = None
    if candidate.word_boundary_weight:
        word_prior = WordBoundaryPrior(candidate.word_boundary_smoothing).fit(texts)
    return lm, word_prior


def _decode_rows_for_candidates(
    rows: Sequence[LabeledRow],
    store: Any,
    candidates: Sequence[DecoderCandidate],
    fit_texts: Sequence[str],
) -> dict[str, list[str]]:
    fitted = {
        candidate.name: _fit_decoder_models(candidate, fit_texts)
        for candidate in candidates
    }
    predictions = {candidate.name: [] for candidate in candidates}
    for row in rows:
        log_probabilities = store.load_log_probabilities(row.identifier)
        for candidate in candidates:
            lm, word_prior = fitted[candidate.name]
            prediction = decode_candidate(
                log_probabilities,
                store.label_texts,
                candidate,
                lm,
                word_prior,
            )
            predictions[candidate.name].append(
                clean_prediction_preserving_internal_spaces(prediction)
            )
    return predictions


def crossfit_decode(
    rows: Sequence[LabeledRow],
    stores: Mapping[int, Any],
    candidates: Sequence[DecoderCandidate],
    *,
    excluded_lm_ids: set[str] | None = None,
    normalization: str = "NFD",
    fixed_candidate: str | None = None,
    tuning_stores: Mapping[tuple[int, int], Any] | None = None,
) -> CrossfitResult:
    """Decode every outer fold without consulting that fold's labels.

    In tuned mode, candidate selection for outer fold ``f`` scores the other
    folds using separate inner stores whose checkpoints excluded both ``f``
    and tuning fold ``g``.  The corresponding LM has the same exclusions.
    Consequently changing any label in ``f`` cannot alter the base emissions,
    candidate, LM, boundary prior, or prediction applied to ``f``.
    """

    if normalization not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError("normalization must be NFC, NFD, NFKC, or NFKD")
    if not rows:
        raise ValueError("cross-fitted decoding needs labeled rows")
    identifiers = [row.identifier for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise SchemaError("cross-fitted rows contain duplicate IDs")
    folds = sorted({row.fold for row in rows})
    if set(stores) != set(folds):
        raise ValueError("one and only one probability store is required per fold")
    if not candidates or len(candidates) > MAX_GRID_CANDIDATES:
        raise ValueError(
            f"decoder grid must contain 1..{MAX_GRID_CANDIDATES} candidates"
        )
    names = [candidate.name for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("decoder candidate names must be unique")
    for candidate in candidates:
        candidate.validate()
    by_name = {candidate.name: candidate for candidate in candidates}
    if fixed_candidate is not None and fixed_candidate not in by_name:
        raise ValueError(f"unknown fixed decoder candidate {fixed_candidate!r}")
    if fixed_candidate is None:
        if len(folds) < 3:
            raise ValueError("nested other-fold tuning needs at least three folds")
        expected_tuning_keys = {
            (outer_fold, tuning_fold)
            for outer_fold in folds
            for tuning_fold in folds
            if tuning_fold != outer_fold
        }
        if tuning_stores is None or set(tuning_stores) != expected_tuning_keys:
            raise ValueError(
                "true nested tuning requires one pair-excluded probability "
                "store for every ordered (outer_fold, tuning_fold) pair"
            )
    excluded = excluded_lm_ids or set()

    rows_by_fold = {
        fold: [row for row in rows if row.fold == fold] for fold in folds
    }
    for fold, fold_rows in rows_by_fold.items():
        store_ids = tuple(getattr(stores[fold], "identifiers", (row.identifier for row in fold_rows)))
        if store_ids != tuple(row.identifier for row in fold_rows):
            raise SchemaError(f"probability store ID order mismatch for fold {fold}")
        training_exclusions = set(getattr(stores[fold], "training_excluded_folds", (fold,)))
        if fold not in training_exclusions:
            raise ValueError(f"outer probability store checkpoint trained on fold {fold}")

    if tuning_stores is not None:
        for (outer_fold, tuning_fold), store in tuning_stores.items():
            expected_ids = tuple(
                row.identifier for row in rows_by_fold[tuning_fold]
            )
            if tuple(getattr(store, "identifiers", ())) != expected_ids:
                raise SchemaError(
                    f"inner probability store ID mismatch for outer {outer_fold}, "
                    f"tuning {tuning_fold}"
                )
            training_exclusions = set(
                getattr(store, "training_excluded_folds", ())
            )
            if not {outer_fold, tuning_fold}.issubset(training_exclusions):
                raise ValueError(
                    f"inner checkpoint for ({outer_fold},{tuning_fold}) did not "
                    "exclude both folds from training"
                )

    outputs: dict[str, str] = {}
    greedy_outputs: dict[str, str] = {}
    audits: dict[int, Mapping[str, Any]] = {}
    internal_greedy = next(
        (candidate for candidate in candidates if candidate.mode == "greedy"),
        DecoderCandidate(name="__model29_greedy__", mode="greedy", lm_alpha=0.0),
    )

    for outer_fold in folds:
        if fixed_candidate is None:
            references: list[str] = []
            candidate_predictions = {candidate.name: [] for candidate in candidates}
            nested_fits: list[dict[str, Any]] = []
            for tuning_fold in folds:
                if tuning_fold == outer_fold:
                    continue
                fit_folds = [
                    fold for fold in folds if fold not in {outer_fold, tuning_fold}
                ]
                fit_texts = [
                    _normalized_text(row.target, normalization)
                    for row in rows
                    if row.fold in fit_folds and row.identifier not in excluded
                ]
                if not fit_texts:
                    raise ValueError(
                        f"no LM texts remain outside outer fold {outer_fold} "
                        f"and tuning fold {tuning_fold}"
                    )
                decoded = _decode_rows_for_candidates(
                    rows_by_fold[tuning_fold],
                    tuning_stores[(outer_fold, tuning_fold)],  # type: ignore[index]
                    candidates,
                    fit_texts,
                )
                references.extend(row.target for row in rows_by_fold[tuning_fold])
                for candidate in candidates:
                    candidate_predictions[candidate.name].extend(decoded[candidate.name])
                nested_fits.append(
                    {
                        "tuning_fold": tuning_fold,
                        "fit_folds": fit_folds,
                        "fit_rows": len(fit_texts),
                        "outer_fold_labels_excluded": True,
                        "tuning_fold_labels_excluded": True,
                    }
                )
            selection_scores = {
                candidate.name: competition_error(
                    references, candidate_predictions[candidate.name]
                )
                for candidate in candidates
            }
            selected_index = min(
                range(len(candidates)),
                key=lambda index: (
                    float(selection_scores[candidates[index].name]["mean_error"]),
                    index,
                ),
            )
            selected = candidates[selected_index]
            selection_audit: dict[str, Any] = {
                "mode": "nested_other_fold_tuning",
                "selection_folds": [fold for fold in folds if fold != outer_fold],
                "selection_rows": len(references),
                "nested_fits": nested_fits,
                "candidate_scores": selection_scores,
            }
        else:
            selected = by_name[fixed_candidate]
            selection_audit = {
                "mode": "fixed_candidate",
                "selection_folds": [],
                "selection_rows": 0,
                "candidate_scores": {},
            }

        final_fit_folds = [fold for fold in folds if fold != outer_fold]
        final_fit_texts = [
            _normalized_text(row.target, normalization)
            for row in rows
            if row.fold in final_fit_folds and row.identifier not in excluded
        ]
        target_candidates = [selected]
        if internal_greedy.name != selected.name:
            target_candidates.append(internal_greedy)
        decoded = _decode_rows_for_candidates(
            rows_by_fold[outer_fold],
            stores[outer_fold],
            target_candidates,
            final_fit_texts,
        )
        for row, prediction in zip(
            rows_by_fold[outer_fold], decoded[selected.name], strict=True
        ):
            outputs[row.identifier] = prediction
        greedy_name = internal_greedy.name
        greedy_values = (
            decoded[selected.name]
            if selected.name == greedy_name
            else decoded[greedy_name]
        )
        for row, prediction in zip(rows_by_fold[outer_fold], greedy_values, strict=True):
            greedy_outputs[row.identifier] = prediction
        audits[outer_fold] = {
            "outer_fold": outer_fold,
            "heldout_rows": len(rows_by_fold[outer_fold]),
            "heldout_labels_used_for_selection_or_fit": False,
            "selected_candidate": selected.to_dict(),
            "final_lm_fit_folds": final_fit_folds,
            "final_lm_fit_rows": len(final_fit_texts),
            "excluded_lm_ids": sorted(excluded),
            "selection": selection_audit,
        }

    if set(outputs) != set(identifiers) or set(greedy_outputs) != set(identifiers):
        raise RuntimeError("cross-fitted decoder did not emit every canonical ID")
    return CrossfitResult(outputs, greedy_outputs, audits)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read Model29 config {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("Model29 config must be a mapping")
    return config


def _load_labeled_rows(path: Path) -> tuple[Any, list[LabeledRow]]:
    table = read_table(path, require_target=True)
    if "fold" not in table.fieldnames:
        raise SchemaError("Model29 OOF truth must contain a fold column")
    assert table.target_column is not None
    try:
        rows = [
            LabeledRow(
                row[table.id_column].strip(),
                row[table.target_column],
                int(row["fold"]),
            )
            for row in table.rows
        ]
    except (TypeError, ValueError) as error:
        raise SchemaError("every Model29 fold value must be an integer") from error
    return table, rows


def _candidates(config: dict[str, Any]) -> list[DecoderCandidate]:
    raw_grid = config.get("decoder_grid")
    if not isinstance(raw_grid, list):
        raise ValueError("decoder_grid must be a list")
    candidates: list[DecoderCandidate] = []
    for value in raw_grid:
        if not isinstance(value, dict):
            raise ValueError("each decoder_grid item must be a mapping")
        candidates.append(DecoderCandidate.from_mapping(value))
    return candidates


def _score_by_fold(
    rows: Sequence[LabeledRow], predictions: Mapping[str, str]
) -> dict[str, Any]:
    folds = sorted({row.fold for row in rows})
    return {
        "global": competition_error(
            [row.target for row in rows],
            [predictions[row.identifier] for row in rows],
        ),
        "per_fold": {
            str(fold): competition_error(
                [row.target for row in rows if row.fold == fold],
                [predictions[row.identifier] for row in rows if row.fold == fold],
            )
            for fold in folds
        },
    }


def _prepare_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved == PROTECTED_MODEL22_OUTPUT or PROTECTED_MODEL22_OUTPUT in resolved.parents:
        raise ValueError("Model29 may read but never write active Model22 outputs")
    if resolved.exists() and not resolved.is_dir():
        raise FileExistsError(f"Model29 output path is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"refusing nonempty Model29 output directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)


def run_oof(
    config_path: str | Path,
    truth_path: str | Path,
    probabilities_root: str | Path,
    output_dir: str | Path,
    *,
    fixed_candidate: str | None = None,
    tune: bool = False,
    tuning_probabilities_root: str | Path | None = None,
) -> dict[str, Any]:
    config_file = _project_path(config_path)
    truth_file = _project_path(truth_path)
    stores_root = _project_path(probabilities_root)
    destination = _project_path(output_dir)
    for path in (config_file, truth_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not stores_root.is_dir():
        raise FileNotFoundError(stores_root)
    config = _load_config(config_file)
    table, rows = _load_labeled_rows(truth_file)
    truth_sha256 = stable_sha256(truth_file)
    contract = config.get("oof_contract", {})
    if not isinstance(contract, dict):
        raise ValueError("oof_contract must be a mapping")
    if truth_sha256 != contract.get("input_sha256"):
        raise SchemaError("truth SHA-256 does not match Model29's canonical OOF contract")
    fold_counts = {
        fold: sum(row.fold == fold for row in rows) for fold in sorted({row.fold for row in rows})
    }
    if len(rows) != contract.get("total_rows") or any(
        count != contract.get("fold_rows") for count in fold_counts.values()
    ):
        raise SchemaError(
            f"canonical OOF row contract mismatch: total={len(rows)}, folds={fold_counts}"
        )
    expected_folds = list(range(int(contract.get("fold_count", 5))))
    if sorted(fold_counts) != expected_folds:
        raise SchemaError(
            f"canonical fold values must be {expected_folds}, found {sorted(fold_counts)}"
        )

    candidates = _candidates(config)
    normalization = str(config.get("text_normalization", "NFD"))
    excluded_path = config.get("exclude_ids")
    exclusion_file = _project_path(excluded_path) if excluded_path else None
    exclusion_sha256 = stable_sha256(exclusion_file) if exclusion_file else None
    excluded = read_id_file(exclusion_file) if exclusion_file else set()
    unknown_exclusions = excluded - {row.identifier for row in rows}
    if unknown_exclusions:
        raise SchemaError(
            f"LM exclusion file contains IDs absent from truth: {sorted(unknown_exclusions)}"
        )
    stores: dict[int, ProbabilityStore] = {}
    for fold in expected_folds:
        fold_ids = [row.identifier for row in rows if row.fold == fold]
        stores[fold] = ProbabilityStore(
            stores_root / f"fold{fold}", fold, fold_ids, truth_sha256
        )
    tuning_stores: dict[tuple[int, int], ProbabilityStore] | None = None
    if tune:
        if fixed_candidate is not None:
            raise ValueError("--tune and --fixed-candidate are mutually exclusive")
        if tuning_probabilities_root is None:
            raise ValueError("--tune requires --tuning-probabilities-root")
        tuning_root = _project_path(tuning_probabilities_root)
        tuning_stores = {}
        for outer_fold in expected_folds:
            for tuning_fold in expected_folds:
                if tuning_fold == outer_fold:
                    continue
                tuning_ids = [
                    row.identifier for row in rows if row.fold == tuning_fold
                ]
                tuning_stores[(outer_fold, tuning_fold)] = ProbabilityStore(
                    tuning_root / f"outer{outer_fold}" / f"fold{tuning_fold}",
                    tuning_fold,
                    tuning_ids,
                    truth_sha256,
                )
    else:
        if tuning_probabilities_root is not None:
            raise ValueError("--tuning-probabilities-root is valid only with --tune")
        if fixed_candidate is None:
            default_candidate = config.get("default_fixed_candidate")
            if not isinstance(default_candidate, str) or not default_candidate:
                raise ValueError(
                    "config must declare default_fixed_candidate unless --tune is used"
                )
            fixed_candidate = default_candidate

    # Validate the destination before the expensive CPU decode, but create it
    # only after all predictions are ready so a failure leaves no partial run.
    resolved_destination = destination.resolve()
    if (
        resolved_destination == PROTECTED_MODEL22_OUTPUT
        or PROTECTED_MODEL22_OUTPUT in resolved_destination.parents
    ):
        raise ValueError("Model29 may read but never write active Model22 outputs")
    if resolved_destination.exists() and not resolved_destination.is_dir():
        raise FileExistsError(
            f"Model29 output path is not a directory: {resolved_destination}"
        )
    if resolved_destination.exists() and any(resolved_destination.iterdir()):
        raise FileExistsError(
            f"refusing nonempty Model29 output directory: {resolved_destination}"
        )
    config_sha256 = stable_sha256(config_file)
    implementation_sha256 = _implementation_sha256()
    result = crossfit_decode(
        rows,
        stores,
        candidates,
        excluded_lm_ids=excluded,
        normalization=normalization,
        fixed_candidate=fixed_candidate,
        tuning_stores=tuning_stores,
    )
    if stable_sha256(truth_file) != truth_sha256 or stable_sha256(config_file) != config_sha256:
        raise RuntimeError("truth or Model29 config changed during decoding")
    for store in stores.values():
        if (
            stable_sha256(store.manifest_path) != store.manifest_sha256
            or stable_sha256(store.index_path) != store.index_sha256
        ):
            raise RuntimeError(f"probability store changed during decoding: {store.directory}")
    if tuning_stores is not None:
        for store in tuning_stores.values():
            if (
                stable_sha256(store.manifest_path) != store.manifest_sha256
                or stable_sha256(store.index_path) != store.index_sha256
            ):
                raise RuntimeError(
                    f"inner probability store changed during decoding: {store.directory}"
                )
    if exclusion_file is not None and stable_sha256(exclusion_file) != exclusion_sha256:
        raise RuntimeError("LM exclusion file changed during decoding")
    if _implementation_sha256() != implementation_sha256:
        raise RuntimeError("Model29 implementation changed during decoding")

    _prepare_output_dir(destination)
    prediction_path = destination / "predictions.csv"
    greedy_path = destination / "greedy_predictions.csv"
    _atomic_prediction_csv(prediction_path, table.id_column, rows, result.predictions)
    _atomic_prediction_csv(greedy_path, table.id_column, rows, result.greedy_predictions)
    fold_artifacts: dict[str, str] = {}
    for fold, audit in result.fold_audits.items():
        path = destination / "folds" / f"fold{fold}.json"
        _atomic_json(path, audit)
        fold_artifacts[str(fold)] = str(path)

    decoded_scores = _score_by_fold(rows, result.predictions)
    greedy_scores = _score_by_fold(rows, result.greedy_predictions)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": (
            "nested pair-excluded other-fold decoder selection"
            if tune
            else "prespecified fixed decoder; outer-fold LM fit"
        ),
        "rows": len(rows),
        "fold_counts": fold_counts,
        "fixed_candidate": fixed_candidate,
        "tuned": tune,
        "exclude_ids": str(exclusion_file) if exclusion_file else None,
        "exclude_ids_sha256": exclusion_sha256,
        "candidate_grid": [candidate.to_dict() for candidate in candidates],
        "decoded": decoded_scores,
        "greedy": greedy_scores,
        "multi_score_delta": (
            float(decoded_scores["global"]["multi_score"])
            - float(greedy_scores["global"]["multi_score"])
        ),
        "predictions": str(prediction_path),
        "greedy_predictions": str(greedy_path),
        "fold_artifacts": fold_artifacts,
        "fold_audits": {str(key): value for key, value in result.fold_audits.items()},
    }
    report_path = destination / "report.json"
    _atomic_json(report_path, report)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "truth": str(truth_file),
        "truth_sha256": truth_sha256,
        "config": str(config_file),
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "probability_stores": {
            str(fold): store.provenance() for fold, store in stores.items()
        },
        "tuning_probability_stores": (
            {
                f"outer{outer_fold}/fold{tuning_fold}": store.provenance()
                for (outer_fold, tuning_fold), store in tuning_stores.items()
            }
            if tuning_stores is not None
            else {}
        ),
        "exclude_ids": str(exclusion_file) if exclusion_file else None,
        "exclude_ids_sha256": exclusion_sha256,
        "fixed_candidate": fixed_candidate,
        "tuned": tune,
        "predictions_sha256": stable_sha256(prediction_path),
        "greedy_predictions_sha256": stable_sha256(greedy_path),
        "report_sha256": stable_sha256(report_path),
        "fold_artifact_sha256": {
            fold: stable_sha256(Path(path)) for fold, path in fold_artifacts.items()
        },
    }
    _atomic_json(destination / "receipt.json", receipt)
    return report


def main(default_config: str | Path | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Exact-score leakage-safe Model29 OOF CTC beam decoding"
    )
    parser.add_argument(
        "--config", type=Path, default=default_config, required=default_config is None
    )
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--probabilities-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fixed-candidate",
        help="override config's prespecified fixed grid member",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="select on true pair-excluded inner-fold probability stores",
    )
    parser.add_argument(
        "--tuning-probabilities-root",
        type=Path,
        help="root containing outerF/foldG pair-excluded stores",
    )
    arguments = parser.parse_args()
    report = run_oof(
        arguments.config,
        arguments.truth,
        arguments.probabilities_root,
        arguments.output_dir,
        fixed_candidate=arguments.fixed_candidate,
        tune=arguments.tune,
        tuning_probabilities_root=arguments.tuning_probabilities_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
