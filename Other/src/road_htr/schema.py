"""CSV schema discovery and safe image lookup.

The authenticated CSVs are intentionally not committed. These helpers avoid
hard-coding an unverified schema while still failing loudly on ambiguity.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


ID_NAMES = ("id", "image_id", "image", "filename", "file_name")
TARGET_NAMES = ("target", "text", "transcription", "transcript", "label")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


class SchemaError(ValueError):
    """Raised when a competition CSV is ambiguous or invalid."""


@dataclass(frozen=True)
class CsvTable:
    path: Path
    fieldnames: tuple[str, ...]
    id_column: str
    target_column: str | None
    rows: tuple[dict[str, str], ...]


def _match_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {name.casefold().strip(): name for name in fieldnames}
    matches = [lookup[name] for name in candidates if name in lookup]
    if len(matches) > 1:
        raise SchemaError(f"ambiguous columns: {matches}")
    return matches[0] if matches else None


def read_table(path: str | Path, require_target: bool = False) -> CsvTable:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SchemaError(f"{path} has no header")
        fieldnames = tuple(reader.fieldnames)
        rows = tuple(dict(row) for row in reader)

    id_column = _match_column(fieldnames, ID_NAMES)
    if id_column is None:
        if len(fieldnames) == 1:
            id_column = fieldnames[0]
        else:
            raise SchemaError(
                f"cannot infer ID column in {path}; columns are {fieldnames}"
            )

    target_column = _match_column(fieldnames, TARGET_NAMES)
    if require_target and target_column is None:
        non_id = [name for name in fieldnames if name != id_column]
        if len(non_id) == 1:
            target_column = non_id[0]
        else:
            raise SchemaError(
                f"cannot infer target column in {path}; columns are {fieldnames}"
            )

    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        identifier = (row.get(id_column) or "").strip()
        if not identifier:
            raise SchemaError(f"blank ID at {path}:{line_number}")
        if identifier in seen:
            raise SchemaError(f"duplicate ID {identifier!r} at {path}:{line_number}")
        seen.add(identifier)
        if require_target and not (row.get(target_column or "") or "").strip():
            raise SchemaError(f"blank target for {identifier!r} at {path}:{line_number}")

    return CsvTable(path, fieldnames, id_column, target_column, rows)


def find_file(directory: str | Path, names: Iterable[str]) -> Path:
    directory = Path(directory)
    by_casefold = {
        item.name.casefold(): item for item in directory.iterdir() if item.is_file()
    }
    for name in names:
        match = by_casefold.get(name.casefold())
        if match is not None:
            return match
    expected = ", ".join(names)
    raise FileNotFoundError(f"none of [{expected}] found in {directory}")


class ImageIndex:
    """Resolve competition IDs to images without accepting arbitrary paths."""

    def __init__(self, image_dir: str | Path):
        self.image_dir = Path(image_dir).resolve()
        if not self.image_dir.is_dir():
            raise FileNotFoundError(self.image_dir)
        self._by_name: dict[str, Path] = {}
        self._by_stem: dict[str, Path] = {}
        for path in self.image_dir.iterdir():
            if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            key = path.name.casefold()
            stem = path.stem.casefold()
            if key in self._by_name or stem in self._by_stem:
                raise SchemaError(f"ambiguous image name/stem near {path}")
            self._by_name[key] = path
            self._by_stem[stem] = path

    def __len__(self) -> int:
        return len(self._by_name)

    def resolve(self, identifier: str) -> Path:
        safe_name = Path(identifier.strip()).name
        if safe_name != identifier.strip():
            raise SchemaError(f"ID must be a basename, got {identifier!r}")
        match = self._by_name.get(safe_name.casefold())
        if match is None:
            match = self._by_stem.get(Path(safe_name).stem.casefold())
        if match is None:
            raise FileNotFoundError(
                f"no image for ID {identifier!r} under {self.image_dir}"
            )
        return match

    def paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self._by_name.values()))


def predictions_by_id(
    table: CsvTable, prediction_column: str | None = None
) -> Mapping[str, str]:
    column = prediction_column or table.target_column
    if column is None or column not in table.fieldnames:
        raise SchemaError(
            f"prediction column is not known; available columns: {table.fieldnames}"
        )
    return {
        row[table.id_column].strip(): row.get(column, "") or "" for row in table.rows
    }

