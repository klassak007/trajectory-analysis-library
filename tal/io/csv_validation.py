from __future__ import annotations

import csv
from itertools import chain
from pathlib import Path
from typing import TextIO


def _csv_name_failure(name: str, *, seen: set[str]) -> str | None:
    if not name:
        return "empty"
    if name.isspace():
        return "blank"
    if "\ufeff" in name:
        return "bom"
    if "\x00" in name:
        return "nul"
    if name in seen:
        return "duplicate"
    seen.add(name)
    return None


def _is_pandas_blank_line(line: str) -> bool:
    """Match pandas' leading space/tab-only blank-line treatment."""
    return not line.rstrip("\r\n").strip(" \t")


def _read_effective_header(stream: TextIO) -> tuple[str, ...]:
    """Read only the first logical nonblank record with Python's CSV parser."""
    for line in stream:
        if _is_pandas_blank_line(line):
            continue
        return tuple(next(csv.reader(chain((line,), stream))))
    return ()


def _read_csv_header(path: str, *, owner: str) -> tuple[str, ...]:
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            return _read_effective_header(stream)
    except (csv.Error, OSError, UnicodeError) as exc:
        raise ValueError(f"{owner}: failed reading CSV header from {path!r}.") from exc


def require_valid_csv_header(path: str, *, owner: str) -> tuple[str, ...]:
    """Validate TAL-owned header identities before pandas normalizes them."""
    fields = _read_csv_header(path, owner=owner)
    if not fields:
        raise ValueError(f"{owner}: empty CSV header at column 0 in {path!r}.")
    seen: set[str] = set()
    for index, name in enumerate(fields):
        failure = _csv_name_failure(name, seen=seen)
        if failure == "empty":
            raise ValueError(f"{owner}: empty CSV header at column {index} in {path!r}.")
        if failure == "blank":
            raise ValueError(
                f"{owner}: CSV header at column {index} in {path!r} must contain "
                "a non-whitespace character."
            )
        if failure == "bom":
            raise ValueError(
                f"{owner}: CSV header at column {index} in {path!r} contains a BOM character."
            )
        if failure == "nul":
            raise ValueError(
                f"{owner}: CSV header at column {index} in {path!r} contains a NUL byte."
            )
        if failure == "duplicate":
            raise ValueError(f"{owner}: duplicate CSV header {name!r} in {path!r}.")
    return fields


def register_csv_export_name(name: str, *, seen: set[str], owner: str) -> None:
    """Register one normalized export name under the shared CSV policy."""
    failure = _csv_name_failure(name, seen=seen)
    if failure == "empty":
        raise ValueError(
            f"{owner}: CSV column names must be non-empty after string normalization."
        )
    if failure == "blank":
        raise ValueError(
            f"{owner}: CSV column names must contain a non-whitespace character."
        )
    if failure == "bom":
        raise ValueError(f"{owner}: CSV column names must not contain BOM characters.")
    if failure == "nul":
        raise ValueError(f"{owner}: CSV column name {name!r} must not contain NUL bytes.")
    if failure == "duplicate":
        raise ValueError(
            f"{owner}: duplicate CSV column name {name!r} after string normalization."
        )


__all__ = ["register_csv_export_name", "require_valid_csv_header"]
