from __future__ import annotations

import pandas as pd

from .adapter_metadata import normalize_metadata_scalar


def _is_missing_metadata(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _metadata_scalars_equal(left: object, right: object) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _collect_scalar_column(
    column: pd.Series,
    *,
    name: str,
    owner: str,
    source_path: str,
) -> object:
    missing = object()
    first: object = missing
    for value in column:
        if _is_missing_metadata(value):
            continue
        value = normalize_metadata_scalar(value)
        if first is missing:
            first = value
            continue
        if not _metadata_scalars_equal(first, value):
            raise ValueError(
                f"{owner}: metadata column {name!r} must be scalar per input file; "
                f"found multiple values in {source_path!r}."
            )
    return None if first is missing else first


def collect_csv_scalar_metadata(
    frame: pd.DataFrame,
    *,
    metadata_columns: tuple[str, ...],
    owner: str,
    source_path: str,
) -> dict[str, object]:
    """Collect per-file scalar CSV metadata under the CSV-domain owner."""
    out: dict[str, object] = {}
    for name in metadata_columns:
        if name not in frame.columns:
            raise ValueError(f"{owner}: metadata column {name!r} was not found in {source_path!r}.")
        out[name] = _collect_scalar_column(
            frame[name],
            name=name,
            owner=owner,
            source_path=source_path,
        )
    return out


__all__ = ["collect_csv_scalar_metadata"]
