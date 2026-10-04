from __future__ import annotations

from numbers import Integral, Rational, Real
from typing import NoReturn

import numpy as np
import xarray as xr

from tal.core.validity_values import (
    SequenceSizeValueError,
    normalize_sequence_size_values,
    sequence_size_dtype,
)


def _size_error_detail(exc: SequenceSizeValueError, *, width: int) -> str:
    return {
        "unsupported_dtype": "must contain real numeric values",
        "chunked": "must be materialized before export",
        "non_finite": "must be finite",
        "non_integer": "must be integer-valued",
        "out_of_range": f"must be in [0, {width}]",
    }[exc.reason]


def _raise_csv_size_error(
    exc: SequenceSizeValueError,
    *,
    width: int,
    size_name: str,
    owner: str,
) -> NoReturn:
    detail = _size_error_detail(exc, width=width)
    raise ValueError(
        f"{owner}: sequence_size_coord {size_name!r} values {detail}."
    ) from exc


def _size_value_error(
    *,
    size_name: str,
    index: int,
    owner: str,
    detail: str,
) -> ValueError:
    return ValueError(
        f"{owner}: sequence_size_coord {size_name!r} value at flat index {index} {detail}."
    )


def _inspect_object_size(raw_value: object) -> tuple[int | None, str | None]:
    if isinstance(raw_value, Integral):
        return int(raw_value), None
    if isinstance(raw_value, Rational):
        if not bool(raw_value.denominator == 1):
            return None, "must be integer-valued"
        return int(raw_value), None
    if not isinstance(raw_value, Real):
        return None, "is not numeric"
    numeric = float(raw_value)
    if not np.isfinite(numeric):
        return None, "must be finite"
    value = int(raw_value)
    if not bool(raw_value == value):
        return None, "must be integer-valued"
    return value, None


def _coerce_object_size(
    raw_value: object,
    *,
    size_name: str,
    index: int,
    owner: str,
) -> int:
    if isinstance(raw_value, (bool, np.bool_)):
        raise _size_value_error(
            size_name=size_name,
            index=index,
            owner=owner,
            detail="is not an integer",
        )
    try:
        value, detail = _inspect_object_size(raw_value)
    except Exception as exc:
        raise ValueError(
            f"{owner}: failed inspecting sequence_size_coord {size_name!r} value "
            f"at flat index {index}."
        ) from exc
    if detail is not None:
        raise _size_value_error(
            size_name=size_name,
            index=index,
            owner=owner,
            detail=detail,
        )
    if value is None:  # pragma: no cover - helper invariant.
        raise RuntimeError("CSV object sequence-size inspection omitted a normalized value")
    return value


def _coerce_object_sizes(
    size: xr.DataArray,
    *,
    width: int,
    size_name: str,
    owner: str,
) -> xr.DataArray:
    values = np.asarray(size.data)
    normalized = np.empty(values.shape, dtype=np.int64)
    for index, raw_value in enumerate(values.flat):
        coerced = _coerce_object_size(
            raw_value,
            size_name=size_name,
            index=index,
            owner=owner,
        )
        try:
            normalized.flat[index] = coerced
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"{owner}: sequence_size_coord {size_name!r} values must be in [0, {width}]."
            ) from exc
    return size.copy(data=normalized)


def require_csv_export_size_dtype(
    size: xr.DataArray,
    *,
    size_name: str,
    owner: str,
) -> None:
    """Validate an effective CSV size dtype without realizing its values."""
    try:
        dtype = np.dtype(size.dtype)
    except (TypeError, ValueError):
        pass
    else:
        if dtype.kind == "O":
            return
    try:
        sequence_size_dtype(size)
    except SequenceSizeValueError as exc:
        _raise_csv_size_error(
            exc,
            width=0,
            size_name=size_name,
            owner=owner,
        )


def normalize_csv_export_sizes(
    size: xr.DataArray,
    *,
    width: int,
    size_name: str,
    owner: str,
) -> xr.DataArray:
    """Normalize materialized CSV size overrides under writer ownership."""
    candidate = size
    if np.dtype(size.dtype).kind == "O":
        candidate = _coerce_object_sizes(
            size,
            width=width,
            size_name=size_name,
            owner=owner,
        )
    try:
        return normalize_sequence_size_values(candidate, sequence_len=width)
    except SequenceSizeValueError as exc:
        _raise_csv_size_error(
            exc,
            width=width,
            size_name=size_name,
            owner=owner,
        )


__all__ = ["normalize_csv_export_sizes", "require_csv_export_size_dtype"]
