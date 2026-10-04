from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandas as pd

# Decimal points and exponents intentionally mark float-domain tokens.
_DECIMAL_INTEGER_TOKEN = re.compile(r"[+-]?\d(?:_?\d)*\Z")
_EXPLICIT_NONFINITE_TOKEN = re.compile(r"[+-]?(?:inf(?:inity)?|nan)\Z", re.IGNORECASE)
_FLOAT64_EXACT_INTEGER_LIMIT = 2 ** 53
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)
_UINT64_MAX = int(np.iinfo(np.uint64).max)


@dataclass(frozen=True)
class CsvTimeValues:
    """Losslessly normalized CSV timestamps plus their retained-row mask."""

    valid_mask: np.ndarray
    values: np.ndarray
    invalid_count: int


@dataclass(frozen=True)
class _CsvTimeProfile:
    valid_mask: np.ndarray
    valid_count: int
    has_float: bool
    integer_min: int | None
    integer_max: int | None


class _TimeTokenKind(Enum):
    MISSING = auto()
    INTEGER = auto()
    FLOAT = auto()
    MALFORMED = auto()


def _parse_integer_token(text: str, *, owner: str, label: str) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{owner}: failed parsing an integer token in {label}.") from exc


def _has_nonzero_decimal_mantissa(text: str) -> bool:
    mantissa = text.partition("e")[0].partition("E")[0]
    return any(unicodedata.decimal(char, -1) > 0 for char in mantissa)


def _raise_unrepresentable_float(*, owner: str, label: str) -> None:
    raise ValueError(
        f"{owner}: {label} contains a finite numeric token outside the float64 range."
    )


def _parse_float_token(
    text: str,
    *,
    owner: str,
    label: str,
) -> tuple[_TimeTokenKind, float | None]:
    try:
        numeric = float(text)
    except (TypeError, ValueError, OverflowError):
        return _TimeTokenKind.MALFORMED, None
    if not np.isfinite(numeric):
        if _EXPLICIT_NONFINITE_TOKEN.fullmatch(text):
            return _TimeTokenKind.MISSING, None
        _raise_unrepresentable_float(owner=owner, label=label)
    if numeric == 0.0 and _has_nonzero_decimal_mantissa(text):
        _raise_unrepresentable_float(owner=owner, label=label)
    return _TimeTokenKind.FLOAT, numeric


def _parse_time_token(
    value: object,
    *,
    owner: str,
    label: str,
) -> tuple[_TimeTokenKind, int | float | None]:
    if pd.isna(value):
        return _TimeTokenKind.MISSING, None
    text = str(value).strip()
    if not text:
        return _TimeTokenKind.MISSING, None
    if _DECIMAL_INTEGER_TOKEN.fullmatch(text):
        return _TimeTokenKind.INTEGER, _parse_integer_token(
            text,
            owner=owner,
            label=label,
        )
    return _parse_float_token(text, owner=owner, label=label)


def _raise_malformed_time(*, owner: str, label: str) -> None:
    raise ValueError(f"{owner}: {label} must contain ordered real numeric values.")


def _profile_time_tokens(values: pd.Series, *, owner: str, label: str) -> _CsvTimeProfile:
    mask = np.zeros(len(values), dtype=bool)
    count = 0
    has_float = False
    minimum: int | None = None
    maximum: int | None = None
    for index, raw_value in enumerate(values.array):
        kind, numeric = _parse_time_token(raw_value, owner=owner, label=label)
        if kind is _TimeTokenKind.MALFORMED:
            _raise_malformed_time(owner=owner, label=label)
        if kind is _TimeTokenKind.MISSING:
            continue
        mask[index] = True
        count += 1
        if kind is _TimeTokenKind.FLOAT:
            has_float = True
            continue
        if numeric is None:
            raise RuntimeError(f"{owner}: CSV time profiling omitted an integer value.")
        integer = int(numeric)
        minimum = integer if minimum is None else min(minimum, integer)
        maximum = integer if maximum is None else max(maximum, integer)
    return _CsvTimeProfile(mask, count, has_float, minimum, maximum)


def _time_output_dtype(profile: _CsvTimeProfile, *, owner: str) -> np.dtype:
    if profile.valid_count == 0 or profile.has_float:
        extrema = (profile.integer_min, profile.integer_max)
        if any(value is not None and abs(value) > _FLOAT64_EXACT_INTEGER_LIMIT for value in extrema):
            raise ValueError(
                f"{owner}: integer CSV time values cannot be combined with floating-point "
                "time values without precision loss."
            )
        return np.dtype(np.float64)
    if profile.integer_min is None or profile.integer_max is None:
        raise RuntimeError(f"{owner}: CSV time profiling omitted integer bounds.")
    if profile.integer_min >= _INT64_MIN and profile.integer_max <= _INT64_MAX:
        return np.dtype(np.int64)
    if profile.integer_min >= 0 and profile.integer_max <= _UINT64_MAX:
        return np.dtype(np.uint64)
    raise ValueError(f"{owner}: integer CSV time values span no lossless NumPy integer dtype.")


def _fill_time_array(
    values: pd.Series,
    *,
    profile: _CsvTimeProfile,
    owner: str,
    label: str,
) -> np.ndarray:
    normalized = np.empty(profile.valid_count, dtype=_time_output_dtype(profile, owner=owner))
    output_index = 0
    for raw_value in values.array:
        kind, numeric = _parse_time_token(raw_value, owner=owner, label=label)
        if kind is _TimeTokenKind.MALFORMED:
            _raise_malformed_time(owner=owner, label=label)
        if kind is _TimeTokenKind.MISSING:
            continue
        if numeric is None:
            raise RuntimeError(f"{owner}: CSV time normalization omitted a numeric value.")
        normalized[output_index] = numeric
        output_index += 1
    if output_index != profile.valid_count:
        raise ValueError(
            f"{owner}: CSV time values changed while they were being normalized."
        )
    return normalized


def normalize_csv_time_tokens(
    values: pd.Series,
    *,
    owner: str,
    label: str,
) -> CsvTimeValues:
    """Classify raw timestamp tokens before any potentially lossy conversion."""
    profile = _profile_time_tokens(values, owner=owner, label=label)
    normalized = _fill_time_array(values, profile=profile, owner=owner, label=label)
    return CsvTimeValues(
        profile.valid_mask,
        normalized,
        len(values) - profile.valid_count,
    )


__all__ = ["CsvTimeValues", "normalize_csv_time_tokens"]
