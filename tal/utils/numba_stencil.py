from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

_INT64_MAX = np.iinfo(np.int64).max
_INT64_MIN = np.iinfo(np.int64).min
_INT64_FLOAT_MAX_EXCLUSIVE = float(_INT64_MAX)
_INT64_FLOAT_MIN_INCLUSIVE = float(_INT64_MIN)


@dataclass(frozen=True)
class WindowBounds:
    """Exclusive integer window bounds for row-local loops.

    Parameters
    ----------
    start
        Inclusive start index for each position.
    stop
        Exclusive stop index for each position.
    """

    start: np.ndarray
    stop: np.ndarray


@dataclass(frozen=True)
class WindowRows:
    """Prepared window metadata for row-local loops.

    Parameters
    ----------
    bounds
        Normalized exclusive start and stop bounds with ``int64`` dtype.
    length
        Number of positions, equal to ``len(bounds.start)``.
    widths
        Per-position window widths, computed as ``stop - start``.
    max_width
        Maximum window width, or zero when ``length`` is zero.
    """

    bounds: WindowBounds
    length: int
    widths: np.ndarray
    max_width: int


def _coerce_int(value: object, *, name: str, owner: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{owner}: {name} must be an integer.")
    return int(value)


def _validate_length(length: int, *, owner: str) -> None:
    if length < 0:
        raise ValueError(f"{owner}: length must be >= 0.")
    if length > _INT64_MAX:
        raise ValueError(f"{owner}: length must fit int64.")


def _raise_bound_value_error(*, name: str, owner: str) -> None:
    raise ValueError(f"{owner}: window {name} bounds must be exact integer-like values.")


def _check_int64_range(value: int, *, name: str, owner: str) -> None:
    if value < _INT64_MIN or value > _INT64_MAX:
        raise ValueError(f"{owner}: window {name} bounds must fit int64.")


def _coerce_object_bound_value(value: object, *, name: str, owner: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        _raise_bound_value_error(name=name, owner=owner)
    if isinstance(value, Integral):
        out = int(value)
        _check_int64_range(out, name=name, owner=owner)
        return out
    if isinstance(value, Real):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            _raise_bound_value_error(name=name, owner=owner)
        if numeric < _INT64_FLOAT_MIN_INCLUSIVE or numeric >= _INT64_FLOAT_MAX_EXCLUSIVE:
            raise ValueError(f"{owner}: window {name} bounds must fit int64.")
        return int(numeric)
    _raise_bound_value_error(name=name, owner=owner)


def _coerce_integer_bounds(array: np.ndarray, *, name: str, owner: str) -> np.ndarray:
    if np.issubdtype(array.dtype, np.unsignedinteger):
        if np.any(array > _INT64_MAX):
            raise ValueError(f"{owner}: window {name} bounds must fit int64.")
    elif np.any((array < _INT64_MIN) | (array > _INT64_MAX)):
        raise ValueError(f"{owner}: window {name} bounds must fit int64.")
    return array.astype(np.int64, copy=False)


def _coerce_float_bounds(array: np.ndarray, *, name: str, owner: str) -> np.ndarray:
    if not np.all(np.isfinite(array)) or not np.all(np.floor(array) == array):
        _raise_bound_value_error(name=name, owner=owner)
    if np.any((array < _INT64_FLOAT_MIN_INCLUSIVE) | (array >= _INT64_FLOAT_MAX_EXCLUSIVE)):
        raise ValueError(f"{owner}: window {name} bounds must fit int64.")
    return array.astype(np.int64)


def _coerce_object_bounds(array: np.ndarray, *, name: str, owner: str) -> np.ndarray:
    out = np.empty(array.shape, dtype=np.int64)
    for idx, value in enumerate(array):
        out[idx] = _coerce_object_bound_value(value, name=name, owner=owner)
    return out


def _coerce_window_bounds(values: object, *, name: str, owner: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{owner}: window {name} bounds must be one-dimensional.")
    if np.issubdtype(array.dtype, np.bool_):
        _raise_bound_value_error(name=name, owner=owner)
    if np.issubdtype(array.dtype, np.integer):
        return _coerce_integer_bounds(array, name=name, owner=owner)
    if np.issubdtype(array.dtype, np.floating):
        return _coerce_float_bounds(array, name=name, owner=owner)
    if array.dtype == np.dtype(object):
        return _coerce_object_bounds(array, name=name, owner=owner)
    _raise_bound_value_error(name=name, owner=owner)


def _stop_bounds(idx: np.ndarray, *, length: int, after: int) -> np.ndarray:
    stop = np.empty(idx.shape, dtype=np.int64)
    cutoff = length - after - 1
    unclipped = idx <= cutoff
    stop[unclipped] = idx[unclipped] + after + 1
    stop[~unclipped] = length
    return stop


def clipped_window_bounds(length: object, *, before: object, after: object, owner: str) -> WindowBounds:
    """Return clipped exclusive bounds around every position.

    Parameters
    ----------
    length
        Number of positions to cover.
    before
        Number of positions to include before each center.
    after
        Number of positions to include after each center.
    owner
        Error-message prefix for the caller-owned boundary.

    Returns
    -------
    WindowBounds
        Exclusive start and stop arrays with ``int64`` dtype.

    Raises
    ------
    ValueError
        If ``length``, ``before``, or ``after`` is non-integer, boolean, or
        negative, or if ``length`` cannot fit an ``int64`` output shape.

    Examples
    --------
    >>> from tal.utils import numba as tal_numba
    >>> bounds = tal_numba.clipped_window_bounds(4, before=2, after=0, owner="docs")
    >>> bounds.start.tolist(), bounds.stop.tolist()
    ([0, 0, 0, 1], [1, 2, 3, 4])
    """

    n = _coerce_int(length, name="length", owner=owner)
    left = _coerce_int(before, name="before", owner=owner)
    right = _coerce_int(after, name="after", owner=owner)
    _validate_length(n, owner=owner)
    if left < 0 or right < 0:
        raise ValueError(f"{owner}: before and after must be >= 0.")
    left = min(left, n)
    right = min(right, n)
    idx = np.arange(n, dtype=np.int64)
    start = np.maximum(idx - left, 0).astype(np.int64, copy=False)
    stop = _stop_bounds(idx, length=n, after=right)
    return WindowBounds(start=start, stop=stop)


def centered_window_bounds(length: object, *, radius: object, owner: str) -> WindowBounds:
    """Return clipped centered bounds around every position.

    Parameters
    ----------
    length
        Number of positions to cover.
    radius
        Number of neighbors to include on each side.
    owner
        Error-message prefix for the caller-owned boundary.

    Returns
    -------
    WindowBounds
        Exclusive start and stop arrays with ``int64`` dtype.

    Raises
    ------
    ValueError
        If ``length`` or ``radius`` is non-integer, boolean, or negative, or
        if ``length`` cannot fit an ``int64`` output shape.

    Examples
    --------
    >>> from tal.utils import numba as tal_numba
    >>> bounds = tal_numba.centered_window_bounds(3, radius=1, owner="docs")
    >>> bounds.start.tolist(), bounds.stop.tolist()
    ([0, 0, 1], [2, 3, 3])
    """

    r = _coerce_int(radius, name="radius", owner=owner)
    if r < 0:
        raise ValueError(f"{owner}: radius must be >= 0.")
    return clipped_window_bounds(length, before=r, after=r, owner=owner)


def forward_window_bounds(length: object, *, width: object, owner: str) -> WindowBounds:
    """Return clipped forward-looking bounds around every position.

    Parameters
    ----------
    length
        Number of positions to cover.
    width
        Maximum number of positions in each forward-looking window.
    owner
        Error-message prefix for the caller-owned boundary.

    Returns
    -------
    WindowBounds
        Exclusive start and stop arrays with ``int64`` dtype.

    Raises
    ------
    ValueError
        If ``length`` or ``width`` is non-integer or boolean, if ``length`` is
        negative or cannot fit an ``int64`` output shape, or if ``width`` is
        less than one.

    Examples
    --------
    >>> from tal.utils import numba as tal_numba
    >>> bounds = tal_numba.forward_window_bounds(4, width=2, owner="docs")
    >>> bounds.start.tolist(), bounds.stop.tolist()
    ([0, 1, 2, 3], [2, 3, 4, 4])
    """

    w = _coerce_int(width, name="width", owner=owner)
    if w < 1:
        raise ValueError(f"{owner}: width must be >= 1.")
    return clipped_window_bounds(length, before=0, after=w - 1, owner=owner)


def backward_window_bounds(length: object, *, width: object, owner: str) -> WindowBounds:
    """Return clipped backward-looking bounds around every position.

    Parameters
    ----------
    length
        Number of positions to cover.
    width
        Maximum number of positions in each backward-looking window.
    owner
        Error-message prefix for the caller-owned boundary.

    Returns
    -------
    WindowBounds
        Exclusive start and stop arrays with ``int64`` dtype.

    Raises
    ------
    ValueError
        If ``length`` or ``width`` is non-integer or boolean, if ``length`` is
        negative or cannot fit an ``int64`` output shape, or if ``width`` is
        less than one.

    Examples
    --------
    >>> from tal.utils import numba as tal_numba
    >>> bounds = tal_numba.backward_window_bounds(4, width=2, owner="docs")
    >>> bounds.start.tolist(), bounds.stop.tolist()
    ([0, 0, 1, 2], [1, 2, 3, 4])
    """

    w = _coerce_int(width, name="width", owner=owner)
    if w < 1:
        raise ValueError(f"{owner}: width must be >= 1.")
    return clipped_window_bounds(length, before=w - 1, after=0, owner=owner)


def prepare_window_rows(bounds: WindowBounds, *, owner: str) -> WindowRows:
    """Normalize window bounds and derive loop metadata.

    Parameters
    ----------
    bounds
        Exclusive start and stop arrays. Exact integer-like values are accepted
        and normalized to ``int64``.
    owner
        Error-message prefix for the caller-owned boundary.

    Returns
    -------
    WindowRows
        Normalized bounds, row count, per-position widths, and maximum width.

    Raises
    ------
    ValueError
        If ``bounds`` is not a ``WindowBounds`` object, if start and stop
        arrays are not one-dimensional matching exact integer-like values, or
        if any bound is outside the normalized mechanical range.

    Examples
    --------
    >>> import numpy as np
    >>> from tal.utils import numba as tal_numba
    >>> bounds = tal_numba.WindowBounds(
    ...     start=np.asarray([0.0, 1.0], dtype=object),
    ...     stop=np.asarray([1, 2], dtype=object),
    ... )
    >>> rows = tal_numba.prepare_window_rows(bounds, owner="docs")
    >>> rows.length, rows.widths.tolist(), rows.max_width
    (2, [1, 1], 1)
    """

    if not isinstance(bounds, WindowBounds):
        raise ValueError(f"{owner}: bounds must be a WindowBounds object.")
    start = _coerce_window_bounds(bounds.start, name="start", owner=owner)
    stop = _coerce_window_bounds(bounds.stop, name="stop", owner=owner)
    if start.shape != stop.shape:
        raise ValueError(f"{owner}: window start and stop bounds must share shape.")
    if np.any(start < 0):
        raise ValueError(f"{owner}: window start bounds must be >= 0.")
    if np.any(stop < start):
        raise ValueError(f"{owner}: window stop bounds must be >= start.")
    length = int(start.shape[0])
    if np.any(stop > length):
        raise ValueError(f"{owner}: window stop bounds must be <= length.")
    widths = stop - start
    max_width = int(np.max(widths)) if widths.size else 0
    return WindowRows(WindowBounds(start=start, stop=stop), length, widths.astype(np.int64, copy=False), max_width)


__all__ = [
    "WindowBounds",
    "WindowRows",
    "backward_window_bounds",
    "centered_window_bounds",
    "clipped_window_bounds",
    "forward_window_bounds",
    "prepare_window_rows",
]
