from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

_INT64_MAX = np.iinfo(np.int64).max


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


def _coerce_int(value: object, *, name: str, owner: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{owner}: {name} must be an integer.")
    return int(value)


def _validate_length(length: int, *, owner: str) -> None:
    if length < 0:
        raise ValueError(f"{owner}: length must be >= 0.")
    if length > _INT64_MAX:
        raise ValueError(f"{owner}: length must fit int64.")


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


__all__ = [
    "WindowBounds",
    "backward_window_bounds",
    "centered_window_bounds",
    "clipped_window_bounds",
    "forward_window_bounds",
]
