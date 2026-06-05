from __future__ import annotations

from math import inf
from numbers import Integral
import os
import subprocess
import sys
import tempfile
from statistics import median
from time import perf_counter
from typing import Sequence


def time_once(func, *args) -> float:
    """Measure one call to a callable.

    Parameters
    ----------
    func
        Callable to execute.
    *args
        Positional arguments passed to ``func``.

    Returns
    -------
    float
        Elapsed wall-clock seconds for one call.

    Examples
    --------
    >>> from tal.utils import numba as tal_numba
    >>> elapsed = tal_numba.time_once(lambda value: value + 1, 1)
    >>> elapsed >= 0.0
    True
    """

    start = perf_counter()
    func(*args)
    return perf_counter() - start


def _validate_repeats(repeats: object) -> int:
    if isinstance(repeats, bool) or not isinstance(repeats, Integral):
        raise ValueError("repeats must be a positive integer.")
    count = int(repeats)
    if count < 1:
        raise ValueError("repeats must be a positive integer.")
    return count


def warm_median(func, *args, repeats: int = 7) -> float:
    """Measure the median elapsed time over repeated warm calls.

    Parameters
    ----------
    func
        Callable to execute.
    *args
        Positional arguments passed to ``func``.
    repeats
        Number of repeated measurements. Must be a positive non-boolean
        integer.

    Returns
    -------
    float
        Median elapsed wall-clock seconds.

    Raises
    ------
    ValueError
        If ``repeats`` is not a positive non-boolean integer.

    Examples
    --------
    >>> from tal.utils import numba as tal_numba
    >>> tal_numba.warm_median(lambda value: value + 1, 1, repeats=1) >= 0.0
    True
    """

    count = _validate_repeats(repeats)
    return median(time_once(func, *args) for _ in range(count))


def cold_subprocess(script_path: str, args: Sequence[str], *, cache_prefix: str) -> float:
    """Measure a fresh-process benchmark script result.

    Parameters
    ----------
    script_path
        Python script executed with the current Python executable.
    args
        Command-line arguments appended after ``script_path``.
    cache_prefix
        Prefix for a fresh temporary ``NUMBA_CACHE_DIR``.

    Returns
    -------
    float
        Float parsed from the subprocess standard output.

    Raises
    ------
    subprocess.CalledProcessError
        If the subprocess exits with a non-zero status. Captured stdout and
        stderr remain attached to the exception.
    ValueError
        If subprocess stdout does not contain a single parseable float.

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile
    >>> from tal.utils import numba as tal_numba
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     script = Path(tmp) / "bench.py"
    ...     _ = script.write_text("print('0.0')\\n", encoding="utf-8")
    ...     tal_numba.cold_subprocess(str(script), (), cache_prefix="docs-numba-")
    0.0
    """

    with tempfile.TemporaryDirectory(prefix=cache_prefix) as cache_dir:
        env = dict(os.environ)
        env["NUMBA_CACHE_DIR"] = cache_dir
        proc = subprocess.run(
            [sys.executable, str(script_path), *(str(arg) for arg in args)],
            check=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    text = proc.stdout.strip()
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"cold subprocess stdout must contain a float, got {text!r}.") from exc


def break_even_calls(baseline: float, cold: float, warm: float) -> float:
    """Estimate warm calls needed to amortize first-call cost.

    Parameters
    ----------
    baseline
        Baseline warm-call time in seconds.
    cold
        First-call fresh-process time in seconds.
    warm
        Accelerated warm-call time in seconds.

    Returns
    -------
    float
        Number of warm calls needed to amortize first-call cost. Returns
        ``math.inf`` when warm calls are not faster than baseline.

    Examples
    --------
    >>> from tal.utils import numba as tal_numba
    >>> tal_numba.break_even_calls(10.0, 25.0, 5.0)
    4.0
    """

    savings = baseline - warm
    if savings <= 0.0:
        return inf
    return max(0.0, cold - warm) / savings


__all__ = ["break_even_calls", "cold_subprocess", "time_once", "warm_median"]
