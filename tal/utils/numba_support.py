from __future__ import annotations

import importlib
from functools import wraps
from threading import RLock

_COMPILATION_LOCK = RLock()


def _cached_compilation(factory):
    """Publish one successful initialization, including nested dependencies.

    A normal lru_cache may run a cold factory concurrently. These factories
    install shared numerical helpers, so publication must be serialized across
    owners. Warm access and execution of the returned dispatchers stay unlocked.
    This helper imports no optional dependency and does not retry failures.
    """
    ready = False
    result = None

    @wraps(factory)
    def compiled():
        nonlocal ready, result
        if ready:
            return result
        with _COMPILATION_LOCK:
            if not ready:
                result = factory()
                ready = True
        return result

    return compiled


def _import_numba():
    return importlib.import_module("numba")


def _numba_available() -> bool:
    if importlib.util.find_spec("numba") is None:
        return False
    _import_numba()
    return True


def require_numba(owner: str):
    """Import Numba lazily for an owner-prefixed optional boundary.

    Parameters
    ----------
    owner
        Error-message prefix for the caller-owned boundary.

    Returns
    -------
    types.ModuleType
        Imported ``numba`` module.

    Raises
    ------
    ImportError
        If Numba is not installed.

    Examples
    --------
    >>> from tal.utils import numba as tal_numba
    >>> try:
    ...     module = tal_numba.require_numba("docs")
    ... except ImportError:
    ...     module = None
    >>> module is None or hasattr(module, "njit")
    True
    """

    try:
        return _import_numba()
    except ImportError as exc:
        raise ImportError(f"{owner}: numba is required for backend='numba'. Install with 'tal[numba]'.") from exc


def njit_kernel(numba, func):
    """Apply TAL's standard Numba compile policy to a function.

    Parameters
    ----------
    numba
        Imported Numba module, usually from ``require_numba(owner)``.
    func
        Python function to compile.

    Returns
    -------
    object
        Numba dispatcher returned by ``numba.njit(cache=True, fastmath=False)``.

    Examples
    --------
    >>> from tal.utils import numba as tal_numba
    >>> def add_one(value):
    ...     return value + 1
    >>> try:
    ...     module = tal_numba.require_numba("docs")
    ... except ImportError:
    ...     compiled = None
    ... else:
    ...     compiled = tal_numba.njit_kernel(module, add_one)
    >>> compiled is None or callable(compiled)
    True
    """

    return numba.njit(cache=True, fastmath=False)(func)


__all__ = ["njit_kernel", "require_numba"]
