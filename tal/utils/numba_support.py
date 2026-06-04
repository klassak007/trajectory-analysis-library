from __future__ import annotations

import importlib


def _import_numba():
    return importlib.import_module("numba")


def require_numba(owner: str):
    """Import Numba lazily for an owner-prefixed optional boundary.

    Parameters
    ----------
    owner
        Error-message prefix for the caller-owned boundary.

    Returns
    -------
    module
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
