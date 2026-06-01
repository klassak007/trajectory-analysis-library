from __future__ import annotations

import importlib


def _import_numba():
    return importlib.import_module("numba")


def require_numba(owner: str):
    try:
        return _import_numba()
    except ImportError as exc:
        raise ImportError(f"{owner}: numba is required for backend='numba'. Install with 'tal[numba]'.") from exc


def njit_kernel(numba, func):
    return numba.njit(cache=True, fastmath=False)(func)


__all__ = ["njit_kernel", "require_numba"]
