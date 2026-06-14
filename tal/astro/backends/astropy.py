from __future__ import annotations

from importlib import import_module
from types import ModuleType


def require_astropy(owner: str) -> ModuleType:
    """Import Astropy at an explicit backend boundary."""
    try:
        return import_module("astropy")
    except ImportError as exc:
        raise ImportError(f"{owner}: Astropy is required for this astro backend. Install with 'tal[astro]'.") from exc


__all__ = ["require_astropy"]
