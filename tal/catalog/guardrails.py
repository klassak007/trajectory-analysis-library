from __future__ import annotations

from typing import NoReturn


def raise_compute_blocked(*, owner: str, operation: str) -> NoReturn:
    raise TypeError(
        f"{owner}: Catalog is browse-only; compute operation {operation!r} is not supported."
    )


__all__ = ["raise_compute_blocked"]
