from __future__ import annotations

import xarray as xr

from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
)

_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")


def require_xyz_core_labels(
    ds: xr.Dataset,
    *,
    core_dim: str,
    owner: str,
    what: str,
) -> None:
    labels = require_explicit_unique_dim_labels(ds, dim=core_dim, owner=owner, what=what)
    require_exact_labels(labels, expected=_XYZ_LABELS, owner=owner, what=f"{what} core")

__all__ = [
    "require_xyz_core_labels",
]
