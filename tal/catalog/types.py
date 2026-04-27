from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import xarray as xr

CatalogBackend = Literal["dataset", "datatree"]
CatalogData = xr.Dataset | xr.DataTree


@dataclass(frozen=True)
class CatalogState:
    backend: CatalogBackend
    batch_dim: str
    data: CatalogData
    template: xr.Dataset | None = None


def make_catalog_state(
    *,
    backend: CatalogBackend,
    batch_dim: str,
    data: CatalogData,
    template: xr.Dataset | None = None,
) -> CatalogState:
    if backend == "dataset" and not isinstance(data, xr.Dataset):
        raise TypeError("CatalogState: dataset backend requires xr.Dataset payload.")
    if backend == "datatree" and not isinstance(data, xr.DataTree):
        raise TypeError("CatalogState: datatree backend requires xr.DataTree payload.")
    if template is not None and not isinstance(template, xr.Dataset):
        raise TypeError("CatalogState: template must be xr.Dataset when provided.")
    return CatalogState(backend=backend, batch_dim=batch_dim, data=data, template=template)


__all__ = ["CatalogBackend", "CatalogData", "CatalogState", "make_catalog_state"]
