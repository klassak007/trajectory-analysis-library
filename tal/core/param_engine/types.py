from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import xarray as xr

ParamMethod = Literal["nearest", "linear"]
ParamDuplicatePolicy = Literal["invalid", "left", "right", "raise"]


@dataclass(frozen=True)
class ParamCoordSpec:
    """Resolved parameter coordinate metadata.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    name: str
    coord: xr.DataArray
    sequence_dim: str
    batch_dims: tuple[str, ...]


@dataclass(frozen=True)
class QueryGrid:
    """Normalized query coordinate.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    values: xr.DataArray
    query_dim: str
    stacked_dims: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ParamMapOptions:
    """Interpolation options for parameter mapping.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    method: ParamMethod = "linear"
    duplicate_policy: ParamDuplicatePolicy = "invalid"


@dataclass(frozen=True)
class ParamMap:
    """Parameter-index mapping for interpolation/resampling.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    i0: xr.DataArray
    i1: xr.DataArray
    alpha: xr.DataArray
    valid: xr.DataArray
    query_dim: str


@dataclass(frozen=True)
class ParamBoundsMap:
    """Parameter-index bounds mapping for slice-style selection.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    i0: xr.DataArray
    i1: xr.DataArray


@dataclass(frozen=True)
class ParamSchemaContext:
    """Schema-derived context for param engine operations.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    ds: xr.Dataset
    sequence_dim: str
    batch_dims: tuple[str, ...]
    core_dims: tuple[str, ...]
    param_name: str | None
    param_declared: bool
    sequence_size_coord: str | None
    validity_declared: bool
