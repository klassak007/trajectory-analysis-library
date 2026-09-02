from __future__ import annotations

import datetime as _datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..param_engine.types import ParamCoordSpec, QueryGrid

ParamKind = Literal["numeric", "datetime64"]
ParamSyncTolerance = float | int | np.timedelta64 | _datetime.timedelta | pd.Timedelta

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


@dataclass(frozen=True)
class ParamSelectOptions:
    """Selection options for param-aware point/slice selection.

    Notes
    -----
    Selection maps query parameter values to existing sequence samples.
    ``layout="packed"`` returns only selected samples.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, ParamSelectOptions
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": ("sample", [0.0, 10.0])}, coords={"sample": [0, 1], "time": ("sample", [0.0, 1.0])}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     param_coord="time",
    ...     validate=True,
    ... )
    >>> ao.param.sel([1.0], on="time", opts=ParamSelectOptions()).as_dataset()["value"].item()
    10.0
    """

    method: Literal["nearest"] = "nearest"
    layout: Literal["packed", "padded"] = "packed"
    query_dim: str = "query"


@dataclass(frozen=True)
class ParamEvalOptions:
    """Evaluation options for param-aware interpolation.

    Notes
    -----
    Evaluation can interpolate between samples. ``method="linear"`` uses the
    declared parameter coordinate as the interpolation axis.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, ParamEvalOptions
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": ("sample", [0.0, 10.0])}, coords={"sample": [0, 1], "time": ("sample", [0.0, 1.0])}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     param_coord="time",
    ...     validate=True,
    ... )
    >>> ao.param.at([0.5], on="time", opts=ParamEvalOptions(method="linear")).as_dataset()["value"].item()
    5.0
    """

    method: Literal["nearest", "linear"] = "linear"
    duplicate_policy: Literal["invalid", "left", "right", "raise"] = "invalid"
    query_dim: str = "query"


@dataclass(frozen=True)
class ParamSyncOptions:
    """Synchronization options for multi-AO param alignment.

    Parameters
    ----------
    join
        Shared grid policy. ``"left"``, ``"right"``, and ``"override"`` use an
        existing or explicit grid; ``"outer"``, ``"inner"``, ``"domain"``, and
        ``"exact"`` synthesize a grid from input parameter domains.
    how
        Evaluation policy for each input on the resolved grid.
    batch_join
        Batch label alignment policy used before synchronization.
    query_dim
        Temporary query dimension name used during evaluation.
    tol
        Numeric tolerance for numeric parameter coordinates, or timedelta-like
        tolerance for datetime64 parameter coordinates. Datetime64 contexts
        accept only exact numeric zero or timedelta-like values.
    fill_value
        Numeric fill value used only when ``how="fill"``.

    Notes
    -----
    Synchronization builds a common param grid for one or more AOs. The
    returned AOs are evaluated on that grid according to ``join`` and ``how``.
    Numeric param coordinates use numeric ``tol`` values. Datetime64 param
    coordinates accept timedelta-like ``tol`` values for synchronization only;
    selection and indexing remain tolerance-free in T1.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, ParamSyncOptions, synchronize
    >>> left = AnalysisObject.from_data(
    ...     xr.Dataset({"value": ("sample", [0.0, 10.0])}, coords={"sample": [0, 1], "time": ("sample", [0.0, 1.0])}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     param_coord="time",
    ...     validate=True,
    ... )
    >>> synced = synchronize([left], opts=ParamSyncOptions(join="left", how="interp"))
    >>> synced[0].as_dataset().sizes["sample"]
    2
    """

    join: Literal["left", "right", "outer", "inner", "domain", "exact", "override"] = "left"
    how: Literal["interp", "nearest", "fill"] = "interp"
    batch_join: Literal["inner", "outer", "exact"] = "inner"
    query_dim: str = "query"
    tol: ParamSyncTolerance = 0.0
    fill_value: float | int | None = np.nan


@dataclass(frozen=True)
class ParamRuntimeContext:
    """Resolved runtime context for param operations.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    ao: AnalysisObject
    ds: xr.Dataset
    spec: ParamCoordSpec
    sequence_dim: str
    batch_dims: tuple[str, ...]
    core_dims: tuple[str, ...]
    valid_mask: xr.DataArray
    sequence_size_coord: str | None
    batch_coords: dict[str, xr.DataArray]
    param_kind: ParamKind = "numeric"


@dataclass(frozen=True)
class ParamIndexResult:
    """Nearest index mapping metadata for point selection/index ops.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    index: xr.DataArray
    valid: xr.DataArray
    grid: QueryGrid
    scalar_query: bool
