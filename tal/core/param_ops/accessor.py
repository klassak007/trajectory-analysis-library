from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import xarray as xr

from ..orchestration.resolve import resolve_param_runtime_context
from .options import coerce_eval_options, coerce_select_options
from .types import ParamEvalOptions, ParamRuntimeContext, ParamSelectOptions


def _resolve_runtime_context(
    ao: "AnalysisObject",
    *,
    on: str | None = None,
    sequence_dim: str | None = None,
    batch_dims: Sequence[str] | None = None,
    sequence_size_coord: str | None = None,
) -> ParamRuntimeContext:
    return resolve_param_runtime_context(
        ao,
        on=on,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
    )


class ParamAccessor:
    """Accessor for param-aware selection and alignment operations.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    def __init__(self, ao: "AnalysisObject") -> None:
        self._ao = ao

    def index(
        self,
        query: xr.DataArray | np.ndarray | Sequence[float] | float,
        *,
        on: str | None = None,
        opts: ParamSelectOptions | None = None,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] | None = None,
        sequence_size_coord: str | None = None,
    ) -> xr.DataArray:
        """Build nearest index mapping for param queries.

        Parameters
        ----------
        query : xr.DataArray | np.ndarray | Sequence[float] | float
            Query coordinate/grid for param-aware selection/evaluation.
        on : str | None, optional
            Coordinate/dimension name used as the operation domain.
        opts : ParamSelectOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``ParamSelectOptions`` key fields: ``method`` (default 'nearest'), ``layout`` (default 'packed'), ``query_dim`` (default 'query').
        sequence_dim : str | None, optional
            Optional override for the sequence dimension used by temporal semantics.
        batch_dims : Sequence[str] | None, optional
            Optional override for batch dimensions used by temporal semantics.
        sequence_size_coord : str | None, optional
            Optional sequence-size coordinate used for ragged validity handling.

        Returns
        -------
        xr.DataArray
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.
        Datetime64 param coordinates accept datetime-like queries. Indexing is
        tolerance-free in T1.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject, ParamSelectOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [0.0, 1.0, 4.0])}, coords={"sample": [0, 1, 2], "time": ("sample", [0.0, 1.0, 2.0])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> ao.param.index([0.2, 1.8], on="time", opts=ParamSelectOptions(method="nearest")).values.tolist()
        [0, 2]
        """
        from .index import build_index

        options = coerce_select_options(opts, owner="param index")
        ctx = _resolve_runtime_context(
            self._ao,
            on=on,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
        )
        return build_index(ctx, query=query, opts=options)

    def sel(
        self,
        query: xr.DataArray | np.ndarray | Sequence[float] | float | slice,
        *,
        on: str | None = None,
        opts: ParamSelectOptions | None = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "AnalysisObject":
        """Select existing samples by param-coordinate query.

        Parameters
        ----------
        query : xr.DataArray | np.ndarray | Sequence[float] | float | slice
            Query coordinate/grid for param-aware selection/evaluation.
        on : str | None, optional
            Coordinate/dimension name used as the operation domain.
        opts : ParamSelectOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``ParamSelectOptions`` key fields: ``method`` (default 'nearest'), ``layout`` (default 'packed'), ``query_dim`` (default 'query').
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.
        sequence_dim : str | None, optional
            Optional override for the sequence dimension used by temporal semantics.
        batch_dims : Sequence[str] | None, optional
            Optional override for batch dimensions used by temporal semantics.
        sequence_size_coord : str | None, optional
            Optional sequence-size coordinate used for ragged validity handling.

        Returns
        -------
        AnalysisObject
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.
        Datetime64 param coordinates accept datetime-like point and slice
        queries. Selection is tolerance-free in T1.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject, ParamSelectOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [0.0, 1.0, 4.0])}, coords={"sample": [0, 1, 2], "time": ("sample", [0.0, 1.0, 2.0])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> selected = ao.param.sel([0.2, 1.8], on="time", opts=ParamSelectOptions(method="nearest"))
        >>> selected.unsafe_data["value"].values.tolist()
        [0.0, 4.0]
        """
        from .select import select_param

        options = coerce_select_options(opts, owner="param sel")
        ctx = _resolve_runtime_context(
            self._ao,
            on=on,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
        )
        return select_param(ctx, query=query, opts=options, validate=validate)

    def at(
        self,
        query: xr.DataArray | np.ndarray | Sequence[float] | float,
        *,
        on: str | None = None,
        opts: ParamEvalOptions | None = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "AnalysisObject":
        """Evaluate AO values at param queries via interpolation.

        Parameters
        ----------
        query : xr.DataArray | np.ndarray | Sequence[float] | float
            Query coordinate/grid for param-aware selection/evaluation.
        on : str | None, optional
            Coordinate/dimension name used as the operation domain.
        opts : ParamEvalOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``ParamEvalOptions`` key fields: ``method`` (default 'linear'), ``duplicate_policy`` (default 'invalid'), ``query_dim`` (default 'query').
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.
        sequence_dim : str | None, optional
            Optional override for the sequence dimension used by temporal semantics.
        batch_dims : Sequence[str] | None, optional
            Optional override for batch dimensions used by temporal semantics.
        sequence_size_coord : str | None, optional
            Optional sequence-size coordinate used for ragged validity handling.

        Returns
        -------
        AnalysisObject
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.
        Datetime64 param coordinates accept datetime-like queries and compute
        interpolation weights from row-local nanosecond deltas.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject, ParamEvalOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [0.0, 1.0, 4.0])}, coords={"sample": [0, 1, 2], "time": ("sample", [0.0, 1.0, 2.0])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> out = ao.param.at([0.5, 1.5], on="time", opts=ParamEvalOptions(method="linear"))
        >>> out.unsafe_data["value"].values.tolist()
        [0.5, 2.5]
        """
        from .evaluate import evaluate_param

        options = coerce_eval_options(opts, owner="param at/resample")
        ctx = _resolve_runtime_context(
            self._ao,
            on=on,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
        )
        return evaluate_param(ctx, query=query, opts=options, validate=validate)

    def resample_to(
        self,
        grid: xr.DataArray | np.ndarray | Sequence[float] | float,
        *,
        on: str | None = None,
        opts: ParamEvalOptions | None = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "AnalysisObject":
        """Resample AO values to a target param grid.

        Parameters
        ----------
        grid : xr.DataArray | np.ndarray | Sequence[float] | float
            Target query grid for resampling/evaluation.
        on : str | None, optional
            Coordinate/dimension name used as the operation domain.
        opts : ParamEvalOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``ParamEvalOptions`` key fields: ``method`` (default 'linear'), ``duplicate_policy`` (default 'invalid'), ``query_dim`` (default 'query').
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.
        sequence_dim : str | None, optional
            Optional override for the sequence dimension used by temporal semantics.
        batch_dims : Sequence[str] | None, optional
            Optional override for batch dimensions used by temporal semantics.
        sequence_size_coord : str | None, optional
            Optional sequence-size coordinate used for ragged validity handling.

        Returns
        -------
        AnalysisObject
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.
        Datetime64 param coordinates accept datetime-like target grids and
        compute interpolation weights from row-local nanosecond deltas.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject, ParamEvalOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [0.0, 1.0, 4.0])}, coords={"sample": [0, 1, 2], "time": ("sample", [0.0, 1.0, 2.0])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> out = ao.param.resample_to([0.0, 0.5, 1.0], on="time", opts=ParamEvalOptions(method="linear"))
        >>> out.unsafe_data["value"].values.tolist()
        [0.0, 0.5, 1.0]
        """
        from .resample import resample_param

        options = coerce_eval_options(opts, owner="param at/resample")
        ctx = _resolve_runtime_context(
            self._ao,
            on=on,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
        )
        return resample_param(ctx, grid=grid, opts=options, validate=validate)

    def interp_like(
        self,
        other: "AnalysisObject | xr.Dataset | xr.DataArray",
        *,
        on: str | None = None,
        opts: ParamEvalOptions | None = None,
        batch_join: Literal["inner", "left"] = "inner",
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "AnalysisObject":
        """Interpolate this AO onto another AO-like object's param grid.

        Parameters
        ----------
        other : AnalysisObject | xr.Dataset | xr.DataArray
            Reference AO-like object that provides the target parameter grid for interpolation.
        on : str | None, optional
            Coordinate/dimension name used as the operation domain.
        opts : ParamEvalOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``ParamEvalOptions`` key fields: ``method`` (default 'linear'), ``duplicate_policy`` (default 'invalid'), ``query_dim`` (default 'query').
        batch_join : Literal['inner', 'left'], optional
            Batch alignment policy used before interpolation (``'inner'`` or ``'left'``).
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.
        sequence_dim : str | None, optional
            Optional override for the sequence dimension used by temporal semantics.
        batch_dims : Sequence[str] | None, optional
            Optional override for batch dimensions used by temporal semantics.
        sequence_size_coord : str | None, optional
            Optional sequence-size coordinate used for ragged validity handling.

        Returns
        -------
        AnalysisObject
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.
        Datetime64 source and target param coordinates remain datetime64
        throughout query normalization and interpolation.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject, ParamEvalOptions
        >>> source = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [0.0, 1.0, 4.0])}, coords={"sample": [0, 1, 2], "time": ("sample", [0.0, 1.0, 2.0])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> target = AnalysisObject.from_data(
        ...     xr.Dataset({"target": ("sample", [10.0, 20.0])}, coords={"sample": [0, 1], "time": ("sample", [0.0, 2.0])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> out = source.param.interp_like(target, on="time", opts=ParamEvalOptions(method="linear"))
        >>> out.unsafe_data["value"].values.tolist()
        [0.0, 4.0]
        """
        from .interp_like import interp_like_param

        options = coerce_eval_options(opts, owner="interp_like")
        ctx = _resolve_runtime_context(
            self._ao,
            on=on,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
        )
        return interp_like_param(
            ctx,
            other=other,
            opts=options,
            batch_join=batch_join,
            validate=validate,
        )


__all__ = ["ParamAccessor", "_resolve_runtime_context"]
