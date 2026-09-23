from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from ..orchestration.indexing import restore_result_coordinates
from .backend_selection import _with_map_backend
from .map_build import build_param_map
from .query_grid import _normalize_query_grid_with_topology
from .query_topology import QueryTopologyPlan
from .types import ParamMap, ParamMapOptions, QueryGrid


@dataclass(frozen=True)
class PreparedParamEvaluation:
    """Request-local normalized query and parameter-map state."""

    grid: QueryGrid
    param_map: ParamMap
    sequence_dim: str
    batch_dims: tuple[str, ...]
    param_kind: str
    options: ParamMapOptions
    source_param: xr.DataArray
    source_valid: xr.DataArray
    query_topology: QueryTopologyPlan

    @property
    def has_no_rows(self) -> bool:
        return int(self.param_map.valid.size) == 0


@dataclass(frozen=True)
class _ParamEvaluationRequest:
    param: xr.DataArray
    query: object
    sequence_dim: str
    batch_dims: tuple[str, ...]
    batch_coords: xr.Coordinates | None
    valid_mask: xr.DataArray
    options: ParamMapOptions
    param_kind: str
    query_dim: str
    map_backend: str | None = None


def _array_is_lazy(value: xr.DataArray) -> bool:
    return value.chunks is not None


def _contains_scalar_missing(values: np.ndarray) -> bool:
    for value in values.flat:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return True
    return False


def _object_comparison_is_determinate(left: np.ndarray, right: np.ndarray) -> bool:
    try:
        if _contains_scalar_missing(left) or _contains_scalar_missing(right):
            return False
        comparison = np.equal(left, right)
        for value in np.asarray(comparison, dtype=object).flat:
            _ = bool(value)
    except Exception:  # noqa: BLE001 - equality uncertainty declines reuse.
        return False
    return True


def _eager_values_equal(left: xr.DataArray, right: xr.DataArray) -> bool:
    if left.data is right.data:
        return True
    left_values = np.asarray(left.data)
    right_values = np.asarray(right.data)
    if left_values.dtype.kind in {"f", "c"}:
        return bool(np.array_equal(left_values, right_values, equal_nan=True))
    if left_values.dtype.kind in {"M", "m"}:
        equal = left_values == right_values
        missing = np.isnat(left_values) & np.isnat(right_values)
        return bool(np.all(equal | missing))
    if left_values.dtype.kind == "O" and not _object_comparison_is_determinate(
        left_values,
        right_values,
    ):
        return False
    try:
        return bool(left.variable.equals(right.variable))
    except Exception:  # noqa: BLE001 - equality uncertainty declines reuse.
        return False


def _index_groups_equal(
    left: xr.Dataset | xr.DataArray,
    right: xr.Dataset | xr.DataArray,
) -> bool:
    left_groups = tuple(left.xindexes.group_by_index())
    right_groups = tuple(right.xindexes.group_by_index())
    if len(left_groups) != len(right_groups):
        return False
    for (left_index, left_coords), (right_index, right_coords) in zip(
        left_groups,
        right_groups,
        strict=True,
    ):
        if tuple(left_coords) != tuple(right_coords):
            return False
        if not left_index.equals(right_index):
            return False
    return True


def _mapping_values_equal(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    for key, left_value in left.items():
        right_value = right[key]
        if left_value is right_value:
            continue
        try:
            equal = left_value == right_value
        except Exception:  # noqa: BLE001 - equality uncertainty declines reuse.
            return False
        if not isinstance(equal, (bool, np.bool_)) or not bool(equal):
            return False
    return True


def _metadata_equal(left: xr.DataArray, right: xr.DataArray) -> bool:
    return _mapping_values_equal(left.attrs, right.attrs) and _mapping_values_equal(
        left.encoding,
        right.encoding,
    )


def _coordinate_values_equal(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    indexed: bool,
) -> bool:
    if indexed:
        return True
    if _array_is_lazy(left) or _array_is_lazy(right):
        return left.data is right.data
    return _eager_values_equal(left, right)


def _coordinate_topology_equal(
    left: xr.Dataset | xr.DataArray,
    right: xr.Dataset | xr.DataArray,
) -> bool:
    if tuple(left.coords) != tuple(right.coords):
        return False
    indexed = set(left.xindexes)
    if indexed != set(right.xindexes):
        return False
    for name in left.coords:
        left_coord = left.coords[name]
        right_coord = right.coords[name]
        if left_coord.dims != right_coord.dims or left_coord.dtype != right_coord.dtype:
            return False
        if not _metadata_equal(left_coord, right_coord):
            return False
        if not _coordinate_values_equal(
            left_coord,
            right_coord,
            indexed=name in indexed,
        ):
            return False
    return True


def _dataarrays_equivalent(left: xr.DataArray, right: xr.DataArray) -> bool:
    if (
        left.name != right.name
        or left.dims != right.dims
        or left.shape != right.shape
        or left.dtype != right.dtype
        or not _metadata_equal(left, right)
    ):
        return False
    if not _index_groups_equal(left, right):
        return False
    if not _coordinate_topology_equal(left, right):
        return False
    if _array_is_lazy(left) or _array_is_lazy(right):
        return left.data is right.data
    return _eager_values_equal(left, right)


def _query_grids_equivalent(left: QueryGrid, right: QueryGrid) -> bool:
    return (
        left.query_dim == right.query_dim
        and left.stacked_dims == right.stacked_dims
        and _dataarrays_equivalent(left.values, right.values)
    )


def _topology_coordinates(plan: QueryTopologyPlan) -> xr.Dataset:
    return restore_result_coordinates(  # type: ignore[return-value]
        xr.Dataset(),
        plan.coordinates,
    )


def _query_topologies_equivalent(
    left: QueryTopologyPlan,
    right: QueryTopologyPlan,
) -> bool:
    if (
        left.query_dim != right.query_dim
        or left.dims != right.dims
        or left.sizes != right.sizes
        or left.stacked_dims != right.stacked_dims
    ):
        return False
    left_coords = _topology_coordinates(left)
    right_coords = _topology_coordinates(right)
    return _index_groups_equal(left_coords, right_coords) and _coordinate_topology_equal(
        left_coords,
        right_coords,
    )


def prepared_evaluation_matches(
    prepared: PreparedParamEvaluation,
    *,
    grid: QueryGrid,
    param: xr.DataArray,
    valid_mask: xr.DataArray,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    param_kind: str,
    options: ParamMapOptions,
    query_topology: QueryTopologyPlan,
) -> bool:
    """Return whether an existing request-local map is exactly reusable."""
    if (
        prepared.sequence_dim != sequence_dim
        or prepared.batch_dims != batch_dims
        or prepared.param_kind != param_kind
        or prepared.options != options
    ):
        return False
    try:
        return (
            _query_topologies_equivalent(
                prepared.query_topology,
                query_topology,
            )
            and _query_grids_equivalent(prepared.grid, grid)
            and _dataarrays_equivalent(prepared.source_param, param)
            and _dataarrays_equivalent(prepared.source_valid, valid_mask)
        )
    except Exception:  # noqa: BLE001 - equality uncertainty declines reuse.
        return False


def prepare_param_evaluation(
    *,
    param: xr.DataArray,
    query: object,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    batch_coords: xr.Coordinates | None,
    valid_mask: xr.DataArray,
    options: ParamMapOptions,
    param_kind: str,
    query_dim: str,
    reuse: Iterable[PreparedParamEvaluation] = (),
) -> PreparedParamEvaluation:
    """Normalize and map one request, reusing an exactly equivalent plan."""
    request = _ParamEvaluationRequest(
        param,
        query,
        sequence_dim,
        batch_dims,
        batch_coords,
        valid_mask,
        options,
        param_kind,
        query_dim,
    )
    return _prepare_param_evaluation(request, reuse=reuse)


def _prepare_param_evaluation(
    request: _ParamEvaluationRequest,
    *,
    reuse: Iterable[PreparedParamEvaluation] = (),
) -> PreparedParamEvaluation:
    grid, query_topology = _normalize_query_grid_with_topology(
        request.query,
        query_dim=request.query_dim,
        batch_dims=request.batch_dims,
        batch_coords=request.batch_coords,
        param_kind=request.param_kind,
    )
    for candidate in reuse:
        if prepared_evaluation_matches(
            candidate,
            grid=grid,
            param=request.param,
            valid_mask=request.valid_mask,
            sequence_dim=request.sequence_dim,
            batch_dims=request.batch_dims,
            param_kind=request.param_kind,
            options=request.options,
            query_topology=query_topology,
        ):
            return candidate
    return _new_prepared_evaluation(
        grid=grid,
        request=request,
        query_topology=query_topology,
    )


def _new_prepared_evaluation(
    *,
    grid: QueryGrid,
    request: _ParamEvaluationRequest,
    query_topology: QueryTopologyPlan,
) -> PreparedParamEvaluation:
    param_map = build_param_map(
        param=request.param,
        query=grid.values,
        sequence_dim=request.sequence_dim,
        query_dim=grid.query_dim,
        valid_mask=request.valid_mask,
        options=_with_map_backend(request.options, request.map_backend),
        param_kind=request.param_kind,
    )
    return PreparedParamEvaluation(
        grid=grid,
        param_map=param_map,
        sequence_dim=request.sequence_dim,
        batch_dims=request.batch_dims,
        param_kind=request.param_kind,
        options=request.options,
        source_param=request.param,
        source_valid=request.valid_mask,
        query_topology=query_topology,
    )


__all__ = [
    "PreparedParamEvaluation",
    "_ParamEvaluationRequest",
    "_prepare_param_evaluation",
    "prepare_param_evaluation",
    "prepared_evaluation_matches",
]
