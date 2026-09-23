from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ..ao_internal import finalize_structural
from ..dataset_ownership import analysis_object_dataset
from ..orchestration.finalize import transfer_dataset_attrs
from ..orchestration.indexing import (
    without_dimension_coordinate,
    without_index_topology,
)
from ..param_engine.query_output_verify import verify_query_output_plan
from ..param_engine.query_topology import (
    QueryOutputPlan,
    QueryTopologyPlan,
    attach_query_coordinates,
    restore_query_topology,
)
from ..schema_update import source_schema_view
from ..validity_finalize import assign_sequence_size_from_valid_mask
from .guards import mark_generated_size_coord, mark_reserved_coord
from .types import ParamRuntimeContext

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


def _positional_coord(size: int) -> np.ndarray:
    return np.arange(int(size), dtype="int64")


def _rename_query_dim(
    da: xr.DataArray,
    *,
    query_dim: str,
    sequence_dim: str,
) -> xr.DataArray:
    if query_dim in da.dims and query_dim != sequence_dim:
        return da.rename({query_dim: sequence_dim})
    return da


def _query_attach_values(
    query: xr.DataArray,
    *,
    query_dim: str,
    sequence_dim: str,
    sequence_size: int,
) -> xr.DataArray:
    projected = without_index_topology(query, dims=(query_dim,))
    out = _rename_query_dim(
        projected.reset_coords(drop=True),  # type: ignore[union-attr]
        query_dim=query_dim,
        sequence_dim=sequence_dim,
    )
    if sequence_dim not in out.dims:
        return out
    if int(out.sizes[sequence_dim]) != int(sequence_size):
        raise ValueError(
            "finalize_param_output: query length does not match output sequence length after normalization."
        )
    out = out.drop_vars(sequence_dim, errors="ignore")
    return out.assign_coords({sequence_dim: _positional_coord(sequence_size)})


def assign_sampled_query_coordinate(
    value: xr.Dataset,
    *,
    query: xr.DataArray,
    query_dim: str,
    name: str,
) -> xr.Dataset:
    """Attach current query values without inheriting caller axis metadata."""
    projected = without_index_topology(query, dims=(query_dim,))
    coordinate = projected.reset_coords(drop=True).drop_vars(query_dim, errors="ignore")
    return value.assign_coords({name: coordinate})


def _finalize_trajectory_coordinates(
    context: ParamRuntimeContext,
    ds: xr.Dataset,
    *,
    query: xr.DataArray | None,
    query_dim: str,
    valid_query: xr.DataArray | None,
    query_topology: QueryTopologyPlan | None,
    output_plan: QueryOutputPlan | None,
    owner: str,
) -> xr.Dataset:
    ds_out = ds
    if query_dim in ds_out.dims and query_dim != context.sequence_dim:
        ds_out = without_dimension_coordinate(ds_out, dim=query_dim)  # type: ignore[assignment]
        if context.sequence_dim in ds_out.coords and context.sequence_dim not in ds_out.dims:
            ds_out = ds_out.drop_vars(context.sequence_dim, errors="ignore")
        ds_out = ds_out.rename({query_dim: context.sequence_dim})
    if context.sequence_dim in ds_out.dims:
        ds_out = ds_out.assign_coords({context.sequence_dim: _positional_coord(ds_out.sizes[context.sequence_dim])})
    if query is not None:
        q = _query_attach_values(
            query,
            query_dim=query_dim,
            sequence_dim=context.sequence_dim,
            sequence_size=int(ds_out.sizes.get(context.sequence_dim, 0)),
        )
        ds_out = ds_out.assign_coords({context.spec.name: q})
    if valid_query is not None and context.sequence_size_coord:
        v = _rename_query_dim(valid_query, query_dim=query_dim, sequence_dim=context.sequence_dim)
        ds_out, _ = assign_sequence_size_from_valid_mask(
            ds_out,
            valid=v,
            sequence_dim=context.sequence_dim,
            batch_dims=context.batch_dims,
            sequence_size_coord=context.sequence_size_coord,
        )
        ds_out = mark_generated_size_coord(ds_out, name=context.sequence_size_coord)
    if output_plan is not None and output_plan.intent == "trajectory" and query_topology is not None:
        if output_plan.topology is not query_topology:
            raise ValueError(f"{owner}: typed output plan does not match the prepared query topology.")
        ds_out = attach_query_coordinates(
            ds_out, topology=query_topology, plan=output_plan, owner=owner,
        )
    return ds_out


def _prepare_param_output_dataset(
    context: ParamRuntimeContext,
    ds: xr.Dataset,
    *,
    query: xr.DataArray | None,
    query_dim: str,
    valid_query: xr.DataArray | None,
    query_topology: QueryTopologyPlan | None,
    trajectory: bool,
    owner: str = "finalize_param_output",
    output_plan: QueryOutputPlan | None = None,
    copy_schema: bool = True,
) -> xr.Dataset:
    ds_out = (
        transfer_dataset_attrs(context.ds, ds, validate=False)
        if copy_schema
        else source_schema_view(context.ds, ds)
    )
    if valid_query is not None and output_plan is not None and "valid" in output_plan.generated_names:
        ds_out = ds_out.assign_coords({
            "valid": mark_reserved_coord(valid_query, name="valid"),
        })
    if query is not None and query_dim in ds_out.dims and not trajectory:
        if query_topology is None:
            raise ValueError("finalize_param_output: stacked query topology is missing.")
        ds_out = restore_query_topology(
            ds_out,
            plan=query_topology,
            owner=owner,
        )  # type: ignore[assignment]
    if trajectory:
        ds_out = _finalize_trajectory_coordinates(
            context, ds_out, query=query, query_dim=query_dim,
            valid_query=valid_query, query_topology=query_topology,
            output_plan=output_plan, owner=owner,
        )
    return ds_out


def finalize_param_output(
    context: ParamRuntimeContext,
    ds: xr.Dataset,
    *,
    query: xr.DataArray | None,
    query_dim: str,
    valid_query: xr.DataArray | None,
    query_topology: QueryTopologyPlan | None,
    validate: bool,
    trajectory: bool,
    owner: str = "finalize_param_output",
    output_plan: QueryOutputPlan | None = None,
) -> AnalysisObject:
    ds_out = _prepare_param_output_dataset(
        context,
        ds,
        query=query,
        query_dim=query_dim,
        valid_query=valid_query,
        query_topology=query_topology,
        trajectory=trajectory,
        owner=owner,
        output_plan=output_plan,
    )
    return _finalize_prepared_param_output(
        context,
        ds_out,
        validate=validate,
        trajectory=trajectory,
        output_plan=output_plan,
        query_topology=query_topology,
    )


def _finalize_prepared_param_output(
    context: ParamRuntimeContext,
    ds_out: xr.Dataset,
    *,
    validate: bool,
    trajectory: bool,
    output_plan: QueryOutputPlan | None,
    query_topology: QueryTopologyPlan | None,
) -> AnalysisObject:
    out = finalize_structural(context.ao, ds_out, validate=validate)
    if not trajectory:
        core = analysis_object_dataset(out).attrs.get("tal", {}).get("core", {})
        if isinstance(core, dict) and "validity" in core:
            out = out.set_validity(sequence_size_coord=None, validate=validate)
    if output_plan is not None and query_topology is not None:
        verify_query_output_plan(
            analysis_object_dataset(out), plan=output_plan, topology=query_topology,
        )
    return out


__all__ = [
    "finalize_param_output",
]
