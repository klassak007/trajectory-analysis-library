from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.utils.xarray_namespace import dataset_namespace_names, unique_temp_dim

from ..orchestration.lazy import is_chunked_dataarray
from ..param_engine.map_apply import gather_dataset_along_sequence
from .when_common import enforce_when_on_empty, selected_when_mask
from .evaluate import evaluate_mask
from .event_primitives import SAMPLE_SENTINEL, batch_dims, lane_data
from .finalize import finalize_event_output
from .intervals import extract_intervals
from .options import coerce_interval_extract_options
from .pack import pack_interval_table
from .resolve import EventEvalContext, resolve_event_eval_context
from .types import Condition, WhenOptions, IntervalExtractOptions

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject

_SEGMENT_META = (
    "segment_start_time",
    "segment_end_time",
    "segment_start_index",
    "segment_end_index",
    "orig_index",
)


def _segment_interval_options(opts: WhenOptions, *, owner: str) -> IntervalExtractOptions:
    options = IntervalExtractOptions(
        eval=opts.eval,
        include_initial=True,
        truth_eval="exact",
        max_segments=opts.max_segments,
    )
    return coerce_interval_extract_options(options, owner=owner)


def _segment_and_edge_dims(table: xr.Dataset, *, context: EventEvalContext, owner: str) -> tuple[str, str]:
    dims = [dim for dim in table["time"].dims if dim not in context.runtime.batch_dims]
    if len(dims) != 2:
        raise ValueError(f"{owner}: expected one segment dim and one edge dim, got {dims!r}.")
    edge_matches = [
        dim
        for dim in dims
        if table.sizes[dim] == 2 and set(map(str, np.asarray(table.coords[dim].values).tolist())) == {"start", "end"}
    ]
    if len(edge_matches) != 1:
        raise ValueError(f"{owner}: unable to resolve interval edge dim from {dims!r}.")
    edge_dim = str(edge_matches[0])
    segment_dim = next(dim for dim in dims if dim != edge_dim)
    return str(segment_dim), edge_dim


def _segment_valid(table: xr.Dataset, *, owner: str) -> xr.DataArray:
    _ = owner
    return table["valid_segment"].astype(bool)

def _orig_index_row(start: int, end: int, *, valid: bool, sequence_size: int) -> np.ndarray:
    out = np.full(sequence_size, SAMPLE_SENTINEL, dtype="int64")
    if not valid:
        return out
    if start < 0 or end < start:
        return out
    stop = min(int(end), sequence_size - 1)
    if stop < start:
        return out
    count = stop - int(start) + 1
    out[:count] = np.arange(int(start), int(start) + count, dtype="int64")
    return out


def _build_orig_index_dynamic(
    start_idx: xr.DataArray,
    end_idx: xr.DataArray,
    segment_valid: xr.DataArray,
    *,
    context: EventEvalContext,
    segment_dim: str,
    sequence_dim: str,
) -> xr.DataArray:
    sequence_size = int(context.clock.sizes[sequence_dim])
    start_lanes = lane_data(start_idx.astype("int64"), context=context, core_dim=segment_dim)
    end_lanes = lane_data(end_idx.astype("int64"), context=context, core_dim=segment_dim)
    valid_lanes = lane_data(segment_valid.astype(bool), context=context, core_dim=segment_dim)
    lane_rows = np.full(
        (start_lanes.shape[0], start_lanes.shape[1], sequence_size),
        SAMPLE_SENTINEL,
        dtype="int64",
    )
    for lane in range(start_lanes.shape[0]):
        for seg in range(start_lanes.shape[1]):
            lane_rows[lane, seg, :] = _orig_index_row(
                int(start_lanes[lane, seg]),
                int(end_lanes[lane, seg]),
                valid=bool(valid_lanes[lane, seg]),
                sequence_size=sequence_size,
            )
    batch_shape = tuple(context.clock.sizes[dim] for dim in batch_dims(context))
    shape = (start_lanes.shape[1], sequence_size) if not batch_shape else batch_shape + (start_lanes.shape[1], sequence_size)
    coords: dict[str, object] = {
        segment_dim: start_idx.coords[segment_dim],
        sequence_dim: context.clock.coords[sequence_dim],
    }
    for dim in context.runtime.batch_dims:
        coords[dim] = context.clock.coords[dim]
    dims = context.runtime.batch_dims + (segment_dim, sequence_dim)
    return xr.DataArray(lane_rows.reshape(shape), dims=dims, coords=coords, name="orig_index").astype("int64")


def _bounded_orig_index_block(
    start_block: np.ndarray,
    end_block: np.ndarray,
    valid_block: np.ndarray,
    *,
    sequence_size: int,
) -> np.ndarray:
    start = start_block.astype("int64", copy=False)
    end = end_block.astype("int64", copy=False)
    valid = valid_block.astype(bool, copy=False)
    clipped_end = np.minimum(end, int(sequence_size) - 1)
    count = clipped_end - start + 1
    count = np.where(valid & (start >= 0) & (clipped_end >= start), count, 0)
    count = np.clip(count, 0, int(sequence_size)).astype("int64")
    pos = np.arange(int(sequence_size), dtype="int64")
    expanded = start[..., None] + pos
    return np.where(pos < count[..., None], expanded, SAMPLE_SENTINEL).astype("int64")


def _build_orig_index_bounded(
    start_idx: xr.DataArray,
    end_idx: xr.DataArray,
    segment_valid: xr.DataArray,
    *,
    context: EventEvalContext,
    segment_dim: str,
    sequence_dim: str,
) -> xr.DataArray:
    sequence_size = int(context.clock.sizes[sequence_dim])
    chunked = any(is_chunked_dataarray(da) for da in (start_idx, end_idx, segment_valid))
    ufunc_kwargs: dict[str, object] = {}
    if chunked:
        ufunc_kwargs["dask_gufunc_kwargs"] = {
            "output_sizes": {sequence_dim: sequence_size},
            "allow_rechunk": True,
        }
    out = xr.apply_ufunc(
        _bounded_orig_index_block,
        start_idx.astype("int64"),
        end_idx.astype("int64"),
        segment_valid.astype(bool),
        input_core_dims=[[segment_dim], [segment_dim], [segment_dim]],
        output_core_dims=[[segment_dim, sequence_dim]],
        kwargs={"sequence_size": sequence_size},
        vectorize=False,
        dask="parallelized" if chunked else "allowed",
        output_dtypes=[np.int64],
        **ufunc_kwargs,
    )
    return out.assign_coords({sequence_dim: context.clock.coords[sequence_dim]}).rename("orig_index").astype("int64")


def _build_orig_index(
    start_idx: xr.DataArray,
    end_idx: xr.DataArray,
    segment_valid: xr.DataArray,
    *,
    context: EventEvalContext,
    segment_dim: str,
    sequence_dim: str,
    bounded: bool,
) -> xr.DataArray:
    if bounded:
        return _build_orig_index_bounded(
            start_idx,
            end_idx,
            segment_valid,
            context=context,
            segment_dim=segment_dim,
            sequence_dim=sequence_dim,
        )
    return _build_orig_index_dynamic(
        start_idx,
        end_idx,
        segment_valid,
        context=context,
        segment_dim=segment_dim,
        sequence_dim=sequence_dim,
    )


def _safe_gather_indexer(orig_index: xr.DataArray, *, sequence_size: int) -> xr.DataArray:
    return orig_index.where(orig_index >= 0, 0).clip(min=0, max=max(sequence_size - 1, 0)).astype("int64")


def _mask_sequence_payload(ds: xr.Dataset, *, valid_samples: xr.DataArray, sequence_dim: str) -> xr.Dataset:
    var_updates = {
        name: var.where(valid_samples)
        for name, var in ds.data_vars.items()
        if sequence_dim in var.dims
    }
    coord_updates = {
        name: coord.where(valid_samples)
        for name, coord in ds.coords.items()
        if name != sequence_dim and sequence_dim in coord.dims
    }
    out = ds.assign(var_updates) if var_updates else ds
    return out.assign_coords(coord_updates) if coord_updates else out


def _segment_metadata_namespace_safe(ds: xr.Dataset, *, owner: str) -> None:
    names = set(dataset_namespace_names(ds))
    conflicts = sorted(name for name in _SEGMENT_META if name in names)
    if conflicts:
        raise ValueError(f"{owner}: segment metadata names conflict with dataset namespace: {conflicts!r}.")


def _size_coord_name(ds: xr.Dataset, *, context: EventEvalContext, owner: str) -> str:
    declared = context.runtime.sequence_size_coord
    if declared is not None:
        if declared in ds.data_vars and declared not in ds.coords:
            raise ValueError(f"{owner}: declared sequence_size_coord {declared!r} conflicts with data variable.")
        return str(declared)
    taken = set(dataset_namespace_names(ds))
    return unique_temp_dim("segment_size", taken_dims=tuple(sorted(taken)))


def _resolve_segment_table(
    ao: "AnalysisObject",
    condition: Condition,
    *,
    opts: WhenOptions,
    owner: str,
) -> tuple[EventEvalContext, xr.Dataset, xr.DataArray, str, str]:
    context = resolve_event_eval_context(ao, opts=opts.eval, owner=owner)
    effective = evaluate_mask(condition, context=context, owner=owner)
    selected = selected_when_mask(
        effective,
        valid_mask=context.valid_mask,
        inside=opts.inside,
    )
    interval_opts = _segment_interval_options(opts, owner=owner)
    payload = extract_intervals(
        condition,
        effective_mask=selected,
        context=context,
        opts=interval_opts,
        owner=owner,
        emit_triggers=False,
    )
    table = pack_interval_table(payload, context=context, owner=owner)
    segment_dim, edge_dim = _segment_and_edge_dims(table, context=context, owner=owner)
    segment_valid = _segment_valid(table, owner=owner)
    enforce_when_on_empty(
        segment_valid,
        opts=opts,
        owner=owner,
        layout="segments",
        unit="segments",
        chunked_message="opts.on_empty='error' requires unchunked when segment selection.",
    )
    return context, table, segment_valid, segment_dim, edge_dim


def _segment_bounds(
    table: xr.Dataset,
    *,
    edge_dim: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    start_time = table["time"].sel({edge_dim: "start"}).astype("float64").rename("segment_start_time")
    end_time = table["time"].sel({edge_dim: "end"}).astype("float64").rename("segment_end_time")
    start_idx = table["sample_index"].sel({edge_dim: "start"}).astype("int64").rename("segment_start_index")
    end_idx = table["sample_index"].sel({edge_dim: "end"}).astype("int64").rename("segment_end_index")
    return start_time, end_time, start_idx, end_idx


def _gather_segment_dataset(
    context: EventEvalContext,
    *,
    segment_dim: str,
    segment_valid: xr.DataArray,
    start_idx: xr.DataArray,
    end_idx: xr.DataArray,
    bounded: bool,
    owner: str,
) -> tuple[xr.Dataset, xr.DataArray, xr.DataArray]:
    seq_dim = context.runtime.sequence_dim
    orig_index = _build_orig_index(
        start_idx,
        end_idx,
        segment_valid,
        context=context,
        segment_dim=segment_dim,
        sequence_dim=seq_dim,
        bounded=bounded,
    )
    safe_indexer = _safe_gather_indexer(orig_index, sequence_size=int(context.clock.sizes[seq_dim]))
    gathered = gather_dataset_along_sequence(
        context.runtime.ds,
        safe_indexer,
        sequence_dim=seq_dim,
        query_dim=seq_dim,
        owner=owner,
    )
    valid_samples = (orig_index >= 0).astype(bool)
    masked = _mask_sequence_payload(gathered, valid_samples=valid_samples, sequence_dim=seq_dim)
    if seq_dim in context.runtime.ds.coords and context.runtime.ds.coords[seq_dim].dims == (seq_dim,):
        masked = masked.assign_coords({seq_dim: context.runtime.ds.coords[seq_dim]})
    return masked, orig_index, valid_samples


def evaluate_when_segments_layout(
    ao: "AnalysisObject",
    condition: Condition,
    *,
    opts: WhenOptions,
    validate: bool = True,
    owner: str = "events.when",
) -> "AnalysisObject":
    """Evaluate condition-driven selection in segment-major layout.

    Parameters
    ----------
    ao : AnalysisObject
        AnalysisObject-like input value.
    condition : Condition
        Condition/expression used for event or mask evaluation.
    opts : WhenOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    context, table, segment_valid, segment_dim, edge_dim = _resolve_segment_table(
        ao,
        condition,
        opts=opts,
        owner=owner,
    )
    start_time, end_time, start_idx, end_idx = _segment_bounds(table, edge_dim=edge_dim)
    masked, orig_index, valid_samples = _gather_segment_dataset(
        context,
        segment_dim=segment_dim,
        segment_valid=segment_valid,
        start_idx=start_idx,
        end_idx=end_idx,
        bounded=opts.max_segments is not None,
        owner=owner,
    )
    masked = masked.assign_coords({segment_dim: table.coords[segment_dim]})
    _segment_metadata_namespace_safe(masked, owner=owner)
    size_name = _size_coord_name(masked, context=context, owner=owner)
    segment_size = valid_samples.sum(dim=context.runtime.sequence_dim).astype("int64").rename(size_name)
    with_meta = masked.assign_coords(
        {
            "segment_start_time": start_time,
            "segment_end_time": end_time,
            "segment_start_index": start_idx,
            "segment_end_index": end_idx,
            "orig_index": orig_index,
            size_name: segment_size,
        }
    )
    return finalize_event_output(
        context.ao,
        with_meta,
        sequence_dim=context.runtime.sequence_dim,
        batch_dims=context.runtime.batch_dims + (segment_dim,),
        core_dims=context.runtime.core_dims,
        param_name=context.runtime.spec.name,
        size_name=size_name,
        validate=validate,
        owner=owner,
    )


__all__ = ["evaluate_when_segments_layout"]
