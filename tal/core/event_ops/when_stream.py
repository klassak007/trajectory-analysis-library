from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ...utils.xarray_namespace import dataset_namespace_names, unique_temp_dim
from ..dataset_ownership import analysis_object_dataset
from ..orchestration.indexing import dimension_coordinates
from ..orchestration.lazy import (
    fail_if_chunked_boundary,
    is_chunked_dataarray,
    is_chunked_variable,
)
from ..schema_read import read_roles
from .event_primitives import SAMPLE_SENTINEL
from .finalize import finalize_event_output
from .resolve import resolve_event_eval_context
from .types import Condition, WhenOptions
from .when_common import enforce_when_on_empty
from .when_segments import evaluate_when_segments_layout
from .window_stack import (
    StackedStreamResult,
    _bounded_stream_indexer,
    gather_packed_samples,
    mask_aligned_samples,
    stack_segment_stream,
)

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject
    from .resolve import EventEvalContext

_STREAM_INDEX_META = ("orig_index", "stream_segment_index", "segment_start_index", "segment_end_index")
_STREAM_TIME_META = ("segment_start_time", "segment_end_time")


def _stream_source_dims(
    ds: xr.Dataset,
    *,
    owner: str,
) -> tuple[tuple[str, ...], str, str]:
    roles_declared, sequence_dim, batch_dims, _ = read_roles(ds)
    if not roles_declared or sequence_dim is None:
        raise ValueError(f"{owner}: expected when segments output with declared roles.")
    if not batch_dims:
        raise ValueError(f"{owner}: expected when segments output with a segment batch axis.")
    return tuple(batch_dims[:-1]), str(batch_dims[-1]), str(sequence_dim)


def _stream_out_len(
    valid_sample: xr.DataArray,
    *,
    stream_dim: str,
    sequence_size: int,
    opts: WhenOptions,
    owner: str,
) -> int:
    chunked = is_chunked_dataarray(valid_sample)
    fail_if_chunked_boundary(
        chunked and opts.max_segments is None,
        owner=owner,
        message="chunked stream extraction requires opts.max_segments for bounded output sizing.",
    )
    if opts.max_segments is not None:
        return int(opts.max_segments) * int(sequence_size)
    counts = valid_sample.astype("int64").sum(dim=stream_dim)
    data = np.asarray(counts.data)
    return int(data.max()) if data.size else 0


def _row_indexer(valid_row: np.ndarray, *, out_len: int) -> np.ndarray:
    out = np.full(out_len, SAMPLE_SENTINEL, dtype="int64")
    idx = np.flatnonzero(np.asarray(valid_row, dtype=bool))
    n = min(int(idx.size), int(out_len))
    if n:
        out[:n] = idx[:n]
    return out


def _dynamic_stream_indexer(
    valid_sample: xr.DataArray,
    *,
    batch_dims: tuple[str, ...],
    stream_dim: str,
    out_len: int,
) -> xr.DataArray:
    lane = valid_sample.transpose(*(batch_dims + (stream_dim,)))
    lane_count = 1 if not batch_dims else int(np.prod([lane.sizes[dim] for dim in batch_dims], dtype=np.int64))
    source_size = int(lane.sizes[stream_dim])
    if lane_count == 0:
        lanes = np.empty((0, source_size), dtype=bool)
    elif source_size == 0:
        lanes = np.empty((lane_count, 0), dtype=bool)
    else:
        lanes = np.asarray(lane.data).reshape(lane_count, source_size)
    rows = np.full((lanes.shape[0], out_len), SAMPLE_SENTINEL, dtype="int64")
    for idx in range(lanes.shape[0]):
        rows[idx, :] = _row_indexer(lanes[idx], out_len=out_len)
    shape = (out_len,) if not batch_dims else tuple(valid_sample.sizes[dim] for dim in batch_dims) + (out_len,)
    coords = xr.Dataset(coords=dimension_coordinates(valid_sample, dims=batch_dims))
    coords = coords.assign_coords({stream_dim: np.arange(out_len, dtype="int64")})
    return xr.DataArray(rows.reshape(shape), dims=batch_dims + (stream_dim,), coords=coords.coords, name="stream_indexer")


def _stream_indexer(
    valid_sample: xr.DataArray,
    *,
    batch_dims: tuple[str, ...],
    stream_dim: str,
    sequence_size: int,
    opts: WhenOptions,
    owner: str,
    out_dim: str,
) -> xr.DataArray:
    out_len = _stream_out_len(
        valid_sample,
        stream_dim=stream_dim,
        sequence_size=sequence_size,
        opts=opts,
        owner=owner,
    )
    if opts.max_segments is None:
        return _dynamic_stream_indexer(valid_sample, batch_dims=batch_dims, stream_dim=stream_dim, out_len=out_len)
    return _bounded_stream_indexer(valid_sample, stream_dim=stream_dim, out_len=out_len, out_dim=out_dim)


def _assign_stream_sentinels(
    ds: xr.Dataset,
    *,
    valid_stream: xr.DataArray,
    stream_dim: str,
) -> xr.Dataset:
    updates: dict[str, xr.DataArray] = {}
    for name in _STREAM_INDEX_META:
        if name in ds.coords and stream_dim in ds.coords[name].dims:
            updates[name] = ds.coords[name].where(valid_stream, -1).fillna(-1).astype("int64")
    for name in _STREAM_TIME_META:
        if name in ds.coords and stream_dim in ds.coords[name].dims:
            updates[name] = ds.coords[name].astype("float64").where(valid_stream, np.nan).astype("float64")
    return ds.assign_coords(updates) if updates else ds


def _repacked_stream_dataset(
    stacked: StackedStreamResult,
    *,
    batch_dims: tuple[str, ...],
    sequence_size: int,
    opts: WhenOptions,
    owner: str,
    source_names: tuple[str, ...],
) -> xr.Dataset:
    stream_dim = stacked.stream_dim
    indexer = _stream_indexer(
        stacked.valid_sample,
        batch_dims=batch_dims,
        stream_dim=stream_dim,
        sequence_size=sequence_size,
        opts=opts,
        owner=owner,
        out_dim=unique_temp_dim("__tal_stream_out__", taken_dims=source_names + dataset_namespace_names(stacked.ds)),
    )
    gathered, valid_stream = gather_packed_samples(
        stacked.ds, indexer, sequence_dim=stream_dim, owner=owner,
    )
    masked = mask_aligned_samples(gathered, valid_stream, sequence_dim=stream_dim)
    return _assign_stream_sentinels(masked, valid_stream=valid_stream, stream_dim=stream_dim)


def _has_chunked_stream_source(context: EventEvalContext) -> bool:
    if is_chunked_dataarray(context.clock):
        return True
    for var in context.runtime.ds.data_vars.values():
        if is_chunked_variable(var):
            return True
    return False


def _stacked_when_stream(
    ao: AnalysisObject,
    condition: Condition,
    *,
    context: EventEvalContext,
    opts: WhenOptions,
    validate: bool,
    owner: str,
) -> StackedStreamResult:
    segments = evaluate_when_segments_layout(
        ao,
        condition,
        opts=replace(opts, on_empty="empty"),
        validate=validate,
        owner=owner,
    )
    segments_ds = analysis_object_dataset(segments)
    batch_dims, segment_dim, sequence_dim = _stream_source_dims(segments_ds, owner=owner)
    if batch_dims != context.runtime.batch_dims:
        raise ValueError(f"{owner}: when stream batch dims do not match context batch dims.")
    return stack_segment_stream(
        segments_ds,
        batch_dims=batch_dims,
        segment_dim=segment_dim,
        sequence_dim=sequence_dim,
        owner=owner,
    )


def evaluate_when_stream_layout(
    ao: AnalysisObject,
    condition: Condition,
    *,
    opts: WhenOptions,
    validate: bool = True,
    owner: str = "events.when",
) -> AnalysisObject:
    """Evaluate condition-driven selection in sequence-stream layout.

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
    context = resolve_event_eval_context(ao, opts=opts.eval, owner=owner)
    fail_if_chunked_boundary(
        _has_chunked_stream_source(context) and opts.max_segments is None,
        owner=owner,
        message="chunked stream extraction requires opts.max_segments for bounded output sizing.",
    )
    stacked = _stacked_when_stream(
        ao,
        condition,
        context=context,
        opts=opts,
        validate=validate,
        owner=owner,
    )
    enforce_when_on_empty(
        stacked.valid_sample,
        opts=opts,
        owner=owner,
        layout="stream",
        unit="stream samples",
        chunked_message="opts.on_empty='error' requires unchunked when stream selection.",
    )
    repacked = _repacked_stream_dataset(
        stacked,
        batch_dims=context.runtime.batch_dims,
        sequence_size=int(context.clock.sizes[context.runtime.sequence_dim]),
        opts=opts,
        owner=owner,
        source_names=dataset_namespace_names(context.runtime.ds),
    )
    return finalize_event_output(
        context.ao,
        repacked,
        sequence_dim=stacked.stream_dim,
        batch_dims=context.runtime.batch_dims,
        core_dims=context.runtime.core_dims,
        param_name=context.runtime.spec.name,
        size_name=stacked.size_name,
        validate=validate,
        owner=owner,
    )


__all__ = ["evaluate_when_stream_layout"]
