from __future__ import annotations

"""Owner for event-locked window extraction via ``EventsAccessor.around``."""

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.utils.xarray_namespace import dataarray_namespace_names, dataset_namespace_names, unique_temp_dim

from ..orchestration.lazy import fail_if_chunked_boundary, is_chunked_dataarray
from ..param_ops.types import ParamEvalOptions
from .boundary import EventBoundaryPayload, extract_event_boundaries
from .boundary_select import select_event_boundaries
from .evaluate import evaluate_mask
from .event_primitives import EDGE_INVALID, EDGE_TRIGGER, SAMPLE_SENTINEL
from .finalize import finalize_event_output
from .options import coerce_around_grid
from .pack import pack_event_table
from .resolve import EventEvalContext, resolve_event_eval_context
from .types import AroundOptions, Condition, EventExtractOptions

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject

_AROUND_META = (
    "event_time",
    "event_edge_code",
    "event_sample_index_before",
    "event_sample_index_after",
)


def _table_event_dim(table: xr.Dataset, *, context: EventEvalContext, owner: str) -> str:
    dims = [dim for dim in table["time"].dims if dim not in context.runtime.batch_dims]
    if len(dims) != 1:
        raise ValueError(f"{owner}: expected exactly one event axis, found {dims!r}.")
    return str(dims[0])


def _chunked_condition_source(context: EventEvalContext, *, effective: xr.DataArray) -> bool:
    return any(is_chunked_dataarray(da) for da in (effective, context.valid_mask, context.clock))


def _event_extract_options(opts: AroundOptions) -> EventExtractOptions:
    return EventExtractOptions(
        eval=opts.eval,
        include_initial=False,
        truth_eval="exact",
        max_events=None,
    )


def _condition_anchor_table(
    condition: Condition,
    *,
    context: EventEvalContext,
    opts: AroundOptions,
    owner: str,
) -> xr.Dataset:
    effective = evaluate_mask(condition, context=context, owner=owner)
    fail_if_chunked_boundary(
        _chunked_condition_source(context, effective=effective),
        owner=owner,
        message="chunked condition-source around extraction requires explicit event anchors.",
    )
    payload = extract_event_boundaries(
        condition,
        effective_mask=effective,
        context=context,
        opts=_event_extract_options(opts),
        owner=owner,
        emit_triggers=False,
    )
    selected = select_event_boundaries(
        payload,
        context=context,
        edges=opts.edge,
        mode="all",
        max_events=None,
        on_empty="empty",
        owner=owner,
    )
    return pack_event_table(selected, context=context, owner=owner)


def _source_time_dataarray(
    source: xr.DataArray | xr.Dataset,
    *,
    owner: str,
) -> xr.DataArray:
    if isinstance(source, xr.Dataset):
        if "time" not in source.data_vars:
            raise ValueError(f"{owner}: explicit dataset source must contain data variable 'time'.")
        data = source["time"]
    else:
        data = source
    try:
        out = data.astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: explicit event times must be numeric (coercible to float64).") from exc
    return out


def _event_dim_name(
    times: xr.DataArray,
    *,
    context: EventEvalContext,
) -> str:
    names = set(dataset_namespace_names(context.runtime.ds))
    names.update(dataarray_namespace_names(times))
    names.update(_AROUND_META)
    return unique_temp_dim("event", taken_dims=tuple(sorted(names)))


def _validate_explicit_batch_labels(
    source: xr.DataArray,
    *,
    context: EventEvalContext,
    owner: str,
) -> None:
    for dim in context.runtime.batch_dims:
        if dim not in source.dims:
            continue
        if source.get_index(dim).equals(context.clock.get_index(dim)):
            continue
        raise ValueError(f"{owner}: explicit event times labels for batch dim {dim!r} must match context labels.")


def _event_axis_name(source: xr.DataArray, *, context: EventEvalContext, owner: str) -> str:
    non_batch = [dim for dim in source.dims if dim not in context.runtime.batch_dims]
    if len(non_batch) > 1:
        if not context.runtime.batch_dims:
            raise ValueError(f"{owner}: explicit event time source must be 0-D or 1-D.")
        raise ValueError(
            f"{owner}: explicit event time source dims must be subset of batch dims plus one optional event axis."
        )
    if non_batch:
        return str(non_batch[0])
    return _event_dim_name(source, context=context)


def _with_missing_batch_dims(
    source: xr.DataArray,
    *,
    context: EventEvalContext,
) -> xr.DataArray:
    out = source
    for dim in context.runtime.batch_dims:
        if dim in out.dims:
            continue
        out = out.expand_dims({dim: context.clock.coords[dim]})
    return out


def _align_explicit_times(
    times: xr.DataArray,
    *,
    context: EventEvalContext,
    owner: str,
) -> tuple[xr.DataArray, str]:
    source = _source_time_dataarray(times, owner=owner)
    if context.runtime.sequence_dim in source.dims:
        raise ValueError(
            f"{owner}: explicit event times must not use sequence dim {context.runtime.sequence_dim!r}."
        )
    _validate_explicit_batch_labels(source, context=context, owner=owner)
    event_dim = _event_axis_name(source, context=context, owner=owner)
    if event_dim not in source.dims:
        source = source.expand_dims({event_dim: [0]})
    source = _with_missing_batch_dims(source, context=context)
    return source.transpose(*(context.runtime.batch_dims + (event_dim,))), event_dim


def _explicit_anchor_payload(
    source: xr.DataArray | xr.Dataset,
    *,
    context: EventEvalContext,
    owner: str,
) -> EventBoundaryPayload:
    if not isinstance(source, (xr.DataArray, xr.Dataset)):
        raise TypeError(f"{owner}: events_or_condition must be Condition, xr.DataArray, or xr.Dataset.")
    if isinstance(source, xr.Dataset) and "time" not in source.data_vars:
        raise ValueError(f"{owner}: explicit dataset source must contain data variable 'time'.")
    base = source if isinstance(source, xr.DataArray) else source["time"]
    times, event_dim = _align_explicit_times(base, context=context, owner=owner)
    finite = xr.apply_ufunc(np.isfinite, times.astype("float64"), dask="allowed").astype(bool)
    edge = xr.where(finite, int(EDGE_TRIGGER), int(EDGE_INVALID)).astype("int8")
    before = xr.ones_like(times, dtype="int64") * int(SAMPLE_SENTINEL)
    after = xr.ones_like(times, dtype="int64") * int(SAMPLE_SENTINEL)
    return EventBoundaryPayload(
        time=times.rename("time"),
        edge_code=edge.rename("edge_code"),
        sample_index_before=before.rename("sample_index_before"),
        sample_index_after=after.rename("sample_index_after"),
        event_dim=event_dim,
    )


def _explicit_anchor_table(
    source: xr.DataArray | xr.Dataset,
    *,
    context: EventEvalContext,
    owner: str,
) -> xr.Dataset:
    payload = _explicit_anchor_payload(source, context=context, owner=owner)
    return pack_event_table(payload, context=context, owner=owner)


def _tau_from_grid(grid: xr.DataArray | np.ndarray, *, tau_dim: str, owner: str) -> xr.DataArray:
    tau = coerce_around_grid(grid, owner=owner)
    return tau.rename({tau.dims[0]: tau_dim}) if tau.dims[0] != tau_dim else tau


def _tau_from_dt(*, pre: float, post: float, dt: float, tau_dim: str, owner: str) -> xr.DataArray:
    tol = max(abs(dt) * 1e-12, 1e-12)
    values = np.arange(-pre, post + tol, dt, dtype="float64")
    if values.size == 0:
        values = np.asarray([0.0], dtype="float64")
    if not np.isclose(values, 0.0, atol=tol, rtol=0.0).any():
        values = np.sort(np.append(values, 0.0)).astype("float64")
    return xr.DataArray(values, dims=(tau_dim,), coords={tau_dim: values}, name=tau_dim)


def _tau_grid(
    opts: AroundOptions,
    *,
    context: EventEvalContext,
    table: xr.Dataset,
    owner: str,
) -> tuple[xr.DataArray, str]:
    names = set(dataset_namespace_names(context.runtime.ds))
    names.update(dataset_namespace_names(table))
    names.update(_AROUND_META)
    tau_dim = unique_temp_dim("tau", taken_dims=tuple(sorted(names)))
    if opts.grid is not None:
        tau = _tau_from_grid(opts.grid, tau_dim=tau_dim, owner=owner)
    else:
        if opts.dt is None:
            raise ValueError(f"{owner}: opts.dt is required when opts.grid is None.")
        tau = _tau_from_dt(pre=float(opts.pre), post=float(opts.post), dt=float(opts.dt), tau_dim=tau_dim, owner=owner)
    return tau, tau_dim


def _query_dim_name(
    *,
    context: EventEvalContext,
    table: xr.Dataset,
    tau_dim: str,
) -> str:
    names = set(dataset_namespace_names(context.runtime.ds))
    names.update(dataset_namespace_names(table))
    names.add(tau_dim)
    names.update(_AROUND_META)
    return unique_temp_dim("around_query", taken_dims=tuple(sorted(names)))


def _query_times(
    event_time: xr.DataArray,
    tau: xr.DataArray,
    *,
    context: EventEvalContext,
    event_dim: str,
    tau_dim: str,
) -> xr.DataArray:
    query = event_time.astype("float64") + tau.astype("float64")
    dims = context.runtime.batch_dims + (event_dim, tau_dim)
    return query.transpose(*dims).rename(context.runtime.spec.name)


def _param_eval_options(*, query_dim: str, opts: AroundOptions) -> ParamEvalOptions:
    return ParamEvalOptions(method=opts.eval.ao_interp, duplicate_policy="invalid", query_dim=query_dim)


def _evaluate_window_dataset(
    *,
    context: EventEvalContext,
    table: xr.Dataset,
    event_dim: str,
    tau: xr.DataArray,
    tau_dim: str,
    opts: AroundOptions,
    validate: bool,
) -> tuple[xr.Dataset, xr.DataArray]:
    valid_event = (table["edge_code"] != int(EDGE_INVALID)).astype(bool)
    if table.sizes.get(event_dim, 0) == 0 or any(context.clock.sizes.get(dim, 0) == 0 for dim in context.runtime.batch_dims):
        query_dim = _query_dim_name(context=context, table=table, tau_dim=tau_dim)
        out = context.ao.param.at(
            tau.astype("float64").rename(context.runtime.spec.name),
            on=opts.eval.coord_name,
            opts=_param_eval_options(query_dim=query_dim, opts=opts),
            validate=validate,
            sequence_dim=context.runtime.sequence_dim,
            batch_dims=context.runtime.batch_dims,
            sequence_size_coord=context.runtime.sequence_size_coord,
        )
        if context.runtime.sequence_dim in out.unsafe_data.dims and context.runtime.sequence_dim != tau_dim:
            out = out.rename({context.runtime.sequence_dim: tau_dim}, validate=False)
        out = out.set_param_coord(name=tau_dim, validate=False)
        empty = out.unsafe_data.expand_dims(
            {event_dim: table.coords[event_dim].values},
            axis=len(context.runtime.batch_dims),
        )
        return empty, valid_event
    query = _query_times(table["time"], tau, context=context, event_dim=event_dim, tau_dim=tau_dim)
    query_dim = _query_dim_name(context=context, table=table, tau_dim=tau_dim)
    out = context.ao.param.at(
        query,
        on=opts.eval.coord_name,
        opts=_param_eval_options(query_dim=query_dim, opts=opts),
        validate=validate,
        sequence_dim=context.runtime.sequence_dim,
        batch_dims=context.runtime.batch_dims,
        sequence_size_coord=context.runtime.sequence_size_coord,
    )
    masked = _mask_invalid_event_rows(out.unsafe_data, valid_event=valid_event, sequence_dim=tau_dim)
    return masked, valid_event


def _mask_invalid_event_rows(
    ds: xr.Dataset,
    *,
    valid_event: xr.DataArray,
    sequence_dim: str,
) -> xr.Dataset:
    var_updates = {
        name: var.where(valid_event)
        for name, var in ds.data_vars.items()
        if sequence_dim in var.dims
    }
    coord_updates = {
        name: coord.where(valid_event)
        for name, coord in ds.coords.items()
        if name != sequence_dim and sequence_dim in coord.dims
    }
    out = ds.assign(var_updates) if var_updates else ds
    return out.assign_coords(coord_updates) if coord_updates else out


def _metadata_namespace_safe(ds: xr.Dataset, *, owner: str) -> None:
    names = set(dataset_namespace_names(ds))
    conflicts = sorted(name for name in _AROUND_META if name in names)
    if conflicts:
        raise ValueError(f"{owner}: around metadata names conflict with dataset namespace: {conflicts!r}.")


def _size_coord_name(ds: xr.Dataset) -> str:
    names = set(dataset_namespace_names(ds))
    names.update(_AROUND_META)
    return unique_temp_dim("tau_len", taken_dims=tuple(sorted(names)))


def _attach_metadata_and_size(
    ds: xr.Dataset,
    *,
    table: xr.Dataset,
    valid_event: xr.DataArray,
    tau: xr.DataArray,
    size_name: str,
    owner: str,
    event_dim: str,
) -> xr.Dataset:
    _metadata_namespace_safe(ds, owner=owner)
    size = xr.where(valid_event, int(tau.size), 0).astype("int64")
    return ds.assign_coords(
        {
            tau.dims[0]: tau,
            "event_time": table["time"].astype("float64"),
            "event_edge_code": table["edge_code"].astype("int8"),
            "event_sample_index_before": table["sample_index_before"].astype("int64"),
            "event_sample_index_after": table["sample_index_after"].astype("int64"),
            event_dim: table.coords[event_dim],
            size_name: size,
        }
    )


def _with_source_schema_attrs(ds: xr.Dataset, *, source: "AnalysisObject") -> xr.Dataset:
    out = ds.copy(deep=False)
    out.attrs = dict(source.unsafe_data.attrs)
    return out


def _anchor_table(
    events_or_condition: Condition | xr.DataArray | xr.Dataset,
    *,
    context: EventEvalContext,
    opts: AroundOptions,
    owner: str,
) -> xr.Dataset:
    if isinstance(events_or_condition, Condition):
        return _condition_anchor_table(events_or_condition, context=context, opts=opts, owner=owner)
    if isinstance(events_or_condition, (xr.DataArray, xr.Dataset)):
        return _explicit_anchor_table(events_or_condition, context=context, owner=owner)
    raise TypeError(f"{owner}: events_or_condition must be Condition, xr.DataArray, or xr.Dataset.")


def evaluate_around_segments_windows(
    ao: "AnalysisObject",
    events_or_condition: Condition | xr.DataArray | xr.Dataset,
    *,
    opts: AroundOptions,
    validate: bool = True,
    owner: str = "events.around",
) -> "AnalysisObject":
    """Evaluate event-locked windows using event-major segments layout.

    Parameters
    ----------
    ao : AnalysisObject
        AnalysisObject-like input value.
    events_or_condition : Condition | xr.DataArray | xr.Dataset
        Condition/expression used for event or mask evaluation.
    opts : AroundOptions, optional
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
    table = _anchor_table(events_or_condition, context=context, opts=opts, owner=owner)
    event_dim = _table_event_dim(table, context=context, owner=owner)
    tau, tau_dim = _tau_grid(opts, context=context, table=table, owner=owner)
    masked, valid_event = _evaluate_window_dataset(
        context=context,
        table=table,
        event_dim=event_dim,
        tau=tau,
        tau_dim=tau_dim,
        opts=opts,
        validate=validate,
    )
    size_name = _size_coord_name(masked)
    enriched = _attach_metadata_and_size(
        masked,
        table=table,
        valid_event=valid_event,
        tau=tau,
        size_name=size_name,
        owner=owner,
        event_dim=event_dim,
    )
    return finalize_event_output(
        context.ao,
        _with_source_schema_attrs(enriched, source=context.ao),
        sequence_dim=tau_dim,
        batch_dims=context.runtime.batch_dims + (event_dim,),
        core_dims=context.runtime.core_dims,
        param_name=tau_dim,
        size_name=size_name,
        validate=validate,
        owner=owner,
    )


def evaluate_around_windows(
    ao: "AnalysisObject",
    events_or_condition: Condition | xr.DataArray | xr.Dataset,
    *,
    opts: AroundOptions,
    validate: bool = True,
    owner: str = "events.around",
) -> "AnalysisObject":
    """Evaluate around windows using the selected around layout policy.

    Parameters
    ----------
    ao : AnalysisObject
        AnalysisObject-like input value.
    events_or_condition : Condition | xr.DataArray | xr.Dataset
        Condition/expression used for event or mask evaluation.
    opts : AroundOptions, optional
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
    if opts.layout == "segments":
        return evaluate_around_segments_windows(
            ao,
            events_or_condition,
            opts=opts,
            validate=validate,
            owner=owner,
        )
    if opts.layout == "stacked":
        from .around_stacked import evaluate_around_stacked_windows

        return evaluate_around_stacked_windows(
            ao,
            events_or_condition,
            opts=opts,
            validate=validate,
            owner=owner,
        )
    raise NotImplementedError(f"{owner}: opts.layout={opts.layout!r} is not supported.")


__all__ = ["evaluate_around_segments_windows", "evaluate_around_windows"]
