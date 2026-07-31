from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from tal.core.group_ops.foundation import resolve_grouping_foundation_context
from tal.core.orchestration.runtime_checks import select_single_numeric_var
from tal.core.reducer_ops.validity import apply_structural_mask, resolve_structural_valid_mask
from tal.utils.xarray_namespace import dataarray_namespace_names, dataset_namespace_names, unique_temp_dim

from .options import VizKind
from .plan import VizRuntimeContext

_NUMERIC_KINDS = {"i", "u", "f", "c"}


@dataclass(frozen=True)
class VizPreparedPayload:
    data: xr.DataArray | xr.Dataset
    kind: VizKind
    x: str | None
    by: tuple[str, ...]
    groupby: tuple[str, ...]
    kwargs: dict[str, object]


def _numeric_data_vars(ds: xr.Dataset) -> tuple[str, ...]:
    return tuple(str(name) for name, var in ds.data_vars.items() if var.dtype.kind in _NUMERIC_KINDS)


def _require_numeric_var(ds: xr.Dataset, *, var_name: str, owner: str) -> xr.DataArray:
    if var_name not in ds.data_vars:
        raise ValueError(f"{owner}: opts.var {var_name!r} was not found in dataset data_vars.")
    data = ds[var_name]
    if data.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"{owner}: opts.var {var_name!r} must be numeric.")
    return data


def _resolve_explicit_or_single(runtime: VizRuntimeContext, *, owner: str) -> xr.DataArray:
    if runtime.opts.var is not None:
        return _require_numeric_var(runtime.context.ds, var_name=runtime.opts.var, owner=owner)
    var_name = select_single_numeric_var(runtime.context.ds, owner=owner, what="AO viz line/scatter")
    return runtime.context.ds[var_name]


def _resolve_explorer_source(runtime: VizRuntimeContext, *, owner: str) -> xr.DataArray | xr.Dataset:
    if runtime.opts.var is None:
        return runtime.context.ds
    return _require_numeric_var(runtime.context.ds, var_name=runtime.opts.var, owner=owner)


def _resolve_plot_dataarray(runtime: VizRuntimeContext, *, owner: str) -> xr.DataArray:
    if runtime.kind in {"line", "scatter"}:
        data = _resolve_explicit_or_single(runtime, owner=owner)
    else:
        resolved = _resolve_explorer_source(runtime, owner=owner)
        if not isinstance(resolved, xr.DataArray):
            raise ValueError(f"{owner}: line/scatter requires a single numeric data variable; set opts.var.")
        data = resolved
    sequence_dim = runtime.context.sequence_dim
    if sequence_dim is None or sequence_dim not in data.dims:
        raise ValueError(f"{owner}: selected plot variable must include declared sequence_dim.")
    return data


def _resolve_explorer_data(runtime: VizRuntimeContext, *, owner: str) -> xr.DataArray | xr.Dataset:
    data = _resolve_explorer_source(runtime, owner=owner)
    if isinstance(data, xr.DataArray):
        sequence_dim = runtime.context.sequence_dim
        if sequence_dim is None or sequence_dim not in data.dims:
            raise ValueError(f"{owner}: selected explorer variable must include declared sequence_dim.")
        return data
    if not data.data_vars:
        raise ValueError(f"{owner}: explorer requires at least one dataset variable.")
    return data


def _coord_or_dim_present(data: xr.DataArray | xr.Dataset, name: str) -> bool:
    return name in data.coords or name in data.dims


def _resolve_explicit_x(data: xr.DataArray | xr.Dataset, *, x: str | None, owner: str) -> str | None:
    if x is None:
        return None
    if _coord_or_dim_present(data, x):
        return x
    raise ValueError(f"{owner}: opts.x {x!r} is not present as a coord or dim on the selected variable.")


def _attach_sequence_index_coord(
    data: xr.DataArray | xr.Dataset,
    *,
    sequence_dim: str,
) -> tuple[xr.DataArray | xr.Dataset, str]:
    index_name = unique_temp_dim(
        f"{sequence_dim}_index",
        taken_dims=(
            dataarray_namespace_names(data)
            if isinstance(data, xr.DataArray)
            else dataset_namespace_names(data)
        ),
    )
    values = np.arange(int(data.sizes[sequence_dim]), dtype=np.int64)
    coord = xr.DataArray(values, dims=(sequence_dim,))
    return data.assign_coords({index_name: coord}), index_name


def _resolve_x(
    data: xr.DataArray | xr.Dataset,
    *,
    runtime: VizRuntimeContext,
    owner: str,
) -> tuple[xr.DataArray | xr.Dataset, str]:
    explicit = _resolve_explicit_x(data, x=runtime.opts.x, owner=owner)
    if explicit is not None:
        return data, explicit
    if runtime.context.param_coord and _coord_or_dim_present(data, runtime.context.param_coord):
        return data, runtime.context.param_coord
    sequence_dim = runtime.context.sequence_dim
    assert sequence_dim is not None
    if sequence_dim in data.coords:
        return data, sequence_dim
    return _attach_sequence_index_coord(data, sequence_dim=sequence_dim)


def _mask_source_var_for_dataset(ds: xr.Dataset, *, runtime: VizRuntimeContext) -> xr.DataArray | None:
    numeric = _numeric_data_vars(ds)
    if not numeric:
        return None
    sequence_dim = runtime.context.sequence_dim
    if sequence_dim is None:
        return None
    for name in numeric:
        var = ds[name]
        if sequence_dim in var.dims:
            return var
    return None


def _apply_validity(
    data: xr.DataArray | xr.Dataset,
    *,
    runtime: VizRuntimeContext,
    owner: str,
) -> xr.DataArray | xr.Dataset:
    if runtime.opts.validity == "ignore":
        return data
    if isinstance(data, xr.Dataset):
        source_var = _mask_source_var_for_dataset(data, runtime=runtime)
        if source_var is None:
            return data
        mask = resolve_structural_valid_mask(
            runtime.context.ds,
            sequence_dim=runtime.context.sequence_dim,
            sequence_size_coord=runtime.context.sequence_size_coord,
            var=source_var,
            owner=owner,
        )
        if mask is None:
            return data
        return data.where(mask)
    mask = resolve_structural_valid_mask(
        runtime.context.ds,
        sequence_dim=runtime.context.sequence_dim,
        sequence_size_coord=runtime.context.sequence_size_coord,
        var=data,
        owner=owner,
    )
    return apply_structural_mask(data, mask=mask)


def _validate_channel_names(
    data: xr.DataArray | xr.Dataset,
    *,
    names: tuple[str, ...],
    owner: str,
    field: str,
) -> tuple[str, ...]:
    for name in names:
        if name in data.dims or name in data.coords:
            continue
        raise ValueError(f"{owner}: opts.{field} channel {name!r} is not present as a dim/coord.")
    return names


def _channel_upper_bound(data: xr.DataArray | xr.Dataset, *, name: str) -> int:
    if name in data.dims:
        return int(data.sizes[name])
    coord = data.coords[name]
    if not coord.dims:
        return 1
    total = 1
    for dim in coord.dims:
        total *= int(data.sizes[dim])
    return total


def _overlay_upper_bound(data: xr.DataArray | xr.Dataset, *, by: tuple[str, ...]) -> int:
    total = 1
    for name in by:
        total *= _channel_upper_bound(data, name=name)
    return total


def _enforce_overlay_cap(data: xr.DataArray | xr.Dataset, *, by: tuple[str, ...], max_items: int, owner: str) -> None:
    if not by:
        return
    upper = _overlay_upper_bound(data, by=by)
    if upper <= max_items:
        return
    raise ValueError(
        f"{owner}: opts.by overlay upper bound {upper} exceeds opts.max_overlay_items={max_items}; "
        "use opts.groupby for high-cardinality channels."
    )


def _attach_group_key_coord(
    data: xr.DataArray | xr.Dataset,
    *,
    runtime: VizRuntimeContext,
    owner: str,
) -> tuple[xr.DataArray | xr.Dataset, str]:
    foundation = resolve_grouping_foundation_context(
        runtime.source,
        runtime.opts.group_key,
        opts=runtime.opts.group_foundation_opts,
        owner=f"{owner}.group_key",
    )
    if len(foundation.keys) != 1:
        raise ValueError(f"{owner}: opts.group_key must resolve to exactly one key in Slice A/B.")
    key = foundation.keys[0].data
    missing = tuple(dim for dim in key.dims if dim not in data.dims)
    if missing:
        raise ValueError(f"{owner}: resolved group key dims {missing!r} are not present in selected data dims.")
    name = unique_temp_dim(
        "__tal_group_key__",
        taken_dims=(
            dataarray_namespace_names(data)
            if isinstance(data, xr.DataArray)
            else dataset_namespace_names(data)
        ),
    )
    return data.assign_coords({name: key}), name


def _default_group_key_channels(
    data: xr.DataArray | xr.Dataset,
    *,
    by: tuple[str, ...],
    groupby: tuple[str, ...],
    group_key_name: str,
    max_items: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if by or groupby:
        return by, groupby
    if _channel_upper_bound(data, name=group_key_name) <= max_items:
        return (group_key_name,), ()
    return (), (group_key_name,)


def _resolve_explorer_x(
    data: xr.DataArray | xr.Dataset,
    *,
    runtime: VizRuntimeContext,
    owner: str,
) -> tuple[xr.DataArray | xr.Dataset, str | None]:
    explicit = _resolve_explicit_x(data, x=runtime.opts.x, owner=owner)
    if explicit is not None:
        return data, explicit
    return data, None


def _prepare_channels_and_group_key(
    data: xr.DataArray | xr.Dataset,
    *,
    runtime: VizRuntimeContext,
    owner: str,
) -> tuple[xr.DataArray | xr.Dataset, tuple[str, ...], tuple[str, ...]]:
    group_key_name: str | None = None
    if runtime.opts.group_key is not None:
        data, group_key_name = _attach_group_key_coord(data, runtime=runtime, owner=owner)
    by = _validate_channel_names(data, names=runtime.opts.by, owner=owner, field="by")
    groupby = _validate_channel_names(data, names=runtime.opts.groupby, owner=owner, field="groupby")
    if runtime.opts.by and runtime.kind in {"line", "scatter"}:
        _enforce_overlay_cap(data, by=by, max_items=runtime.opts.max_overlay_items, owner=owner)
    if group_key_name is not None:
        by, groupby = _default_group_key_channels(
            data,
            by=by,
            groupby=groupby,
            group_key_name=group_key_name,
            max_items=runtime.opts.max_overlay_items,
        )
    return data, by, groupby


def prepare_viz_payload(runtime: VizRuntimeContext, *, owner: str) -> VizPreparedPayload:
    data: xr.DataArray | xr.Dataset
    if runtime.kind == "explorer":
        data = _resolve_explorer_data(runtime, owner=owner)
        data, x_name = _resolve_explorer_x(data, runtime=runtime, owner=owner)
    else:
        data = _resolve_plot_dataarray(runtime, owner=owner)
        data, x_name = _resolve_x(data, runtime=runtime, owner=owner)
    data = _apply_validity(data, runtime=runtime, owner=owner)
    data, by, groupby = _prepare_channels_and_group_key(
        data,
        runtime=runtime,
        owner=owner,
    )
    return VizPreparedPayload(
        data=data,
        kind=runtime.kind,
        x=x_name,
        by=by,
        groupby=groupby,
        kwargs=dict(runtime.opts.kwargs),
    )


__all__ = ["VizPreparedPayload", "prepare_viz_payload"]
