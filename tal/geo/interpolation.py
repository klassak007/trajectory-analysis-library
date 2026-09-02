from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from tal.core import AnalysisObject
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.context import DatasetContextOptions, resolve_dataset_context
from tal.core.orchestration.inputs import query_coord_from_other_input
from tal.core.orchestration.resolve import resolve_param_runtime_context
from tal.core.orchestration.runtime_checks import select_single_numeric_var
from tal.core.param_engine import ParamMapOptions, build_param_map, normalize_query_grid
from tal.core.param_engine.map_apply import gather_along_sequence
from tal.core.param_ops import ParamEvalOptions
from tal.core.param_ops.finalize import finalize_param_output
from tal.core.param_ops.guards import assert_query_dim_safe, assert_reserved_metadata_safe
from tal.core.param_ops.types import ParamRuntimeContext
from tal.spatial import Position

from .backends import geod_interpolate
from .kernels import require_unambiguous_geodesic_interpolation
from .metadata import normalize_geodetic_metadata, options_from_geodetic_metadata
from .options import GeodeticInterpolationOptions

_LLA_LABELS = ("lat", "lon", "alt")
_CTX_OPTIONS = DatasetContextOptions(
    require_roles=True,
    select_numeric_var=True,
    require_single_numeric_var=True,
    allowed_core_arity=(1,),
    require_semantic_dims_in_var=True,
)


def _wrap_error(exc: TypeError | ValueError, *, owner: str) -> TypeError | ValueError:
    text = str(exc)
    if text.startswith(f"{owner}:"):
        return exc
    return type(exc)(f"{owner}: {text}")


def _component(data: xr.DataArray, *, dim: str, label: str) -> xr.DataArray:
    return data.sel({dim: label}, drop=True)


def _order_query_then_core(values: xr.DataArray, *, query_dim: str, core_dim: str) -> xr.DataArray:
    outer = [dim for dim in values.dims if dim not in {query_dim, core_dim}]
    return values.transpose(*(outer + [query_dim, core_dim]))


def _broadcast_query_da(values: xr.DataArray, *, template: xr.DataArray, core_dim: str) -> xr.DataArray:
    target = template.isel({core_dim: 0}, drop=True)
    return values.broadcast_like(target)


def _param_eval_options(opts: GeodeticInterpolationOptions, *, method: str) -> ParamEvalOptions:
    return ParamEvalOptions(
        method=method,  # type: ignore[arg-type]
        duplicate_policy=opts.duplicate_policy,
        query_dim=opts.query_dim,
    )


def _generic_param_eval(
    source: AnalysisObject,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    on: str | None,
    opts: GeodeticInterpolationOptions,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    mode: str,
    method: str,
) -> AnalysisObject:
    eval_opts = _param_eval_options(opts, method=method)
    kwargs = {
        "on": on,
        "opts": eval_opts,
        "validate": validate,
        "sequence_dim": sequence_dim,
        "batch_dims": batch_dims,
        "sequence_size_coord": sequence_size_coord,
    }
    if mode == "at":
        return source.param.at(query, **kwargs)
    return source.param.resample_to(query, **kwargs)


def _resolve_runtime(
    source,
    *,
    query_dim: str,
    on: str | None,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> ParamRuntimeContext:
    runtime_source = source if on is None else source.set_param_coord(name=on, validate=False)
    context = resolve_param_runtime_context(
        runtime_source,
        on=on,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
    )
    assert_query_dim_safe(context.ds, sequence_dim=context.sequence_dim, query_dim=query_dim, owner=owner)
    assert_reserved_metadata_safe(
        context.ds,
        reserved=("valid", "sample_index"),
        param_name=context.spec.name,
        owner=owner,
    )
    return context


def _build_map(
    context: ParamRuntimeContext,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    opts: GeodeticInterpolationOptions,
) -> tuple[xr.DataArray, object, tuple[str, ...] | None]:
    grid = normalize_query_grid(
        query,
        query_dim=opts.query_dim,
        batch_dims=context.batch_dims,
        batch_coords=context.batch_coords,
    )
    pmap = build_param_map(
        param=context.spec.coord,
        query=grid.values,
        sequence_dim=context.sequence_dim,
        query_dim=grid.query_dim,
        valid_mask=context.valid_mask,
        options=ParamMapOptions(method="linear", duplicate_policy=opts.duplicate_policy),
    )
    return grid.values, pmap, grid.stacked_dims


def _build_base_output_dataset(context: ParamRuntimeContext, *, var_name: str, values: xr.DataArray) -> xr.Dataset:
    coords = {
        name: coord
        for name, coord in context.ds.coords.items()
        if context.sequence_dim not in coord.dims
    }
    return xr.Dataset(data_vars={var_name: values}, coords=coords)


def _wrap_lon(lon: xr.DataArray, *, mode: str) -> xr.DataArray:
    if mode == "[0, 360)":
        return (lon % 360.0 + 360.0) % 360.0
    if mode == "[-180, 180)":
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def _assemble_lla(
    components: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    *,
    core_dim: str,
    target_dims: tuple[str, ...],
    var_name: str,
) -> xr.DataArray:
    dim = xr.IndexVariable(core_dim, list(_LLA_LABELS))
    arr = xr.concat(list(components), dim=dim).transpose(*target_dims)
    return arr.rename(var_name)


def _wrap_and_restamp(source, candidate, *, opts: GeodeticInterpolationOptions, validate: bool, owner: str):
    geo_opts = options_from_geodetic_metadata(analysis_object_dataset(source), owner=owner)
    wrap_mode = geo_opts.longitude_wrap if opts.longitude_wrap == "shortest" else opts.longitude_wrap
    candidate_ds = analysis_object_dataset(candidate)
    if opts.longitude_wrap != "preserve":
        ctx = resolve_dataset_context(candidate, owner=owner, options=_CTX_OPTIONS)
        assert ctx.data is not None and ctx.var_name is not None
        core_dim = ctx.core_dims[0]
        lon = _wrap_lon(_component(ctx.data, dim=core_dim, label="lon"), mode=wrap_mode)
        arr = _assemble_lla(
            (
                _component(ctx.data, dim=core_dim, label="lat"),
                lon,
                _component(ctx.data, dim=core_dim, label="alt"),
            ),
            core_dim=core_dim,
            target_dims=ctx.data.dims,
            var_name=ctx.var_name,
        )
        ds = candidate_ds.assign({ctx.var_name: arr})
    else:
        ds = candidate_ds
        wrap_mode = geo_opts.longitude_wrap
    from dataclasses import replace

    ds = normalize_geodetic_metadata(
        ds,
        opts=replace(geo_opts, longitude_wrap=wrap_mode),
        validate=False,
        validate_crs=geo_opts.crs != "EPSG:4979",
        owner=owner,
    )
    from .geodetic import GeodeticPosition

    if validate:
        return GeodeticPosition._from_validated(ds)
    return GeodeticPosition._from_unvalidated(ds)


def _finalize_geodesic_output(
    source,
    context: ParamRuntimeContext,
    *,
    var_name: str,
    values: xr.DataArray,
    query: xr.DataArray,
    valid_query: xr.DataArray,
    query_dim: str,
    stacked_dims: tuple[str, ...] | None,
    opts: GeodeticInterpolationOptions,
    validate: bool,
    owner: str,
):
    ds_out = _build_base_output_dataset(context, var_name=var_name, values=values)
    evaluated = finalize_param_output(
        context,
        ds_out,
        query=query,
        query_dim=query_dim,
        valid_query=valid_query,
        validate=False,
        trajectory=(stacked_dims is None),
    )
    from .geodetic import GeodeticPosition

    candidate = GeodeticPosition._from_unvalidated(analysis_object_dataset(evaluated))
    return _wrap_and_restamp(source, candidate, opts=opts, validate=validate, owner=owner)


def _geodesic_output_values(
    data: xr.DataArray,
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    core_dim: str,
    pmap,
    crs: str,
    owner: str,
) -> xr.DataArray:
    alpha = _broadcast_query_da(pmap.alpha, template=left, core_dim=core_dim)
    safe_alpha = xr.apply_ufunc(
        require_unambiguous_geodesic_interpolation,
        _component(left, dim=core_dim, label="lat"),
        _component(left, dim=core_dim, label="lon"),
        _component(right, dim=core_dim, label="lat"),
        _component(right, dim=core_dim, label="lon"),
        alpha,
        input_core_dims=((), (), (), (), ()),
        output_core_dims=((),),
        output_dtypes=(np.float64,),
        dask="parallelized",
    )
    lat, lon = xr.apply_ufunc(
        geod_interpolate,
        _component(left, dim=core_dim, label="lat"),
        _component(left, dim=core_dim, label="lon"),
        _component(right, dim=core_dim, label="lat"),
        _component(right, dim=core_dim, label="lon"),
        safe_alpha,
        input_core_dims=((), (), (), (), ()),
        output_core_dims=((), ()),
        output_dtypes=(np.float64, np.float64),
        kwargs={"crs": crs, "owner": owner},
        dask="parallelized",
    )
    alt = (1.0 - safe_alpha) * _component(left, dim=core_dim, label="alt") + safe_alpha * _component(
        right, dim=core_dim, label="alt"
    )
    out = _assemble_lla(
        (lat.where(pmap.valid), lon.where(pmap.valid), alt.where(pmap.valid)),
        core_dim=core_dim,
        target_dims=tuple(left.dims),
        var_name=str(data.name),
    )
    out = out.assign_coords({core_dim: data.coords[core_dim]})
    return _order_query_then_core(out, query_dim=pmap.query_dim, core_dim=core_dim)


def _geodesic_linear(
    source,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    on: str | None,
    opts: GeodeticInterpolationOptions,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
):
    context = _resolve_runtime(
        source,
        query_dim=opts.query_dim,
        on=on,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
    )
    var_name = select_single_numeric_var(context.ds, owner=owner, what="GeodeticPosition")
    core_dim = context.core_dims[0]
    query_values, pmap, stacked_dims = _build_map(context, query, opts=opts)
    data = context.ds[var_name]
    left = gather_along_sequence(data, pmap.i0, sequence_dim=context.sequence_dim, query_dim=pmap.query_dim, owner=owner)
    right = gather_along_sequence(data, pmap.i1, sequence_dim=context.sequence_dim, query_dim=pmap.query_dim, owner=owner)
    left = _order_query_then_core(left, query_dim=pmap.query_dim, core_dim=core_dim)
    right = _order_query_then_core(right, query_dim=pmap.query_dim, core_dim=core_dim)
    geo_opts = options_from_geodetic_metadata(context.ds, owner=owner)
    out = _geodesic_output_values(data, left, right, core_dim=core_dim, pmap=pmap, crs=geo_opts.crs, owner=owner)
    return _finalize_geodesic_output(
        source,
        context,
        var_name=var_name,
        values=out,
        query=query_values,
        valid_query=pmap.valid,
        query_dim=pmap.query_dim,
        stacked_dims=stacked_dims,
        opts=opts,
        validate=validate,
        owner=owner,
    )


def _nearest(
    source,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    on: str | None,
    opts: GeodeticInterpolationOptions,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    mode: str,
    owner: str,
):
    carrier = AnalysisObject._from_unvalidated(analysis_object_dataset(source))
    evaluated = _generic_param_eval(
        carrier,
        query,
        on=on,
        opts=opts,
        validate=False,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        mode=mode,
        method="nearest",
    )
    from .geodetic import GeodeticPosition

    candidate = GeodeticPosition._from_unvalidated(analysis_object_dataset(evaluated))
    return _wrap_and_restamp(source, candidate, opts=opts, validate=validate, owner=owner)


def _ecef_linear(
    source,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    on: str | None,
    opts: GeodeticInterpolationOptions,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    mode: str,
    owner: str,
):
    ecef = source.to_ecef(validate=False)
    evaluated = _generic_param_eval(
        AnalysisObject._from_unvalidated(analysis_object_dataset(ecef)),
        query,
        on=on,
        opts=opts,
        validate=False,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        mode=mode,
        method="linear",
    )
    lla = Position(analysis_object_dataset(evaluated)).geo.to_lla(validate=False)
    return _wrap_and_restamp(source, lla, opts=opts, validate=validate, owner=owner)


def _local_enu_linear(
    source,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    on: str | None,
    opts: GeodeticInterpolationOptions,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    mode: str,
    owner: str,
):
    from .options import ENUOptions

    enu = source.to_enu(opts=ENUOptions(origin=opts.local_origin), validate=False)
    evaluated = _generic_param_eval(
        AnalysisObject._from_unvalidated(analysis_object_dataset(enu)),
        query,
        on=on,
        opts=opts,
        validate=False,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        mode=mode,
        method="linear",
    )
    ecef = Position(analysis_object_dataset(evaluated)).geo.to_ecef(origin=opts.local_origin, validate=False)
    lla = ecef.geo.to_lla(validate=False)
    return _wrap_and_restamp(source, lla, opts=opts, validate=validate, owner=owner)


def geodetic_param_at(
    source,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    on: str | None,
    opts: GeodeticInterpolationOptions,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
):
    try:
        return _run(
            source,
            query,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            mode="at",
            owner=owner,
        )
    except (TypeError, ValueError) as exc:
        raise _wrap_error(exc, owner=owner) from exc


def geodetic_param_resample_to(
    source,
    *,
    grid: xr.DataArray | np.ndarray | Sequence[float] | float,
    on: str | None,
    opts: GeodeticInterpolationOptions,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
):
    try:
        return _run(
            source,
            grid,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            mode="resample_to",
            owner=owner,
        )
    except (TypeError, ValueError) as exc:
        raise _wrap_error(exc, owner=owner) from exc


def geodetic_param_interp_like(
    source,
    *,
    other,
    on: str | None,
    opts: GeodeticInterpolationOptions,
    batch_join: str,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
):
    try:
        if batch_join not in {"inner", "left"}:
            raise ValueError("batch_join must be 'inner' or 'left'.")
        context = _resolve_runtime(
            source,
            query_dim=opts.query_dim,
            on=on,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner=owner,
        )
        if context.batch_dims and batch_join == "inner":
            raise ValueError(
                "batch_join='inner' with batch dimensions is not supported by geodetic interp_like. "
                "Use batch_join='left' or pass an explicit aligned query to param.resample_to(...)."
            )
        query = query_coord_from_other_input(other, coord_name=context.spec.name, owner=owner)
        return _run(
            source,
            query,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            mode="resample_to",
            owner=owner,
        )
    except (TypeError, ValueError) as exc:
        raise _wrap_error(exc, owner=owner) from exc


def _run(
    source,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    on: str | None,
    opts: GeodeticInterpolationOptions,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    mode: str,
    owner: str,
):
    common = {
        "on": on,
        "opts": opts,
        "validate": validate,
        "sequence_dim": sequence_dim,
        "batch_dims": batch_dims,
        "sequence_size_coord": sequence_size_coord,
    }
    if opts.method == "nearest":
        return _nearest(source, query, mode=mode, owner=owner, **common)
    if opts.method == "ecef_linear":
        return _ecef_linear(source, query, mode=mode, owner=owner, **common)
    if opts.method == "local_enu_linear":
        return _local_enu_linear(source, query, mode=mode, owner=owner, **common)
    return _geodesic_linear(source, query, owner=owner, **common)


__all__ = [
    "geodetic_param_at",
    "geodetic_param_interp_like",
    "geodetic_param_resample_to",
]
