from __future__ import annotations

import numpy as np
import xarray as xr

from .kernels import require_unambiguous_geodesic_interpolation

_LLA_LABELS = ("lat", "lon", "alt")


def component(data: xr.DataArray, *, dim: str, label: str) -> xr.DataArray:
    return data.sel({dim: label}, drop=True)


def assemble_lla(
    components: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    *,
    core_dim: str,
    target_dims: tuple[str, ...],
    var_name: str,
) -> xr.DataArray:
    dim = xr.IndexVariable(core_dim, list(_LLA_LABELS))
    return xr.concat(list(components), dim=dim).transpose(*target_dims).rename(var_name)


def order_query_then_core(
    values: xr.DataArray,
    *,
    query_dim: str,
    core_dim: str,
) -> xr.DataArray:
    outer = [dim for dim in values.dims if dim not in {query_dim, core_dim}]
    return values.transpose(*(outer + [query_dim, core_dim]))


def _broadcast_query_da(
    values: xr.DataArray,
    *,
    template: xr.DataArray,
    core_dim: str,
) -> xr.DataArray:
    target = template.isel({core_dim: 0}, drop=True)
    return values.broadcast_like(target)


def geodesic_output_values(
    data: xr.DataArray,
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    core_dim: str,
    pmap,
    crs: str,
    owner: str,
    interpolate,
) -> xr.DataArray:
    alpha = _broadcast_query_da(pmap.alpha, template=left, core_dim=core_dim)
    safe_alpha = xr.apply_ufunc(
        require_unambiguous_geodesic_interpolation,
        component(left, dim=core_dim, label="lat"),
        component(left, dim=core_dim, label="lon"),
        component(right, dim=core_dim, label="lat"),
        component(right, dim=core_dim, label="lon"),
        alpha,
        input_core_dims=((), (), (), (), ()),
        output_core_dims=((),),
        output_dtypes=(np.float64,),
        dask="parallelized",
    )
    lat, lon = xr.apply_ufunc(
        interpolate,
        component(left, dim=core_dim, label="lat"),
        component(left, dim=core_dim, label="lon"),
        component(right, dim=core_dim, label="lat"),
        component(right, dim=core_dim, label="lon"),
        safe_alpha,
        input_core_dims=((), (), (), (), ()),
        output_core_dims=((), ()),
        output_dtypes=(np.float64, np.float64),
        kwargs={"crs": crs, "owner": owner},
        dask="parallelized",
    )
    alt = (1.0 - safe_alpha) * component(
        left, dim=core_dim, label="alt"
    ) + safe_alpha * component(right, dim=core_dim, label="alt")
    out = assemble_lla(
        (lat.where(pmap.valid), lon.where(pmap.valid), alt.where(pmap.valid)),
        core_dim=core_dim,
        target_dims=tuple(left.dims),
        var_name=str(data.name),
    )
    out = out.assign_coords({core_dim: data.coords[core_dim]})
    return order_query_then_core(out, query_dim=pmap.query_dim, core_dim=core_dim)
