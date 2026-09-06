from __future__ import annotations

from collections.abc import Callable

import xarray as xr

from tal.core.orchestration.finalize import transfer_dataset_attrs
from tal.core.schema import set_roles
from tal.utils.xarray_namespace import dataset_namespace_names

RepWriter = Callable[..., xr.Dataset]


def dataset_dim_names(ds: xr.Dataset) -> set[str]:
    """Collect names that may collide with new conversion core dims.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.

    Returns
    -------
    set[str]
        String result produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return set(dataset_namespace_names(ds))


def allocate_free_dim_name(
    *,
    existing_dims: set[str],
    candidates: tuple[str, ...],
    base: str,
    owner: str,
    what: str,
) -> str:
    for candidate in candidates:
        if candidate not in existing_dims:
            return candidate
    suffix = 2
    while True:
        generated = f"{base}_{suffix}"
        if generated not in existing_dims:
            return generated
        suffix += 1
        if suffix > 1000:
            raise ValueError(f"{owner}: unable to allocate deterministic core dim for {what}.")


def allocate_dim_pair(
    *,
    existing_dims: set[str],
    first_candidates: tuple[str, ...],
    first_base: str,
    first_what: str,
    second_candidates: tuple[str, ...],
    second_base: str,
    second_what: str,
    owner: str,
) -> tuple[str, str]:
    first = allocate_free_dim_name(
        existing_dims=existing_dims,
        candidates=first_candidates,
        base=first_base,
        owner=owner,
        what=first_what,
    )
    used = set(existing_dims)
    used.add(first)
    second = allocate_free_dim_name(
        existing_dims=used,
        candidates=second_candidates,
        base=second_base,
        owner=owner,
        what=second_what,
    )
    return first, second


def conversion_dataset_from_array(
    converted: xr.DataArray,
    *,
    var_name: str,
    source_ds: xr.Dataset,
) -> xr.Dataset:
    # The payload need not use every declared sequence/batch dimension. Preserve
    # those coordinates and their xarray indexes while replacing the core layout.
    core_dims = tuple(dim for dim in source_ds[var_name].dims if dim not in converted.dims)
    topology = source_ds.drop_dims(core_dims).drop_vars(var_name, errors="ignore")
    out = topology.assign({var_name: converted})
    return transfer_dataset_attrs(source_ds, out, validate=False)


def finalize_conversion_dataset(
    ds: xr.Dataset,
    *,
    core_dims: tuple[str, ...],
    rep_value: str,
    set_rep: RepWriter,
    owner: str,
) -> xr.Dataset:
    out = set_roles(ds, core_dims=core_dims, validate=False)
    return set_rep(out, rep=rep_value, validate=False, owner=owner)


__all__ = [
    "allocate_dim_pair",
    "allocate_free_dim_name",
    "conversion_dataset_from_array",
    "dataset_dim_names",
    "finalize_conversion_dataset",
]
