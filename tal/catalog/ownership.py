from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr


def copy_catalog_payload_for_public_access(
    payload: xr.Dataset | xr.DataTree,
    *,
    owner: str,
) -> xr.Dataset | xr.DataTree:
    """Return an independently owned public view without realizing Dask arrays."""
    if isinstance(payload, xr.Dataset):
        return _isolate_dataset(payload, owner=owner)
    root = _isolate_dataset(payload.to_dataset(inherit=False), owner=owner)
    children = {
        name: xr.DataTree(
            dataset=_isolate_dataset(child.to_dataset(inherit=False), owner=owner),
            name=name,
        )
        for name, child in payload.children.items()
    }
    return xr.DataTree(dataset=root, children=children, name=payload.name)


def isolate_dataset_result_values(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    detached_vars = {
        name: _copy_dataset_result_array(variable, owner=owner, name=name).variable
        for name, variable in ds.data_vars.items()
    }
    out = ds.assign(detached_vars) if detached_vars else ds
    out = _copy_registered_xindexes(out, owner=owner)
    detached_coords = {
        name: _copy_dataset_result_array(coord, owner=owner, name=name).variable
        for name, coord in ds.coords.items()
        if name not in ds.xindexes
    }
    return _assign_unindexed_coords(out, detached_coords)


def _copy_dataset_result_array(
    array: xr.DataArray,
    *,
    owner: str,
    name: str,
) -> xr.DataArray:
    if isinstance(array.dtype, pd.CategoricalDtype):
        return _deepcopy_categorical_array(array, owner=owner, name=name)
    if _dtype_has_object_values(array.dtype):
        return _deepcopy_object_array(array, owner=owner, name=name)
    return array.copy(deep=array.chunks is None)


def isolate_result_values(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    owner: str,
) -> xr.Dataset:
    detached_vars = {
        name: _copy_nested_value_array(variable, owner=owner, name=name).variable
        for name, variable in ds.data_vars.items()
        if _dtype_requires_nested_value_copy(variable.dtype)
    }
    out = ds.assign(detached_vars) if detached_vars else ds
    out = _copy_registered_xindexes(out, owner=owner)
    detached_coords: dict[str, xr.Variable] = {}
    for name, coord in ds.coords.items():
        if name in ds.xindexes:
            continue
        if _dtype_requires_nested_value_copy(coord.dtype):
            detached_coords[name] = _copy_nested_value_array(
                coord,
                owner=owner,
                name=name,
            ).variable
        elif batch_dim not in coord.dims:
            detached_coords[name] = coord.variable.copy(deep=True)
    if not detached_coords:
        return out
    return _assign_unindexed_coords(out, detached_coords)


def _assign_unindexed_coords(
    ds: xr.Dataset,
    coords: Mapping[object, xr.Variable],
) -> xr.Dataset:
    if not coords:
        return ds
    return ds.assign_coords(xr.Coordinates(coords, indexes={}))


def _copy_registered_xindexes(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    if not ds.xindexes:
        return ds
    indexes: dict[object, xr.Index] = {}
    variables: dict[object, xr.Variable] = {}
    metadata_sources: dict[object, xr.Variable] = {}
    for index, coord_vars in ds.xindexes.group_by_index():
        source_vars = {name: ds.variables[name] for name in coord_vars}
        detached_vars = _detach_index_coord_variables(
            source_vars,
            owner=owner,
        )
        copied = _copy_registered_xindex(
            index,
            variables=detached_vars,
            owner=owner,
        )
        indexes.update(dict.fromkeys(detached_vars, copied))
        variables.update(copied.create_variables(detached_vars))
        metadata_sources.update(source_vars)
    coords = xr.Coordinates(variables, indexes=indexes)
    out = ds.drop_indexes(tuple(ds.xindexes)).assign_coords(coords)
    return _restore_index_coordinate_metadata(out, sources=metadata_sources)


def _restore_index_coordinate_metadata(
    ds: xr.Dataset,
    *,
    sources: Mapping[object, xr.Variable],
) -> xr.Dataset:
    for name, source in sources.items():
        ds[name].attrs = dict(source.attrs)
        ds[name].encoding = dict(source.encoding)
    return ds


def _detach_index_coord_variables(
    variables: Mapping[object, xr.Variable],
    *,
    owner: str,
) -> dict[object, xr.Variable]:
    return {
        name: _copy_dataset_result_array(
            xr.DataArray(variable, name=name),
            owner=owner,
            name=str(name),
        ).variable
        for name, variable in variables.items()
    }


def _copy_registered_xindex(
    index: object,
    *,
    variables: Mapping[object, xr.Variable],
    owner: str,
) -> xr.Index:
    rebuilt = _rebuild_pandas_xindex(index, variables=variables, owner=owner)
    if rebuilt is not None:
        return rebuilt
    if any(
        _dtype_requires_nested_value_copy(variable.dtype)
        for variable in variables.values()
    ):
        raise ValueError(
            f"{owner}: registered index type {type(index).__name__!r} has object-bearing "
            "coordinates that cannot be recursively isolated."
        )
    if isinstance(index, xr.Index):
        return index.copy(deep=True)
    raise TypeError(
        f"{owner}: unsupported registered index type {type(index).__name__!r}."
    )


def _rebuild_pandas_xindex(
    index: object,
    *,
    variables: Mapping[object, xr.Variable],
    owner: str,
) -> xr.Index | None:
    description = f"registered index coordinates {tuple(variables)!r}"
    if isinstance(index, xr.indexes.PandasMultiIndex):
        copied = _deepcopy_pandas_index(index.index, owner=owner, description=description)
        return xr.indexes.PandasMultiIndex(
            copied,
            index.dim,
            level_coords_dtype=index.level_coords_dtype,
        )
    if not isinstance(index, xr.indexes.PandasIndex):
        return None
    copied = _deepcopy_pandas_index(index.index, owner=owner, description=description)
    return xr.indexes.PandasIndex(
        copied,
        index.dim,
        coord_dtype=index.coord_dtype,
    )


def _deepcopy_pandas_index(
    index: pd.Index,
    *,
    owner: str,
    description: str,
) -> pd.Index:
    if isinstance(index, pd.MultiIndex):
        levels = [
            _deepcopy_pandas_index(level, owner=owner, description=description)
            for level in index.levels
        ]
        return pd.MultiIndex(
            levels=levels,
            codes=[code.copy() for code in index.codes],
            sortorder=index.sortorder,
            names=index.names,
            verify_integrity=True,
        )
    if isinstance(index, pd.CategoricalIndex):
        categories = _deepcopy_pandas_index(
            index.categories,
            owner=owner,
            description=description,
        )
        values = pd.Categorical.from_codes(
            index.codes.copy(),
            categories=categories,
            ordered=index.ordered,
        )
        return pd.CategoricalIndex(values, name=index.name)
    if bool(getattr(index.dtype, "hasobject", False)):
        values = _deepcopy_index_object_values(
            index,
            owner=owner,
            description=description,
        )
        return pd.Index(values, dtype=object, name=index.name, tupleize_cols=False)
    return index.copy(deep=True)


def _deepcopy_index_object_values(
    values: pd.Index,
    *,
    owner: str,
    description: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    copied = _deepcopy_or_error(
        array,
        owner=owner,
        description=description,
    )
    return np.asarray(copied, dtype=object)


def _dtype_has_object_values(dtype: object) -> bool:
    """Return NumPy object containment without assuming a NumPy dtype."""
    return bool(getattr(dtype, "hasobject", False))


def _dtype_requires_nested_value_copy(dtype: object) -> bool:
    return isinstance(dtype, pd.CategoricalDtype) or _dtype_has_object_values(dtype)


def _copy_nested_value_array(
    array: xr.DataArray,
    *,
    owner: str,
    name: str,
) -> xr.DataArray:
    if isinstance(array.dtype, pd.CategoricalDtype):
        return _deepcopy_categorical_array(array, owner=owner, name=name)
    return _deepcopy_object_array(array, owner=owner, name=name)


def _deepcopy_categorical_array(
    array: xr.DataArray,
    *,
    owner: str,
    name: str,
) -> xr.DataArray:
    dtype = array.dtype
    if not isinstance(dtype, pd.CategoricalDtype):
        raise TypeError(f"{owner}: array {name!r} is not categorical.")
    categories = _deepcopy_pandas_index(
        dtype.categories,
        owner=owner,
        description=f"categories for array {name!r}",
    )
    try:
        # Dask arrays do not support CategoricalDtype; this is an eager-only
        # extension-array boundary and cannot realize a chunked payload.
        flat = np.asarray(array.data, dtype=object).reshape(-1)
        codes = pd.Categorical(flat, dtype=dtype).codes.reshape(array.shape)
    except Exception as exc:
        raise ValueError(
            f"{owner}: could not isolate categorical values for array {name!r}."
        ) from exc
    copied = pd.Categorical.from_codes(
        codes,
        categories=categories,
        ordered=dtype.ordered,
    )
    out = array.copy(data=copied)
    out.encoding = _deepcopy_or_error(
        array.encoding,
        owner=owner,
        description=f"encoding for array {name!r}",
    )
    return out


def _deepcopy_object_array(
    array: xr.DataArray,
    *,
    owner: str,
    name: str,
) -> xr.DataArray:
    out = xr.apply_ufunc(
        _deepcopy_object_values,
        array,
        kwargs={"owner": owner, "name": name},
        dask="parallelized",
        output_dtypes=[array.dtype],
        keep_attrs=True,
    )
    out.encoding = _deepcopy_or_error(
        array.encoding,
        owner=owner,
        description=f"encoding for array {name!r}",
    )
    return out


def _deepcopy_object_values(
    values: object,
    *,
    owner: str,
    name: str,
) -> object:
    return _deepcopy_or_error(
        values,
        owner=owner,
        description=f"object values for array {name!r}",
    )


def isolate_result_metadata(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    out = ds.copy(deep=False)
    out.attrs = _deepcopy_or_error(ds.attrs, owner=owner, description="dataset attributes")
    out.encoding = _deepcopy_or_error(ds.encoding, owner=owner, description="dataset encoding")
    for name, variable in ds.variables.items():
        out[name].attrs = _deepcopy_or_error(
            variable.attrs,
            owner=owner,
            description=f"attributes for array {name!r}",
        )
        out[name].encoding = _deepcopy_or_error(
            variable.encoding,
            owner=owner,
            description=f"encoding for array {name!r}",
        )
    return out


def _isolate_dataset(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    isolated = isolate_dataset_result_values(ds, owner=owner)
    return isolate_result_metadata(isolated, owner=owner)


def _deepcopy_or_error(
    value: object,
    *,
    owner: str,
    description: str,
) -> object:
    try:
        return deepcopy(value)
    except Exception as exc:
        raise ValueError(
            f"{owner}: could not isolate {description}; values and metadata must support deep copy."
        ) from exc


__all__ = [
    "copy_catalog_payload_for_public_access",
    "isolate_dataset_result_values",
    "isolate_result_metadata",
    "isolate_result_values",
]
