from __future__ import annotations

import xarray as xr

from .var_naming import fallback_var_name


def _default_var_name(da: xr.DataArray) -> str:
    return fallback_var_name(da.name)


def ensure_dataset(data: xr.Dataset | xr.DataArray) -> xr.Dataset:
    """Normalize DataArray/Dataset input to Dataset.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Input data payload used to construct/derive an output object.

    Returns
    -------
    xr.Dataset
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if isinstance(data, xr.Dataset):
        return data
    if isinstance(data, xr.DataArray):
        return data.to_dataset(name=_default_var_name(data))
    actual = type(data).__name__
    raise TypeError(f"Expected xr.Dataset or xr.DataArray, got {actual}.")


def require_single_data_var(ds: xr.Dataset) -> str:
    """Return the only data variable name or raise with actionable guidance.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.

    Returns
    -------
    str
        String result produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    names = list(ds.data_vars)
    if len(names) != 1:
        raise ValueError(
            "to_dataarray() requires exactly one data variable. "
            f"Found {len(names)} variables: {names}. "
            "Select one variable first, then convert."
        )
    return names[0]


def dataset_to_dataarray(ds: xr.Dataset, *, name: str | None = None) -> xr.DataArray:
    """Extract a single variable as DataArray.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    name : str | None, optional
        Identifier/name used for lookup or registration.

    Returns
    -------
    xr.DataArray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    var_name = require_single_data_var(ds)
    da = ds[var_name]
    if name is not None:
        return da.rename(name)
    return da
