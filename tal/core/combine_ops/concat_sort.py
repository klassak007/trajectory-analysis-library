from __future__ import annotations

import numpy as np
import xarray as xr


def _restore_sequence_axis(ds: xr.Dataset, *, sequence_dim: str) -> xr.Dataset:
    size = int(ds.sizes.get(sequence_dim, 0))
    out = ds.drop_vars(sequence_dim, errors="ignore")
    return out.assign_coords({sequence_dim: np.arange(size, dtype="int64")})


def _sort_packed_rows(
    ds: xr.Dataset,
    *,
    flat_dim: str,
    sequence_dim: str,
    param_name: str,
) -> xr.Dataset:
    """Sort each packed row by param on valid domain and keep invalid tail stable."""
    param_coord = ds.coords[param_name]
    if flat_dim in param_coord.dims:
        param = np.asarray(param_coord.transpose(flat_dim, sequence_dim).data, dtype="float64")
    else:
        row = np.asarray(param_coord.transpose(sequence_dim).data, dtype="float64")
        param = np.broadcast_to(row[None, :], (int(ds.sizes[flat_dim]), row.shape[0]))
    if int(param.shape[0]) == 0:
        return ds
    valid = np.asarray(ds.coords["valid"].transpose(flat_dim, sequence_dim).data, dtype=bool)
    if np.any(~np.isfinite(param) & valid):
        raise ValueError("concat_sequence: overlap='sort' requires finite param_coord values on the valid domain.")
    sort_key = np.where(valid, param, np.inf)
    perms = np.argsort(sort_key, axis=1, kind="stable")
    indexer = xr.DataArray(
        perms.astype("int64"),
        dims=(flat_dim, sequence_dim),
        coords={flat_dim: ds.coords[flat_dim]},
    )
    sorted_ds = ds.isel({sequence_dim: indexer})
    return _restore_sequence_axis(sorted_ds, sequence_dim=sequence_dim)


def apply_overlap_sort(
    ds: xr.Dataset,
    *,
    overlap: str,
    flat_dim: str,
    sequence_dim: str,
    param_name: str | None,
) -> xr.Dataset:
    """Apply concat-sequence overlap sort policy when enabled.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    overlap : str, optional
        Resolved overlap policy payload used for sequence concat sorting.
    flat_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    sequence_dim : str, optional
        Optional override for the sequence dimension used by temporal semantics.
    param_name : str | None, optional
        Parameter-domain input used for temporal evaluation/alignment.

    Returns
    -------
    xr.Dataset
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if overlap != "sort" or param_name is None:
        return ds
    return _sort_packed_rows(ds, flat_dim=flat_dim, sequence_dim=sequence_dim, param_name=param_name)


__all__ = ["apply_overlap_sort"]
