from __future__ import annotations

from collections.abc import Sequence

import xarray as xr


def shared_optional_name(values: Sequence[str | None]) -> str | None:
    """Return a shared optional metadata name or ``None`` when not unanimous.

    Parameters
    ----------
    values : Sequence[str | None]
        Input values consumed by this operation.

    Returns
    -------
    str | None
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if not values or any(name is None for name in values):
        return None
    first = values[0]
    if all(name == first for name in values):
        return first
    return None


def canonical_param_dims(
    *,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
) -> tuple[str, ...] | None:
    if sequence_dim is None:
        return None
    return batch_dims + (sequence_dim,)


def canonical_size_dims(
    *,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
) -> tuple[str, ...] | None:
    if sequence_dim is None:
        return None
    return batch_dims if batch_dims else ()


def align_param_coord_to_canonical(
    ds: xr.Dataset,
    *,
    name: str,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
) -> xr.Dataset | None:
    if sequence_dim is None or name not in ds.coords:
        return None
    coord = ds.coords[name]
    dims = tuple(coord.dims)
    expected = canonical_param_dims(sequence_dim=sequence_dim, batch_dims=batch_dims)
    if expected is None:
        return None
    if dims == (sequence_dim,):
        return ds
    if set(dims) != set(expected):
        return None
    return ds.assign_coords({name: coord.transpose(*expected)})


def align_size_coord_to_canonical(
    ds: xr.Dataset,
    *,
    name: str,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
) -> xr.Dataset | None:
    if sequence_dim is None or name not in ds.coords:
        return None
    coord = ds.coords[name]
    dims = tuple(coord.dims)
    expected = canonical_size_dims(sequence_dim=sequence_dim, batch_dims=batch_dims)
    if expected is None:
        return None
    if dims == expected:
        return ds
    if set(dims) != set(expected):
        return None
    if not expected:
        return ds
    return ds.assign_coords({name: coord.transpose(*expected)})


def canonicalize_optional_names(
    ds: xr.Dataset,
    *,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    param_name: str | None,
    size_name: str | None,
) -> tuple[xr.Dataset, str | None, str | None]:
    """Align optional metadata coordinates to canonical semantic dims.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    sequence_dim : str | None, optional
        Optional override for the sequence dimension used by temporal semantics.
    batch_dims : tuple[str, ...], optional
        Optional override for batch dimensions used by temporal semantics.
    param_name : str | None, optional
        Parameter-domain input used for temporal evaluation/alignment.
    size_name : str | None, optional
        Output naming metadata used during finalization.

    Returns
    -------
    tuple[xr.Dataset, str | None, str | None]
        Tuple of output values produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    out = ds
    if param_name is not None:
        aligned = align_param_coord_to_canonical(
            out,
            name=param_name,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
        )
        if aligned is None:
            param_name = None
        else:
            out = aligned
    if size_name is not None:
        aligned = align_size_coord_to_canonical(
            out,
            name=size_name,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
        )
        if aligned is None:
            size_name = None
        else:
            out = aligned
    return out, param_name, size_name


__all__ = [
    "align_param_coord_to_canonical",
    "align_size_coord_to_canonical",
    "canonicalize_optional_names",
    "canonical_param_dims",
    "canonical_size_dims",
    "shared_optional_name",
]
