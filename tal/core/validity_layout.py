from __future__ import annotations

import numpy as np
import xarray as xr


def _is_chunked(da: xr.DataArray) -> bool:
    return getattr(da.data, "chunks", None) is not None


def is_left_packed_mask(
    valid: xr.DataArray,
    *,
    sequence_dim: str,
) -> bool:
    """Return True when a validity mask is left-packed along ``sequence_dim``.

    Chunked masks are treated conservatively as not provably left-packed to
    preserve laziness in orchestration/finalization paths.
    """
    if sequence_dim not in valid.dims:
        return True
    if _is_chunked(valid):
        return False
    v = valid.fillna(False).astype("int8")
    prefix = v.cumprod(dim=sequence_dim)
    ok = (prefix == v).all()
    return bool(np.asarray(ok.data).item())


def sequence_size_from_mask(
    valid: xr.DataArray,
    *,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
) -> xr.DataArray:
    """Derive sequence-size coordinates from a boolean validity mask.

    Parameters
    ----------
    valid : xr.DataArray
        Validity/mask payload used by this operation.
    sequence_dim : str, optional
        Optional override for the sequence dimension used by temporal semantics.
    batch_dims : tuple[str, ...], optional
        Optional override for batch dimensions used by temporal semantics.

    Returns
    -------
    xr.DataArray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    out = valid.fillna(False).astype("int64")
    if sequence_dim in out.dims:
        out = out.sum(dim=sequence_dim)
    if batch_dims:
        return out.transpose(*batch_dims)
    if out.dims:
        out = out.sum(dim=tuple(out.dims))
    return out.astype("int64")


def scalar_int_boundary(
    value: xr.DataArray,
    *,
    owner: str,
    field: str,
    allow_chunked_compute: bool,
) -> int:
    """Extract an integer scalar with explicit chunked-compute policy.

    Parameters
    ----------
    value : xr.DataArray
        Input value to normalize/coerce/process.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.
    field : str, optional
        Label/name selection used by this operation.
    allow_chunked_compute : bool, optional
        Behavior flag/policy controlling boundary semantics.

    Returns
    -------
    int
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    scalar = value
    if getattr(scalar.data, "chunks", None) is not None:
        if not allow_chunked_compute:
            raise ValueError(
                f"{owner}: chunked scalar extraction for {field!r} is not allowed without explicit compute policy."
            )
        scalar = scalar.compute()
    return int(np.asarray(scalar.data).item())


__all__ = ["is_left_packed_mask", "scalar_int_boundary", "sequence_size_from_mask"]
