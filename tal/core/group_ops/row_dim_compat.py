from __future__ import annotations

import xarray as xr


def require_row_dim_compatibility(
    data: xr.DataArray,
    *,
    row_dims: tuple[str, ...],
    allow_missing_sequence_dim: bool = False,
    allow_missing_row_dims: bool = False,
    required_dims: tuple[str, ...] = (),
    owner: str,
    what: str,
) -> None:
    dims = tuple(data.dims)
    if dims == row_dims:
        return
    if allow_missing_row_dims:
        if set(dims) <= set(row_dims):
            if all(dim in dims for dim in required_dims):
                return
    if allow_missing_sequence_dim and row_dims:
        sequence_required = row_dims[:-1]
        if set(dims) <= set(row_dims) and all(dim in dims for dim in sequence_required):
            return
    if len(dims) != len(row_dims) or set(dims) != set(row_dims):
        missing = tuple(dim for dim in row_dims if dim not in dims)
        extra = tuple(dim for dim in dims if dim not in row_dims)
        raise ValueError(
            f"{owner}: {what} must align to row dims {row_dims!r}; got {dims!r} "
            f"(missing={missing!r}, extra={extra!r})."
        )


__all__ = ["require_row_dim_compatibility"]
