"""Consumption of inherited operation-owned query coordinates."""

from __future__ import annotations

import xarray as xr

from .guards import reserved_coord_is_owned


def inherited_query_metadata_names(
    query: object, *, size_coord: xr.DataArray | None = None,
) -> tuple[str, ...]:
    """Identify generated auxiliaries without consuming caller axes or indexes."""
    if not isinstance(query, xr.DataArray):
        return ()
    size_name = str(size_coord.name) if size_coord is not None else None
    return tuple(
        name for name in query.coords
        if isinstance(name, str) and name not in query.dims and name not in query.xindexes
        and (reserved_coord_is_owned(query, name=name)
             or (name == size_name and query.coords[name].dims == size_coord.dims))
    )


def without_inherited_query_metadata(query: object) -> object:
    """Keep query data (including deferred validation) while dropping old outputs."""
    names = inherited_query_metadata_names(query)
    return query.drop_vars(names) if names else query
