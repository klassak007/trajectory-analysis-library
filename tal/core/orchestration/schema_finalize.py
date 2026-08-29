from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import xarray as xr

from ..metadata_optional import canonicalize_optional_names
from ..schema import merge_schema, set_param_coord, set_roles, set_validity
from .finalize import finalize_like, rewrap_unvalidated_like

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


@dataclass(frozen=True)
class CoreSchemaFinalizeSpec:
    """Core schema metadata to stamp during operation finalization.

    Parameters
    ----------
    sequence_dim : str | None
        Output sequence dimension, or ``None`` for core-only outputs.
    batch_dims : tuple[str, ...]
        Output batch dimensions.
    core_dims : tuple[str, ...]
        Output core dimensions.
    param_name : str | None
        Optional parameter coordinate name to preserve or stamp.
    size_name : str | None
        Optional sequence-size validity coordinate name to preserve or stamp.

    Notes
    -----
    This spec is an immutable declaration of output schema intent. It does not
    mutate a dataset until passed to a finalization helper.

    Examples
    --------
    >>> from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec
    >>> spec = CoreSchemaFinalizeSpec(
    ...     sequence_dim="sample",
    ...     batch_dims=(),
    ...     core_dims=(),
    ...     param_name=None,
    ...     size_name=None,
    ... )
    >>> (spec.sequence_dim, spec.batch_dims, spec.core_dims)
    ('sample', (), ())
    """

    sequence_dim: str | None
    batch_dims: tuple[str, ...]
    core_dims: tuple[str, ...]
    param_name: str | None
    size_name: str | None


def restore_optional_coord_from_sources(
    ds: xr.Dataset,
    *,
    name: str | None,
    sources: Sequence[xr.DataArray],
) -> xr.Dataset:
    if name is None or name in ds.coords:
        return ds
    for source in sources:
        if name not in source.coords:
            continue
        coord = source.coords[name]
        if any(dim not in ds.dims for dim in coord.dims):
            continue
        try:
            return ds.assign_coords({name: coord})
        except ValueError:
            continue
    return ds


def clear_core_schema_blocks(ds: xr.Dataset, *, validate: bool = False) -> xr.Dataset:
    _ = validate
    return merge_schema(ds, {"core": {"roles": None, "param_coord": None, "validity": None}}, validate=False)


def stamp_core_schema(
    ds: xr.Dataset,
    *,
    spec: CoreSchemaFinalizeSpec,
    validate: bool = False,
) -> xr.Dataset:
    _ = validate
    roles_kwargs: dict[str, object] = {
        "batch_dims": spec.batch_dims,
        "core_dims": spec.core_dims,
    }
    if spec.sequence_dim is not None:
        roles_kwargs["sequence_dim"] = spec.sequence_dim
    out = set_roles(
        ds,
        validate=False,
        **roles_kwargs,
    )
    out = set_param_coord(out, name=spec.param_name, validate=False)
    return set_validity(out, sequence_size_coord=spec.size_name, validate=False)


def finalize_with_schema(
    source_ao: "AnalysisObject",
    ds: xr.Dataset,
    *,
    spec: CoreSchemaFinalizeSpec,
    validate: bool,
    owner: str,
    optional_sources: Sequence[xr.DataArray] = (),
    clear_returns_unvalidated: bool = False,
) -> "AnalysisObject":
    """Stamp core schema metadata and finalize relative to a source AO.

    Parameters
    ----------
    source_ao : AnalysisObject
        Source object whose structural finalization and rewrap behavior are used.
    ds : xr.Dataset
        Candidate output dataset produced by an operation kernel.
    spec : CoreSchemaFinalizeSpec
        Output schema declaration for roles, parameter coordinate, and validity
        coordinate metadata.
    validate : bool
        Whether to validate the finalized output schema.
    owner : str
        Public owner string used to build deterministic diagnostics.
    optional_sources : Sequence[xr.DataArray], optional
        Source arrays from which missing optional coordinates may be restored
        when their dimensions fit the output dataset.
    clear_returns_unvalidated : bool, optional
        When ``True`` and ``spec.sequence_dim`` is ``None``, return the
        core-cleared output without the final validation pass.

    Returns
    -------
    AnalysisObject
        Output AO with core schema metadata stamped and structural metadata
        finalized.

    Notes
    -----
    This helper separates schema declaration from kernel math. It does not read
    payload values; optional coordinate restoration and schema stamping operate
    on xarray metadata and coordinates.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema
    >>> source = AnalysisObject.from_data(
    ...     xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> out_ds = xr.Dataset({"kelvin": ("sample", [293.15])}, coords={"sample": [0]})
    >>> spec = CoreSchemaFinalizeSpec(
    ...     sequence_dim="sample",
    ...     batch_dims=(),
    ...     core_dims=(),
    ...     param_name=None,
    ...     size_name=None,
    ... )
    >>> out = finalize_with_schema(
    ...     source,
    ...     out_ds,
    ...     spec=spec,
    ...     validate=True,
    ...     owner="thermal.to_kelvin",
    ... )
    >>> out.unsafe_data.attrs["tal"]["core"]["roles"]["sequence_dim"]
    'sample'
    """
    candidate = ds
    resolved_spec = spec
    if spec.sequence_dim is None:
        resolved_spec = replace(spec, param_name=None, size_name=None)
        candidate = clear_core_schema_blocks(candidate, validate=False)
        candidate = stamp_core_schema(candidate, spec=resolved_spec, validate=False)
    else:
        candidate = restore_optional_coord_from_sources(candidate, name=spec.param_name, sources=optional_sources)
        candidate = restore_optional_coord_from_sources(candidate, name=spec.size_name, sources=optional_sources)
        candidate, param_name, size_name = canonicalize_optional_names(
            candidate,
            sequence_dim=spec.sequence_dim,
            batch_dims=spec.batch_dims,
            param_name=spec.param_name,
            size_name=spec.size_name,
        )
        resolved_spec = replace(spec, param_name=param_name, size_name=size_name)
        candidate = stamp_core_schema(candidate, spec=resolved_spec, validate=False)
    tmp = rewrap_unvalidated_like(source_ao, candidate, owner=owner)
    if clear_returns_unvalidated and resolved_spec.sequence_dim is None:
        return tmp
    return finalize_like(tmp, tmp.unsafe_data, validate=validate, owner=owner)


__all__ = [
    "CoreSchemaFinalizeSpec",
    "clear_core_schema_blocks",
    "finalize_with_schema",
    "restore_optional_coord_from_sources",
    "stamp_core_schema",
]
