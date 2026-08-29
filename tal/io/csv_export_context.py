from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import xarray as xr

from tal.core import AnalysisObject, SchemaError
from tal.core.schema import set_validity
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.core.schema_validate import validate_schema_structure

from .options import CsvExportOptions


@dataclass(frozen=True)
class CsvExportContext:
    """Schema-resolved state for one grouped CSV export."""

    ds: xr.Dataset
    batch_dim: str
    sequence_dim: str
    param_name: str | None
    declared_size_name: str | None
    size_name: str | None


def _read_overridden_declared_size_name(ds: xr.Dataset) -> str | None:
    """Read usable former validity metadata without reviving its validation role."""
    try:
        return read_sequence_size_coord_name(ds)
    except SchemaError as exc:
        if exc.path != "tal.core.validity" and not exc.path.startswith(
            "tal.core.validity."
        ):
            raise
    tal = ds.attrs.get("tal")
    if not isinstance(tal, Mapping):
        return None
    core = tal.get("core")
    if not isinstance(core, Mapping):
        return None
    validity = core.get("validity")
    if not isinstance(validity, Mapping):
        return None
    name = validity.get("sequence_size_coord")
    return name if isinstance(name, str) and bool(name) else None


def resolve_csv_export_context(
    ao: AnalysisObject,
    *,
    options: CsvExportOptions,
    owner: str,
) -> CsvExportContext:
    """Resolve roles and effective validity without reading array payloads."""
    source_ds = ao.unsafe_data
    try:
        schema_ds = source_ds
        if options.sequence_size_coord is not None:
            schema_ds = set_validity(
                source_ds,
                sequence_size_coord=None,
                validate=False,
            )
        validate_schema_structure(schema_ds)
        if options.sequence_size_coord is None:
            declared_size_name = read_sequence_size_coord_name(source_ds)
        else:
            declared_size_name = _read_overridden_declared_size_name(source_ds)
        roles_declared, sequence_dim, batch_dims, _core_dims = read_roles(schema_ds)
        param_name = read_param_coord_name(schema_ds)
    except (SchemaError, TypeError) as exc:
        raise ValueError(f"{owner}: invalid AnalysisObject schema payload.") from exc
    if not roles_declared:
        raise ValueError(f"{owner}: input requires declared roles.")
    if sequence_dim is None:
        raise ValueError(f"{owner}: input requires declared roles with sequence_dim.")
    if len(batch_dims) != 1:
        raise ValueError(
            f"{owner}: grouped CSV export requires exactly one batch dim; got {batch_dims!r}."
        )
    batch_dim = batch_dims[0]
    if batch_dim not in source_ds.coords:
        raise ValueError(
            f"{owner}: grouped CSV export derives labels from batch coordinate values; "
            f"coord {batch_dim!r} is missing."
        )
    size_name = options.sequence_size_coord
    if size_name is None:
        size_name = declared_size_name
    return CsvExportContext(
        ds=source_ds,
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
        param_name=param_name,
        declared_size_name=declared_size_name,
        size_name=size_name,
    )


__all__ = ["CsvExportContext", "resolve_csv_export_context"]
