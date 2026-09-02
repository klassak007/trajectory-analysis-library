from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import xarray as xr

from .common import SCHEMA_VERSION
from .finalize import finalize_validated_schema
from .phase_param import phase_param_coord
from .phase_roles import phase_roles_structure, phase_roles_vs_dims
from .phase_root import phase_core_envelope, phase_extension_envelope, phase_root_shape, phase_version
from .phase_validity import check_validity_coord_values, phase_validity_structure

__all__ = ["SCHEMA_VERSION", "validate_schema", "validate_schema_structure"]


@dataclass(frozen=True)
class _SchemaStructure:
    tal: Mapping[str, Any]
    sequence_dim: str | None
    sequence_size_coord: str | None


def _resolve_schema_root(
    ds: xr.Dataset,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(ds, xr.Dataset):
        actual = type(ds).__name__
        raise TypeError(f"validate_schema expects xr.Dataset, got {actual}.")
    tal = phase_root_shape(ds)
    phase_version(tal)
    core = phase_core_envelope(tal)
    return tal, core


def _validate_schema_envelope(ds: xr.Dataset) -> None:
    tal, _ = _resolve_schema_root(ds)
    phase_extension_envelope(tal)


def _resolve_schema_structure(ds: xr.Dataset) -> _SchemaStructure:
    tal, core = _resolve_schema_root(ds)
    sequence_dim, batch_dims, core_dims = phase_roles_structure(core)
    phase_roles_vs_dims(
        ds,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=core_dims,
    )
    phase_param_coord(
        ds,
        core=core,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
    )
    size_name = phase_validity_structure(
        ds,
        core=core,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
    )
    return _SchemaStructure(
        tal=tal,
        sequence_dim=sequence_dim,
        sequence_size_coord=size_name,
    )


def validate_schema_structure(ds: xr.Dataset) -> str | None:
    """Validate schema structure without inspecting coordinate values.

    This lazy-safe persistence preflight inspects metadata, dimensions,
    coordinate presence, and dtypes, but never materializes array payloads.
    """
    structure = _resolve_schema_structure(ds)
    phase_extension_envelope(structure.tal)
    name = structure.sequence_size_coord
    return None if name is None else str.__str__(name)


def validate_schema(ds: xr.Dataset) -> xr.Dataset:
    """Validate TAL schema and return a dataset carrying canonical schema."""
    structure = _resolve_schema_structure(ds)
    if structure.sequence_size_coord is not None:
        if structure.sequence_dim is None:  # pragma: no cover - structural invariant.
            raise RuntimeError("validated validity structure is missing sequence_dim")
        check_validity_coord_values(
            ds,
            name=structure.sequence_size_coord,
            sequence_dim=structure.sequence_dim,
        )
    phase_extension_envelope(structure.tal)
    return finalize_validated_schema(ds, structure.tal)
