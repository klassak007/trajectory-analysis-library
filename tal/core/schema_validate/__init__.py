from __future__ import annotations

import xarray as xr

from .common import SCHEMA_VERSION
from .finalize import finalize_validated_schema
from .phase_param import phase_param_coord
from .phase_roles import phase_roles_structure, phase_roles_vs_dims
from .phase_root import phase_core_envelope, phase_extension_envelope, phase_root_shape, phase_version
from .phase_validity import phase_validity

__all__ = ["SCHEMA_VERSION", "validate_schema"]


def validate_schema(ds: xr.Dataset) -> xr.Dataset:
    """Validate TAL schema and return a dataset carrying canonical schema.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.

    Returns
    -------
    xr.Dataset
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if not isinstance(ds, xr.Dataset):
        actual = type(ds).__name__
        raise TypeError(f"validate_schema expects xr.Dataset, got {actual}.")
    tal = phase_root_shape(ds)
    phase_version(tal)
    core = phase_core_envelope(tal)
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
    phase_validity(ds, core=core, sequence_dim=sequence_dim, batch_dims=batch_dims)
    phase_extension_envelope(tal)
    return finalize_validated_schema(ds, tal)
