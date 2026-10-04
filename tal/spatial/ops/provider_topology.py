from __future__ import annotations

from typing import Literal

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.schema_read import read_param_coord_name, read_roles
from tal.core.schema_validate import validate_schema_structure

ProviderTopology = Literal["static", "dynamic", "exact"]


def classify_provider_topology(value: object) -> ProviderTopology:
    """Classify one spatial provider from declared TAL sequence/parameter roles."""
    ds = analysis_object_dataset(value)
    validate_schema_structure(ds)
    _, sequence_dim, _, _ = read_roles(ds)
    param_coord = read_param_coord_name(ds)
    if sequence_dim is None:
        return "static"
    return "dynamic" if param_coord is not None else "exact"


__all__ = ["ProviderTopology", "classify_provider_topology"]
