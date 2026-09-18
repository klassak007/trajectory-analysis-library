"""Fixtures for exercising downstream behavior on deliberately invalid AO state."""

from collections.abc import Sequence

import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.schema import set_param_coord, set_roles, set_validity


@pytest.fixture
def unsafe_from_data():
    """Build malformed state only for tests of downstream validation owners.

    Public ``from_data`` now rejects this state at ingress. These tests need a
    trusted, unvalidated wrapper to exercise later operation boundaries.
    """

    def build(
        data: xr.Dataset,
        *,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] = (),
        core_dims: Sequence[str] = (),
        param_coord: str | None = None,
        sequence_size_coord: str | None = None,
        validate: bool = False,
    ) -> AnalysisObject:
        _ = validate
        candidate = set_roles(
            data,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            core_dims=core_dims,
            validate=False,
        )
        if param_coord is not None:
            candidate = set_param_coord(candidate, name=param_coord, validate=False)
        if sequence_size_coord is not None:
            candidate = set_validity(candidate, sequence_size_coord=sequence_size_coord, validate=False)
        return AnalysisObject._from_unvalidated(candidate, schema_prepared=True)

    return build
