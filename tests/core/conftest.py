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


@pytest.fixture
def tutorial_audit_source():
    """Public irregular clocks for tutorial ownership/laziness regressions."""
    import numpy as np

    def build(*, lazy=False, batch=(2,), index_name=None, axis=False, shared=True,
              n=6, ragged=False, lazy_part="all", clock_values=None,
              kind="numeric", labels="indexed", aux=None):
        dims = tuple(f"b{i}" for i in range(len(batch)))
        values = np.broadcast_to(np.array([0., 1., 1., 0., 1., 0.])[:n], batch + (n,)).copy()
        variables = {dim: xr.Variable(dim, np.arange(size) + 10 * i)
                     for i, (dim, size) in enumerate(zip(dims, batch, strict=True))}
        if not axis:
            variables["sample"] = xr.Variable("sample", np.arange(n) + 100)
        clock = np.arange(n, dtype=float) ** 1.5 if clock_values is None else np.asarray(clock_values)[:n]
        if kind == "datetime":
            clock = np.datetime64("2026-01-01", "ns") + (clock * 1e9).astype("timedelta64[ns]")
        clock_dims = ("sample",) if shared else dims + ("sample",)
        if not shared:
            clock = np.broadcast_to(clock, batch + (n,)).copy()
        parameter = "sample" if axis else "clock"
        variables[parameter] = xr.Variable(clock_dims, clock)
        coords = xr.Coordinates(variables, indexes={})
        ds = xr.Dataset({"value": (dims + ("sample",), values)}, coords=coords)
        for dim in dims:
            ds = ds.set_xindex(dim)
        if not axis and labels == "indexed":
            ds = ds.set_xindex("sample")
        if aux:
            name, form = aux
            variable = xr.Variable(dims[0], np.arange(batch[0]) + 20) if form == "batch" else xr.Variable((), 20)
            ds = ds.assign_coords(xr.Coordinates({name: variable}, indexes={}))
        if index_name:
            ds = ds.assign_coords({index_name: (dims[0], np.arange(batch[0]) + 20)}).set_xindex(index_name)
        if ragged:
            counts = np.full(batch, n, dtype="int64")
            counts.flat[-1] = max(n - 2, 0)
            ds = ds.assign_coords(count=xr.Variable(dims, counts))
        if lazy:
            chunks = {dim: max(1, min(ds.sizes[dim], 2)) for dim in ds.dims}
            if lazy_part in ("payload", "all"):
                ds["value"] = ds.value.chunk(chunks)
            if lazy_part in ("clock", "all"):
                ds = ds.assign_coords(xr.Coordinates({parameter: ds[parameter].chunk(
                    {dim: chunks[dim] for dim in clock_dims}).variable}, indexes={}))
        if labels == "lazy":
            ds = ds.assign_coords(xr.Coordinates({"sample": ds["sample"].chunk(sample=2).variable}, indexes={}))
        return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=dims,
                                        core_dims=(), param_coord=parameter,
                                        sequence_size_coord="count" if ragged else None)
    return build


@pytest.fixture
def window_review_source():
    """Linear public trajectory for window ownership and packing oracles."""
    import numpy as np

    def build(lazy=False, batch=None, extra_index=False):
        dims = () if batch is None else ("trial",)
        shape = (4,) if batch is None else (batch, 4)
        values = np.broadcast_to(np.arange(4.0, dtype=float), shape).copy()
        ds = xr.Dataset(
            {"value": (dims + ("sample",), values)},
            coords={"sample": [10, 20, 30, 40], "time": ("sample", np.arange(4.0))},
        )
        if batch is not None:
            ds = ds.assign_coords(trial=np.arange(batch) + 10)
        if extra_index:
            lane = "sample" if batch is None else "trial"
            ds = ds.assign_coords(
                alias=(lane, np.arange(ds.sizes[lane]) + 100)
            ).set_xindex("alias")
        if lazy:
            ds["value"] = ds.value.chunk(
                {d: max(1, min(2, s)) for d, s in ds.sizes.items()}
            )
            ds = ds.assign_coords(time=ds.time.chunk(sample=2))
        return AnalysisObject.from_data(
            ds, sequence_dim="sample", batch_dims=dims, param_coord="time"
        )

    return build
