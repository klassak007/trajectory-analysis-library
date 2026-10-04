from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from tal.core import AnalysisObject, BatchGroupReduceOptions
from tal.core.schema_read import read_roles
from tal.spatial import Rotation
from tal.spatial.metadata import set_rotation_rep


def _matrix_source(*, dependent: bool, lazy: bool, calls: list[str]) -> Rotation:
    dims = ("trial", "lane", "row", "col") if dependent else ("lane", "row", "col")
    values = np.tile(np.eye(3), (4, 2, 1, 1) if dependent else (2, 1, 1))

    @delayed
    def payload() -> np.ndarray:
        calls.append("payload")
        return values

    data = da.from_delayed(payload(), shape=values.shape, dtype=float) if lazy else values
    source = AnalysisObject.from_data(
        xr.Dataset({"rotation": (dims, data)}, coords={
            "trial": np.arange(4), "lane": ["left", "right"],
            "row": ["x", "y", "z"], "col": ["x", "y", "z"],
            "key": ("trial", ["a", "b", "c", "d"]),
            "score": ("trial", [0.2, 1.2, 2.2, 3.2]),
        }),
        batch_dims=("trial", "lane"), core_dims=("row", "col"),
    )
    return Rotation(set_rotation_rep(source.as_dataset(), rep="matrix", validate=True, owner="test.rotation"))


@pytest.mark.parametrize("topology", ["observed", "empty_bins", "zero_rows"])
@pytest.mark.parametrize("dependent", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("validate", [False, True])
def test_batch_matrix_reductions_use_bound_result_representation(
    topology: str, dependent: bool, lazy: bool, validate: bool,
) -> None:
    """ID: BATCH_ROTATION_067_group_assembly_preserves_bound_reducer_representation."""
    calls: list[str] = []
    source = _matrix_source(dependent=dependent, lazy=lazy, calls=calls)
    if topology == "zero_rows":
        source = source.isel(trial=[])
    grouped = source.group.groupby_bins("score", [0, 1, 2, 3, 4, 5])
    actual = grouped.mean(
        opts=BatchGroupReduceOptions(include_empty_groups=topology == "empty_bins"),
        validate=validate,
    )
    ds = actual.as_dataset(copy="none")
    assert calls == []
    assert isinstance(actual, Rotation)
    assert ds.sizes["group_key"] == {"observed": 4, "empty_bins": 5, "zero_rows": 0}[topology]
    core = ("quat",) if dependent else ("row", "col")
    assert read_roles(ds)[1:] == (None, ("group_key", "lane"), core)
    assert ds["rotation"].dims == (("group_key", "lane", *core) if dependent else ("lane", *core))
    assert isinstance(ds["rotation"].data, da.Array) == lazy
    if topology != "zero_rows":
        expected = source.isel(trial=[0]).mean(dim="trial").as_dataset(copy="none")["rotation"]
        first = ds["rotation"].isel(group_key=0, drop=True) if dependent else ds["rotation"]
        xr.testing.assert_allclose(first, expected)
    if topology == "empty_bins" and dependent:
        assert bool(ds["rotation"].isel(group_key=-1).isnull().all().compute())
    if dependent:
        # Rotation delegates gufunc chunk requirements to xarray/Dask.
        chained = Rotation(ds.chunk({"group_key": -1})) if lazy else actual
        assert isinstance(chained.mean(dim="group_key", validate=validate), Rotation)


@pytest.mark.parametrize("lazy", [False, True])
def test_primary_independent_matrix_mean_keeps_new_quaternion_core(lazy: bool) -> None:
    """ID: BATCH_ROTATION_068_independent_active_reduction_keeps_result_core_roles."""
    calls: list[str] = []
    source = _matrix_source(dependent=False, lazy=lazy, calls=calls)
    actual = source.group.groupby("key").mean(dim=("trial", "lane"))
    ds = actual.as_dataset(copy="none")
    assert calls == []
    assert read_roles(ds)[1:] == (None, ("group_key",), ("quat",))
    assert ds["rotation"].dims == ("quat",)
    np.testing.assert_allclose(ds["rotation"].compute(), [0., 0., 0., 1.])


@pytest.mark.parametrize("lazy", [False, True])
def test_explicit_group_name_is_reserved_from_reducer_component_allocation(
    lazy: bool,
) -> None:
    """ID: BATCH_ROTATION_069_explicit_group_name_precedes_component_allocation."""
    calls: list[str] = []
    source = _matrix_source(dependent=True, lazy=lazy, calls=calls)
    actual = source.group.groupby("key").mean(
        opts=BatchGroupReduceOptions(group_dim="quat")
    )
    ds = actual.as_dataset(copy="none")
    assert calls == []
    assert isinstance(actual, Rotation)
    assert read_roles(ds)[1:] == (None, ("quat", "lane"), ("quat_component",))
    assert ds["rotation"].dims == ("quat", "lane", "quat_component")
    assert isinstance(ds["rotation"].data, da.Array) == lazy
    np.testing.assert_allclose(
        ds["rotation"].compute(),
        np.broadcast_to([0.0, 0.0, 0.0, 1.0], ds["rotation"].shape),
    )


@pytest.mark.parametrize("name", ["quat", "quat_component", "rotation_component"])
@pytest.mark.parametrize("mode", ["conversion", "mean", "groups", "empty_bins", "zero_rows"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("validate", [False, True])
def test_matrix_payload_names_cannot_collide_with_conversion_dims(
    name: str, mode: str, lazy: bool, validate: bool,
) -> None:
    """ID: ROTATION_NAMES_074_conversion_reserves_payload_namespace."""
    calls: list[str] = []
    source = _matrix_source(dependent=True, lazy=lazy, calls=calls)
    candidates = ["quat", "quat_component", "rotation_component"]
    ds = source.as_dataset(copy="none").rename({"rotation": name})
    ds = ds.assign_coords({candidate: 1 for candidate in candidates[:candidates.index(name)]})
    source = Rotation(ds)
    if mode == "zero_rows":
        source = source.isel(trial=[])
    before = source.as_dataset(copy="none")
    if mode == "conversion":
        actual = source.as_quat(validate=validate)
    elif mode == "mean":
        actual = source.mean("trial", validate=validate)
    else:
        actual = source.group.groupby_bins("score", [0, 1, 2, 3, 4, 5]).mean(
            opts=BatchGroupReduceOptions(include_empty_groups=mode == "empty_bins"),
            validate=validate,
        )
    result = actual.as_dataset(copy="none")
    assert calls == []
    assert list(result.data_vars) == [name]
    core = read_roles(result)[3]
    assert len(core) == 1 and core[0] != name
    assert not set(core).intersection(before.variables)
    assert result.sizes[core[0]] == 4
    assert isinstance(result[name].data, da.Array) == lazy
    values = result[name].compute()
    if mode == "empty_bins":
        assert bool(values.isel(group_key=-1).isnull().all())
        values = values.isel(group_key=slice(0, -1))
    np.testing.assert_allclose(values, np.broadcast_to([0., 0., 0., 1.], values.shape))
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("name", ["row", "col"])
@pytest.mark.parametrize("lazy", [False, True])
def test_quaternion_payload_names_survive_matrix_conversion(name: str, lazy: bool) -> None:
    """ID: ROTATION_NAMES_075_matrix_conversion_reserves_payload_namespace."""
    calls: list[str] = []
    source = _matrix_source(dependent=True, lazy=lazy, calls=calls).as_quat()
    source = source.rename({"rotation": name})
    actual = source.as_matrix().as_dataset(copy="none")
    assert calls == []
    assert list(actual.data_vars) == [name]
    assert name not in read_roles(actual)[3]
    assert isinstance(actual[name].data, da.Array) == lazy
    values = actual[name].compute()
    np.testing.assert_allclose(values, np.broadcast_to(np.eye(3), values.shape))
