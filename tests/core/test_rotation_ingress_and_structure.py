from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from tal.core import AnalysisObject, set_roles
from tal.core.schema_read import read_roles
from tal.spatial import Rotation
from tal.spatial.metadata import set_rotation_rep


def _ingress_source(topology: str, *, declared_core: bool) -> AnalysisObject:
    ds = xr.Dataset(
        {"rotation": ("label", [0.0, 0.0, 0.0, 1.0])},
        coords={"label": ["x", "y", "z", "w"]},
    )
    sequence = "sample" if topology == "sequence" else None
    batch = ("trial",) if topology == "batch" else ()
    if sequence:
        ds = ds.assign_coords(sample=[0, 1])
    if batch:
        ds = ds.assign_coords(trial=[0, 1])
    return AnalysisObject(set_roles(
        ds, sequence_dim=sequence, batch_dims=batch,
        core_dims=("label",) if declared_core else (),
    ))


def _construct(source: AnalysisObject, entry: str) -> Rotation:
    if entry == "constructor":
        return Rotation(source)
    return Rotation.from_data(source.as_dataset(), validate=entry == "validated")


@pytest.mark.parametrize("topology", ["sequence", "batch", "core"])
@pytest.mark.parametrize("entry", ["constructor", "validated", "unvalidated"])
def test_rotation_public_ingress_requires_declared_core(topology: str, entry: str) -> None:
    """ID: ROTATION_INGRESS_063_public_ingress_rejects_missing_core_roles."""
    source = _ingress_source(topology, declared_core=False)
    before = source.as_dataset()
    with pytest.raises(ValueError, match="requires exactly one core dim"):
        _construct(source, entry)
    xr.testing.assert_identical(source.as_dataset(), before)


@pytest.mark.parametrize("topology", ["sequence", "batch", "core"])
@pytest.mark.parametrize("entry", ["constructor", "validated", "unvalidated"])
def test_rotation_public_ingress_accepts_declared_core(topology: str, entry: str) -> None:
    """ID: ROTATION_INGRESS_064_public_ingress_retains_declared_core_roles."""
    source = _ingress_source(topology, declared_core=True)
    actual = _construct(source, entry)
    assert isinstance(actual, Rotation)
    xr.testing.assert_identical(actual.as_dataset()["rotation"], source.as_dataset()["rotation"])
    assert read_roles(actual.as_dataset())[3] == ("label",)


def test_role_cleared_reductions_remain_internal_and_support_chaining() -> None:
    """ID: ROTATION_INGRESS_065_trusted_role_cleared_reducer_and_magnitude_paths."""
    source = Rotation(AnalysisObject.from_data(
        xr.Dataset(
            {"rotation": (("trial", "sample", "quat"), np.tile([0., 0., 0., 1.], (2, 2, 1)))},
            coords={"trial": [0, 1], "sample": [0, 1], "quat": ["x", "y", "z", "w"]},
        ),
        sequence_dim="sample", batch_dims=("trial",), core_dims=("quat",),
    ))
    reduced = source.mean(dim="sample")
    assert read_roles(reduced.as_dataset())[1:] == (None, (), ())
    np.testing.assert_allclose(reduced.mean(dim="trial").as_dataset()["rotation"], [0., 0., 0., 1.])
    np.testing.assert_allclose(reduced.magnitude().as_dataset()["datavar"], [0., 0.])
    with pytest.raises(ValueError, match="required component dims"):
        reduced.mean(dim="quat")
    for entry in ("constructor", "validated", "unvalidated"):
        with pytest.raises(ValueError, match="requires exactly one core dim"):
            _construct(reduced, entry)


def _structural_source(rep: str, *, lazy: bool, calls: list[str]) -> Rotation:
    core = ("quat",) if rep == "quat" else ("row", "col")
    values = np.tile([0., 0., 0., 1.], (2, 1)) if rep == "quat" else np.tile(np.eye(3), (2, 1, 1))

    @delayed
    def payload() -> np.ndarray:
        calls.append("payload")
        return values

    data = da.from_delayed(payload(), shape=values.shape, dtype=float) if lazy else values
    coords = {dim: ["x", "y", "z", "w"] if dim == "quat" else ["x", "y", "z"] for dim in core}
    coords.update(sample=[0, 1], trial=[0, 1], time=("sample", [0., 1.]), size=("trial", [2, 1]))
    source = AnalysisObject.from_data(
        xr.Dataset({"rotation": (("trial", *core), data)}, coords=coords),
        sequence_dim="sample", batch_dims=("trial",), core_dims=core,
        param_coord="time", sequence_size_coord="size",
    )
    return Rotation(set_rotation_rep(source.as_dataset(), rep=rep, validate=True, owner="test.rotation"))


@pytest.mark.parametrize("rep", ["quat", "matrix"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("validate", [False, True])
def test_structural_sequence_reduction_preserves_unaffected_rotation_roles(
    rep: str, lazy: bool, validate: bool,
) -> None:
    """ID: ROTATION_FINALIZE_066_structural_sequence_reduction_retains_representation."""
    calls: list[str] = []
    source = _structural_source(rep, lazy=lazy, calls=calls)
    actual = source.mean(dim="sample", validate=validate)
    ds = actual.as_dataset(copy="none")
    assert calls == []
    core = ("quat",) if rep == "quat" else ("row", "col")
    assert read_roles(ds)[1:] == (None, ("trial",), core)
    assert not {"sample", "time", "size"}.intersection(ds.coords)
    assert ds.attrs["tal"]["ext"]["spatial"]["representation"]["rep"] == rep
    assert isinstance(ds["rotation"].data, da.Array) == lazy
    xr.testing.assert_equal(ds["rotation"], source.as_dataset(copy="none")["rotation"].drop_vars("size"))
    assert isinstance(actual.mean(dim="trial", validate=validate), Rotation)


@pytest.mark.parametrize("rep", ["quat", "matrix"])
@pytest.mark.parametrize("lazy", [False, True])
def test_rotation_conversion_preserves_coordinate_only_topology(rep: str, lazy: bool) -> None:
    """ID: ROTATION_FINALIZE_072_conversion_preserves_independent_topology."""
    calls: list[str] = []
    source = _structural_source(rep, lazy=lazy, calls=calls)
    before = source.as_dataset(copy="none")
    target = "matrix" if rep == "quat" else "quat"
    converted = source.to_rep(target)
    ds = converted.as_dataset(copy="none")
    assert calls == []
    assert read_roles(ds)[1:3] == ("sample", ("trial",))
    assert "sample" not in ds["rotation"].dims
    for coord in ("sample", "time", "size", "trial"):
        xr.testing.assert_identical(ds.coords[coord], before.coords[coord])
    assert isinstance(ds["rotation"].data, da.Array) == lazy
    roundtrip = converted.to_rep(rep).as_dataset(copy="none")
    assert calls == []
    xr.testing.assert_allclose(roundtrip["rotation"], before["rotation"])
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("topology", ["sequence", "batch", "core"])
def test_rotation_from_data_accepts_explicit_core_role_options(topology: str) -> None:
    """ID: ROTATION_INGRESS_073_explicit_from_data_roles_are_authoritative."""
    source = _ingress_source(topology, declared_core=False)
    sequence = "sample" if topology == "sequence" else None
    batch = ("trial",) if topology == "batch" else ()
    result = Rotation.from_data(
        source.as_dataset(), sequence_dim=sequence, batch_dims=batch,
        core_dims=("label",), validate=False,
    )
    assert read_roles(result.as_dataset())[1:] == (sequence, batch, ("label",))
