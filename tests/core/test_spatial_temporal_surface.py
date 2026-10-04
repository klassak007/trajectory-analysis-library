from __future__ import annotations

import itertools

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal import AnalysisObject
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.spatial import (
    AO_TEMPORAL_KIND_VALUES,
    Acceleration,
    AngularAcceleration,
    AngularVelocity,
    LinearAcceleration,
    LinearVelocity,
    Pose,
    Position,
    Rotation,
    Velocity,
    differentiate,
    integrate,
    smooth,
)
from tal.spatial.metadata import (
    KINEMATICS_KIND_VALUES,
    get_acceleration_rep,
    get_angular_acceleration_rep,
    get_angular_velocity_rep,
    get_kinematics_kind,
    get_linear_acceleration_rep,
    get_linear_velocity_rep,
    get_position_rep,
    get_velocity_rep,
)

_XYZ = ("x", "y", "z")


def _temporal_vector3_dataset(*, var_name: str, core_dim: str, values: np.ndarray, param: np.ndarray) -> xr.Dataset:
    arr = xr.DataArray(
        values,
        dims=("sample", core_dim),
        coords={"sample": list(range(values.shape[0])), core_dim: list(_XYZ), "time_s": ("sample", param)},
        name=var_name,
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name=var_name),
        sequence_dim="sample",
        core_dims=(core_dim,),
        param_coord="time_s",
        validate=True,
    )
    return ao.as_dataset(copy="none").copy(deep=True)


def _typed_sources() -> dict[str, object]:
    param = np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float64")
    pos = Position(
        _temporal_vector3_dataset(
            var_name="position",
            core_dim="axis",
            values=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype="float64"),
            param=param,
        )
    )
    lv = LinearVelocity(
        _temporal_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            values=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype="float64"),
            param=param,
        )
    )
    av = AngularVelocity(
        _temporal_vector3_dataset(
            var_name="angular_velocity",
            core_dim="angular_axis",
            values=np.asarray([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype="float64"),
            param=param,
        )
    )
    la = LinearAcceleration(
        _temporal_vector3_dataset(
            var_name="linear_acceleration",
            core_dim="linear_axis",
            values=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype="float64"),
            param=param,
        )
    )
    aa = AngularAcceleration(
        _temporal_vector3_dataset(
            var_name="angular_acceleration",
            core_dim="angular_axis",
            values=np.asarray([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype="float64"),
            param=param,
        )
    )
    vel = Velocity.from_linear_angular(lv, av, validate=True)
    acc = Acceleration.from_linear_angular(la, aa, validate=True)
    return {
        "position": pos,
        "linear_velocity": lv,
        "angular_velocity": av,
        "velocity": vel,
        "linear_acceleration": la,
        "angular_acceleration": aa,
        "acceleration": acc,
    }


def _spatial_rep(ds: xr.Dataset) -> str:
    kind = get_kinematics_kind(ds, owner="test")
    if kind is None:
        return get_position_rep(ds, owner="test")
    if kind == "linear_velocity":
        return get_linear_velocity_rep(ds, owner="test")
    if kind == "angular_velocity":
        return get_angular_velocity_rep(ds, owner="test")
    if kind == "velocity":
        return get_velocity_rep(ds, owner="test")
    if kind == "linear_acceleration":
        return get_linear_acceleration_rep(ds, owner="test")
    if kind == "angular_acceleration":
        return get_angular_acceleration_rep(ds, owner="test")
    if kind == "acceleration":
        return get_acceleration_rep(ds, owner="test")
    raise AssertionError(f"unsupported spatial kind for test parity: {kind!r}")


def test_spatial_core_155_d6_ao_temporal_dispatch_uses_single_canonical_allowed_kind_constant() -> None:
    """ID: SPATIAL_CORE_155_d6_ao_temporal_dispatch_uses_single_canonical_allowed_kind_constant."""
    assert AO_TEMPORAL_KIND_VALUES == ("position", *KINEMATICS_KIND_VALUES)


def test_spatial_core_156_d6_ao_temporal_surface_delegates_to_existing_typed_or_family_owners() -> None:
    """ID: SPATIAL_CORE_156_d6_ao_temporal_surface_delegates_to_existing_typed_or_family_owners."""
    sources = _typed_sources()
    for kind, source in sources.items():
        typed_out = smooth(source, validate=True, target_cls=source.__class__)
        assert isinstance(typed_out, source.__class__)
        ao = AnalysisObject._from_validated(source.as_dataset(copy="none"))
        ao_out = smooth(ao, kind=kind, validate=True)
        assert isinstance(ao_out, AnalysisObject)
    diff_expected = {
        "position": LinearVelocity,
        "linear_velocity": LinearAcceleration,
        "angular_velocity": AngularAcceleration,
        "velocity": Acceleration,
    }
    for kind, expected_cls in diff_expected.items():
        out = differentiate(sources[kind], validate=True, target_cls=expected_cls)
        assert isinstance(out, expected_cls)
    int_expected = {
        "linear_velocity": Position,
        "linear_acceleration": LinearVelocity,
        "angular_acceleration": AngularVelocity,
        "acceleration": Velocity,
    }
    for kind, expected_cls in int_expected.items():
        out = integrate(sources[kind], validate=True, target_cls=expected_cls)
        assert isinstance(out, expected_cls)
    assert not hasattr(Velocity, "integrate")
    assert not hasattr(Acceleration, "differentiate")


def test_spatial_hard_173_d6_ao_temporal_kind_validation_fails_closed_on_unsupported_or_ambiguous_kind() -> None:
    """ID: SPATIAL_HARD_173_d6_ao_temporal_kind_validation_fails_closed_on_unsupported_or_ambiguous_kind."""
    src = _typed_sources()["linear_velocity"]
    ao = AnalysisObject._from_validated(src.as_dataset(copy="none"))
    with pytest.raises(ValueError, match="spatial\\.temporal\\.smooth"):
        _ = smooth(ao, validate=True)
    with pytest.raises(ValueError, match="spatial\\.temporal\\.smooth"):
        _ = smooth(ao, kind="rotation", validate=True)
    with pytest.raises(ValueError, match="spatial\\.temporal\\.smooth"):
        _ = smooth(src, kind="angular_velocity", validate=True)
    with pytest.raises(ValueError, match="spatial\\.temporal\\.integrate"):
        _ = integrate(_typed_sources()["position"], validate=True)
    with pytest.raises(ValueError, match="spatial\\.temporal\\.differentiate"):
        _ = differentiate(src, validate=True, target_cls=AngularAcceleration)


def test_spatial_core_159_d6_ao_default_target_cls_none_returns_analysis_object_with_schema_role_continuity() -> None:
    """ID: SPATIAL_CORE_159_d6_ao_default_target_cls_none_returns_analysis_object_with_schema_role_continuity."""
    sources = _typed_sources()
    op_matrix = (
        (
            differentiate,
            {
                "position": LinearVelocity,
                "linear_velocity": LinearAcceleration,
                "angular_velocity": AngularAcceleration,
                "velocity": Acceleration,
            },
        ),
        (
            integrate,
            {
                "linear_velocity": Position,
                "linear_acceleration": LinearVelocity,
                "angular_acceleration": AngularVelocity,
                "acceleration": Velocity,
            },
        ),
        (smooth, {kind: source.__class__ for kind, source in sources.items()}),
    )
    for op, expected in op_matrix:
        for kind, typed_target in expected.items():
            typed_source = sources[kind]
            typed_out = op(typed_source, validate=True, target_cls=typed_target)
            ao_source = AnalysisObject._from_validated(typed_source.as_dataset(copy="none"))
            ao_out = op(ao_source, kind=kind, validate=True, target_cls=None)
            assert type(ao_out) is AnalysisObject
            assert read_roles(ao_out.as_dataset(copy="none")) == read_roles(typed_out.as_dataset(copy="none"))
            assert read_param_coord_name(ao_out.as_dataset(copy="none")) == read_param_coord_name(typed_out.as_dataset(copy="none"))
            assert read_sequence_size_coord_name(ao_out.as_dataset(copy="none")) == read_sequence_size_coord_name(typed_out.as_dataset(copy="none"))
            assert get_kinematics_kind(ao_out.as_dataset(copy="none"), owner="test") == get_kinematics_kind(
                typed_out.as_dataset(copy="none"), owner="test"
            )
            assert _spatial_rep(ao_out.as_dataset(copy="none")) == _spatial_rep(typed_out.as_dataset(copy="none"))
            xr.testing.assert_identical(ao_out.as_dataset(copy="none"), typed_out.as_dataset(copy="none"))


def _typed_query_source(kind, lazy):
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
    raw = xr.Dataset(
        {"quat": (("sample", "q"), quat)},
        coords={
            "sample": [10, 20, 30],
            "q": ["x", "y", "z", "w"],
            "time": ("sample", [0.0, 1.0, 2.0]),
        },
    )
    rotation = Rotation(
        AnalysisObject.from_data(
            raw, sequence_dim="sample", core_dims=("q",), param_coord="time"
        )
    )
    if lazy:
        rotation = Rotation(rotation.as_dataset().chunk({"sample": 1}))
    if kind == "rotation":
        return rotation
    raw = xr.Dataset(
        {"position": (("sample", "axis"), np.tile(np.arange(3.0)[:, None], (1, 3)))},
        coords={
            "sample": [10, 20, 30],
            "axis": ["x", "y", "z"],
            "time": ("sample", [0.0, 1.0, 2.0]),
        },
    )
    position = Position(
        AnalysisObject.from_data(
            raw, sequence_dim="sample", core_dims=("axis",), param_coord="time"
        )
    )
    return Pose.from_components(rotation, position)


@pytest.mark.parametrize("kind", ["rotation", "pose"])
@pytest.mark.parametrize("operation", ["at", "resample_to"])
@pytest.mark.parametrize(
    "lazy_source,validate,labels",
    [(lazy, validate, "lazy") for lazy, validate in itertools.product([False, True], repeat=2)]
    + [(False, True, "eager")],
)
def test_tut_022_typed_query_caller_labels_remain_lazy(
    kind, operation, lazy_source, validate, labels
):
    ao = _typed_query_source(kind, lazy_source)
    values = np.array([[0.2, 0.3], [1.2, 1.3]])
    row = np.array([100, 200])
    if labels == "lazy":
        row = da.from_array(row, chunks=1)
    query = xr.DataArray(
        da.from_array(values, chunks=1),
        dims=("row", "col"),
        coords=xr.Coordinates(
            {
                "row": xr.Variable("row", row),
                "tag": xr.Variable("row", np.array([4, 5])),
            },
            indexes={},
        ),
    )
    before, source_before = query.copy(deep=True), ao.as_dataset()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = getattr(ao.param, operation)(query, validate=validate).as_dataset()
    assert not tasks
    np.testing.assert_array_equal(result.time.compute(), values.ravel())
    np.testing.assert_array_equal(result.row.compute(), [100, 100, 200, 200])
    np.testing.assert_array_equal(result.tag.compute(), [4, 4, 5, 5])
    xr.testing.assert_identical(query, before)
    xr.testing.assert_identical(ao.as_dataset(), source_before)
    assert "row" not in result.xindexes
    assert result.row.dims == result.time.dims == ("sample",)


@pytest.mark.parametrize("kind", ["rotation", "pose"])
@pytest.mark.parametrize("operation", ["at", "resample_to"])
@pytest.mark.parametrize("shape,validate", [((0, 2), False), ((2, 0), True), ((0, 0), True)])
def test_tut_022_empty_typed_queries_keep_lazy_carrier_types(kind, operation, shape, validate):
    """Typed empty products keep reviewed labels and parameter topology without planning work."""
    source = _typed_query_source(kind, True)
    labels = xr.Variable("row", da.from_array(np.arange(shape[0], dtype="int64"), chunks=1), attrs={"meaning": "row"})
    query = xr.DataArray(da.from_array(np.empty(shape), chunks=1), dims=("row", "column"), coords=xr.Coordinates({"row": labels}, indexes={}))
    before, original = source.as_dataset(), query.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = getattr(source.param, operation)(query, validate=validate).as_dataset()
    assert not tasks
    assert result.sizes["sample"] == 0
    assert result.row.dtype == np.dtype("int64")
    assert result.row.attrs == labels.attrs
    assert result.row.dims == result.time.dims == ("sample",)
    assert "row" not in result.xindexes
    xr.testing.assert_identical(source.as_dataset(), before)
    xr.testing.assert_identical(query, original)
