from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal.astro import TopocentricDirection
from tal.core import AnalysisObject, SchemaError
from tal.core.typed_lifecycle import (
    TypedAnalysisObject,
    TypedLifecycleContext,
    TypedLifecycleSpec,
)
from tal.geo import GeodeticPosition, ProjectedPosition
from tal.io import AOZarrReadOptions
from tal.linalg import Array, Matrix, Vector, Vector3
from tal.spatial import (
    Acceleration,
    AngularAcceleration,
    AngularVelocity,
    LinearAcceleration,
    LinearVelocity,
    Pose,
    Position,
    Rotation,
    Velocity,
)

TypedConstructor = Callable[[object], AnalysisObject]
TypedSourceFactory = Callable[[], AnalysisObject]


class _TaskCounter(Callback):
    def __init__(self) -> None:
        self.keys: list[object] = []
        super().__init__(pretask=self._record)

    def _record(self, key: object, _dsk: object, _state: object) -> None:
        self.keys.append(key)


def _analysis_object(
    values: object,
    *,
    dims: tuple[str, ...],
    coords: dict[str, object],
    var_name: str,
    core_dims: tuple[str, ...],
) -> AnalysisObject:
    ds = xr.Dataset({var_name: (dims, values)}, coords=coords)
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        core_dims=core_dims,
        validate=True,
    )


def _scalar_source(*, lazy: bool = False) -> AnalysisObject:
    values: object = da.arange(2, chunks=1) if lazy else np.arange(2.0)
    return _analysis_object(
        values,
        dims=("sample",),
        coords={"sample": [0, 1]},
        var_name="value",
        core_dims=(),
    )


def _vector_source(
    *,
    var_name: str = "value",
    core_dim: str = "axis",
    labels: tuple[str, ...] = ("x", "y", "z"),
    lazy: bool = False,
) -> AnalysisObject:
    eager = np.arange(2 * len(labels), dtype=float).reshape(2, len(labels))
    values: object = (
        da.from_array(eager, chunks=(1, len(labels))) if lazy else eager
    )
    return _analysis_object(
        values,
        dims=("sample", core_dim),
        coords={"sample": [0, 1], core_dim: list(labels)},
        var_name=var_name,
        core_dims=(core_dim,),
    )


def _matrix_source() -> AnalysisObject:
    return _analysis_object(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("sample", "row", "col"),
        coords={"sample": [0, 1], "row": ["r0", "r1"], "col": ["c0", "c1"]},
        var_name="value",
        core_dims=("row", "col"),
    )


def _rotation_source() -> AnalysisObject:
    values = np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )
    return _analysis_object(
        values,
        dims=("sample", "quat"),
        coords={"sample": [0, 1], "quat": ["x", "y", "z", "w"]},
        var_name="rotation",
        core_dims=("quat",),
    )


def _generic_source(value: AnalysisObject) -> AnalysisObject:
    return AnalysisObject._from_validated(value.as_dataset(copy="deep"))


def _velocity_source() -> AnalysisObject:
    linear = LinearVelocity(
        _vector_source(var_name="linear_velocity", core_dim="linear_axis")
    )
    angular = AngularVelocity(
        _vector_source(var_name="angular_velocity", core_dim="angular_axis")
    )
    return _generic_source(Velocity.from_linear_angular(linear, angular))


def _acceleration_source() -> AnalysisObject:
    linear = LinearAcceleration(
        _vector_source(var_name="linear_acceleration", core_dim="linear_axis")
    )
    angular = AngularAcceleration(
        _vector_source(var_name="angular_acceleration", core_dim="angular_axis")
    )
    return _generic_source(Acceleration.from_linear_angular(linear, angular))


def _pose_source() -> AnalysisObject:
    pose = Pose.from_components(
        Rotation(_rotation_source()),
        Position(_vector_source(var_name="position")),
    )
    return _generic_source(pose)


def _projected_source() -> AnalysisObject:
    source = _vector_source(
        var_name="position",
        core_dim="projected",
        labels=("easting", "northing"),
    )
    return _generic_source(
        ProjectedPosition.from_projected(source, crs="EPSG:32611")
    )


_PROMOTION_CASES: dict[str, tuple[TypedConstructor, TypedSourceFactory]] = {
    "array": (Array, _scalar_source),
    "vector": (Vector, _vector_source),
    "matrix": (Matrix, _matrix_source),
    "vector3": (Vector3, _vector_source),
    "position": (Position, lambda: _vector_source(var_name="position")),
    "rotation": (Rotation, _rotation_source),
    "linear_velocity": (
        LinearVelocity,
        lambda: _vector_source(var_name="linear_velocity"),
    ),
    "angular_velocity": (
        AngularVelocity,
        lambda: _vector_source(var_name="angular_velocity"),
    ),
    "linear_acceleration": (
        LinearAcceleration,
        lambda: _vector_source(var_name="linear_acceleration"),
    ),
    "angular_acceleration": (
        AngularAcceleration,
        lambda: _vector_source(var_name="angular_acceleration"),
    ),
    "geodetic": (
        GeodeticPosition,
        lambda: _vector_source(
            var_name="position",
            core_dim="lla",
            labels=("lat", "lon", "alt"),
        ),
    ),
    "topocentric": (
        TopocentricDirection,
        lambda: _vector_source(
            var_name="direction",
            core_dim="enu",
            labels=("east", "north", "up"),
        ),
    ),
    "velocity": (Velocity, _velocity_source),
    "acceleration": (Acceleration, _acceleration_source),
    "pose": (Pose, _pose_source),
    "projected": (ProjectedPosition, _projected_source),
}


def _promotion_case(name: str) -> tuple[TypedConstructor, AnalysisObject]:
    constructor, source_factory = _PROMOTION_CASES[name]
    return constructor, source_factory()


def _decorate_owned_metadata(ao: AnalysisObject) -> None:
    ds = ao.as_dataset(copy="none")
    ds.attrs["nested"] = {"items": ["dataset"]}
    ds.encoding["nested"] = {"items": ["dataset-encoding"]}
    for name in ds.variables:
        ds[name].attrs["nested"] = {"items": [f"{name}-attrs"]}
        ds[name].encoding["nested"] = {"items": [f"{name}-encoding"]}


@pytest.mark.parametrize(
    "family",
    (
        "array",
        "vector",
        "matrix",
        "vector3",
        "position",
        "rotation",
        "pose",
        "linear_velocity",
        "angular_velocity",
        "velocity",
        "linear_acceleration",
        "angular_acceleration",
        "acceleration",
        "geodetic",
        "projected",
        "topocentric",
    ),
)
def test_typed_ownership_001_all_constructor_families_share_owned_payloads(
    family: str,
) -> None:
    """ID: TYPED_OWNERSHIP_001_all_constructor_families_share_owned_payloads."""
    constructor, source = _promotion_case(family)
    _decorate_owned_metadata(source)
    source_ds = source.as_dataset(copy="none")
    before = source.as_dataset(copy="deep")

    promoted = constructor(source)
    promoted_ds = promoted.as_dataset(copy="none")

    xr.testing.assert_identical(source_ds, before)
    assert promoted_ds is not source_ds
    for name in source_ds.data_vars:
        assert promoted_ds.variables[name] is not source_ds.variables[name]
        assert np.shares_memory(promoted_ds[name].data, source_ds[name].data)
    for name in source_ds.coords:
        assert promoted_ds.variables[name] is not source_ds.variables[name]
        source_data = source_ds.coords[name].data
        if np.shares_memory(source_data, source_ds.coords[name].data):
            assert np.shares_memory(promoted_ds.coords[name].data, source_data)
    for name, source_index in source_ds.xindexes.items():
        assert promoted_ds.xindexes[name] is not source_index
        assert type(promoted_ds.xindexes[name]) is type(source_index)
        assert promoted_ds.xindexes[name].equals(source_index)
    assert promoted_ds.attrs["nested"] is not source_ds.attrs["nested"]
    promoted_ds.attrs["nested"]["items"].append("promoted")
    first_var = next(iter(source_ds.data_vars))
    promoted_ds[first_var].encoding["nested"]["items"].append("promoted")
    first_coord = next(iter(source_ds.coords))
    promoted_ds[first_coord].attrs["nested"]["items"].append("promoted")
    assert source_ds.attrs["nested"]["items"] == ["dataset"]
    assert source_ds[first_var].encoding["nested"]["items"] == [
        f"{first_var}-encoding"
    ]
    assert source_ds[first_coord].attrs["nested"]["items"] == [
        f"{first_coord}-attrs"
    ]
    if family == "topocentric":
        assert set(promoted_ds.data_vars) - set(source_ds.data_vars) == {
            "altitude_deg",
            "azimuth_deg",
        }


@pytest.mark.parametrize("kind", ("dataset", "dataarray"))
def test_typed_ownership_002_external_ingress_isolated_before_promotion(
    kind: str,
) -> None:
    """ID: TYPED_OWNERSHIP_002_external_ingress_isolated_before_promotion."""
    source = xr.Dataset(
        {"value": ("sample", np.arange(3.0))},
        coords={"sample": np.arange(3)},
        attrs={"nested": {"items": ["external"]}},
    )
    input_value: xr.Dataset | xr.DataArray = source
    if kind == "dataarray":
        input_value = source["value"]
        input_value.attrs["nested"] = {"items": ["external"]}
    captured: list[AnalysisObject] = []

    def coerce(value: object, ctx: TypedLifecycleContext) -> AnalysisObject:
        from tal.core.typed_lifecycle import default_coerce_source

        owned = default_coerce_source(value, ctx)
        captured.append(owned)
        return owned

    class Probe(TypedAnalysisObject):
        LIFECYCLE = TypedLifecycleSpec(
            type_name="Probe",
            owner_prefix="typed.probe",
            coerce_source=coerce,
        )

    promoted = Probe(input_value)
    owned_ds = captured[0].as_dataset(copy="none")
    promoted_ds = promoted.as_dataset(copy="none")
    assert not np.shares_memory(owned_ds["value"].data, source["value"].data)
    assert np.shares_memory(promoted_ds["value"].data, owned_ds["value"].data)
    source["value"].data[0] = 99.0
    assert float(promoted_ds["value"].data[0]) == 0.0
    if kind == "dataset":
        source.attrs["nested"]["items"].append("changed")
        assert promoted_ds.attrs["nested"]["items"] == ["external"]
    else:
        input_value.attrs["nested"]["items"].append("changed")
        assert promoted_ds["value"].attrs["nested"]["items"] == ["external"]


@pytest.mark.parametrize("family", ("array", "position"))
def test_typed_ownership_003_dask_promotion_is_lazy_and_graph_preserving(
    family: str,
) -> None:
    """ID: TYPED_OWNERSHIP_003_dask_promotion_is_lazy_and_graph_preserving."""
    constructor: TypedConstructor = Array
    source = _scalar_source(lazy=True)
    var_name = "value"
    if family == "position":
        constructor = Position
        source = _vector_source(var_name="position", lazy=True)
        var_name = "position"
    source_data = source.as_dataset(copy="none")[var_name].data
    counter = _TaskCounter()

    with counter:
        promoted = constructor(source)

    promoted_data = promoted.as_dataset(copy="none")[var_name].data
    assert counter.keys == []
    assert isinstance(promoted_data, da.Array)
    assert promoted_data is source_data
    assert promoted_data.dask is source_data.dask


def test_typed_ownership_009_normalizing_dask_promotion_adds_only_derived_graphs() -> None:
    """ID: TYPED_OWNERSHIP_009_normalizing_dask_promotion_adds_only_derived_graphs."""
    source = _vector_source(
        var_name="direction",
        core_dim="enu",
        labels=("east", "north", "up"),
        lazy=True,
    )
    source_data = source.as_dataset(copy="none")["direction"].data
    source_keys = set(source_data.__dask_graph__())
    counter = _TaskCounter()

    with counter:
        promoted = TopocentricDirection(source)

    promoted_ds = promoted.as_dataset(copy="none")
    assert counter.keys == []
    assert promoted_ds["direction"].data is source_data
    assert set(promoted_ds.data_vars) == {
        "direction",
        "altitude_deg",
        "azimuth_deg",
    }
    for name in ("altitude_deg", "azimuth_deg"):
        derived = promoted_ds[name].data
        assert isinstance(derived, da.Array)
        assert source_keys <= set(derived.__dask_graph__())


@pytest.mark.parametrize("close_first", ("source", "typed"))
def test_typed_ownership_004_coupled_resource_closes_once(
    close_first: str,
) -> None:
    """ID: TYPED_OWNERSHIP_004_coupled_resource_closes_once."""
    source = _scalar_source()
    source_ds = source.as_dataset(copy="none")
    closed: list[str] = []
    source_ds.set_close(lambda: closed.append("backend"))
    first = Array(source)
    second = Array(source)
    deep = first.as_dataset(copy="deep")
    shallow = first.as_dataset(copy="shallow")
    deep.close()
    shallow.close()
    assert closed == []

    if close_first == "source":
        source.close()
        first.close()
    else:
        first.as_dataset(copy="none").close()
        source.close()
    second.close()
    first.close()
    assert closed == ["backend"]


@pytest.mark.parametrize("topology", ("fanout", "chain"))
def test_typed_ownership_010_deep_promotion_graph_closes_without_recursion(
    topology: str,
) -> None:
    """ID: TYPED_OWNERSHIP_010_deep_promotion_graph_closes_without_recursion."""
    source = _scalar_source()
    closed: list[str] = []
    source.as_dataset(copy="none").set_close(lambda: closed.append("backend"))
    owner: AnalysisObject = source
    aliases: list[AnalysisObject] = []
    for _ in range(1_100):
        owner = Array(source if topology == "fanout" else owner)
        aliases.append(owner)

    owner.close()
    source.close()
    for alias in aliases:
        alias.close()
    assert closed == ["backend"]


def test_typed_ownership_008_lazy_zarr_promotion_uses_coupled_lifetime(
    tmp_path: Path,
) -> None:
    """ID: TYPED_OWNERSHIP_008_lazy_zarr_promotion_uses_coupled_lifetime."""
    store = tmp_path / "typed-promotion.zarr"
    _scalar_source().io.to_zarr(str(store))
    source = AnalysisObject.from_zarr(
        str(store),
        opts=AOZarrReadOptions(chunks={}),
    )
    source_ds = source.as_dataset(copy="none")
    backend_close = getattr(source_ds, "_close", None)
    assert backend_close is not None
    closed: list[str] = []

    def close_backend() -> None:
        closed.append("backend")
        backend_close()

    source_ds.set_close(close_backend)
    promoted = Array(source)
    promoted_ds = promoted.as_dataset(copy="none")
    assert isinstance(promoted_ds["value"].data, da.Array)
    np.testing.assert_array_equal(promoted_ds["value"].compute(), [0, 1])

    promoted.close()
    source.close()
    assert closed == ["backend"]


def test_typed_ownership_005_lifecycle_failure_leaves_source_open() -> None:
    """ID: TYPED_OWNERSHIP_005_lifecycle_failure_leaves_source_open."""
    source = _scalar_source()
    closed: list[str] = []
    source.as_dataset(copy="none").set_close(lambda: closed.append("backend"))

    with pytest.raises(ValueError, match="Vector requires exactly one core dim"):
        Vector(source)

    assert closed == []
    source.close()
    assert closed == ["backend"]


def test_typed_ownership_006_existing_target_cleanup_remains_primary() -> None:
    """ID: TYPED_OWNERSHIP_006_existing_target_cleanup_remains_primary."""
    events: list[str] = []
    source = _scalar_source()

    def close_source() -> None:
        events.append("source")
        raise RuntimeError("secondary")

    source.as_dataset(copy="none").set_close(close_source)

    def normalize(ds: xr.Dataset, _ctx: TypedLifecycleContext) -> xr.Dataset:
        def close_target() -> None:
            events.append("target")
            raise ValueError("primary")

        ds.set_close(close_target)
        return ds

    class Probe(TypedAnalysisObject):
        LIFECYCLE = TypedLifecycleSpec(
            type_name="Probe",
            owner_prefix="typed.probe",
            normalize=normalize,
        )

    promoted = Probe(source)
    with pytest.raises(ValueError, match="primary"):
        promoted.close()
    promoted.close()
    source.close()
    assert events == ["target", "source"]


def test_typed_ownership_007_schema_failure_retains_owner_and_source() -> None:
    """ID: TYPED_OWNERSHIP_007_schema_failure_retains_owner_and_source."""
    ds = _scalar_source().as_dataset(copy="deep")
    ds.attrs["tal"] = deepcopy(ds.attrs["tal"])
    ds.attrs["tal"]["version"] = True
    source = AnalysisObject._from_unvalidated(ds, schema_prepared=True)
    closed: list[str] = []
    source.as_dataset(copy="none").set_close(lambda: closed.append("backend"))

    with pytest.raises(SchemaError) as error:
        Array(source)

    assert error.value.code == "schema.version.invalid"
    assert error.value.path == "tal.version"
    assert closed == []
    source.close()
    assert closed == ["backend"]
