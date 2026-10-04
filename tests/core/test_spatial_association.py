from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation as SciRotation

from tal import AnalysisObject, ufuncs
from tal.core import (
    BatchConcatOptions,
    ComponentPatchOptions,
    ComponentRegistryOptions,
    ComponentSpec,
    CoreConcatOptions,
    CoreOverlayOptions,
    MergeOptions,
    SequenceConcatOptions,
    define_components,
    patch_components,
)
from tal.core.schema import UNSET
from tal.frames import FrameGraph
from tal.spatial import (
    Acceleration,
    AngularAcceleration,
    AngularVelocity,
    LinearAcceleration,
    LinearVelocity,
    PathSolveOptions,
    Pose,
    Position,
    Rotation,
    Velocity,
    bind_pose,
    solve_pose_path_transform,
    solve_rotation_path_transform,
)
from tal.spatial.metadata import get_expressed_in, set_expressed_in
from tal.utils.frame_schema import get_frames, set_frames

_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")


def _vector_ao(name: str, *, core_dim: str = "axis") -> AnalysisObject:
    data = xr.DataArray(
        [[1.0, 2.0, 3.0]],
        dims=("sample", core_dim),
        coords={"sample": [0], core_dim: list(_XYZ)},
        name=name,
    )
    return AnalysisObject.from_data(
        data,
        sequence_dim="sample",
        core_dims=(core_dim,),
    )


def _rotation() -> Rotation:
    data = xr.DataArray(
        [[0.0, 0.0, 0.0, 1.0]],
        dims=("sample", "quat"),
        coords={"sample": [0], "quat": list(_QUAT)},
        name="rotation",
    )
    return Rotation(
        AnalysisObject.from_data(
            data,
            sequence_dim="sample",
            core_dims=("quat",),
        )
    )


def _velocity() -> Velocity:
    linear = LinearVelocity(_vector_ao("linear_velocity", core_dim="linear_axis"))
    angular = AngularVelocity(_vector_ao("angular_velocity", core_dim="angular_axis"))
    return Velocity.from_linear_angular(linear, angular)


def _acceleration() -> Acceleration:
    linear = LinearAcceleration(
        _vector_ao("linear_acceleration", core_dim="linear_acceleration_axis")
    )
    angular = AngularAcceleration(
        _vector_ao("angular_acceleration", core_dim="angular_acceleration_axis")
    )
    return Acceleration.from_linear_angular(linear, angular)


def _pose() -> Pose:
    return Pose.from_components(_rotation(), Position(_vector_ao("position")))


def _component_factory_inputs(kind: str) -> tuple[object, object, object]:
    if kind == "pose":
        return Pose.from_components, _rotation(), Position(_vector_ao("position"))
    if kind == "velocity":
        linear = LinearVelocity(_vector_ao("linear_velocity", core_dim="linear_axis"))
        angular = AngularVelocity(_vector_ao("angular_velocity", core_dim="angular_axis"))
        return Velocity.from_linear_angular, linear, angular
    linear = LinearAcceleration(
        _vector_ao("linear_acceleration", core_dim="linear_acceleration_axis")
    )
    angular = AngularAcceleration(
        _vector_ao("angular_acceleration", core_dim="angular_acceleration_axis")
    )
    return Acceleration.from_linear_angular, linear, angular


def _with_frame_declarations(
    value: object,
    *,
    parent: str | None,
    child: str | None,
    expressed_in: str | None,
) -> object:
    ds = set_frames(
        value.as_dataset(copy="none"),
        parent=parent,
        child=child,
        validate=False,
    )
    ds = set_expressed_in(
        ds,
        expressed_in=expressed_in,
        validate=False,
        owner="test",
    )
    return type(value)(ds)


def _registered_position(graph: FrameGraph | None = None) -> Position:
    position = Position(_vector_ao("position"), graph=graph)
    return define_components(
        position,
        opts=ComponentRegistryOptions(
            {"xyz": ComponentSpec(core_dim="axis", labels=_XYZ)}
        ),
    )


class _RepeatedPatchMapping(dict[str, object]):
    def __init__(self, items: list[tuple[str, object]]) -> None:
        super().__init__(items)
        self._items = items

    def items(self) -> list[tuple[str, object]]:
        return list(self._items)


def _with_effective_basis(value, basis: str):
    ds = set_expressed_in(
        value.as_dataset(copy="deep"),
        expressed_in=basis,
        validate=False,
        owner="test",
    )
    return type(value)(ds)


def _explicit_basis_is_present(value: object) -> bool:
    tal = value.as_dataset(copy="none").attrs["tal"]
    relation = tal.get("ext", {}).get("spatial", {}).get("relation", {})
    return "expressed_in" in relation


def _assert_partial_inverse_preflight(source, *, graph: FrameGraph) -> None:
    from dask.callbacks import Callback

    for represented in (source, source.as_matrix(validate=False)):
        before = represented.as_dataset(copy="deep")
        partials = (
            type(represented)(represented, parent=None),
            type(represented)(represented, parent=None, child=None),
            type(represented)(represented, parent=None, expressed_in=None),
        )
        for partial in partials:
            lazy = type(partial)(
                partial.as_dataset(copy="none").chunk({"sample": 1}),
                graph=graph,
            )
            tasks: list[object] = []
            with (
                Callback(pretask=lambda key, *_, tasks=tasks: tasks.append(key)),
                pytest.raises(ValueError, match="inverse requires a parent frame"),
            ):
                lazy.inverse(validate=False)
            assert tasks == []

        unframed = type(represented)(
            represented,
            parent=None,
            child=None,
            expressed_in=None,
            graph=None,
        )
        result = unframed.inverse(validate=False)
        assert result.graph is None
        assert get_frames(result.as_dataset(copy="none")) == (None, None)
        assert get_expressed_in(result.as_dataset(copy="none"), owner="test") is None
        xr.testing.assert_identical(represented.as_dataset(copy="none"), before)


def _factory_case(name: str):
    if name == "pose":
        return Pose.from_matrix, _pose().as_matrix(validate=False)
    if name == "velocity":
        return Velocity.from_vector6, _velocity().as_vector6(validate=False)
    return Acceleration.from_vector6, _acceleration().as_vector6(validate=False)


def _spatial_sources() -> tuple[object, ...]:
    return (
        Position(_vector_ao("position")),
        _rotation(),
        _pose(),
        LinearVelocity(_vector_ao("linear_velocity")),
        AngularVelocity(_vector_ao("angular_velocity")),
        _velocity(),
        LinearAcceleration(_vector_ao("linear_acceleration")),
        AngularAcceleration(_vector_ao("angular_acceleration")),
        _acceleration(),
    )


@pytest.mark.parametrize("source", _spatial_sources(), ids=lambda value: type(value).__name__)
def test_spatial_core_129b_001_direct_constructors_share_frame_plan(source) -> None:
    """ID: SPATIAL_CORE_129B_001_direct_constructors_share_frame_plan."""
    graph = FrameGraph()
    result = type(source)(
        source,
        parent="world",
        child="body",
        expressed_in="world",
        graph=graph,
    )
    assert result.graph is graph
    assert get_frames(result.as_dataset(copy="none")) == ("world", "body")
    assert get_expressed_in(result.as_dataset(copy="none"), owner="test") == "world"
    assert source.graph is None
    assert get_frames(source.as_dataset(copy="none")) == (None, None)
    assert graph.get_frame("world") is None
    assert graph.get_frame("body") is None


def test_spatial_core_129b_002_construction_clears_and_rejects_conflicts() -> None:
    """ID: SPATIAL_CORE_129B_002_construction_clears_and_rejects_conflicts."""
    graph = FrameGraph()
    source = Position(
        _vector_ao("position"),
        parent="world",
        child="body",
        expressed_in="world",
        graph=graph,
    )
    cleared = Position(
        source,
        parent=None,
        child=None,
        expressed_in=None,
        graph=None,
    )
    assert cleared.graph is None
    assert get_frames(cleared.as_dataset(copy="none")) == (None, None)
    assert get_expressed_in(cleared.as_dataset(copy="none"), owner="test") is None
    with pytest.raises(ValueError, match="parent='camera'.*existing parent='world'"):
        Position(source, parent="camera")
    with pytest.raises(ValueError, match="expressed_in='camera'.*effective basis 'world'"):
        Position(source, expressed_in="camera")
    with pytest.raises(TypeError, match="spatial.position.__init__: parent"):
        Position(object(), parent=1)
    with pytest.raises(TypeError, match="spatial.position.__init__: graph"):
        Position(source, graph=object())


def test_spatial_core_129b_003_factories_share_construction_and_association() -> None:
    """ID: SPATIAL_CORE_129B_003_factories_share_construction_and_association."""
    graph = FrameGraph()
    position = Position(_vector_ao("position"), graph=graph)
    rotation = Rotation(_rotation(), graph=graph)
    pose = Pose.from_components(
        rotation,
        position,
        parent="world",
        child="body",
    )
    velocity = Velocity.from_linear_angular(
        LinearVelocity(_vector_ao("linear_velocity", core_dim="linear_axis"), graph=graph),
        AngularVelocity(_vector_ao("angular_velocity", core_dim="angular_axis"), graph=graph),
        parent="world",
    )
    acceleration = Acceleration.from_linear_angular(
        LinearAcceleration(
            _vector_ao("linear_acceleration", core_dim="linear_acceleration_axis"),
            graph=graph,
        ),
        AngularAcceleration(
            _vector_ao("angular_acceleration", core_dim="angular_acceleration_axis"),
            graph=graph,
        ),
        parent="world",
    )
    assert pose.graph is velocity.graph is acceleration.graph is graph
    assert get_frames(pose.as_dataset(copy="none")) == ("world", "body")
    assert get_frames(velocity.as_dataset(copy="none"))[0] == "world"
    assert get_frames(acceleration.as_dataset(copy="none"))[0] == "world"

    matrix_pose = Pose.from_matrix(
        pose.as_matrix().as_dataset(copy="none"),
        graph=graph,
    )
    vector_velocity = Velocity.from_vector6(
        velocity.to_rep("vector6").as_dataset(copy="none"),
        graph=graph,
    )
    vector_acceleration = Acceleration.from_vector6(
        acceleration.to_rep("vector6").as_dataset(copy="none"),
        graph=graph,
    )
    assert matrix_pose.graph is graph
    assert vector_velocity.graph is graph
    assert vector_acceleration.graph is graph


@pytest.mark.parametrize("kind", ("pose", "velocity", "acceleration"))
@pytest.mark.parametrize(
    ("left_frames", "right_frames", "overrides", "message"),
    (
        (
            ("world", "body", "world"),
            ("map", "body", "map"),
            {"parent": None, "expressed_in": None},
            "frame tags must match exactly for parent",
        ),
        (
            ("world", "body", "world"),
            ("world", "tool", "world"),
            {"child": None},
            "frame tags must match exactly for child",
        ),
        (
            ("world", "body", "map"),
            ("world", "body", "camera"),
            {"expressed_in": None},
            "different effective expressed_in bases",
        ),
    ),
)
def test_spatial_hard_129b_035_output_clearing_does_not_hide_source_frame_conflicts(
    kind: str,
    left_frames: tuple[str, str, str],
    right_frames: tuple[str, str, str],
    overrides: dict[str, object],
    message: str,
) -> None:
    """ID: SPATIAL_HARD_129B_035_output_clearing_does_not_hide_source_frame_conflicts."""
    from dask.callbacks import Callback

    factory, left_source, right_source = _component_factory_inputs(kind)
    left = _with_frame_declarations(
        left_source,
        parent=left_frames[0],
        child=left_frames[1],
        expressed_in=left_frames[2],
    )
    right = _with_frame_declarations(
        right_source,
        parent=right_frames[0],
        child=right_frames[1],
        expressed_in=right_frames[2],
    )
    left = type(left)(left.as_dataset(copy="none").chunk({"sample": 1}))
    right = type(right)(right.as_dataset(copy="none").chunk({"sample": 1}))
    left_before = left.as_dataset(copy="deep")
    right_before = right.as_dataset(copy="deep")
    tasks: list[object] = []

    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        pytest.raises(ValueError, match=message),
    ):
        factory(left, right, graph=FrameGraph(), **overrides)

    assert tasks == []
    xr.testing.assert_identical(left.as_dataset(copy="none"), left_before)
    xr.testing.assert_identical(right.as_dataset(copy="none"), right_before)


@pytest.mark.parametrize("kind", ("pose", "velocity", "acceleration"))
@pytest.mark.parametrize(
    (
        "right_frames",
        "overrides",
        "expected_frames",
        "expected_basis",
        "explicit_basis",
    ),
    (
        (
            ("world", "body", "world"),
            {"parent": None},
            (None, "body"),
            "world",
            True,
        ),
        (
            ("world", "body", "world"),
            {"child": None},
            ("world", None),
            "world",
            False,
        ),
        (
            ("world", "body", "world"),
            {"expressed_in": None},
            ("world", "body"),
            "world",
            False,
        ),
        (
            ("world", "body", "world"),
            {"parent": None, "child": None, "expressed_in": None},
            (None, None),
            None,
            False,
        ),
        (
            (None, None, None),
            {},
            ("world", "body"),
            "world",
            False,
        ),
    ),
    ids=(
        "clear-parent",
        "clear-child",
        "clear-explicit-basis",
        "clear-all",
        "one-framed-input",
    ),
)
def test_spatial_core_129b_038_factory_output_clearing_success_matrix(
    kind: str,
    right_frames: tuple[str | None, str | None, str | None],
    overrides: dict[str, object],
    expected_frames: tuple[str | None, str | None],
    expected_basis: str | None,
    explicit_basis: bool,
) -> None:
    """ID: SPATIAL_CORE_129B_038_factory_output_clearing_success_matrix."""
    from dask.callbacks import Callback

    factory, left_source, right_source = _component_factory_inputs(kind)
    graph = FrameGraph()
    left = _with_frame_declarations(
        left_source,
        parent="world",
        child="body",
        expressed_in="world",
    )
    right = _with_frame_declarations(
        right_source,
        parent=right_frames[0],
        child=right_frames[1],
        expressed_in=right_frames[2],
    )
    left = type(left)(left.as_dataset(copy="none").chunk({"sample": 1}), graph=graph)
    right = type(right)(right.as_dataset(copy="none").chunk({"sample": 1}))
    left_before = left.as_dataset(copy="deep")
    right_before = right.as_dataset(copy="deep")
    tasks: list[object] = []

    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = factory(left, right, **overrides)

    result_ds = result.as_dataset(copy="none")
    assert tasks == []
    assert result.graph is graph
    assert get_frames(result_ds) == expected_frames
    assert get_expressed_in(result_ds, owner="test") == expected_basis
    assert _explicit_basis_is_present(result) is explicit_basis
    xr.testing.assert_identical(left.as_dataset(copy="none"), left_before)
    xr.testing.assert_identical(right.as_dataset(copy="none"), right_before)


def test_spatial_core_129b_036_component_patch_resolves_all_associations() -> None:
    """ID: SPATIAL_CORE_129B_036_component_patch_resolves_all_associations."""
    graph = FrameGraph()
    base = _registered_position()
    patch = Position(_vector_ao("position"), graph=graph)
    options = ComponentPatchOptions(on_overlap="replace", output_var="patched")

    unassociated = patch_components(
        base,
        {"xyz": Position(_vector_ao("position"))},
        opts=ComponentPatchOptions(on_overlap="replace"),
    )
    right_only = patch_components(base, {"xyz": patch}, opts=options)
    shared = patch_components(
        base.with_graph(graph),
        {"xyz": patch},
        opts=ComponentPatchOptions(on_overlap="replace"),
    )
    generic_base = AnalysisObject(base.as_dataset(copy="none"))
    demoted = patch_components(generic_base, {"xyz": patch}, opts=options)

    assert isinstance(unassociated, Position) and unassociated.graph is None
    assert isinstance(right_only, Position) and right_only.graph is graph
    assert "patched" in right_only.as_dataset(copy="none")
    assert isinstance(shared, Position) and shared.graph is graph
    assert type(demoted) is AnalysisObject
    assert not hasattr(demoted, "graph")

    velocity_base = _velocity()
    velocity_patch = Velocity(_velocity(), graph=graph)
    velocity_result = patch_components(
        velocity_base,
        {"linear": velocity_patch},
        opts=ComponentPatchOptions(on_overlap="replace"),
    )
    assert isinstance(velocity_result, Velocity)
    assert velocity_result.graph is graph
    xr.testing.assert_identical(
        velocity_result.as_dataset(copy="none")["angular_velocity"],
        velocity_base.as_dataset(copy="none")["angular_velocity"],
    )


def test_spatial_perf_129b_037_component_patch_conflict_precedes_lazy_work() -> None:
    """ID: SPATIAL_PERF_129B_037_component_patch_conflict_precedes_lazy_work."""
    from dask.callbacks import Callback

    first = FrameGraph()
    second = FrameGraph()
    base = _registered_position(first)
    left_patch = Position(_vector_ao("position"), graph=first)
    right_patch = Position(_vector_ao("position"), graph=second)
    base = Position(base.as_dataset(copy="none").chunk({"sample": 1}), graph=first)
    right_patch = Position(
        right_patch.as_dataset(copy="none").chunk({"sample": 1}),
        graph=second,
    )
    velocity_base = Velocity(
        _velocity().as_dataset(copy="none").chunk({"sample": 1}),
        graph=first,
    )
    velocity_patch = Velocity(
        _velocity().as_dataset(copy="none").chunk({"sample": 1}),
        graph=second,
    )
    tasks: list[object] = []

    with Callback(pretask=lambda key, *_: tasks.append(key)):
        with pytest.raises(ValueError, match="components.patch:.*different FrameGraph"):
            patch_components(
                base,
                {"xyz": right_patch},
                opts=ComponentPatchOptions(on_overlap="replace"),
            )
        with pytest.raises(ValueError, match="components.patch:.*different FrameGraph"):
            patch_components(
                _registered_position(),
                _RepeatedPatchMapping(
                    [("xyz", left_patch), ("xyz", right_patch)]
                ),
                opts=ComponentPatchOptions(on_overlap="replace"),
            )
        with pytest.raises(ValueError, match="components.patch:.*different FrameGraph"):
            patch_components(
                velocity_base,
                {"linear": velocity_patch},
                opts=ComponentPatchOptions(on_overlap="replace"),
            )

    assert tasks == []


def test_spatial_core_129b_004_with_graph_isolates_metadata_and_shares_payload() -> None:
    """ID: SPATIAL_CORE_129B_004_with_graph_isolates_metadata_and_shares_payload."""
    first = FrameGraph()
    second = FrameGraph()
    source = Position(_vector_ao("position"), graph=first)
    source_ds = source.as_dataset(copy="none")
    source_ds.attrs["nested"] = {"items": [1]}
    result = source.with_graph(second)
    result_ds = result.as_dataset(copy="none")
    assert result is not source
    assert result.graph is second
    assert source.graph is first
    assert result_ds is not source_ds
    assert result_ds["position"].variable is not source_ds["position"].variable
    assert np.shares_memory(result_ds["position"].data, source_ds["position"].data)
    result_ds.attrs["nested"]["items"].append(2)
    assert source_ds.attrs["nested"] == {"items": [1]}
    with pytest.raises(AttributeError):
        result.graph = first


def test_spatial_core_129b_005_single_and_cross_type_results_propagate() -> None:
    """ID: SPATIAL_CORE_129B_005_single_and_cross_type_results_propagate."""
    graph = FrameGraph()
    pose = Pose(_pose(), parent="world", child="body", graph=graph)
    selected = pose.isel(sample=slice(None)).a()
    position, rotation = pose.decompose()
    assert selected.graph is graph
    assert pose.as_matrix().graph is graph
    assert pose.inverse().graph is graph
    assert position.graph is graph
    assert rotation.graph is graph
    assert _velocity().with_graph(graph).linear().graph is graph
    assert _acceleration().with_graph(graph).angular().graph is graph
    assert ufuncs.negative(position).graph is graph

    param_data = xr.DataArray(
        [[1.0, 2.0, 3.0]],
        dims=("sample", "axis"),
        coords={
            "sample": [0],
            "time": ("sample", [0.0]),
            "axis": list(_XYZ),
        },
        name="position",
    )
    param_position = Position(
        AnalysisObject.from_data(
            param_data,
            sequence_dim="sample",
            core_dims=("axis",),
            param_coord="time",
        ),
        graph=graph,
    )
    assert param_position.param.at([0.0]).graph is graph

    point = Position(_vector_ao("position"), graph=graph)
    identity = Rotation(_rotation(), graph=graph)
    assert identity.apply(point).graph is graph
    assert pose.apply(point).graph is graph
    assert identity.compose(Rotation(_rotation(), graph=graph)).graph is graph


def test_spatial_hard_129b_006_ordinary_multi_input_graph_conflicts_fail() -> None:
    """ID: SPATIAL_HARD_129B_006_ordinary_multi_input_graph_conflicts_fail."""
    first = FrameGraph()
    second = FrameGraph()
    left = Position(_vector_ao("position"), graph=first)
    right = Position(_vector_ao("position"), graph=second)
    with pytest.raises(ValueError, match="different FrameGraph instances"):
        _ = left + right
    rotation = Rotation(_rotation(), graph=second)
    with pytest.raises(ValueError, match="different FrameGraph instances"):
        rotation.apply(left)
    overridden = Pose.from_components(
        Rotation(_rotation(), graph=first),
        Position(_vector_ao("position"), graph=second),
        graph=first,
    )
    assert overridden.graph is first


def test_spatial_core_129b_007_dataset_roundtrip_loses_association() -> None:
    """ID: SPATIAL_CORE_129B_007_dataset_roundtrip_loses_association."""
    source = Position(_vector_ao("position"), graph=FrameGraph())
    assert Position(source.as_dataset()).graph is None
    array_source = AnalysisObject.from_data(
        source.to_dataarray(),
        sequence_dim="sample",
        core_dims=("axis",),
    )
    assert Position(array_source).graph is None


def test_spatial_core_129b_014_active_graph_is_not_passively_acquired() -> None:
    """ID: SPATIAL_CORE_129B_014_active_graph_is_not_passively_acquired."""
    active = FrameGraph()
    with active:
        source = Position(_vector_ao("position"))
        result = source + Position(_vector_ao("position")).as_delta()
        pose = Pose.from_components(_rotation(), source)
    assert source.graph is None
    assert result.graph is None
    assert pose.graph is None


def test_spatial_core_129b_008_remembered_graph_selects_paths_and_results() -> None:
    """ID: SPATIAL_CORE_129B_008_remembered_graph_selects_paths_and_results."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        graph.get_or_create_frame("sensor", parent=world)
    edge = Pose(_pose(), parent="world", child="sensor")
    source = Position(_vector_ao("position"), parent="sensor", graph=graph)
    result = source.to_frame("world", edge_pose_fn=lambda *_: edge)
    solved = solve_pose_path_transform(
        "sensor",
        "world",
        graph=graph,
        edge_pose_fn=lambda *_: edge,
    )
    assert result.graph is graph
    assert solved.graph is graph


def test_spatial_core_129b_009_object_identity_applies_graph_matrix() -> None:
    """ID: SPATIAL_CORE_129B_009_object_identity_applies_graph_matrix."""
    remembered = FrameGraph()
    foreign = FrameGraph()
    source = Position(_vector_ao("position"), parent="sensor", graph=remembered)
    assert source.to_frame("sensor").graph is remembered
    with foreign:
        foreign_sensor = foreign.get_or_create_frame("sensor")
    with pytest.raises(ValueError, match="different FrameGraph instances"):
        source.to_frame(foreign_sensor)
    unassociated = source.with_graph(None)
    assert unassociated.to_frame(foreign_sensor).graph is foreign
    with remembered:
        remembered.get_or_create_frame("sensor")
    assert source.to_frame("sensor", graph=remembered).graph is remembered
    assert source.to_frame("sensor", graph=foreign).graph is foreign


def test_spatial_perf_129b_010_with_graph_does_not_compute_dask() -> None:
    """ID: SPATIAL_PERF_129B_010_with_graph_does_not_compute_dask."""
    from dask.callbacks import Callback

    source = Position(
        _vector_ao("position").as_dataset(copy="none").chunk({"sample": 1})
    )
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        constructed = Position(source, parent="world", graph=FrameGraph())
        result = constructed.with_graph(FrameGraph())
    assert tasks == []
    assert result.as_dataset(copy="none")["position"].data.__dask_graph__() == (
        source.as_dataset(copy="none")["position"].data.__dask_graph__()
    )


@pytest.mark.parametrize("close_first", ["source", "alias"])
def test_spatial_core_129b_013_with_graph_couples_resource_once(close_first) -> None:
    """ID: SPATIAL_CORE_129B_013_with_graph_couples_resource_once."""
    source = Position(_vector_ao("position"))
    closed: list[str] = []
    source.as_dataset(copy="none").set_close(lambda: closed.append("backend"))
    alias = source.with_graph(FrameGraph())
    first = source if close_first == "source" else alias
    second = alias if close_first == "source" else source
    first.close()
    second.close()
    first.close()
    assert closed == ["backend"]


def test_spatial_core_129b_015_rebuilt_results_preserve_effective_basis() -> None:
    """ID: SPATIAL_CORE_129B_015_rebuilt_results_preserve_effective_basis."""
    position = _with_effective_basis(
        Position(_vector_ao("position"), parent="world", child="body"),
        "camera",
    )
    rotation = _with_effective_basis(
        Rotation(_rotation(), parent="world", child="body"),
        "camera",
    )
    position_before = position.as_dataset(copy="deep")
    rotation_before = rotation.as_dataset(copy="deep")
    pose = Pose.from_components(rotation, position)
    pose_variants = (pose, pose.as_matrix(), pose.as_matrix().as_components())
    for value in pose_variants:
        assert get_expressed_in(value.as_dataset(copy="none"), owner="test") == "camera"
    decomposed = pose.as_matrix().decompose()
    for value in decomposed:
        assert get_expressed_in(value.as_dataset(copy="none"), owner="test") == "camera"
    xr.testing.assert_identical(position.as_dataset(copy="none"), position_before)
    xr.testing.assert_identical(rotation.as_dataset(copy="none"), rotation_before)

    linear_velocity = _with_effective_basis(
        LinearVelocity(
            _vector_ao("linear_velocity", core_dim="linear_axis"),
            parent="world",
            child="body",
        ),
        "camera",
    )
    angular_velocity = _with_effective_basis(
        AngularVelocity(
            _vector_ao("angular_velocity", core_dim="angular_axis"),
            parent="world",
            child="body",
        ),
        "camera",
    )
    velocity = Velocity.from_linear_angular(linear_velocity, angular_velocity)
    acceleration = _with_effective_basis(
        Acceleration(
            _acceleration(),
            parent="world",
            child="body",
        ),
        "camera",
    )
    for value in (
        velocity.as_vector6(),
        velocity.as_vector6().as_components(),
        velocity.as_vector6().linear(),
        velocity.as_vector6().angular(),
        acceleration.as_vector6(),
        acceleration.as_vector6().as_components(),
        acceleration.as_vector6().linear(),
        acceleration.as_vector6().angular(),
    ):
        assert get_expressed_in(value.as_dataset(copy="none"), owner="test") == "camera"


def test_spatial_perf_129b_024_basis_preservation_is_lazy() -> None:
    """ID: SPATIAL_PERF_129B_024_basis_preservation_is_lazy."""
    from dask.callbacks import Callback

    source = _with_effective_basis(
        Pose(_pose(), parent="world", child="body"),
        "camera",
    )
    lazy = Pose(source.as_dataset(copy="none").chunk({"sample": 1}))
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        matrix = lazy.as_matrix(validate=False)
        position, rotation = matrix.decompose(validate=False)

    assert tasks == []
    for value in (matrix, position, rotation):
        assert get_expressed_in(value.as_dataset(copy="none"), owner="test") == "camera"


def test_spatial_core_129b_016_generic_binary_results_resolve_association() -> None:
    """ID: SPATIAL_CORE_129B_016_generic_binary_results_resolve_association."""
    graph = FrameGraph()
    associated = Position(_vector_ao("position"), graph=graph)
    unassociated = Position(_vector_ao("position"))

    assert ufuncs.subtract(associated, unassociated).graph is graph
    assert ufuncs.subtract(unassociated, associated).graph is graph
    assert (associated - unassociated).graph is graph
    assert (unassociated - associated).graph is graph
    assert ufuncs.multiply(associated, 2.0).graph is graph
    assert ufuncs.multiply(2.0, associated).graph is graph
    assert ufuncs.add(associated, associated.as_dataset(copy="none")).graph is graph


def test_spatial_hard_129b_017_generic_binary_conflict_precedes_dask() -> None:
    """ID: SPATIAL_HARD_129B_017_generic_binary_conflict_precedes_dask."""
    from dask.callbacks import Callback

    first = Position(
        _vector_ao("position").as_dataset(copy="none").chunk({"sample": 1}),
        graph=FrameGraph(),
    )
    second = Position(first, graph=FrameGraph())
    tasks: list[object] = []
    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        pytest.raises(ValueError, match="different FrameGraph instances"),
    ):
        ufuncs.subtract(first, second)
    assert tasks == []


@pytest.mark.parametrize("validate", (False, True))
@pytest.mark.parametrize("case", ("pose", "velocity", "acceleration"))
def test_spatial_core_129b_018_factories_use_shallow_typed_promotion(
    case: str,
    validate: bool,
) -> None:
    """ID: SPATIAL_CORE_129B_018_factories_use_shallow_typed_promotion."""
    factory, source = _factory_case(case)
    source_ds = source.as_dataset(copy="none")
    source_ds.attrs["nested"] = {"items": ["source"]}
    var_name = next(iter(source_ds.data_vars))
    coord_name = next(iter(source_ds.coords))
    source_ds[var_name].encoding["nested"] = {"items": ["source"]}
    source_ds[coord_name].attrs["nested"] = {"items": ["source"]}
    source_ds[coord_name].encoding["nested"] = {"items": ["source"]}
    result = factory(source, validate=validate)
    result_ds = result.as_dataset(copy="none")

    assert result_ds is not source_ds
    assert result_ds.variables[var_name] is not source_ds.variables[var_name]
    assert np.shares_memory(result_ds[var_name].data, source_ds[var_name].data)
    for name in source_ds.coords:
        assert result_ds.variables[name] is not source_ds.variables[name]
    for name, source_index in source_ds.xindexes.items():
        assert result_ds.xindexes[name] is not source_index
        assert result_ds.xindexes[name].equals(source_index)
    result_ds.attrs["nested"]["items"].append("result")
    result_ds[var_name].encoding["nested"]["items"].append("result")
    result_ds[coord_name].attrs["nested"]["items"].append("result")
    result_ds[coord_name].encoding["nested"]["items"].append("result")
    assert source_ds.attrs["nested"]["items"] == ["source"]
    assert source_ds[var_name].encoding["nested"]["items"] == ["source"]
    assert source_ds[coord_name].attrs["nested"]["items"] == ["source"]
    assert source_ds[coord_name].encoding["nested"]["items"] == ["source"]


@pytest.mark.parametrize("case", ("pose", "velocity", "acceleration"))
@pytest.mark.parametrize("close_first", ("source", "result"))
def test_spatial_core_129b_019_factories_couple_source_resource_once(
    case: str,
    close_first: str,
) -> None:
    """ID: SPATIAL_CORE_129B_019_factories_couple_source_resource_once."""
    factory, source = _factory_case(case)
    closed: list[str] = []
    source.as_dataset(copy="none").set_close(lambda: closed.append("backend"))
    result = factory(source)
    first = source if close_first == "source" else result
    second = result if close_first == "source" else source
    first.close()
    second.close()
    result.close()
    assert closed == ["backend"]


@pytest.mark.parametrize("invalid", (UNSET, object()))
def test_spatial_hard_129b_020_with_graph_rejects_invalid_required_value(
    invalid,
) -> None:
    """ID: SPATIAL_HARD_129B_020_with_graph_rejects_invalid_required_value."""
    source = Position(_vector_ao("position"))
    with pytest.raises(TypeError, match="Position.with_graph: graph must be FrameGraph or None"):
        source.with_graph(invalid)
    assert source.with_graph(FrameGraph()).graph is not None
    assert source.with_graph(None).graph is None


@pytest.mark.parametrize("case", ("pose", "velocity", "acceleration"))
@pytest.mark.parametrize("kind", ("dataset", "dataarray"))
def test_spatial_core_129b_021_factory_external_ingress_isolated(
    case: str,
    kind: str,
) -> None:
    """ID: SPATIAL_CORE_129B_021_factory_external_ingress_isolated."""
    factory, source = _factory_case(case)
    external_ds = source.as_dataset(copy="deep")
    var_name = next(iter(external_ds.data_vars))
    external: xr.Dataset | xr.DataArray = external_ds
    if kind == "dataarray":
        external = external_ds[var_name]
        external.attrs["tal"] = external_ds.attrs["tal"]
    result = factory(external)
    result_data = result.as_dataset(copy="none")[var_name].data
    external_data = external_ds[var_name].data

    assert not np.shares_memory(result_data, external_data)
    expected = np.array(result_data, copy=True)
    external_data.reshape(-1)[0] += 10.0
    np.testing.assert_array_equal(result_data, expected)


@pytest.mark.parametrize("case", ("pose", "velocity", "acceleration"))
def test_spatial_hard_129b_022_factory_failure_leaves_source_open(case: str) -> None:
    """ID: SPATIAL_HARD_129B_022_factory_failure_leaves_source_open."""
    factory, typed_source = _factory_case(case)
    source_ds = typed_source.as_dataset(copy="deep")
    var_name = next(iter(source_ds.data_vars))
    if case == "pose":
        source_ds[var_name].data[..., 3, 3] = 0.0
    else:
        core_dim = source_ds[var_name].dims[-1]
        labels = list(source_ds.coords[core_dim].data)
        labels[0] = "invalid"
        source_ds = source_ds.assign_coords({core_dim: labels})
    source = AnalysisObject(source_ds)
    closed: list[str] = []
    source.as_dataset(copy="none").set_close(lambda: closed.append("backend"))

    with pytest.raises(ValueError):
        factory(source, validate=True)
    assert closed == []
    source.close()
    assert closed == ["backend"]


@pytest.mark.parametrize("case", ("pose", "velocity", "acceleration"))
def test_spatial_perf_129b_023_factory_promotion_preserves_lazy_graph(case: str) -> None:
    """ID: SPATIAL_PERF_129B_023_factory_promotion_preserves_lazy_graph."""
    from dask.callbacks import Callback

    factory, eager = _factory_case(case)
    eager_ds = eager.as_dataset(copy="none")
    var_name = next(iter(eager_ds.data_vars))
    chunked = eager_ds.chunk({eager_ds[var_name].dims[0]: 1})
    source = type(eager)(chunked)
    source_data = source.as_dataset(copy="none")[var_name].data
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = factory(source, validate=False)
    result_data = result.as_dataset(copy="none")[var_name].data

    assert tasks == []
    assert result_data.__dask_graph__() == source_data.__dask_graph__()


def test_spatial_core_129b_025_direct_identity_skips_path_only_policy() -> None:
    """ID: SPATIAL_CORE_129B_025_direct_identity_skips_path_only_policy."""
    graph = FrameGraph()
    frame = graph.get_or_create_frame("world")
    cases = (
        (solve_pose_path_transform, "edge_pose_fn"),
        (solve_rotation_path_transform, "edge_rotation_fn"),
        (Pose.solve_path_transform, "edge_pose_fn"),
        (Rotation.solve_path_transform, "edge_rotation_fn"),
    )
    for solver, resolver_name in cases:
        result = solver(
            frame,
            frame,
            graph=graph,
            opts=PathSolveOptions(strict=False),
            **{resolver_name: object()},
        )
        assert result.graph is graph
        assert get_frames(result.as_dataset(copy="none")) == ("world", "world")


def test_spatial_core_129b_026_inverse_establishes_new_parent_basis_lazily() -> None:
    """ID: SPATIAL_CORE_129B_026_inverse_establishes_new_parent_basis_lazily."""
    from dask.callbacks import Callback

    graph = FrameGraph()
    for value in (_rotation().as_matrix(validate=False), _pose().as_matrix(validate=False)):
        framed = type(value)(
            value,
            parent="world",
            child="body",
            expressed_in="world",
            graph=graph,
        )
        lazy = type(value)(
            framed.as_dataset(copy="none").chunk({"sample": 1}),
            graph=graph,
        )
        before = lazy.as_dataset(copy="deep")
        tasks: list[object] = []
        with Callback(pretask=lambda key, *_, tasks=tasks: tasks.append(key)):
            result = lazy.inverse(validate=False)

        assert tasks == []
        assert result.graph is graph
        assert get_frames(result.as_dataset(copy="none")) == ("body", "world")
        assert get_expressed_in(result.as_dataset(copy="none"), owner="test") == "body"
        xr.testing.assert_identical(lazy.as_dataset(copy="none"), before)


def test_spatial_hard_129b_029_inverse_requires_parent_basis_before_kernel() -> None:
    """ID: SPATIAL_HARD_129B_029_inverse_requires_parent_basis_before_kernel."""
    from dask.callbacks import Callback

    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        graph.get_or_create_frame("body", parent=world)
        graph.get_or_create_frame("camera", parent=world)
    quat = SciRotation.from_euler("xyz", [20.0, -15.0, 35.0], degrees=True).as_quat()
    camera_quat = SciRotation.from_euler("z", 40.0, degrees=True).as_quat()
    rotation_data = xr.DataArray(
        [quat],
        dims=("sample", "quat"),
        coords={"sample": [0], "quat": list(_QUAT)},
        name="rotation",
    )
    camera_rotation_data = rotation_data.copy(data=np.asarray([camera_quat]))
    rotation = Rotation(
        AnalysisObject.from_data(rotation_data, sequence_dim="sample", core_dims=("quat",)),
        parent="world",
        child="body",
        graph=graph,
    )
    position = Position(
        _vector_ao("position"),
        parent="world",
        child="body",
        graph=graph,
    )
    pose = Pose.from_components(rotation, position, graph=graph)
    camera_rotation = Rotation(
        AnalysisObject.from_data(camera_rotation_data, sequence_dim="sample", core_dims=("quat",)),
        parent="world",
        child="camera",
    )
    camera_pose = Pose.from_components(
        camera_rotation,
        Position(
            _vector_ao("position"),
            parent="world",
            child="camera",
        ),
    )
    third_rotation = rotation.express_in(
        "camera",
        graph=graph,
        edge_rotation_fn=lambda *_: camera_rotation,
    )
    third_pose = pose.express_in(
        "camera",
        graph=graph,
        edge_pose_fn=lambda *_: camera_pose,
    )
    for source, resolver_name, resolver in (
        (third_rotation, "edge_rotation_fn", camera_rotation),
        (third_pose, "edge_pose_fn", camera_pose),
    ):
        lazy = type(source)(
            source.as_dataset(copy="none").chunk({"sample": 1}),
            graph=graph,
        )
        tasks: list[object] = []
        with (
            Callback(pretask=lambda key, *_, tasks=tasks: tasks.append(key)),
            pytest.raises(ValueError, match="inverse requires expressed_in to equal parent"),
        ):
            lazy.inverse(validate=False)
        assert tasks == []
        _assert_partial_inverse_preflight(source, graph=graph)
        canonical = source.express_in(
            "world",
            graph=graph,
            **{resolver_name: lambda *_args, value=resolver: value},
        )
        expected = rotation.inverse() if isinstance(source, Rotation) else pose.inverse()
        xr.testing.assert_allclose(
            canonical.inverse().as_matrix().as_dataset(copy="none"),
            expected.as_matrix().as_dataset(copy="none"),
        )


@pytest.mark.parametrize("kind", ("rotation", "pose"))
@pytest.mark.parametrize("representation", ("native", "matrix"))
def test_spatial_core_129b_034_nonidentity_reexpression_uses_neutral_basis_projection(
    kind: str,
    representation: str,
) -> None:
    """ID: SPATIAL_CORE_129B_034_nonidentity_reexpression_uses_neutral_basis_projection."""
    from dask.base import is_dask_collection
    from dask.callbacks import Callback

    graph = FrameGraph()
    source_quat = SciRotation.from_euler("xyz", [23.0, -17.0, 31.0], degrees=True).as_quat()
    basis_quat = SciRotation.from_euler("zyx", [37.0, 11.0, -19.0], degrees=True).as_quat()
    source_rotation = Rotation(
        AnalysisObject.from_data(
            xr.DataArray(
                [source_quat],
                dims=("sample", "quat"),
                coords={"sample": [0], "quat": list(_QUAT)},
                name="rotation",
            ),
            sequence_dim="sample",
            core_dims=("quat",),
        ),
        parent="world",
        child="body",
        graph=graph,
    )
    source_position = Position(
        _vector_ao("position"),
        parent="world",
        child="body",
        graph=graph,
    )
    source_pose = Pose.from_components(source_rotation, source_position, graph=graph)
    basis_rotation = Rotation(
        AnalysisObject.from_data(
            xr.DataArray(
                [basis_quat],
                dims=("sample", "quat"),
                coords={"sample": [0], "quat": list(_QUAT)},
                name="rotation",
            ),
            sequence_dim="sample",
            core_dims=("quat",),
        ),
        parent="world",
        child="camera",
    )
    basis_pose = Pose.from_components(
        basis_rotation,
        Position(_vector_ao("position"), parent="world", child="camera"),
    )
    bind_pose(graph, "world", "camera", basis_pose)

    source = source_rotation if kind == "rotation" else source_pose
    represented = source if representation == "native" else source.as_matrix(validate=False)
    lazy = type(represented)(
        represented.as_dataset(copy="none").chunk({"sample": 1}),
        graph=graph,
    )
    before = lazy.as_dataset(copy="deep")
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_, tasks=tasks: tasks.append(key)):
        in_camera = lazy.express_in("camera", validate=False)
        restored = in_camera.express_in("world", validate=False)

    assert tasks == []
    assert in_camera.graph is graph
    assert restored.graph is graph
    assert get_frames(in_camera.as_dataset(copy="none")) == ("world", "body")
    assert get_frames(restored.as_dataset(copy="none")) == ("world", "body")
    assert get_expressed_in(in_camera.as_dataset(copy="none"), owner="test") == "camera"
    assert get_expressed_in(restored.as_dataset(copy="none"), owner="test") == "world"
    assert set(in_camera.as_dataset(copy="none").data_vars) == set(before.data_vars)
    assert all(is_dask_collection(array.data) for array in restored.as_dataset(copy="none").data_vars.values())
    xr.testing.assert_identical(lazy.as_dataset(copy="none"), before)
    expected = lazy.as_matrix(validate=False).as_dataset(copy="none")
    actual = restored.as_matrix(validate=False).as_dataset(copy="none")
    xr.testing.assert_allclose(actual.compute(), expected.compute(), rtol=1e-12, atol=1e-12)


def test_spatial_core_129b_032_combine_resolves_all_spatial_associations() -> None:
    """ID: SPATIAL_CORE_129B_032_combine_resolves_all_spatial_associations."""
    from dask.callbacks import Callback

    graph = FrameGraph()
    left = Position(_vector_ao("position"))
    right = Position(_vector_ao("position"), graph=graph)
    batch = left.combine.concat_batch(
        [right],
        opts=BatchConcatOptions(batch_dim="run", sequence_join="exact"),
    )
    assert isinstance(batch, Position)
    assert batch.graph is graph
    assert left.combine.concat_batch([Position(_vector_ao("position"))]).graph is None
    assert right.combine.concat_batch([left]).graph is graph

    second_ds = right.as_dataset(copy="none").assign_coords(sample=[1])
    second = Position(second_ds, graph=graph)
    sequence = left.combine.concat_sequence(
        [second],
        opts=SequenceConcatOptions(overlap="error"),
    )
    assert isinstance(sequence, Position)
    assert sequence.graph is graph
    assert left.combine.merge([right], opts=MergeOptions()).graph is graph
    overlay = left.combine.overlay_core(
        [right],
        opts=CoreOverlayOptions(core_dim="axis", on_overlap="replace"),
    )
    assert isinstance(overlay, Position)
    assert overlay.graph is graph
    assert left.with_graph(graph).combine.merge([right]).graph is graph

    lazy_left = Position(left.as_dataset(copy="none").chunk({"sample": 1}), graph=FrameGraph())
    lazy_right = Position(right.as_dataset(copy="none").chunk({"sample": 1}), graph=FrameGraph())
    tasks: list[object] = []
    conflict_operations = (
        lambda: lazy_left.combine.concat_batch([lazy_right]),
        lambda: lazy_left.combine.concat_sequence([lazy_right]),
        lambda: lazy_left.combine.concat_core(
            [lazy_right],
            opts=CoreConcatOptions(
                core_dim="axis",
                core_labels=("x0", "y0", "z0", "x1", "y1", "z1"),
            ),
        ),
        lambda: lazy_left.combine.merge([lazy_right]),
        lambda: lazy_left.combine.overlay_core(
            [lazy_right],
            opts=CoreOverlayOptions(core_dim="axis", on_overlap="replace"),
        ),
    )
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        for operation in conflict_operations:
            with pytest.raises(ValueError, match="different FrameGraph instances"):
                operation()
    assert tasks == []


def test_spatial_core_129b_033_with_graph_documented_boundary() -> None:
    """ID: SPATIAL_CORE_129B_033_with_graph_documented_boundary."""
    source = Position(_vector_ao("position"), parent="world")
    graph = FrameGraph()
    associated = source.with_graph(graph)

    assert associated.graph is graph
    assert source.graph is None
    with pytest.raises(TypeError, match="Position.with_graph: graph must be FrameGraph or None"):
        source.with_graph(object())  # type: ignore[arg-type]


def test_spatial_hard_129b_027_position_destination_precedes_graph_work() -> None:
    """ID: SPATIAL_HARD_129B_027_position_destination_precedes_graph_work."""
    from dask.callbacks import Callback

    source = Position(
        _vector_ao("position").as_dataset(copy="none").chunk({"sample": 1}),
        parent="world",
        graph=FrameGraph(),
    )
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        for invalid in (object(), "  "):
            with pytest.raises(TypeError, match="spatial.position.to_frame: dst must be Frame"):
                source.to_frame(invalid)  # type: ignore[arg-type]
    assert tasks == []


def test_spatial_core_129b_028_public_constructor_annotations_resolve() -> None:
    """ID: SPATIAL_CORE_129B_028_public_constructor_annotations_resolve."""
    constructors = tuple(type(value).__init__ for value in _spatial_sources())
    factories = (
        Pose.from_components,
        Pose.from_matrix,
        Velocity.from_linear_angular,
        Velocity.from_vector6,
        Acceleration.from_linear_angular,
        Acceleration.from_vector6,
    )
    for target in (*constructors, *factories):
        hints = get_type_hints(target)
        assert {"parent", "child", "expressed_in", "graph"} <= hints.keys()
