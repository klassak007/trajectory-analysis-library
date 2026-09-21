"""Public typed scalar-field construction and immutable recipe behavior."""

from __future__ import annotations

import gc
import inspect
import tracemalloc
import weakref
from dataclasses import FrozenInstanceError
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal.core import AnalysisLayoutSpec, AnalysisObject, SchemaError
from tal.core.component_ops import read_components
from tal.core.schema import merge_schema
from tal.frames import FrameGraph
from tal.spatial import Pose, Position, Rotation, SpatialFieldRecipe
from tal.spatial.metadata import (
    get_expressed_in,
    set_instantaneous_inertial,
    set_kinematics_kind,
    set_position_intent,
)
from tal.spatial.metadata.relation import set_expressed_in
from tal.utils.frame_schema import get_frames, set_frames


class _CopyBomb:
    def __init__(self) -> None:
        self.calls = 0

    def __deepcopy__(self, memo: object) -> object:
        self.calls += 1
        raise RuntimeError("metadata copy must not run")


class _CopyProbe:
    def __init__(self) -> None:
        self.calls = 0

    def __deepcopy__(self, memo: object) -> object:
        self.calls += 1
        return _CopyProbe()


class _SharedCopyProbe:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls

    def __deepcopy__(self, memo: object) -> object:
        self.calls.append(object())
        return _SharedCopyProbe(self.calls)


class _UnrenderableName:
    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        return self is other

    def __str__(self) -> str:
        raise RuntimeError("name string formatting must not escape")

    def __repr__(self) -> str:
        raise RuntimeError("name repr formatting must not escape")


def _field_dataset(*, lazy: bool = False) -> xr.Dataset:
    values = np.arange(14.0).reshape(7, 2)
    if lazy:
        values = da.from_array(values, chunks=(7, 1))
    names = (
        "camera.position.x",
        "camera.position.y",
        "camera.position.z",
        "camera.orientation.x",
        "camera.orientation.y",
        "camera.orientation.z",
        "camera.orientation.w",
    )
    arrays = {name: ("sample", values[index]) for index, name in enumerate(names)}
    arrays["camera.orientation.x"] = ("sample", values[3] * 0.0)
    arrays["camera.orientation.y"] = ("sample", values[4] * 0.0)
    arrays["camera.orientation.z"] = ("sample", values[5] * 0.0)
    arrays["camera.orientation.w"] = ("sample", values[6] * 0.0 + 1.0)
    return xr.Dataset(
        arrays,
        coords={"sample": [0, 1], "time": ("sample", [0.0, 1.0])},
        attrs={"nested": {"items": ["source"]}},
    )


def _layout() -> AnalysisLayoutSpec:
    return AnalysisLayoutSpec(sequence_dim="sample", param_coord="time")


def _pose_from_raw(ds: xr.Dataset, **kwargs: object) -> Pose:
    return Pose.from_fields(
        ds,
        position="camera.position.{x,y,z}",
        rotation="camera.orientation.{x,y,z,w}",
        source_layout=_layout(),
        **kwargs,
    )


def _build_field_target(
    target: str,
    source: object,
    *,
    recipe: bool,
    source_layout: AnalysisLayoutSpec | None,
) -> Position | Rotation | Pose:
    options = {} if source_layout is None else {"source_layout": source_layout}
    if target == "position":
        if recipe:
            return Position.fields("camera.position.{x,y,z}").build(source, **options)
        return Position.from_fields(source, "camera.position.{x,y,z}", **options)
    if target == "rotation":
        if recipe:
            return Rotation.fields("camera.orientation.{x,y,z,w}").build(source, **options)
        return Rotation.from_fields(source, "camera.orientation.{x,y,z,w}", **options)
    if recipe:
        return Pose.fields(
            position="camera.position.{x,y,z}",
            rotation="camera.orientation.{x,y,z,w}",
        ).build(source, **options)
    return Pose.from_fields(
        source,
        position="camera.position.{x,y,z}",
        rotation="camera.orientation.{x,y,z,w}",
        **options,
    )


def test_spatial_core_field_factory_001_public_signatures_and_typed_values() -> None:
    """ID: SPATIAL_CORE_FIELD_FACTORY_001."""
    ds = _field_dataset()
    position = Position.from_fields(
        ds,
        "camera.position.{x,y,z}",
        source_layout=_layout(),
    )
    rotation = Rotation.from_fields(
        ds,
        {
            "w": "camera.orientation.w",
            "x": "camera.orientation.x",
            "z": "camera.orientation.z",
            "y": "camera.orientation.y",
        },
        source_layout=_layout(),
    )
    pose = _pose_from_raw(ds)
    np.testing.assert_array_equal(
        position.as_dataset(copy="none")["position"],
        [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]],
    )
    np.testing.assert_array_equal(
        rotation.as_dataset(copy="none")["rotation"],
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
    )
    pos_out, rot_out = pose.decompose()
    xr.testing.assert_identical(
        pos_out.as_dataset(copy="none")["position"],
        position.as_dataset(copy="none")["position"],
    )
    xr.testing.assert_identical(
        rot_out.as_dataset(copy="none")["rotation"],
        rotation.as_dataset(copy="none")["rotation"],
    )
    components = read_components(pose.as_dataset(copy="none"))
    assert set(components) == {"position", "rotation"}
    assert components["position"].var == "position"
    assert components["rotation"].var == "rotation"
    assert isinstance(rotation.inverse(), Rotation)
    matrix_ds = pose.as_matrix().as_dataset(copy="none")
    matrix_var = next(iter(matrix_ds.data_vars))
    assert matrix_ds[matrix_var].shape[-2:] == (4, 4)
    assert "validate" not in inspect.signature(Position.from_fields).parameters
    assert "validate" not in inspect.signature(SpatialFieldRecipe.build).parameters
    with pytest.raises(TypeError, match="must be created by Position.fields"):
        SpatialFieldRecipe()


def test_spatial_core_field_recipe_001_reuse_is_immutable_and_prefix_local() -> None:
    """ID: SPATIAL_CORE_FIELD_RECIPE_001."""
    mapping = {"x": "position.x", "y": "position.y", "z": "position.z"}
    recipe = Position.fields(mapping)
    mapping["x"] = "missing"
    source = _field_dataset().rename(
        {
            "camera.position.x": "tool.position.x",
            "camera.position.y": "tool.position.y",
            "camera.position.z": "tool.position.z",
        }
    )
    camera = recipe.build(_field_dataset(), prefix="camera.", source_layout=_layout())
    tool = recipe.build(source, prefix="tool.", source_layout=_layout())
    xr.testing.assert_equal(
        camera.as_dataset(copy="none")["position"],
        tool.as_dataset(copy="none")["position"],
    )
    with pytest.raises(ValueError, match=r"SpatialFieldRecipe.build.*absent\.position\.x"):
        recipe.build(_field_dataset(), prefix="absent.", source_layout=_layout())
    retry = recipe.build(_field_dataset(), prefix="camera.", source_layout=_layout())
    xr.testing.assert_identical(retry.as_dataset(copy="none"), camera.as_dataset(copy="none"))
    with pytest.raises(FrozenInstanceError):
        recipe._target = "rotation"


def test_spatial_hard_field_source_001_layout_authority_and_scalar_projection() -> None:
    """ID: SPATIAL_HARD_FIELD_SOURCE_001."""
    ds = _field_dataset().assign(
        packed=(("sample", "old_axis"), np.ones((2, 2)))
    ).assign_coords(old_axis=["a", "b"])
    source = AnalysisLayoutSpec(
        sequence_dim="sample",
        core_dims=("old_axis",),
        param_coord="time",
    ).wrap(ds)
    result = Position.from_fields(source, "camera.position.{x,y,z}")
    roles = result.as_dataset(copy="none").attrs["tal"]["core"]["roles"]
    assert roles["core_dims"] == ["axis"]
    assert "packed" not in result.as_dataset(copy="none")
    with pytest.raises(ValueError, match="source_layout cannot accompany"):
        Position.from_fields(source, "camera.position.{x,y,z}", source_layout=_layout())
    with pytest.raises(ValueError, match="requires source_layout"):
        Position.from_fields(_field_dataset(), "camera.position.{x,y,z}")
    bad = ds.assign(**{"camera.position.x": ds["packed"]})
    bad_source = AnalysisLayoutSpec(
        sequence_dim="sample",
        core_dims=("old_axis",),
        param_coord="time",
    ).wrap(bad)
    with pytest.raises(ValueError, match="camera.position.x.*semantic dimensions"):
        Position.from_fields(bad_source, "camera.position.{x,y,z}")
    with pytest.raises(ValueError, match="source_layout.core_dims must be empty"):
        Position.from_fields(
            _field_dataset(),
            "camera.position.{x,y,z}",
            source_layout=AnalysisLayoutSpec(
                sequence_dim="sample",
                core_dims=("axis",),
            ),
        )

    multi = pd.MultiIndex.from_product([["a"], [0, 1]], names=["run", "number"])
    coords = xr.Coordinates.from_pandas_multiindex(multi, "observation")
    indexed = xr.Dataset(
        {
            "p.x": ("observation", [1.0, 2.0]),
            "p.y": ("observation", [3.0, 4.0]),
            "p.z": ("observation", [5.0, 6.0]),
        },
        coords=coords,
    )
    with pytest.raises(ValueError, match="PandasMultiIndex dimensions are not supported"):
        Position.from_fields(
            indexed,
            "p.{x,y,z}",
            source_layout=AnalysisLayoutSpec(sequence_dim="observation"),
        )


@pytest.mark.parametrize(
    "source",
    (
        xr.DataArray([1.0], dims="sample"),
        Path("run.csv"),
        pd.DataFrame({"x": [1.0]}),
    ),
)
def test_spatial_hard_field_source_002_direct_source_boundaries(source: object) -> None:
    with pytest.raises(TypeError, match="Position.from_fields.*AnalysisObject or xarray.Dataset"):
        Position.from_fields(source, "p.{x,y,z}", source_layout=_layout())


def test_spatial_hard_field_diagnostics_001_grammar_membership_and_precedence() -> None:
    """ID: SPATIAL_HARD_FIELD_DIAGNOSTICS_001."""
    with pytest.raises(ValueError, match="exactly one brace group"):
        Position.fields("position.x")
    with pytest.raises(ValueError, match="labels must be exactly"):
        Position.fields("position.{x,y}")
    with pytest.raises(TypeError, match="mapping labels and source names"):
        Position.fields({"x": 1, "y": "y", "z": "z"})
    with pytest.raises(ValueError, match="repeats a source field"):
        Position.fields({"x": "value", "y": "value", "z": "other"})
    with pytest.raises(TypeError, match="prefix must be a string"):
        Position.fields("position.{x,y,z}").build(
            _field_dataset(), prefix=1, source_layout=_layout()
        )
    wide = xr.Dataset({f"v{index}": ("sample", [float(index)]) for index in range(12)})
    with pytest.raises(ValueError, match=r"Position.from_fields.*position.*missing.*\+4 more"):
        Position.from_fields(wide, "missing.{x,y,z}", source_layout=AnalysisLayoutSpec(sequence_dim="sample"))
    malformed = _field_dataset()
    malformed.attrs["tal"] = {
        "version": 1,
        "core": {"roles": {"sequence_dim": "absent", "batch_dims": [], "core_dims": []}},
    }
    with pytest.raises(SchemaError) as failure:
        Position.from_fields(malformed, "camera.position.{x,y,z}")
    assert failure.value.code == "schema.roles.sequence_dim.not_in_dataset"
    assert failure.value.path == "tal.core.roles.sequence_dim"
    assert failure.value.hint.count("Position.from_fields") == 1
    assert isinstance(failure.value.__cause__, SchemaError)

    recipe = Position.fields("camera.position.{x,y,z}")
    with pytest.raises(SchemaError) as recipe_failure:
        recipe.build(malformed)
    assert recipe_failure.value.code == failure.value.code
    assert recipe_failure.value.path == failure.value.path
    assert recipe_failure.value.hint.count("SpatialFieldRecipe.build") == 1
    assert isinstance(recipe_failure.value.__cause__, SchemaError)


def test_spatial_core_field_factory_002_topology_validity_and_metadata() -> None:
    ds = xr.Dataset(
        {
            "p.x": (("trial", "sample"), [[1.0, 2.0], [3.0, 4.0]]),
            "p.y": (("trial", "sample"), [[5.0, 6.0], [7.0, 8.0]]),
            "p.z": (("trial", "sample"), [[9.0, 10.0], [11.0, 12.0]]),
        },
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1],
            "time": (("trial", "sample"), [[0.0, 1.0], [0.0, 1.0]]),
            "group_size": ("trial", [2, 1]),
            "label": ("trial", ["A", "B"]),
        },
        attrs={"ordinary": {"items": ["source"]}},
    ).set_xindex("label")
    ds["p.x"].attrs["units"] = "m"
    ds["p.x"].encoding["dtype"] = "float32"
    ds["time"].attrs["clock"] = "monotonic"
    layout = AnalysisLayoutSpec(
        sequence_dim="sample",
        batch_dims=("trial",),
        param_coord="time",
        sequence_size_coord="group_size",
    )
    result = Position.from_fields(ds, "p.{x,y,z}", source_layout=layout)
    out = result.as_dataset(copy="none")
    assert out["position"].dims == ("trial", "sample", "axis")
    assert out.attrs["tal"]["core"]["validity"]["sequence_size_coord"] == "group_size"
    assert type(out.xindexes["label"]) is type(ds.xindexes["label"])
    assert out.xindexes["label"].equals(ds.xindexes["label"])
    assert out["position"].attrs == {}
    assert out["position"].encoding == {}
    assert out["time"].attrs == {"clock": "monotonic"}
    out.attrs["ordinary"]["items"].append("result")
    out["time"].attrs["clock"] = "changed"
    assert ds.attrs["ordinary"]["items"] == ["source"]
    assert ds["time"].attrs == {"clock": "monotonic"}


def test_spatial_core_field_factory_003_static_empty_and_mixed_topology() -> None:
    static = xr.Dataset({"x": 1.0, "y": 2.0, "z": 3.0})
    result = Position.from_fields(
        static,
        {"x": "x", "y": "y", "z": "z"},
        source_layout=AnalysisLayoutSpec(),
    )
    assert result.as_dataset(copy="none")["position"].dims == ("axis",)
    empty = _field_dataset().isel(sample=slice(0, 0))
    out = Position.from_fields(empty, "camera.position.{x,y,z}", source_layout=_layout())
    assert out.as_dataset(copy="none").sizes == {"sample": 0, "axis": 3}
    mixed = _field_dataset().assign(**{"camera.position.z": 1.0})
    with pytest.raises(ValueError, match="semantic dimensions"):
        Position.from_fields(mixed, "camera.position.{x,y,z}", source_layout=_layout())

    batch = xr.Dataset(
        {
            "p.x": ("trial", [1.0, 2.0]),
            "p.y": ("trial", [3.0, 4.0]),
            "p.z": ("trial", [5.0, 6.0]),
        },
        coords={"trial": ["a", "b"]},
    )
    batch_out = Position.from_fields(
        batch,
        "p.{x,y,z}",
        source_layout=AnalysisLayoutSpec(batch_dims=("trial",)),
    )
    assert batch_out.as_dataset(copy="none")["position"].dims == ("trial", "axis")


def test_spatial_hard_field_diagnostics_002_generated_namespace_conflicts() -> None:
    ds = _field_dataset().assign_coords(axis=("sample", ["a", "b"]))
    with pytest.raises(ValueError, match="generated output name 'axis'.*rename"):
        Position.from_fields(ds, "camera.position.{x,y,z}", source_layout=_layout())
    ds = _field_dataset().assign_coords(position=("sample", ["a", "b"]))
    with pytest.raises(ValueError, match="generated output name 'position'.*rename"):
        Position.from_fields(ds, "camera.position.{x,y,z}", source_layout=_layout())


@pytest.mark.parametrize("failure", ("namespace", "dtype", "dimensions"))
def test_spatial_hard_field_diagnostics_003_target_preflight_precedes_copy(
    failure: str,
) -> None:
    bomb = _CopyBomb()
    ds = _field_dataset()
    ds.attrs["copy_bomb"] = bomb
    if failure == "namespace":
        ds = ds.assign_coords(axis=("sample", ["a", "b"]))
        expected = "generated output name 'axis'"
    elif failure == "dtype":
        ds["camera.position.x"] = ("sample", ["a", "b"])
        expected = "real numeric dtype"
    else:
        ds = ds.assign_coords(extra=[0])
        ds["camera.position.x"] = (("sample", "extra"), [[1.0], [2.0]])
        expected = "semantic dimensions"
    with pytest.raises((TypeError, ValueError), match=expected):
        Position.from_fields(
            ds,
            "camera.position.{x,y,z}",
            source_layout=_layout(),
        )
    assert bomb.calls == 0


def test_spatial_hard_field_diagnostics_004_frame_preflight_precedes_copy() -> None:
    bomb = _CopyBomb()
    source_ds = set_frames(
        _layout().wrap(_field_dataset()).as_dataset(copy="none"),
        parent="world",
        child="camera",
        validate=False,
    )
    source_ds.attrs["copy_bomb"] = bomb
    source = AnalysisObject._from_validated(source_ds)
    with pytest.raises(ValueError, match="child='tool'.*conflicts"):
        Position.from_fields(source, "camera.position.{x,y,z}", child="tool")
    assert bomb.calls == 0

    basis_bomb = _CopyBomb()
    malformed = merge_schema(
        source_ds,
        {"ext": {"spatial": {"relation": {"expressed_in": 1}}}},
        validate=False,
    )
    malformed.attrs["copy_bomb"] = basis_bomb
    with pytest.raises(ValueError, match="expressed_in"):
        Position.from_fields(
            AnalysisObject._from_validated(malformed),
            "camera.position.{x,y,z}",
        )
    assert basis_bomb.calls == 0


def test_spatial_hard_field_diagnostics_005_extension_copy_waits_for_preflight() -> None:
    bomb = _CopyBomb()
    source = _layout().wrap(_field_dataset()).as_dataset(copy="none")
    source.attrs["tal"].setdefault("ext", {})["probe"] = bomb
    source["camera.position.x"] = ("sample", ["a", "b"])
    with pytest.raises(TypeError, match="real numeric dtype"):
        Position.from_fields(source, "camera.position.{x,y,z}")
    assert bomb.calls == 0


@pytest.mark.parametrize("target", ("position", "rotation", "pose"))
@pytest.mark.parametrize("recipe", (False, True))
@pytest.mark.parametrize("owned", (False, True))
@pytest.mark.parametrize(
    ("patch", "expected"),
    (
        ({"roles": {"position_intent": 5}}, "position_intent"),
        (
            {"relation": {"instantaneous_inertial": ["bogus"]}},
            "instantaneous_inertial",
        ),
        ({"representation": {"rep": 5}}, "representation.rep"),
        ({"representation": {"rep": "future"}}, "unsupported spatial representation"),
    ),
)
def test_spatial_hard_field_diagnostics_006_malformed_spatial_source_precedes_copy(
    target: str,
    recipe: bool,
    owned: bool,
    patch: dict[str, object],
    expected: str,
) -> None:
    bomb = _CopyBomb()
    source_ds = _layout().wrap(_field_dataset(lazy=True)).as_dataset(copy="none")
    source_ds = merge_schema(
        source_ds,
        {"ext": {"spatial": patch}},
        validate=False,
    )
    if owned:
        source: object = AnalysisObject(source_ds)
        source.as_dataset(copy="none").attrs["copy_bomb"] = bomb
    else:
        source_ds.attrs["copy_bomb"] = bomb
        source = source_ds
    tasks: list[object] = []
    owner = "SpatialFieldRecipe.build" if recipe else f"{target.title()}.from_fields"
    with (
        Callback(pretask=lambda key, dsk, state: tasks.append(key)),
        pytest.raises(ValueError, match=rf"{owner}.*{expected}"),
    ):
        _build_field_target(
            target,
            source,
            recipe=recipe,
            source_layout=None,
        )
    assert bomb.calls == 0
    assert tasks == []


@pytest.mark.parametrize("target", ("position", "rotation", "pose"))
@pytest.mark.parametrize("recipe", (False, True))
@pytest.mark.parametrize("owned", (False, True))
def test_spatial_ownership_field_build_007_consumed_channel_metadata_is_not_copied(
    target: str,
    recipe: bool,
    owned: bool,
) -> None:
    bomb = _CopyBomb()
    dataset = _field_dataset()
    layout = None
    if owned:
        source: object = _layout().wrap(dataset)
        source_ds = source.as_dataset(copy="none")
    else:
        source = dataset
        source_ds = dataset
        layout = _layout()
    source_ds["camera.position.x"].attrs["discarded"] = bomb
    source_ds["camera.position.x"].encoding["discarded"] = bomb
    result = _build_field_target(
        target,
        source,
        recipe=recipe,
        source_layout=layout,
    )
    out = result.as_dataset(copy="none")
    assert bomb.calls == 0
    assert all(variable.attrs == {} for variable in out.data_vars.values())
    assert all(variable.encoding == {} for variable in out.data_vars.values())


@pytest.mark.parametrize("recipe", (False, True))
def test_spatial_hard_field_diagnostics_007_available_name_preview_is_exception_safe(
    recipe: bool,
) -> None:
    name = _UnrenderableName()
    source = xr.Dataset(
        {
            name: ("sample", [1.0]),
            "other": ("sample", [2.0]),
        }
    )
    layout = AnalysisLayoutSpec(sequence_dim="sample")
    owner = "SpatialFieldRecipe.build" if recipe else "Position.from_fields"
    with pytest.raises(
        ValueError,
        match=rf"{owner}.*camera.position.x.*available=.*unrepr:_UnrenderableName",
    ):
        _build_field_target(
            "position",
            source,
            recipe=recipe,
            source_layout=layout,
        )


def test_spatial_lazy_field_build_001_construction_executes_no_dask_tasks() -> None:
    """ID: SPATIAL_LAZY_FIELD_BUILD_001."""
    source = _field_dataset(lazy=True)
    tasks: list[object] = []
    with Callback(pretask=lambda key, dsk, state: tasks.append(key)):
        result = _pose_from_raw(source)
    assert not tasks
    assert isinstance(result.as_dataset(copy="none")["position"].data, da.Array)
    assert isinstance(result.as_dataset(copy="none")["rotation"].data, da.Array)


@pytest.mark.parametrize("target", ("position", "pose"))
def test_spatial_ownership_field_build_001_frames_graph_and_resource_lifetime(
    target: str,
) -> None:
    """ID: SPATIAL_OWNERSHIP_FIELD_BUILD_001."""
    base = _layout().wrap(_field_dataset(lazy=True))
    framed_ds = set_frames(
        base.as_dataset(copy="none"),
        parent="world",
        child="camera",
        validate=False,
    )
    source = AnalysisObject._from_validated(framed_ds)
    closed: list[str] = []
    source.as_dataset(copy="none").set_close(lambda: closed.append("source"))
    graph = FrameGraph()
    if target == "position":
        result = Position.from_fields(
            source,
            "camera.position.{x,y,z}",
            graph=graph,
        )
    else:
        result = Pose.from_fields(
            source,
            position="camera.position.{x,y,z}",
            rotation="camera.orientation.{x,y,z,w}",
            graph=graph,
        )
    assert get_frames(result.as_dataset(copy="none")) == ("world", "camera")
    assert result.graph is graph
    assert graph.get_frame("world") is None
    result.close()
    source.close()
    assert closed == ["source"]


def test_spatial_ownership_field_build_002_failed_build_leaves_source_open() -> None:
    source = _layout().wrap(_field_dataset(lazy=True))
    closed: list[str] = []
    source.as_dataset(copy="none").set_close(lambda: closed.append("source"))
    with pytest.raises(ValueError, match="missing"):
        Position.from_fields(source, "missing.{x,y,z}")
    assert closed == []
    source.close()
    assert closed == ["source"]


@pytest.mark.parametrize("target", ("position", "rotation", "pose"))
@pytest.mark.parametrize("owned", (False, True))
def test_spatial_ownership_field_build_006_success_isolates_metadata_once(
    target: str,
    owned: bool,
) -> None:
    calls: list[object] = []
    probe = _SharedCopyProbe(calls)
    dataset = _field_dataset()
    if owned:
        source: object = _layout().wrap(dataset)
        source.as_dataset(copy="none").attrs["probe"] = probe
        layout = None
    else:
        dataset.attrs["probe"] = probe
        source = dataset
        layout = _layout()
    result = _build_field_target(
        target,
        source,
        recipe=False,
        source_layout=layout,
    )
    assert len(calls) == 1
    assert result.as_dataset(copy="none").attrs["probe"] is not probe


@pytest.mark.parametrize("owned", (False, True))
@pytest.mark.parametrize("lazy", (False, True))
def test_spatial_ownership_field_build_008_pose_preserves_and_isolates_custom_extensions(
    owned: bool,
    lazy: bool,
) -> None:
    declared = _layout().wrap(_field_dataset(lazy=lazy)).as_dataset(copy="none")
    declared = merge_schema(
        declared,
        {
            "ext": {
                "calibration": {
                    "offsets": [1.0, 2.0],
                }
            }
        },
        validate=False,
    )
    source: object = AnalysisObject._from_validated(declared) if owned else declared
    tasks: list[object] = []
    with Callback(pretask=lambda key, dsk, state: tasks.append(key)):
        result = _build_field_target(
            "pose",
            source,
            recipe=False,
            source_layout=None,
        )
    source_calibration = declared.attrs["tal"]["ext"]["calibration"]
    result_calibration = result.as_dataset(copy="none").attrs["tal"]["ext"][
        "calibration"
    ]
    assert set(result_calibration) == {"offsets"}
    assert result_calibration["offsets"] == source_calibration["offsets"]
    assert result_calibration is not source_calibration
    assert result_calibration["offsets"] is not source_calibration["offsets"]
    result_calibration["offsets"].append(3.0)
    assert source_calibration["offsets"] == [1.0, 2.0]
    assert tasks == []


def test_spatial_ownership_field_build_009_pose_retains_one_working_buffer() -> None:
    rows = 50_000
    names = (
        "camera.position.x",
        "camera.position.y",
        "camera.position.z",
        "camera.orientation.x",
        "camera.orientation.y",
        "camera.orientation.z",
        "camera.orientation.w",
    )
    source = xr.Dataset(
        {
            name: ("sample", np.arange(rows, dtype=np.float64) + index)
            for index, name in enumerate(names)
        },
        coords={"sample": np.arange(rows)},
    )
    layout = AnalysisLayoutSpec(sequence_dim="sample", param_coord="sample")
    _ = _pose_from_raw(_field_dataset())
    gc.collect()
    tracemalloc.start()
    result = _build_field_target(
        "pose",
        source,
        recipe=False,
        source_layout=layout,
    )
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result_ds = result.as_dataset(copy="none")
    payload_bytes = sum(variable.nbytes for variable in result_ds.data_vars.values())
    assert peak - retained < payload_bytes + 2 * 1024 * 1024


def test_spatial_ownership_field_build_003_inherits_association_and_target_metadata() -> None:
    raw = _field_dataset()
    graph = FrameGraph()
    pose = _pose_from_raw(raw, parent="world", child="camera", graph=graph)
    source_ds = pose.as_dataset(copy="none").assign(
        {name: raw[name] for name in raw.data_vars}
    )
    source = Pose(source_ds).with_graph(graph)

    position = Position.from_fields(source, "camera.position.{x,y,z}")
    rotation = Rotation.from_fields(source, "camera.orientation.{x,y,z,w}")

    assert position.graph is graph
    assert rotation.graph is graph
    assert get_frames(position.as_dataset(copy="none")) == ("world", "camera")
    assert get_frames(rotation.as_dataset(copy="none")) == ("world", "camera")
    assert position.as_dataset(copy="none").attrs["tal"]["ext"]["spatial"]["representation"] == {"rep": "cart"}
    assert rotation.as_dataset(copy="none").attrs["tal"]["ext"]["spatial"]["representation"] == {"rep": "quat"}
    delta_source = position.as_delta().as_dataset(copy="none").assign(
        {name: raw[name] for name in raw.data_vars}
    )
    rebuilt_rotation = Rotation.from_fields(
        AnalysisObject._from_validated(delta_source),
        "camera.orientation.{x,y,z,w}",
    )
    spatial = rebuilt_rotation.as_dataset(copy="none").attrs["tal"]["ext"]["spatial"]
    assert "roles" not in spatial
    assert graph.get_frame("world") is None


@pytest.mark.parametrize("target", ("position", "rotation", "pose"))
def test_spatial_ownership_field_build_005_projects_source_type_metadata(
    target: str,
) -> None:
    source_ds = _layout().wrap(_field_dataset()).as_dataset(copy="none")
    source_ds = set_frames(
        source_ds,
        parent="world",
        child="camera",
        validate=False,
    )
    source_ds = set_expressed_in(
        source_ds,
        expressed_in="world",
        validate=False,
        owner="test",
    )
    source_ds = set_position_intent(
        source_ds,
        intent="delta",
        validate=False,
        owner="test",
    )
    source_ds = set_kinematics_kind(
        source_ds,
        kind="velocity",
        validate=False,
        owner="test",
    )
    source_ds = set_instantaneous_inertial(
        source_ds,
        instantaneous_inertial={"parent"},
        validate=False,
        owner="test",
    )
    before = source_ds.copy(deep=True)
    source = AnalysisObject._from_validated(source_ds)
    if target == "position":
        result = Position.from_fields(source, "camera.position.{x,y,z}")
    elif target == "rotation":
        result = Rotation.from_fields(source, "camera.orientation.{x,y,z,w}")
    else:
        result = Pose.from_fields(
            source,
            position="camera.position.{x,y,z}",
            rotation="camera.orientation.{x,y,z,w}",
        )
    out = result.as_dataset(copy="none")
    spatial = out.attrs["tal"]["ext"]["spatial"]
    assert "roles" not in spatial
    assert "instantaneous_inertial" not in spatial.get("relation", {})
    assert get_expressed_in(out, owner="test") == "world"
    assert get_frames(out) == ("world", "camera")
    xr.testing.assert_identical(source_ds, before)


def test_spatial_ownership_field_build_004_frame_clearing_conflict_and_recipe_source_lifetime() -> None:
    source_ds = set_frames(
        _layout().wrap(_field_dataset()).as_dataset(copy="none"),
        parent="world",
        child="camera",
        validate=False,
    )
    source = AnalysisObject._from_validated(source_ds)
    with pytest.raises(ValueError, match="child='tool'.*conflicts"):
        Position.from_fields(source, "camera.position.{x,y,z}", child="tool")
    cleared = Position.from_fields(
        source,
        "camera.position.{x,y,z}",
        parent=None,
        child=None,
        expressed_in=None,
    )
    assert get_frames(cleared.as_dataset(copy="none")) == (None, None)

    recipe = Position.fields("camera.position.{x,y,z}")
    reference = weakref.ref(source)
    del source
    gc.collect()
    assert reference() is None
    assert recipe._target == "position"
