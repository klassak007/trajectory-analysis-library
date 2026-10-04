"""Public reusable-layout and schema-safe selection behavior."""

import inspect
from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import Any

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal.core import AnalysisLayoutSpec, AnalysisObject, SchemaError
from tal.frames import FrameGraph
from tal.spatial import Pose, Position, Rotation


def _source() -> xr.Dataset:
    return xr.Dataset(
        {
            "position": (("trial", "sample", "axis"), np.arange(12.0).reshape(2, 2, 3)),
            "noise": (("trial", "sample"), np.ones((2, 2))),
        },
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1],
            "axis": ["x", "y", "z"],
            "time": ("sample", [0.0, 1.0]),
        },
    )


def test_ao_layout_core_002_complete_wrap_matches_from_data() -> None:
    """ID: AO_LAYOUT_CORE_002_complete_wrap_matches_from_data."""
    layout = AnalysisLayoutSpec(
        sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",), param_coord="time"
    )
    wrapped = layout.wrap(_source())
    overlaid = AnalysisObject.from_data(
        _source(), sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",), param_coord="time"
    )
    xr.testing.assert_identical(wrapped.as_dataset(copy="none"), overlaid.as_dataset(copy="none"))


def test_ao_layout_core_001_frozen_validation_and_reuse() -> None:
    """ID: AO_LAYOUT_CORE_001_frozen_validation_and_reuse."""
    layout = AnalysisLayoutSpec(
        sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",), param_coord="time"
    )
    with pytest.raises(FrozenInstanceError):
        layout.sequence_dim = "other"
    with pytest.raises(TypeError, match="tuple"):
        AnalysisLayoutSpec(batch_dims=["trial"])
    with pytest.raises(ValueError, match="disjoint"):
        AnalysisLayoutSpec(sequence_dim="sample", core_dims=("sample",))
    with pytest.raises(ValueError, match="require sequence_dim"):
        AnalysisLayoutSpec(param_coord="time")
    first = layout.wrap(_source(), data_vars=("noise", "position"))
    second = layout.wrap(_source(), data_vars="position")
    assert list(first.as_dataset().data_vars) == ["noise", "position"]
    assert list(second.as_dataset().data_vars) == ["position"]
    selected = first.select_vars("position")
    assert list(selected.as_dataset().data_vars) == ["position"]
    assert selected.as_dataset().attrs["tal"]["core"]["roles"]["sequence_dim"] == "sample"


def test_ao_select_core_001_ordered_data_variable_grammar() -> None:
    """ID: AO_SELECT_CORE_001_ordered_data_variable_grammar."""
    source = AnalysisObject.from_data(_source(), sequence_dim="sample")
    assert tuple(source.select_vars(("noise", "position")).as_dataset(copy="none").data_vars) == (
        "noise", "position"
    )
    assert tuple(source.select_vars("position").as_dataset(copy="none").data_vars) == ("position",)


@pytest.mark.parametrize("bad", [(), [], {"position"}, iter(["position"]), ("position", "position")])
def test_ao_select_hard_001_invalid_name_forms_fail_early(bad: object) -> None:
    """ID: AO_SELECT_HARD_001_invalid_name_forms_fail_early."""
    ao = AnalysisObject.from_data(_source())
    with pytest.raises((TypeError, ValueError), match="select_vars"):
        ao.select_vars(bad)


@pytest.mark.parametrize("as_array", (False, True))
def test_ao_layout_core_003_target_defaults_do_not_inherit_source_schema(as_array: bool) -> None:
    """ID: AO_LAYOUT_CORE_003_target_defaults_do_not_inherit_source_schema."""
    ds = _source().assign_coords(group_size=("trial", [2, 2]))
    source = AnalysisObject.from_data(
        ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",),
        param_coord="time", sequence_size_coord="group_size",
    )
    payload: xr.Dataset | xr.DataArray = source.as_dataset()
    if as_array:
        payload = payload["position"].copy(deep=True)
        payload.attrs["tal"] = deepcopy(source.as_dataset(copy="none").attrs["tal"])
    result = AnalysisLayoutSpec().wrap(payload)
    assert result.as_dataset().attrs["tal"]["core"] == {}
    overlay = AnalysisObject.from_data(payload)
    assert overlay.as_dataset(copy="none").attrs["tal"]["core"]["roles"]["sequence_dim"] == "sample"


def test_ao_select_ownership_001_owned_selection_shares_payload_only() -> None:
    """ID: AO_SELECT_OWNERSHIP_001_owned_selection_shares_payload_only."""
    ds = _source()
    layout = AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    external = layout.wrap(ds, data_vars="position")
    assert not np.shares_memory(ds["position"].data, external.as_dataset(copy="none")["position"].data)
    owned = external.select_vars("position")
    assert np.shares_memory(
        external.as_dataset(copy="none")["position"].data,
        owned.as_dataset(copy="none")["position"].data,
    )
    assert owned.as_dataset(copy="none") is not external.as_dataset(copy="none")
    owned.as_dataset(copy="none").attrs["tal"]["core"]["roles"]["core_dims"].append("extra")
    assert external.as_dataset().attrs["tal"]["core"]["roles"]["core_dims"] == ["axis"]


def test_ao_select_core_002_semantic_lane_and_optional_metadata_truth() -> None:
    """ID: AO_SELECT_CORE_002_semantic_lane_and_optional_metadata_truth."""
    source = AnalysisObject.from_data(
        _source(), sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",)
    )
    result = source.select_vars("noise")
    roles = result.as_dataset().attrs["tal"]["core"]["roles"]
    assert roles["sequence_dim"] == "sample"
    assert roles["batch_dims"] == ["trial"]
    assert roles["core_dims"] == ["axis"]  # axis coordinate is a valid carrier
    no_axis_carrier = AnalysisObject.from_data(
        _source().drop_vars("axis"), sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",)
    )
    pruned = no_axis_carrier.select_vars("noise")
    assert pruned.as_dataset(copy="none").attrs["tal"]["core"]["roles"]["core_dims"] == []


def test_ao_layout_hard_001_invalid_target_precedes_copy() -> None:
    """ID: AO_LAYOUT_HARD_001_invalid_target_precedes_copy."""
    ds = _source().drop_vars("axis")
    layout = AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",))
    with pytest.raises(ValueError, match="AnalysisLayoutSpec.wrap.*axis.*carrier"):
        layout.wrap(ds, data_vars="noise")


def test_ao_select_hard_003_source_schema_precedes_pruning() -> None:
    """ID: AO_SELECT_HARD_003_source_schema_precedes_pruning."""
    ds = _source()
    ds.attrs["tal"] = {"version": 1, "core": {"roles": {"sequence_dim": "missing", "batch_dims": [], "core_dims": []}}}
    with pytest.raises(ValueError, match="missing"):
        AnalysisLayoutSpec().wrap(ds, validate=False)
    with pytest.raises(ValueError, match="missing"):
        AnalysisObject.from_data(ds, sequence_dim="sample", validate=False)


def test_ao_select_resource_001_owned_alias_couples_cleanup_after_success() -> None:
    """ID: AO_SELECT_RESOURCE_001_owned_alias_couples_cleanup_after_success."""
    source = Position(
        AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",)).wrap(
            _source().isel(trial=0, drop=True), data_vars="position"
        )
    ).with_graph(FrameGraph())
    closed: list[str] = []
    source.as_dataset(copy="none").set_close(lambda: closed.append("backend"))
    selected = source.select_vars("position")
    assert type(selected) is Position
    assert selected.graph is source.graph
    selected.close()
    source.close()
    assert closed == ["backend"]


def test_ao_select_spatial_001_exact_association_or_neutral_boundary() -> None:
    """ID: AO_SELECT_SPATIAL_001_exact_association_or_neutral_boundary."""
    base = AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",)).wrap(
        _source().isel(trial=0, drop=True), data_vars="position"
    )
    graph = FrameGraph()
    source = Position(base).with_graph(graph)
    assert source.select_vars("position").graph is graph
    assert Position(AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",)).wrap(
        source.as_dataset(copy="none"), data_vars="position"
    )).graph is None


def test_ao_select_lazy_001_planning_executes_no_payload_or_transform() -> None:
    """ID: AO_SELECT_LAZY_001_planning_executes_no_payload_or_transform."""
    ds = _source().chunk({"trial": 1})
    count: list[object] = []
    with Callback(pretask=lambda key, dsk, state: count.append(key)):
        result = AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",)).wrap(
            ds, data_vars="position"
        )
        selected = result.select_vars("position")
    assert not count
    assert isinstance(selected.as_dataset(copy="none")["position"].data, da.Array)


def test_complete_target_and_overlay_diverge_only_at_default_semantics() -> None:
    tagged = AnalysisObject.from_data(_source(), sequence_dim="sample", core_dims=("axis",))
    target = AnalysisLayoutSpec().wrap(tagged.as_dataset(copy="none"))
    overlay = AnalysisObject.from_data(tagged.as_dataset(copy="none"))
    assert target.as_dataset(copy="none").attrs["tal"]["core"] == {}
    assert overlay.as_dataset(copy="none").attrs["tal"]["core"]["roles"]["sequence_dim"] == "sample"


def test_ao_select_core_003_caller_coordinates_and_indexes_survive() -> None:
    """ID: AO_SELECT_CORE_003_caller_coordinates_and_indexes_survive."""
    ds = xr.Dataset(
        {"value": ("sample", [1.0, 2.0]), "discard": ("sample", [3.0, 4.0])},
        coords={"label": ("sample", ["a", "b"])},
    ).set_xindex("label")
    result = AnalysisLayoutSpec(sequence_dim="sample").wrap(ds, data_vars="value")
    out = result.as_dataset(copy="none")
    assert tuple(out.data_vars) == ("value",)
    assert type(out.xindexes["label"]) is type(ds.xindexes["label"])
    assert out.xindexes["label"].equals(ds.xindexes["label"])
    assert out.xindexes["label"] is not ds.xindexes["label"]


def test_ao_select_hard_002_name_ownership_prevents_reclassification() -> None:
    """ID: AO_SELECT_HARD_002_name_ownership_prevents_reclassification."""
    source = AnalysisObject.from_data(_source(), sequence_dim="sample", core_dims=("axis",))
    for coordinate in ("trial", "axis", "time"):
        with pytest.raises(ValueError, match="not a data variable"):
            source.select_vars(coordinate, validate=False)
    assert tuple(source.as_dataset(copy="none").data_vars) == ("position", "noise")


def test_ao_select_component_001_strict_source_then_truthful_prune() -> None:
    """ID: AO_SELECT_COMPONENT_001_strict_source_then_truthful_prune."""
    from tal.core import ComponentRegistryOptions, ComponentSpec, define_components

    source = AnalysisObject.from_data(_source(), sequence_dim="sample", core_dims=("axis",))
    registered = define_components(
        source,
        opts=ComponentRegistryOptions(
            registry={"position_x": ComponentSpec(core_dim="axis", labels=("x",), var="position")}
        ),
    )
    result = registered.select_vars("noise")
    assert "components" not in result.as_dataset(copy="none").attrs["tal"].get("ext", {})
    assert "components" in registered.as_dataset(copy="none").attrs["tal"]["ext"]


def test_ao_select_index_001_native_index_groups_are_atomic() -> None:
    """ID: AO_SELECT_INDEX_001_native_index_groups_are_atomic."""
    ds = xr.Dataset(
        {
            "value": (("y", "x"), np.arange(4.0).reshape(2, 2)),
            "discard": (("y", "x"), np.ones((2, 2))),
        },
        coords={
            "xx": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]]),
            "yy": (("y", "x"), [[5.0, 6.0], [7.0, 8.0]]),
        },
    ).set_xindex(("xx", "yy"), xr.indexes.NDPointIndex)
    selected = AnalysisLayoutSpec(batch_dims=("y", "x")).wrap(ds, data_vars="value")
    out = selected.as_dataset(copy="none")
    assert out.xindexes["xx"] is out.xindexes["yy"]
    assert out.xindexes["xx"].equals(ds.xindexes["xx"])


def test_ao_select_index_002_multiindex_boundary_is_unchanged() -> None:
    """ID: AO_SELECT_INDEX_002_multiindex_boundary_is_unchanged."""
    source = xr.Dataset(
        {"value": (("row", "col"), np.ones((2, 2))), "discard": (("row", "col"), np.zeros((2, 2)))},
        coords={"row": ["a", "b"], "col": [0, 1]},
    ).stack(sample=("row", "col"))
    with pytest.raises(ValueError, match="PandasMultiIndex"):
        AnalysisLayoutSpec(sequence_dim="sample").wrap(source, data_vars="value")
    with pytest.raises(ValueError, match="PandasMultiIndex"):
        AnalysisObject.from_data(source, sequence_dim="sample")


def test_dataarray_complete_wrap_and_selection_boundary() -> None:
    array = xr.DataArray([1.0, 2.0], dims="sample", coords={"sample": [0, 1]}, name="value")
    layout = AnalysisLayoutSpec(sequence_dim="sample")
    wrapped = layout.wrap(array)
    assert tuple(wrapped.as_dataset(copy="none").data_vars) == ("value",)
    with pytest.raises(TypeError, match="AnalysisLayoutSpec.wrap.*Dataset"):
        layout.wrap(array, data_vars="value")


@pytest.mark.parametrize("validate", [None, 0, "yes"])
def test_ingress_requires_exact_validate_bool(validate: object) -> None:
    with pytest.raises(TypeError, match="validate must be a bool"):
        AnalysisLayoutSpec().wrap(_source(), validate=validate)
    with pytest.raises(TypeError, match="validate must be a bool"):
        AnalysisObject.from_data(_source(), validate=validate)


def test_ao_select_ownership_002_external_selection_isolates_retained_only() -> None:
    """ID: AO_SELECT_OWNERSHIP_002_external_selection_isolates_retained_only."""
    class CopyBomb:
        def __deepcopy__(self, memo: object) -> object:
            raise AssertionError("discarded metadata must not be copied")

    ds = _source()
    ds["noise"].attrs["bomb"] = CopyBomb()
    ds.attrs["tal"] = {"version": 1, "core": {"roles": {"sequence_dim": "missing", "batch_dims": [], "core_dims": []}}}
    with pytest.raises(ValueError, match="AnalysisLayoutSpec.wrap.*duplicate"):
        AnalysisLayoutSpec().wrap(ds, data_vars=("position", "position"))
    ds.attrs.pop("tal")
    result = AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",)).wrap(
        ds, data_vars="position"
    )
    assert tuple(result.as_dataset(copy="none").data_vars) == ("position",)


def test_source_component_registry_is_checked_before_rewrite() -> None:
    from tal.core import ComponentRegistryOptions, ComponentSpec, define_components

    source = AnalysisObject.from_data(_source(), sequence_dim="sample", core_dims=("axis",))
    registered = define_components(
        source,
        opts=ComponentRegistryOptions(
            registry={"position_x": ComponentSpec(core_dim="axis", labels=("x",), var="position")}
        ),
    )
    malformed = registered.as_dataset(copy="shallow")
    malformed.attrs["tal"]["ext"]["components"]["registry"]["position_x"]["var"] = "missing"
    for validate in (False, True):
        with pytest.raises(ValueError, match="missing"):
            AnalysisLayoutSpec().wrap(malformed, validate=validate)
        with pytest.raises(ValueError, match="missing"):
            AnalysisLayoutSpec().wrap(malformed, data_vars="noise", validate=validate)


def test_ao_select_core_004_empty_dimension_topology() -> None:
    """ID: AO_SELECT_CORE_004_empty_dimension_topology."""
    coords = xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(0, dim="trial"))
    ds = xr.Dataset(
        {
            "value": (("trial", "sample"), np.empty((0, 2))),
            "discard": (("trial", "sample"), np.empty((0, 2))),
        },
        coords=coords.assign(sample=("sample", [0, 1])),
    )
    layout = AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",))
    selected = layout.wrap(ds, data_vars="value")
    result = selected.as_dataset(copy="none")
    assert result.sizes == {"trial": 0, "sample": 2}
    assert isinstance(result.xindexes["trial"], xr.indexes.RangeIndex)
    assert result.xindexes["trial"].equals(ds.xindexes["trial"])


def test_ao_select_typed_001_subtype_preserves_or_fails_closed() -> None:
    """ID: AO_SELECT_TYPED_001_subtype_preserves_or_fails_closed."""
    position = Position(
        AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",)).wrap(
            _source().isel(trial=0, drop=True), data_vars="position"
        )
    )
    rotation = Rotation(
        AnalysisLayoutSpec(sequence_dim="sample", core_dims=("quat",)).wrap(
            xr.Dataset(
                {"rotation": (("sample", "quat"), [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])},
                coords={"sample": [0, 1], "quat": ["x", "y", "z", "w"]},
            )
        )
    )
    pose = Pose.from_components(rotation, position)
    assert type(pose.select_vars(("rotation", "position"))) is Pose
    with pytest.raises(ValueError, match="Pose.select_vars") as failure:
        pose.select_vars("position")
    assert isinstance(failure.value.__cause__, ValueError)


def test_selected_caller_only_coordinate_keeps_nonstring_xarray_name() -> None:
    ds = _source().assign_coords({17: ("sample", [4, 5])})
    selected = AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",)).wrap(
        ds, data_vars="position"
    )
    assert 17 in selected.as_dataset(copy="none").coords


class _UntouchedTransform(xr.indexes.CoordinateTransform):
    def __init__(self, calls: list[str], *, dim: str = "axis", size: int = 3) -> None:
        self.calls = calls
        self.dim = dim
        super().__init__((dim,), {dim: size})

    def forward(self, dim_positions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("forward")
        return {self.dim: dim_positions[self.dim]}

    def reverse(self, coord_labels: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("reverse")
        return {self.dim: coord_labels[self.dim]}

    def equals(self, other: object, **kwargs: object) -> bool:
        return isinstance(other, _UntouchedTransform)


def test_ao_select_component_002_transform_core_index_fails_without_execution() -> None:
    """ID: AO_SELECT_COMPONENT_002_transform_core_index_fails_without_execution."""
    from tal.core import merge_schema

    calls: list[str] = []
    coords = xr.Coordinates.from_xindex(xr.indexes.CoordinateTransformIndex(_UntouchedTransform(calls)))
    ds = xr.Dataset({"position": (("sample", "axis"), np.ones((1, 3)))}, coords=coords.assign(sample=("sample", [0])))
    tagged = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",)).as_dataset(copy="none")
    malformed = merge_schema(
        tagged,
        {"ext": {"components": {"version": 1, "registry": {
            "x": {"core_dim": "axis", "labels": ["x"], "var": "position"}
        }}}},
        validate=False,
    )
    with pytest.raises(TypeError, match="pandas-compatible"):
        AnalysisLayoutSpec().wrap(malformed, data_vars="position")
    assert calls == []

    unrelated = xr.Coordinates.from_xindex(
        xr.indexes.CoordinateTransformIndex(_UntouchedTransform(calls, dim="aux", size=2))
    )
    ds = xr.Dataset(
        {"position": (("sample", "axis", "aux"), np.ones((1, 3, 2))),
         "discard": (("sample", "aux"), np.zeros((1, 2)))},
        coords=unrelated.assign(sample=("sample", [0]), axis=("axis", ["x", "y", "z"])),
    )
    from tal.core import ComponentRegistryOptions, ComponentSpec, define_components

    source = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",))
    registered = define_components(
        source,
        opts=ComponentRegistryOptions(
            registry={"x": ComponentSpec(core_dim="axis", labels=("x",), var="position")}
        ),
    )
    calls.clear()
    selected = registered.select_vars("position")
    output = selected.as_dataset(copy="none")
    assert isinstance(output.xindexes["aux"], xr.indexes.CoordinateTransformIndex)
    assert output.xindexes["aux"].equals(registered.as_dataset(copy="none").xindexes["aux"])
    assert list(output.data_vars) == ["position"]
    assert calls == []


def test_external_selected_ingress_does_not_take_caller_resource() -> None:
    ds = _source()
    closed: list[str] = []
    ds.set_close(lambda: closed.append("caller"))
    selected = AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",)).wrap(
        ds, data_vars="position"
    )
    selected.close()
    assert closed == []
    ds.close()
    assert closed == ["caller"]


def test_ao_select_ext_001_unknown_extensions_preserve_and_isolate() -> None:
    """ID: AO_SELECT_EXT_001_unknown_extensions_preserve_and_isolate."""
    from tal.core import merge_schema

    source = AnalysisObject.from_data(_source(), sequence_dim="sample", core_dims=("axis",))
    shared = {"items": ["source"]}
    tagged = merge_schema(
        source.as_dataset(copy="none"),
        {"ext": {"unknown": {"first": shared, "second": shared}}},
        validate=False,
    )
    result = AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",)).wrap(
        tagged, data_vars="position"
    )
    original = tagged.attrs["tal"]["ext"]["unknown"]
    extension = result.as_dataset(copy="none").attrs["tal"]["ext"]["unknown"]
    assert extension["first"] is extension["second"]
    extension["first"]["items"].append("result")
    assert original["first"]["items"] == ["source"]


def test_ao_layout_api_001_no_new_variable_selecting_typed_constructor_surface() -> None:
    """ID: AO_LAYOUT_API_001_no_new_variable_selecting_typed_constructor_surface."""
    assert "data_vars" not in inspect.signature(Position.from_data).parameters
    assert "data_vars" not in inspect.signature(Rotation.from_data).parameters
    assert "layout" in inspect.signature(Position.from_data).parameters
    assert "layout" in inspect.signature(Rotation.from_data).parameters
    assert not hasattr(Position, "from_dataset")


def test_ao_layout_hard_004_public_failure_precedence_and_owner() -> None:
    """ID: AO_LAYOUT_HARD_004_public_failure_precedence_and_owner."""
    ds = _source()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {"roles": {"sequence_dim": "missing", "batch_dims": [], "core_dims": []}},
        "ext": {"components": {"version": 999, "registry": {}}},
    }
    layout = AnalysisLayoutSpec()
    with pytest.raises(ValueError, match="duplicate"):
        layout.wrap(ds, data_vars=("position", "position"))
    with pytest.raises(ValueError, match="not a data variable"):
        layout.wrap(ds, data_vars="unknown")
    with pytest.raises(SchemaError) as error:
        layout.wrap(ds, data_vars="position")
    assert error.value.code == "schema.roles.sequence_dim.not_in_dataset"
    assert error.value.path == "tal.core.roles.sequence_dim"


def test_ao_select_atomic_001_public_failures_leave_all_source_state() -> None:
    """ID: AO_SELECT_ATOMIC_001_public_failures_leave_all_source_state."""
    source = AnalysisObject.from_data(_source(), sequence_dim="sample", core_dims=("axis",))
    before = source.as_dataset()
    with pytest.raises(ValueError, match="not a data variable"):
        source.select_vars("unknown")
    with pytest.raises(ValueError, match="no selected variable or carrier"):
        AnalysisLayoutSpec(sequence_dim="sample", core_dims=("missing",)).wrap(
            source.as_dataset(copy="none"), data_vars="position"
        )
    xr.testing.assert_identical(source.as_dataset(), before)


def test_ao_select_perf_001_retained_payload_isolation_is_bounded() -> None:
    """ID: AO_SELECT_PERF_001_retained_payload_isolation_is_bounded."""
    from benchmarks.bench_layout_selection import fixture

    layout = AnalysisLayoutSpec(sequence_dim="sample")
    eager = fixture(lazy=False)
    lazy = fixture(lazy=True)
    selected = layout.wrap(eager, data_vars=("v00", "v01"))
    lazy_selected = layout.wrap(lazy, data_vars=("v00", "v01"))
    assert tuple(selected.as_dataset(copy="none").data_vars) == ("v00", "v01")
    assert not np.shares_memory(eager["v00"].data, selected.as_dataset(copy="none")["v00"].data)
    selected_tasks = sum(
        len(variable.data.dask) for variable in lazy_selected.as_dataset(copy="none").data_vars.values()
    )
    source_tasks = sum(len(variable.data.dask) for variable in lazy.data_vars.values())
    assert 0 < selected_tasks < source_tasks


@pytest.mark.parametrize("validate", (False, True))
@pytest.mark.parametrize("route", ("from_data", "complete", "selected"))
def test_ao_layout_hard_005_target_failure_precedes_extension_copy(
    route: str, validate: bool
) -> None:
    """ID: AO_LAYOUT_HARD_005_target_failure_precedes_extension_copy."""
    class CopyBomb:
        def __deepcopy__(self, memo: object) -> object:
            raise RuntimeError("extension copy ran before target validation")

    source = AnalysisObject.from_data(_source(), sequence_dim="sample")
    ds = source.as_dataset(copy="none")
    ds.attrs["tal"]["ext"] = {"unknown": CopyBomb()}
    layout = AnalysisLayoutSpec(sequence_dim="sample", param_coord="missing")
    with pytest.raises(SchemaError) as failure:
        if route == "from_data":
            AnalysisObject.from_data(ds, sequence_dim="sample", param_coord="missing", validate=validate)
        elif route == "complete":
            layout.wrap(ds, validate=validate)
        else:
            layout.wrap(ds, data_vars="position", validate=validate)
    assert failure.value.code == "schema.param_coord.not_found"


@pytest.mark.parametrize("validate", (False, True))
@pytest.mark.parametrize("route", ("from_data", "complete", "selected"))
def test_ao_layout_hard_006_source_core_precedes_extension_namespace(
    route: str, validate: bool
) -> None:
    """ID: AO_LAYOUT_HARD_006_core_precedes_extension_namespace."""
    ds = _source()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {"roles": {"sequence_dim": "missing", "batch_dims": [], "core_dims": []}},
        "ext": {0: {}},
    }
    with pytest.raises(SchemaError) as failure:
        if route == "from_data":
            AnalysisObject.from_data(ds, sequence_dim="sample", validate=validate)
        elif route == "complete":
            AnalysisLayoutSpec(sequence_dim="sample").wrap(ds, validate=validate)
        else:
            AnalysisLayoutSpec(sequence_dim="sample").wrap(ds, data_vars="position", validate=validate)
    assert failure.value.code == "schema.roles.sequence_dim.not_in_dataset"
    assert failure.value.path == "tal.core.roles.sequence_dim"


@pytest.mark.parametrize("validate", (False, True))
@pytest.mark.parametrize(
    "route", ("from_data", "complete", "selected", "from_data_array", "complete_array")
)
def test_ao_select_perf_002_component_ingress_copy_bound(route: str, validate: bool) -> None:
    """ID: AO_SELECT_PERF_002_component_ingress_copy_bound."""
    from tal.core import ComponentRegistryOptions, ComponentSpec, define_components

    counter: list[int] = []

    class CopyProbe:
        def __deepcopy__(self, memo: object) -> object:
            counter.append(1)
            return CopyProbe()

    source = AnalysisObject.from_data(_source(), sequence_dim="sample", core_dims=("axis",))
    registered = define_components(source, opts=ComponentRegistryOptions(
        registry={"x": ComponentSpec(core_dim="axis", labels=("x",), var="position")}
    ))
    ds = registered.as_dataset(copy="none")
    ds.attrs["tal"]["ext"]["unknown"] = CopyProbe()
    layout = AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",))
    data = ds
    if route.endswith("array"):
        data = ds["position"].copy(deep=False)
        data.attrs = {**data.attrs, "tal": ds.attrs["tal"]}
    if route == "from_data":
        result = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), validate=validate)
    elif route == "from_data_array":
        result = AnalysisObject.from_data(data, sequence_dim="sample", core_dims=("axis",), validate=validate)
    elif route == "complete":
        result = layout.wrap(ds, validate=validate)
    elif route == "complete_array":
        result = layout.wrap(data, validate=validate)
    else:
        result = layout.wrap(data, data_vars="position", validate=validate)
    assert len(counter) == (3 if validate else 2)
    assert "x" in result.as_dataset(copy="none").attrs["tal"]["ext"]["components"]["registry"]


@pytest.mark.parametrize("field", ("batch_dims", "core_dims"))
@pytest.mark.parametrize("invalid", (None, {}, set(), "", False, 0))
def test_ao_select_hard_004_false_role_options_reject_before_defaulting(field: str, invalid: object) -> None:
    """ID: AO_SELECT_HARD_004_false_role_options_reject."""
    with pytest.raises(TypeError, match="AnalysisObject.from_data"):
        AnalysisObject.from_data(_source(), **{field: invalid})


@pytest.mark.parametrize("empty", ([], ()))
def test_ao_select_hard_005_empty_role_sequences_inherit(empty: object) -> None:
    """ID: AO_SELECT_HARD_005_empty_role_sequences_inherit."""
    source = AnalysisObject.from_data(_source(), sequence_dim="sample", core_dims=("axis",))
    result = AnalysisObject.from_data(source.as_dataset(copy="none"), core_dims=empty)
    assert result.as_dataset(copy="none").attrs["tal"]["core"]["roles"]["core_dims"] == ["axis"]


@pytest.mark.parametrize("names,expected", [([1], TypeError), ([""], ValueError)])
def test_ao_select_hard_006_item_error_classification(names: object, expected: type[Exception]) -> None:
    """ID: AO_SELECT_HARD_006_item_error_classification."""
    source = AnalysisObject.from_data(_source())
    with pytest.raises(expected, match="AnalysisObject.select_vars"):
        source.select_vars(names)
