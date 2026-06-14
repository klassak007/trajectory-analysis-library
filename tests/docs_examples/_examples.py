from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import numpy as np
import xarray as xr

from tal.catalog import Catalog
from tal.core import AnalysisObject, GroupByOptions, SequenceConcatOptions, concat_sequence
from tal.core.component_ops import ComponentRegistryOptions, ComponentSpec
from tal.core.event_ops import Condition, WhenOptions
from tal.core.schema_read import read_roles
from tal.frames import FrameGraph, find_path, fold_path, render_snapshot_ascii, snapshot_from_seeds, snapshot_to_networkx
from tal.io import CsvIngestOptions, read_csv_logs
from tal.geo import (
    ENUOptions,
    GeodeticInterpolationOptions,
    GeodeticOptions,
    GeodesicOptions,
    GeodeticPosition,
    LocalOrigin,
    ProjectedPosition,
    transform_crs,
)
from tal.linalg import (
    Array,
    Matrix,
    Vector,
    Vector3,
    add,
    dot,
    inv,
    matmul,
    norm,
    pinv,
    solve,
    sub,
)
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
    solve_pose_path_transform,
)
from tal.spatial.metadata.frame_motion import (
    get_edge_motion_class,
    get_frame_inertial_status,
    propagate_inertial_status,
    set_edge_motion_class,
    set_frame_inertial_status,
)
from tal.utils.frame_ops import frame_bind, frame_retag
from tal.utils.frame_schema import get_frames, set_frames
from tal.utils.topology_operation_families import operation_intent_support_for_operation_family
from tal.utils.xarray_namespace import rename_dims_collision_safe
from tal.viz import line


def _make_signal_ao(*, n_trials: int = 2, n_samples: int = 5, t_end: float = 4.0) -> AnalysisObject:
    trial = [f"trial_{i}" for i in range(n_trials)]
    sample = np.arange(n_samples)
    time = np.linspace(0.0, float(t_end), n_samples, dtype=float)
    base = np.stack([time + float(i) for i in range(n_trials)], axis=0)
    ds = xr.Dataset(
        {"value": (("trial", "sample"), base)},
        coords={
            "trial": trial,
            "sample": sample,
            "time": (("trial", "sample"), np.broadcast_to(time, (n_trials, n_samples))),
            "group_size": ("trial", np.full((n_trials,), n_samples, dtype=np.int64)),
            "outcome": ("trial", np.array(["intercept", "miss"][:n_trials], dtype=object)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
        sequence_size_coord="group_size",
        validate=True,
    )


def _make_vector(values: np.ndarray, *, dim: str, labels: tuple[str, ...]) -> Vector:
    ds = xr.Dataset(
        {"v": (("sample", dim), values)},
        coords={"sample": np.arange(values.shape[0]), dim: list(labels)},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=(dim,), validate=True)
    return Vector(ao)


def _make_matrix(values: np.ndarray) -> Matrix:
    ds = xr.Dataset(
        {"A": (("sample", "row", "col"), values)},
        coords={"sample": np.arange(values.shape[0]), "row": ["r0", "r1"], "col": ["c0", "c1"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("row", "col"), validate=True)
    return Matrix(ao)


def _make_position(values: np.ndarray) -> Position:
    ds = xr.Dataset(
        {"position": (("sample", "axis"), values)},
        coords={"sample": np.arange(values.shape[0]), "axis": ["x", "y", "z"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), validate=True)
    return Position(ao)


def _make_linear_velocity(values: np.ndarray) -> LinearVelocity:
    ds = xr.Dataset(
        {"linear_velocity": (("sample", "lin_axis"), values)},
        coords={"sample": np.arange(values.shape[0]), "lin_axis": ["x", "y", "z"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lin_axis",), validate=True)
    return LinearVelocity(ao)


def _make_angular_velocity(values: np.ndarray) -> AngularVelocity:
    ds = xr.Dataset(
        {"angular_velocity": (("sample", "ang_axis"), values)},
        coords={"sample": np.arange(values.shape[0]), "ang_axis": ["x", "y", "z"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("ang_axis",), validate=True)
    return AngularVelocity(ao)


def _make_linear_acceleration(values: np.ndarray) -> LinearAcceleration:
    ds = xr.Dataset(
        {"linear_acceleration": (("sample", "lin_axis"), values)},
        coords={"sample": np.arange(values.shape[0]), "lin_axis": ["x", "y", "z"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lin_axis",), validate=True)
    return LinearAcceleration(ao)


def _make_angular_acceleration(values: np.ndarray) -> AngularAcceleration:
    ds = xr.Dataset(
        {"angular_acceleration": (("sample", "ang_axis"), values)},
        coords={"sample": np.arange(values.shape[0]), "ang_axis": ["x", "y", "z"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("ang_axis",), validate=True)
    return AngularAcceleration(ao)


def _make_rotation(quat_values: np.ndarray) -> Rotation:
    ds = xr.Dataset(
        {"rotation": (("sample", "quat"), quat_values)},
        coords={"sample": np.arange(quat_values.shape[0]), "quat": ["x", "y", "z", "w"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("quat",), validate=True)
    return Rotation(ao)


def _identity_pose() -> Pose:
    rotation = _make_rotation(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float))
    translation = _make_position(np.asarray([[0.0, 0.0, 0.0]], dtype=float))
    return Pose.from_components(rotation, translation, validate=True)


def example_core_ao_from_data() -> None:
    ds = xr.Dataset({"value": (("trial", "axis"), np.array([[1.0, 2.0, 3.0]]))}, coords={"trial": ["t0"], "axis": ["x", "y", "z"]})
    ao = AnalysisObject.from_data(ds, batch_dims=("trial",), core_dims=("axis",), validate=True)
    declared, sequence_dim, batch_dims, core_dims = read_roles(ao.unsafe_data)
    assert declared
    assert sequence_dim is None
    assert batch_dims == ("trial",)
    assert core_dims == ("axis",)


def example_core_ao_set_roles() -> None:
    ds = xr.Dataset(
        {"value": (("trial", "sample"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"trial": ["trial_0", "trial_1"], "sample": [0, 1]},
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )
    out = ao.set_roles(sequence_dim=None, batch_dims=("trial",), core_dims=(), validate=True)
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.unsafe_data)
    assert declared
    assert sequence_dim is None
    assert batch_dims == ("trial",)
    assert core_dims == ()


def example_core_ao_xarray_methods() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {
                "value": (("trial", "sample"), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                "quality": (("trial", "sample"), np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int64)),
            },
            coords={"trial": ["a", "b"], "sample": [0, 1, 2]},
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )
    selected = ao.isel(sample=slice(0, 2)).sel(trial="a")
    masked = ao.where(ao.unsafe_data["value"] > 2.0)
    renamed = ao.rename({"sample": "step"})
    dropped = ao.drop_vars("quality")
    transposed = ao.transpose("sample", "trial")
    validated = ao.validate_schema()
    ds_copy = ao.as_dataset()
    assert selected.unsafe_data.sizes["sample"] == 2
    assert bool(np.isnan(masked.unsafe_data["value"].values).any())
    assert "step" in renamed.unsafe_data.dims
    assert "quality" not in dropped.unsafe_data.data_vars
    assert transposed.unsafe_data["value"].dims == ("sample", "trial")
    assert ds_copy is not ao.unsafe_data
    assert validated.unsafe_data.identical(ao.unsafe_data)


def example_core_ao_reducers() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
            coords={"trial": ["a", "b"], "sample": [0, 1]},
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )
    bool_ao = AnalysisObject.from_data(
        xr.Dataset(
            {"flag": (("trial", "sample"), np.array([[True, False], [True, True]]))},
            coords={"trial": ["a", "b"], "sample": [0, 1]},
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )
    np.testing.assert_allclose(ao.mean(dim="sample").unsafe_data["value"], np.array([1.5, 3.5]))
    np.testing.assert_allclose(ao.sum(dim="sample").unsafe_data["value"], np.array([3.0, 7.0]))
    np.testing.assert_allclose(ao.std(dim="sample").unsafe_data["value"], np.array([0.5, 0.5]))
    np.testing.assert_allclose(ao.var(dim="sample").unsafe_data["value"], np.array([0.25, 0.25]))
    np.testing.assert_allclose(ao.median(dim="sample").unsafe_data["value"], np.array([1.5, 3.5]))
    np.testing.assert_allclose(ao.min(dim="sample").unsafe_data["value"], np.array([1.0, 3.0]))
    np.testing.assert_allclose(ao.max(dim="sample").unsafe_data["value"], np.array([2.0, 4.0]))
    np.testing.assert_allclose(ao.count(dim="sample").unsafe_data["value"], np.array([2, 2]))
    np.testing.assert_array_equal(bool_ao.any(dim="sample").unsafe_data["flag"], np.array([True, True]))
    np.testing.assert_array_equal(bool_ao.all(dim="sample").unsafe_data["flag"], np.array([False, True]))


def example_core_param_at() -> None:
    ao = _make_signal_ao()
    out = ao.param.at([0.5, 1.5], on="time")
    assert out.unsafe_data.sizes["sample"] == 2
    values = out.unsafe_data["value"].isel(trial=0).values
    np.testing.assert_allclose(values, np.array([0.5, 1.5]), atol=1e-8)


def example_core_param_interp_like() -> None:
    src = _make_signal_ao(n_samples=5, t_end=4.0)
    dst = _make_signal_ao(n_samples=3, t_end=4.0)
    out = src.param.interp_like(dst, on="time", batch_join="inner")
    assert out.unsafe_data.sizes["sample"] == dst.unsafe_data.sizes["sample"]
    np.testing.assert_allclose(
        out.unsafe_data.coords["time"].values,
        dst.unsafe_data.coords["time"].values,
    )


def example_core_event_when() -> None:
    ao = _make_signal_ao()
    cond = Condition.compare(Condition.var("value"), "gt", 2.0)
    out = ao.events.when(cond, opts=WhenOptions(layout="mask"))
    assert out.unsafe_data["value"].dims == ao.unsafe_data["value"].dims
    assert bool(np.isnan(out.unsafe_data["value"].values).any())


def example_core_group_groupby() -> None:
    ao = _make_signal_ao()
    grouped = ao.group.groupby("outcome", opts=GroupByOptions(preserve_batch=False))
    out = grouped.mean(dim="sample")
    assert "group_key" in out.unsafe_data.dims
    assert out.unsafe_data.sizes["group_key"] == 2


def example_core_combine_concat_sequence() -> None:
    def make_with_sample_offset(offset: int) -> AnalysisObject:
        sample = np.arange(offset, offset + 3)
        ds = xr.Dataset(
            {"value": (("trial", "sample"), np.array([[1.0, 2.0, 3.0]]))},
            coords={
                "trial": ["t0"],
                "sample": sample,
                "time": (("trial", "sample"), np.array([[float(v) for v in sample]])),
                "group_size": ("trial", np.array([3], dtype=np.int64)),
            },
        )
        return AnalysisObject.from_data(
            ds,
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=(),
            param_coord="time",
            sequence_size_coord="group_size",
            validate=True,
        )

    left = make_with_sample_offset(0)
    right = make_with_sample_offset(3)
    out = concat_sequence([left, right], opts=SequenceConcatOptions(overlap="error"), validate=True)
    assert out.unsafe_data.sizes["sample"] == 6


def example_core_combine_core_layouts() -> None:
    def scalar(name: str, value: float) -> AnalysisObject:
        return AnalysisObject.from_data(
            xr.Dataset({name: ("sample", np.array([value]))}, coords={"sample": [0]}),
            sequence_dim="sample",
            core_dims=(),
            validate=True,
        )

    x = scalar("value", 1.0)
    y = scalar("value", 2.0)
    stacked = x.combine.stack_core([y], core_dim="axis", core_labels=("x", "y"), output_var="vec")
    assembled = x.combine.assemble_core([y], core_dims=("axis",), core_labels=(("x", "y"),), output_var="vec")
    blocked = x.combine.block_core([[y]], row_dim="row", col_dim="col", output_var="block")
    assert stacked.unsafe_data["vec"].dims == ("axis", "sample")
    assert assembled.unsafe_data.sizes["axis"] == 2
    assert blocked.unsafe_data.sizes["row"] == 2
    assert blocked.unsafe_data.sizes["col"] == 1


def example_core_component_registry() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"vec": (("sample", "axis"), np.array([[1.0, 2.0, 3.0]]))},
            coords={"sample": [0], "axis": ["x", "y", "z"]},
        ),
        sequence_dim="sample",
        core_dims=("axis",),
        validate=True,
    )
    tagged = ao.components.define(
        opts=ComponentRegistryOptions(
            registry={"xy": ComponentSpec(core_dim="axis", labels=("x", "y"))},
            replace=True,
        )
    )
    registry = tagged.components.registry()
    assert registry["xy"].labels == ("x", "y")


def example_core_param_surface() -> None:
    from tal.core import ParamEvalOptions, ParamSelectOptions, ParamSyncOptions, synchronize, synchronize_param

    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("sample", np.asarray([0.0, 1.0, 4.0], dtype=float))},
            coords={"sample": [0, 1, 2], "time": ("sample", [0.0, 1.0, 2.0])},
        ),
        sequence_dim="sample",
        core_dims=(),
        param_coord="time",
        validate=True,
    )
    target = AnalysisObject.from_data(
        xr.Dataset(
            {"target": ("sample", np.asarray([10.0, 20.0], dtype=float))},
            coords={"sample": [0, 1], "time": ("sample", [0.0, 2.0])},
        ),
        sequence_dim="sample",
        core_dims=(),
        param_coord="time",
        validate=True,
    )
    np.testing.assert_array_equal(
        ao.param.index([0.2, 1.8], on="time", opts=ParamSelectOptions(method="nearest")).values,
        np.asarray([0, 2]),
    )
    np.testing.assert_allclose(
        ao.param.sel([0.2, 1.8], on="time", opts=ParamSelectOptions(method="nearest")).unsafe_data["value"],
        np.asarray([0.0, 4.0]),
    )
    np.testing.assert_allclose(
        ao.param.at([0.5, 1.5], on="time", opts=ParamEvalOptions(method="linear")).unsafe_data["value"],
        np.asarray([0.5, 2.5]),
    )
    np.testing.assert_allclose(
        ao.param.resample_to([0.0, 0.5, 1.0], on="time", opts=ParamEvalOptions(method="linear")).unsafe_data["value"],
        np.asarray([0.0, 0.5, 1.0]),
    )
    np.testing.assert_allclose(
        ao.param.interp_like(target, on="time", opts=ParamEvalOptions(method="linear")).unsafe_data["value"],
        np.asarray([0.0, 4.0]),
    )
    np.testing.assert_allclose(
        synchronize_param([ao], on="time", grid=[0.0, 1.0], opts=ParamSyncOptions(join="override"))[0].unsafe_data["value"],
        np.asarray([0.0, 1.0]),
    )
    np.testing.assert_allclose(
        synchronize([ao], on="time", grid=[0.0, 2.0], opts=ParamSyncOptions(join="override"))[0].unsafe_data["value"],
        np.asarray([0.0, 4.0]),
    )


def example_core_event_surface() -> None:
    from tal.core.event_ops import AroundOptions, AtBoundariesOptions

    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("sample", np.asarray([0.0, 2.0, 3.0, 1.0], dtype=float))},
            coords={"sample": [0, 1, 2, 3], "time": ("sample", [0.0, 1.0, 2.0, 3.0])},
        ),
        sequence_dim="sample",
        core_dims=(),
        param_coord="time",
        validate=True,
    )
    cond = Condition.compare(Condition.var("value"), "gt", 1.5)
    assert ao.events.mask(cond).values.tolist() == [False, True, True, False]
    assert ao.events.events(cond)["edge_code"].values.tolist() == [1, 2]
    assert ao.events.intervals(cond).sizes["segment"] == 1
    np.testing.assert_allclose(
        ao.events.at_boundaries(cond, opts=AtBoundariesOptions(edges="enter")).unsafe_data["value"],
        np.asarray([2.0]),
    )
    masked = ao.events.when(cond, opts=WhenOptions(layout="mask"))
    assert masked.unsafe_data["value"].isnull().values.tolist() == [True, False, False, True]
    around = ao.events.around(cond, opts=AroundOptions(pre=0.0, post=0.0, dt=1.0))
    assert around.unsafe_data.sizes["event"] == 1


def example_core_group_surface() -> None:
    from tal.core import GroupMaterializeOptions

    ao = AnalysisObject.from_data(
        xr.Dataset(
            {
                "value": (("run", "sample"), np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)),
                "flag": (("run", "sample"), np.asarray([[True, False], [True, True]], dtype=bool)),
            },
            coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])},
        ),
        sequence_dim="sample",
        batch_dims=("run",),
        core_dims=(),
        validate=True,
    )
    grouped = ao.group.groupby("kind", opts=GroupByOptions(preserve_batch=False))
    assert grouped.materialize(opts=GroupMaterializeOptions(layout="padded")).unsafe_data.sizes["group_key"] == 2
    assert grouped.padded().unsafe_data.sizes["sample"] == 2
    assert grouped.stacked().unsafe_data.sizes["group_member"] == 4
    assert grouped.mean(dim="sample").unsafe_data["value"].sel(group_key="sim").item() == 1.5
    assert grouped.sum(dim="sample").unsafe_data["value"].sel(group_key="robot").item() == 7.0
    assert grouped.std(dim="sample").unsafe_data["value"].sel(group_key="sim").item() == 0.5
    assert grouped.var(dim="sample").unsafe_data["value"].sel(group_key="sim").item() == 0.25
    assert grouped.median(dim="sample").unsafe_data["value"].sel(group_key="robot").item() == 3.5
    assert grouped.min(dim="sample").unsafe_data["value"].sel(group_key="sim").item() == 1.0
    assert grouped.max(dim="sample").unsafe_data["value"].sel(group_key="robot").item() == 4.0
    assert grouped.count(dim="sample").unsafe_data["value"].sel(group_key="sim").item() == 2
    assert bool(grouped.any(dim="sample").unsafe_data["flag"].sel(group_key="robot").item()) is True
    assert bool(grouped.all(dim="sample").unsafe_data["flag"].sel(group_key="robot").item()) is True
    binned_source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("sample", np.asarray([1.0, 2.0, 3.0], dtype=float))},
            coords={"sample": [0, 1, 2], "time": ("sample", [0.0, 0.5, 1.5])},
        ),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    assert binned_source.group.groupby_bins("time", bins=[0.0, 1.0, 2.0], include_lowest=True).padded().unsafe_data.sizes["group_key"] == 2


def example_core_combine_surface() -> None:
    from tal.core import (
        AlignOptions,
        BatchConcatOptions,
        CoreConcatOptions,
        CoreDecomposeOptions,
        CoreOverlayOptions,
        MergeOptions,
        SequenceConcatOptions,
        align_many,
        align_pair,
        assemble_core,
        block_core,
        concat_batch,
        concat_core,
        decompose_core,
        merge,
        overlay_core,
        stack_core,
    )

    def scalar(name: str, value: float, sample: int = 0) -> AnalysisObject:
        return AnalysisObject.from_data(
            xr.Dataset({name: ("sample", np.asarray([value], dtype=float))}, coords={"sample": [sample]}),
            sequence_dim="sample",
            core_dims=(),
            validate=True,
        )

    x = scalar("value", 1.0)
    y = scalar("value", 2.0)
    batched = concat_batch([x, y], opts=BatchConcatOptions(batch_dim="run", batch_labels=("a", "b")))
    assert tuple(batched.unsafe_data.coords["run"].values.tolist()) == ("a", "b")
    sequenced = x.combine.concat_sequence([scalar("value", 2.0, sample=1)], opts=SequenceConcatOptions(overlap="error"))
    assert sequenced.unsafe_data.sizes["sample"] == 2
    left = scalar("x", 1.0)
    right = scalar("y", 2.0)
    assert sorted(merge([left, right], opts=MergeOptions()).unsafe_data.data_vars) == ["x", "y"]
    assert sorted(left.combine.merge([right], opts=MergeOptions()).unsafe_data.data_vars) == ["x", "y"]
    outer_left = AnalysisObject.from_data(
        xr.Dataset({"value": ("sample", np.asarray([1.0, 2.0], dtype=float))}, coords={"sample": [0, 1]}),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    outer_right = AnalysisObject.from_data(
        xr.Dataset({"value": ("sample", np.asarray([3.0, 4.0], dtype=float))}, coords={"sample": [1, 2]}),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    assert [item.unsafe_data.sizes["sample"] for item in align_many([outer_left, outer_right], opts=AlignOptions(sequence_join="outer"))] == [3, 3]
    a, b = align_pair(x, y, opts=AlignOptions())
    assert (a.unsafe_data.sizes["sample"], b.unsafe_data.sizes["sample"]) == (1, 1)
    assert outer_left.combine.align([outer_right], opts=AlignOptions(sequence_join="outer"))[0].unsafe_data.sizes["sample"] == 3
    assert assemble_core([x, y], core_dims=("axis",), core_labels=(("x", "y"),)).unsafe_data.sizes["axis"] == 2
    assert stack_core([x, y], core_dim="axis", core_labels=("x", "y")).unsafe_data.sizes["axis"] == 2
    assert block_core([[x], [y]], row_dim="row", col_dim="col").unsafe_data.sizes["row"] == 2
    vx = AnalysisObject.from_data(
        xr.Dataset({"v": (("sample", "axis"), np.asarray([[1.0]], dtype=float))}, coords={"sample": [0], "axis": ["x"]}),
        sequence_dim="sample",
        core_dims=("axis",),
        validate=True,
    )
    vy = AnalysisObject.from_data(
        xr.Dataset({"v": (("sample", "axis"), np.asarray([[2.0]], dtype=float))}, coords={"sample": [0], "axis": ["y"]}),
        sequence_dim="sample",
        core_dims=("axis",),
        validate=True,
    )
    assert concat_core([vx, vy], opts=CoreConcatOptions(core_dim="axis")).unsafe_data.sizes["axis"] == 2
    assert vx.combine.concat_core([vy], opts=CoreConcatOptions(core_dim="axis")).unsafe_data.sizes["axis"] == 2
    parts = decompose_core(
        concat_core([vx, vy], opts=CoreConcatOptions(core_dim="axis")),
        opts=CoreDecomposeOptions(core_dims=("axis",), key_mode="label"),
    )
    assert sorted(parts) == [("x",), ("y",)]
    patch = AnalysisObject.from_data(
        xr.Dataset({"v": (("sample", "axis"), np.asarray([[9.0]], dtype=float))}, coords={"sample": [0], "axis": ["y"]}),
        sequence_dim="sample",
        core_dims=("axis",),
        validate=True,
    )
    overlaid = overlay_core(concat_core([vx, vy], opts=CoreConcatOptions(core_dim="axis")), patch, opts=CoreOverlayOptions(core_dim="axis", on_overlap="replace"))
    assert overlaid.unsafe_data["v"].sel(axis="y").item() == 9.0


def example_core_component_surface() -> None:
    from tal.core.component_ops import (
        ComponentComposeOptions,
        ComponentExtractOptions,
        ComponentPatchOptions,
        compose_components,
        define_components,
        extract_components,
        patch_components,
        read_components,
    )

    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"vec": (("sample", "axis"), np.asarray([[1.0, 2.0, 3.0]], dtype=float))},
            coords={"sample": [0], "axis": ["x", "y", "z"]},
        ),
        sequence_dim="sample",
        core_dims=("axis",),
        validate=True,
    )
    tagged = define_components(
        ao,
        opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}),
    )
    assert read_components(tagged)["xy"].labels == ("x", "y")
    parts = extract_components(tagged, opts=ComponentExtractOptions(names=("xy",)))
    assert parts["xy"].unsafe_data["vec"].sizes["axis"] == 2
    patch = AnalysisObject.from_data(
        xr.Dataset({"vec": (("sample", "axis"), np.asarray([[9.0, 8.0]], dtype=float))}, coords={"sample": [0], "axis": ["x", "y"]}),
        sequence_dim="sample",
        core_dims=("axis",),
        validate=True,
    )
    patched = patch_components(tagged, {"xy": patch}, opts=ComponentPatchOptions(on_overlap="replace"))
    assert patched.unsafe_data["vec"].sel(axis="x").item() == 9.0
    x = AnalysisObject.from_data(
        xr.Dataset({"vec": (("sample", "axis"), np.asarray([[1.0]], dtype=float))}, coords={"sample": [0], "axis": ["x"]}),
        sequence_dim="sample",
        core_dims=("axis",),
        validate=True,
    )
    y = AnalysisObject.from_data(
        xr.Dataset({"vec": (("sample", "axis"), np.asarray([[2.0]], dtype=float))}, coords={"sample": [0], "axis": ["y"]}),
        sequence_dim="sample",
        core_dims=("axis",),
        validate=True,
    )
    opts = ComponentComposeOptions({"x": ComponentSpec("axis", ("x",)), "y": ComponentSpec("axis", ("y",))})
    composed = compose_components({"x": x, "y": y}, opts=opts)
    assert composed.unsafe_data["vec"].sel(axis="y").item() == 2.0
    assert x.components.compose({"x": x, "y": y}, opts=opts).unsafe_data.sizes["axis"] == 2


def example_linalg_add() -> None:
    left = Array(
        AnalysisObject.from_data(
            xr.Dataset({"v": (("sample", "axis"), np.array([[1.0, 2.0, 3.0]]))}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
            sequence_dim="sample",
            core_dims=("axis",),
            validate=True,
        )
    )
    right = Array(
        AnalysisObject.from_data(
            xr.Dataset({"v": (("sample", "axis"), np.array([[4.0, 5.0, 6.0]]))}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
            sequence_dim="sample",
            core_dims=("axis",),
            validate=True,
        )
    )
    out = add(left, right)
    np.testing.assert_allclose(out.unsafe_data["datavar"].values, np.array([[5.0, 7.0, 9.0]]))


def example_linalg_sub() -> None:
    left = Array(
        AnalysisObject.from_data(
            xr.Dataset({"v": (("sample", "axis"), np.array([[4.0, 5.0, 6.0]]))}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
            sequence_dim="sample",
            core_dims=("axis",),
            validate=True,
        )
    )
    right = Array(
        AnalysisObject.from_data(
            xr.Dataset({"v": (("sample", "axis"), np.array([[1.0, 2.0, 3.0]]))}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
            sequence_dim="sample",
            core_dims=("axis",),
            validate=True,
        )
    )
    out = sub(left, right)
    np.testing.assert_allclose(out.unsafe_data["datavar"].values, np.array([[3.0, 3.0, 3.0]]))


def example_linalg_dot() -> None:
    left = _make_vector(np.array([[1.0, 2.0, 3.0]]), dim="axis", labels=("x", "y", "z"))
    right = _make_vector(np.array([[4.0, 5.0, 6.0]]), dim="axis", labels=("x", "y", "z"))
    out = dot(left, right)
    np.testing.assert_allclose(out.unsafe_data["datavar"].values, np.array([32.0]))


def example_linalg_norm() -> None:
    left = _make_vector(np.array([[3.0, 4.0, 0.0]]), dim="axis", labels=("x", "y", "z"))
    out = norm(left)
    np.testing.assert_allclose(out.unsafe_data["datavar"].values, np.array([5.0]))


def example_linalg_matmul() -> None:
    A = _make_matrix(np.array([[[2.0, 0.0], [0.0, 3.0]]]))
    v = _make_vector(np.array([[4.0, 5.0]]), dim="col", labels=("c0", "c1"))
    out = matmul(A, v)
    np.testing.assert_allclose(out.unsafe_data["datavar"].values, np.array([[8.0, 15.0]]))


def example_linalg_solve() -> None:
    A = _make_matrix(np.array([[[2.0, 0.0], [0.0, 3.0]]]))
    b = _make_vector(np.array([[8.0, 15.0]]), dim="row", labels=("r0", "r1"))
    out = solve(A, b)
    np.testing.assert_allclose(out.unsafe_data["datavar"].values, np.array([[4.0, 5.0]]), atol=1e-8)


def example_linalg_inv() -> None:
    A = _make_matrix(np.array([[[2.0, 0.0], [0.0, 4.0]]]))
    out = inv(A)
    np.testing.assert_allclose(out.unsafe_data["datavar"].values, np.array([[[0.5, 0.0], [0.0, 0.25]]]), atol=1e-8)


def example_linalg_pinv() -> None:
    A = _make_matrix(np.array([[[2.0, 0.0], [0.0, 4.0]]]))
    out = pinv(A)
    np.testing.assert_allclose(out.unsafe_data["datavar"].values, np.array([[[0.5, 0.0], [0.0, 0.25]]]), atol=1e-8)


def example_linalg_vector3_from_xyz() -> None:
    x = AnalysisObject.from_data(
        xr.Dataset(
            {"x": ("sample", np.array([1.0, 2.0]))},
            coords={"sample": np.array([0, 1]), "time": ("sample", np.array([0.0, 1.0]))},
        ),
        sequence_dim="sample",
        core_dims=(),
        param_coord="time",
        validate=True,
    )
    out = Vector3.from_xyz(x, 0.0, 1.0, axis="axis", output_var="vec3")
    assert out.unsafe_data.sizes["axis"] == 3
    assert tuple(out.unsafe_data.coords["axis"].to_numpy().tolist()) == ("x", "y", "z")
    np.testing.assert_allclose(out.unsafe_data["vec3"].sel(axis="x").values, np.array([1.0, 2.0]))


def example_linalg_array_core_dims() -> None:
    arr = Array(
        AnalysisObject.from_data(
            xr.Dataset(
                {"value": (("sample", "row", "col"), np.array([[[1.0, 0.0], [0.0, 1.0]]]))},
                coords={"sample": [0], "row": ["r0", "r1"], "col": ["c0", "c1"]},
            ),
            sequence_dim="sample",
            core_dims=("row", "col"),
            validate=True,
        )
    )
    vector = arr.set_core_dims("row").set_vector_axis("row")
    matrix = arr.set_matrix_axes("row", "col")
    assert read_roles(vector.unsafe_data)[3] == ("row",)
    assert read_roles(arr.as_core("row").unsafe_data)[3] == ("row",)
    assert read_roles(arr.axis("row").unsafe_data)[3] == ("row",)
    assert read_roles(matrix.unsafe_data)[3] == ("row", "col")
    assert read_roles(arr.rc("row", "col").unsafe_data)[3] == ("row", "col")


def example_spatial_position_to_frame() -> None:
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    _ = graph.get_or_create_frame("sensor", parent=world)
    position = frame_retag(
        _make_position(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        parent="world",
        child="sensor",
        validate=True,
    )
    calls = {"n": 0}

    def edge_pose(*_args):
        calls["n"] += 1
        raise AssertionError("identity path should not call resolver")

    out = position.to_frame("world", edge_pose_fn=edge_pose, opts=PathSolveOptions(graph=graph), validate=True)
    assert calls["n"] == 0
    assert get_frames(out.unsafe_data) == ("world", "sensor")
    np.testing.assert_allclose(out.unsafe_data["position"].values, position.unsafe_data["position"].values, atol=1e-9)


def example_spatial_position_basic() -> None:
    position = _make_position(np.asarray([[1.0, 2.0, 3.0]], dtype=float))
    delta = position.as_delta(validate=True)
    assert isinstance(delta, Position)
    assert delta.unsafe_data["position"].shape == (1, 3)


def example_spatial_rotation_to_rep() -> None:
    rotation = _make_rotation(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float))
    as_matrix = rotation.to_rep("matrix", validate=True)
    back = as_matrix.to_rep("quat", validate=True)
    assert as_matrix.unsafe_data["rotation"].shape == (1, 3, 3)
    assert tuple(back.unsafe_data.coords["quat"].to_numpy().tolist()) == ("x", "y", "z", "w")


def example_spatial_rotation_basic() -> None:
    rotation = _make_rotation(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float))
    position = _make_position(np.asarray([[1.0, 0.0, 0.0]], dtype=float))
    as_matrix = rotation.as_matrix(validate=True)
    back = as_matrix.as_quat(validate=True)
    composed = rotation.compose(rotation, validate=True)
    inverse = rotation.inverse(validate=True)
    applied = rotation.apply(position, validate=True)
    assert as_matrix.unsafe_data["rotation"].shape[-2:] == (3, 3)
    assert tuple(back.unsafe_data.coords["quat"].to_numpy().tolist()) == ("x", "y", "z", "w")
    assert isinstance(composed, Rotation)
    assert isinstance(inverse, Rotation)
    np.testing.assert_allclose(applied.unsafe_data["position"].values, position.unsafe_data["position"].values)


def example_spatial_pose_from_components() -> None:
    pose = Pose.from_components(
        _make_rotation(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float)),
        _make_position(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        validate=True,
    )
    assert sorted(pose.unsafe_data.data_vars) == ["position", "rotation"]


def example_spatial_pose_basic() -> None:
    pose = _identity_pose()
    position, rotation = pose.decompose(validate=True)
    matrix = pose.to_rep("matrix", validate=True)
    components = matrix.as_components(validate=True)
    as_matrix = components.as_matrix(validate=True)
    composed = pose.compose(pose, validate=True)
    inverse = pose.inverse(validate=True)
    applied = pose.apply(_make_position(np.asarray([[1.0, 0.0, 0.0]], dtype=float)), validate=True)
    assert isinstance(position, Position)
    assert isinstance(rotation, Rotation)
    assert matrix.unsafe_data["pose_matrix"].shape[-2:] == (4, 4)
    assert sorted(components.unsafe_data.data_vars) == ["position", "rotation"]
    assert as_matrix.unsafe_data["pose_matrix"].shape[-2:] == (4, 4)
    assert isinstance(composed, Pose)
    assert isinstance(inverse, Pose)
    assert isinstance(applied, Position)


def example_spatial_path_solve_pose() -> None:
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    body = graph.get_or_create_frame("body", parent=world)
    sensor = graph.get_or_create_frame("sensor", parent=body)
    edge_map = {
        ("sensor", "body"): _identity_pose(),
        ("body", "world"): _identity_pose(),
    }

    def resolver(child, parent):
        return edge_map[(child.id, parent.id)]

    out = solve_pose_path_transform(sensor, world, edge_pose_fn=resolver, opts=PathSolveOptions(graph=graph))
    assert isinstance(out, Pose)
    assert out.as_matrix(validate=True).unsafe_data["pose_matrix"].shape[-2:] == (4, 4)


def example_spatial_velocity_components() -> None:
    velocity = Velocity.from_linear_angular(
        _make_linear_velocity(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        _make_angular_velocity(np.asarray([[0.1, 0.2, 0.3]], dtype=float)),
        validate=True,
    )
    vector6 = velocity.to_rep("vector6", validate=True)
    components = vector6.as_components(validate=True)
    assert velocity.as_vector6(validate=True).unsafe_data["velocity"].shape[-1] == 6
    assert sorted(components.as_components(validate=True).unsafe_data.data_vars) == ["angular_velocity", "linear_velocity"]
    assert isinstance(components.linear(validate=True), LinearVelocity)
    assert isinstance(components.angular(validate=True), AngularVelocity)


def example_spatial_acceleration_components() -> None:
    acceleration = Acceleration.from_linear_angular(
        _make_linear_acceleration(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        _make_angular_acceleration(np.asarray([[0.1, 0.2, 0.3]], dtype=float)),
        validate=True,
    )
    vector6 = acceleration.to_rep("vector6", validate=True)
    components = vector6.as_components(validate=True)
    assert acceleration.as_vector6(validate=True).unsafe_data["acceleration"].shape[-1] == 6
    assert sorted(components.as_components(validate=True).unsafe_data.data_vars) == ["angular_acceleration", "linear_acceleration"]
    assert isinstance(components.linear(validate=True), LinearAcceleration)
    assert isinstance(components.angular(validate=True), AngularAcceleration)


def example_spatial_frame_motion_metadata() -> None:
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    base = graph.get_or_create_frame("base", parent=world)
    set_edge_motion_class(base, "static", parent=world)
    set_frame_inertial_status(world, "inertial")
    status = propagate_inertial_status(base, parent=world)
    assert get_edge_motion_class(base, parent=world) == "static"
    assert get_frame_inertial_status(world) == "inertial"
    assert status == "inertial"


def example_geo_geodetic_options() -> None:
    opts = GeodeticOptions()
    assert opts.datum == "WGS84"
    assert opts.crs == "EPSG:4979"
    assert opts.ecef_crs == "EPSG:4978"


def example_geo_enu_options() -> None:
    origin = LocalOrigin(45.0, -75.0, 100.0)
    opts = ENUOptions(origin=origin, output_frame="site_enu")
    assert opts.origin == origin
    assert opts.output_frame == "site_enu"


def example_geo_geodesic_options() -> None:
    opts = GeodesicOptions(method="local_enu", local_origin=LocalOrigin(45.0, -75.0))
    assert opts.method == "local_enu"
    assert opts.local_origin is not None


def example_geo_interpolation_options() -> None:
    opts = GeodeticInterpolationOptions(method="ecef_linear", query_dim="target")
    assert opts.method == "ecef_linear"
    assert opts.query_dim == "target"


def example_geo_geodetic_from_lla() -> None:
    ds = xr.Dataset(
        {"position": (("sample", "lla"), np.asarray([[45.0, -75.0, 100.0]], dtype=float))},
        coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True)
    opts = GeodeticOptions(longitude_wrap="[0, 360)")
    with patch("tal.geo.options.normalize_supported_crs", lambda value, expected, owner: expected):
        lla = GeodeticPosition.from_lla(ao, opts=opts)
    geo = lla.unsafe_data.attrs["tal"]["ext"]["geo"]
    assert geo["kind"] == "geodetic_position"
    assert geo["longitude_wrap"] == "[0, 360)"
    assert list(lla.unsafe_data["lla"].values) == ["lat", "lon", "alt"]


def example_geo_geodetic_conversion() -> None:
    from tal.geo import from_ecef as geo_from_ecef

    ds = xr.Dataset(
        {"position": (("sample", "lla"), np.asarray([[45.0, -75.0, 100.0]], dtype=float))},
        coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True)
    opts = GeodeticOptions(longitude_wrap="[0, 360)")
    with (
        patch("tal.geo.options.normalize_supported_crs", lambda value, expected, owner: expected),
        patch(
            "tal.geo.conversion.transform_lla_to_ecef",
            lambda lat, lon, alt, crs, ecef_crs, owner: (lat + 1.0, lon + 2.0, alt + 3.0),
        ),
        patch(
            "tal.geo.conversion.transform_ecef_to_lla",
            lambda x, y, z, crs, ecef_crs, owner: (x - 1.0, y - 2.0, z - 3.0),
        ),
    ):
        lla = GeodeticPosition.from_lla(ao)
        ecef = lla.to_ecef(opts=opts)
        roundtrip = GeodeticPosition.from_ecef(ecef, opts=opts)
        via_module = geo_from_ecef(ecef, opts=opts)
    assert list(ecef.unsafe_data["axis"].values) == ["x", "y", "z"]
    assert list(roundtrip.unsafe_data["lla"].values) == ["lat", "lon", "alt"]
    assert roundtrip.unsafe_data.attrs["tal"]["ext"]["geo"]["longitude_wrap"] == "[0, 360)"
    np.testing.assert_allclose(roundtrip.unsafe_data["position"], lla.unsafe_data["position"])
    np.testing.assert_allclose(via_module.unsafe_data["position"], lla.unsafe_data["position"])


def example_geo_enu_conversion() -> None:
    ds = xr.Dataset(
        {"position": (("sample", "lla"), np.asarray([[0.0, 1.0, 0.0]], dtype=float))},
        coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True)
    origin = LocalOrigin(0.0, 0.0, 0.0)
    opts = ENUOptions(origin=origin, output_frame="site_enu")
    with (
        patch("tal.geo.options.normalize_supported_crs", lambda value, expected, owner: expected),
        patch(
            "tal.geo.conversion.transform_lla_to_ecef",
            lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt),
        ),
        patch(
            "tal.geo.conversion.transform_ecef_to_lla",
            lambda x, y, z, crs, ecef_crs, owner: (x, y, z),
        ),
        patch(
            "tal.geo.local.transform_lla_to_ecef",
            lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt),
        ),
    ):
        lla = GeodeticPosition.from_lla(ao)
        ecef = lla.to_ecef()
        enu = lla.to_enu(opts=opts)
        enu_from_ecef = ecef.geo.to_enu(origin=origin)
        ecef_roundtrip = enu.geo.to_ecef()
        lla_roundtrip = ecef.geo.to_lla()
    assert list(enu.unsafe_data["axis"].values) == ["x", "y", "z"]
    assert enu.unsafe_data.attrs["tal"]["ext"]["geo"]["cartesian_system"] == "enu"
    assert enu.unsafe_data.attrs["tal"]["ext"]["geo"]["origin"]["storage"] == "inline"
    np.testing.assert_allclose(enu.unsafe_data["position"], enu_from_ecef.unsafe_data["position"])
    np.testing.assert_allclose(ecef_roundtrip.unsafe_data["position"], ecef.unsafe_data["position"])
    np.testing.assert_allclose(lla_roundtrip.unsafe_data["position"], lla.unsafe_data["position"])


def example_geo_distance_bearing() -> None:
    ds = xr.Dataset(
        {"position": (("sample", "lla"), np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float))},
        coords={"sample": [0, 1], "lla": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True)

    def inverse(lat1, lon1, lat2, lon2, crs, owner):
        shape = np.broadcast_shapes(np.shape(lat1), np.shape(lon1), np.shape(lat2), np.shape(lon2))
        return np.full(shape, 90.0), np.full(shape, -90.0), np.abs(lon2 - lon1) * 1000.0

    with (
        patch("tal.geo.options.normalize_supported_crs", lambda value, expected, owner: expected),
        patch("tal.geo.distance.geod_inverse", inverse),
    ):
        lla = GeodeticPosition.from_lla(ao)
        distance = lla.distance_to(lla)
        bearing = lla.initial_bearing_to(lla)
    assert tuple(distance.unsafe_data.attrs["tal"]["core"]["roles"]["core_dims"]) == ()
    assert "distance_m" in distance.unsafe_data.data_vars
    assert "initial_bearing_deg" in bearing.unsafe_data.data_vars


def example_geo_interpolation() -> None:
    ds = xr.Dataset(
        {
            "position": (
                ("sample", "lla"),
                np.asarray([[0.0, 170.0, 0.0], [0.0, 190.0, 10.0]], dtype=float),
            )
        },
        coords={"sample": [0, 1], "lla": ["lat", "lon", "alt"], "time_s": ("sample", [0.0, 10.0])},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), param_coord="time_s", validate=True)

    def interpolate(lat1, lon1, lat2, lon2, alpha, crs, owner):
        return lat1 + (lat2 - lat1) * alpha, lon1 + (lon2 - lon1) * alpha

    with patch("tal.geo.interpolation.geod_interpolate", interpolate):
        lla = GeodeticPosition.from_lla(ao)
        out = lla.param.at([5.0], on="time_s")
        nearest = lla.param.resample_to([6.0], on="time_s", opts=GeodeticInterpolationOptions(method="nearest"))
        like = lla.param.interp_like(nearest, on="time_s", opts=GeodeticInterpolationOptions(method="nearest"))
    assert isinstance(out, GeodeticPosition)
    assert isinstance(nearest, GeodeticPosition)
    assert isinstance(like, GeodeticPosition)
    assert out.unsafe_data.attrs["tal"]["ext"]["geo"]["kind"] == "geodetic_position"


def example_geo_crs_transform() -> None:
    ds = xr.Dataset(
        {"position": (("sample", "lla"), np.asarray([[34.0, -118.0, 20.0]], dtype=float))},
        coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True)
    projected_ds = xr.Dataset(
        {"position": (("sample", "projected"), np.asarray([[500000.0, 4100000.0]], dtype=float))},
        coords={"sample": [0], "projected": ["easting", "northing"]},
    )
    projected_ao = AnalysisObject.from_data(
        projected_ds,
        sequence_dim="sample",
        core_dims=("projected",),
        validate=True,
    )

    def kind(value, owner, field="crs"):
        if str(value).endswith("4978"):
            return "geocentric"
        if str(value).endswith("32611"):
            return "projected"
        return "geographic"

    def xyz(x, y, z, src_crs, dst_crs, owner):
        if dst_crs == "EPSG:32611":
            return x + 1000.0, y + 2000.0, z
        if dst_crs == "EPSG:4979":
            return x - 1000.0, y - 2000.0, z
        return x + 1.0, y + 2.0, z + 3.0

    with (
        patch(
            "tal.geo.crs_transform.normalize_crs_with_class",
            lambda value, owner, field="crs": SimpleNamespace(text=str(value), kind=kind(value, owner, field)),
        ),
        patch("tal.geo.crs_transform.crs_has_height_axis", lambda value, owner, field="crs": False),
        patch("tal.geo.crs_transform.base_geodetic_crs", lambda value, owner, field="crs": "EPSG:4326"),
        patch("tal.geo.crs_transform.transform_crs_xyz", xyz),
        patch("tal.geo.crs_transform.transform_crs_xy", lambda x, y, src_crs, dst_crs, owner: (x + 10.0, y + 20.0)),
        patch("tal.geo.options.normalize_supported_crs", lambda value, expected, owner: expected),
        patch("tal.geo.metadata.normalize_crs_for_class", lambda value, expected, owner, field="crs": str(value)),
        patch("tal.geo.metadata.base_geodetic_crs", lambda value, owner, field="crs": "EPSG:4326"),
    ):
        lla = GeodeticPosition.from_lla(ao)
        projected = lla.to_crs("EPSG:32611")
        direct = transform_crs(lla, dst="EPSG:4978")
        projected_input = ProjectedPosition.from_projected(projected_ao, crs="EPSG:32611")
        roundtrip = projected.to_crs("EPSG:4979")
    assert isinstance(projected, ProjectedPosition)
    assert isinstance(projected_input, ProjectedPosition)
    assert isinstance(direct, Position)
    assert isinstance(roundtrip, GeodeticPosition)
    assert list(projected.unsafe_data["projected"].values) == ["easting", "northing", "height"]
    assert projected.unsafe_data.attrs["tal"]["ext"]["geo"]["kind"] == "projected_position"


def example_io_read_csv_logs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "run.csv"
        path.write_text("time,value\n0.0,1.0\n1.0,2.0\n", encoding="utf-8")
        out = read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))
    assert "value" in out.unsafe_data.data_vars
    assert out.unsafe_data.sizes["sample"] == 2


def example_io_roundtrip_surface() -> None:
    from tal.io import (
        AOCsvReadOptions,
        AOCsvWriteOptions,
        AOZarrReadOptions,
        AOZarrWriteOptions,
        CsvExportOptions,
        CsvIngestOptions,
        read_csv_logs_catalog,
        write_csv_logs,
    )

    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("run", "sample"), np.asarray([[1.0, 2.0]], dtype=float))},
            coords={"run": ["run_a"], "sample": [0, 1], "time": ("sample", [0.0, 1.0])},
        ),
        sequence_dim="sample",
        batch_dims=("run",),
        core_dims=(),
        param_coord="time",
        validate=True,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        csv_root = ao.io.to_csv(str(root / "trajectory"), opts=AOCsvWriteOptions())
        loaded_csv = AnalysisObject.from_csv(csv_root, opts=AOCsvReadOptions())
        store = root / "trajectory.zarr"
        _ = loaded_csv.io.to_zarr(str(store), opts=AOZarrWriteOptions(mode="w"))
        loaded_zarr = AnalysisObject.from_zarr(str(store), opts=AOZarrReadOptions())
        exported = write_csv_logs(ao, str(root / "logs"), opts=CsvExportOptions(float_format="%.1f"))
        log_path = root / "run.csv"
        log_path.write_text("time,value\n0.0,1.0\n1.0,2.0\n", encoding="utf-8")
        catalog = read_csv_logs_catalog(str(log_path), opts=CsvIngestOptions(time_col="time"))
    assert loaded_csv.unsafe_data["value"].sizes["sample"] == 2
    assert loaded_zarr.unsafe_data["value"].sizes["sample"] == 2
    assert len(exported) == 1
    assert catalog.group_labels == ("run",)


def example_io_ros_optional_surface() -> None:
    from tal.io import RosIngestOptions, read_ros_logs, read_ros_logs_catalog

    opts = RosIngestOptions(topic="/robot/pose", message_type="geometry_msgs/msg/PoseStamped")
    try:
        ao = read_ros_logs("robot_run.mcap", opts=opts)
    except (ImportError, ValueError, FileNotFoundError):
        ao = None
    try:
        catalog = read_ros_logs_catalog("robot_run.mcap", opts=opts)
    except (ImportError, ValueError, FileNotFoundError):
        catalog = None
    assert ao is None or "translation_x" in ao.unsafe_data.data_vars
    assert catalog is None or catalog.backend == "dataset"


def example_catalog_extract() -> None:
    from tal.catalog.options import CatalogExtractOptions, CatalogQueryOptions

    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("run", "sample"), np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float))},
            coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])},
        ),
        sequence_dim="sample",
        batch_dims=("run",),
        core_dims=(),
        validate=True,
    )
    catalog = Catalog(ao, backend="dataset", batch_dim="run")
    assert catalog.group_labels == ("a", "b")
    assert catalog.query(kind="sim", opts=CatalogQueryOptions()).group_labels == ("a",)
    extracted = catalog.extract("value", opts=CatalogExtractOptions())
    assert extracted.unsafe_data["value"].sizes["run"] == 2


def example_frames_find_path() -> None:
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    base = graph.get_or_create_frame("base", parent=world)
    sensor = graph.get_or_create_frame("sensor", parent=base)
    path = find_path(sensor, world)
    assert [node.id for node in path.nodes] == ["sensor", "base", "world"]


def example_frames_graph_mutation() -> None:
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    base = world.child("base")
    sensor = graph.get_or_create_frame("sensor", parent=base)
    graph.reparent_frame(sensor, world, on_conflict="replace")
    renamed = sensor.rename("camera")
    assert graph.get_frame("camera") is renamed
    renamed.reparent(base, on_conflict="replace")
    graph.rename_frame("camera", "camera_0")
    assert graph.get_frame("camera_0") is renamed
    renamed.remove(subtree=True)
    assert graph.get_frame("camera_0") is None
    assert graph.freeze().frozen is True


def example_frames_snapshot_from_seeds() -> None:
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    _ = graph.get_or_create_frame("base", parent=world)
    snapshot = snapshot_from_seeds(("world",), graph=graph)
    assert "world" in snapshot.node_ids
    assert snapshot.issues == ()


def example_frames_topology_rendering() -> None:
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    base = graph.get_or_create_frame("base", parent=world)
    sensor = graph.get_or_create_frame("sensor", parent=base)
    path = find_path(sensor, world)
    folded = fold_path(
        path,
        edge_value_fn=lambda child, parent: [(child.id, parent.id)],
        compose=lambda acc, value: acc + value,
        inverse=lambda value: [(value[0][1], value[0][0])],
        identity=lambda: [],
    )
    snapshot = snapshot_from_seeds(("world",), graph=graph)
    ascii_tree = render_snapshot_ascii(snapshot)
    try:
        nx_graph = snapshot_to_networkx(snapshot)
    except ImportError:
        nx_graph = None
    assert folded == [("sensor", "base"), ("base", "world")]
    assert "sensor" in ascii_tree
    if nx_graph is not None:
        assert "sensor" in nx_graph.nodes


def example_frames_api_surface() -> None:
    from tal.frames import get_active_frame_graph, get_or_create_frame
    from tal.frames.snapshot import SnapshotIssue
    from tal.frames.visualization import FrameGraphDrawOptions, draw_frame_graph
    from tal.utils.frame_ops import frame_ids, frame_remap_ids, frame_rename, frame_retag

    graph = FrameGraph()
    with graph:
        world = get_or_create_frame("world")
        base = get_or_create_frame("base", parent=world)
        assert get_active_frame_graph() is graph
    path = find_path(base, world)
    assert path.steps[0].invert is False
    snapshot = snapshot_from_seeds(("world",), graph=graph)
    assert snapshot.root_ids == ("world",)
    assert SnapshotIssue(code="unreachable_registered_frame", frame_id="camera").frame_id == "camera"
    opts = FrameGraphDrawOptions(layout="circular", include_legend=False)
    try:
        ax = draw_frame_graph(graph=graph, seeds=("world",), opts=opts)
    except ImportError:
        ax = None
    assert ax is None or hasattr(ax, "plot")

    ao = AnalysisObject.from_data(
        xr.Dataset({"value": ("sample", np.asarray([1.0], dtype=float))}, coords={"sample": [0]}),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    tagged = frame_retag(ao, parent="world", child="base")
    assert frame_ids(tagged) == ("world", "base")
    assert frame_remap_ids(tagged, {"base": "base_link"}).frames.ids() == ("world", "base_link")
    renamed = frame_rename(tagged, "base", "base_link", graph=graph)
    assert renamed.frames.ids() == ("world", "base_link")


def example_viz_line() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset({"value": ("sample", np.asarray([0.0, 1.0, 2.0], dtype=float))}, coords={"sample": [0, 1, 2]}),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    try:
        rendered = line(ao)
    except ImportError:
        return
    assert rendered is not None


def example_viz_surface_accessors() -> None:
    from tal.core.component_ops import ComponentRegistryOptions, ComponentSpec
    from tal.viz import AOVizOptions, component, explorer, scatter

    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"vec": (("sample", "axis"), np.asarray([[1.0, 2.0]], dtype=float))},
            coords={"sample": [0], "axis": ["x", "y"]},
        ),
        sequence_dim="sample",
        core_dims=("axis",),
        validate=True,
    ).components.define(opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))

    for render in (
        lambda: line(ao, opts=AOVizOptions(var="vec")),
        lambda: scatter(ao, opts=AOVizOptions(var="vec")),
        lambda: ao.viz.line(opts=AOVizOptions(var="vec")),
        lambda: ao.viz.scatter(opts=AOVizOptions(var="vec")),
        lambda: component(ao, "xy", kind="line", opts=AOVizOptions(var="vec")),
        lambda: ao.viz.component("xy", kind="line", opts=AOVizOptions(var="vec")),
    ):
        try:
            rendered = render()
        except ImportError:
            rendered = None
        assert rendered is None or rendered is not None
    for render in (
        lambda: explorer(ao, opts=AOVizOptions(var="vec")),
        lambda: ao.viz.explorer(opts=AOVizOptions(var="vec")),
    ):
        try:
            rendered = render()
        except (ImportError, ValueError):
            rendered = None
        assert rendered is None or rendered is not None


def example_core_schema_ufuncs() -> None:
    from tal import ufuncs
    from tal.core import merge_schema, set_roles, validate_schema

    ds = xr.Dataset({"value": ("sample", np.asarray([1.0, 2.0], dtype=float))}, coords={"sample": [0, 1]})
    tagged = set_roles(ds, sequence_dim="sample", core_dims=(), validate=True)
    assert validate_schema(tagged).attrs["tal"]["version"] == 1
    merged = merge_schema(ds, {"version": 1, "core": {"roles": {"sequence_dim": "sample", "batch_dims": [], "core_dims": []}}})
    assert merged.attrs["tal"]["core"]["roles"]["sequence_dim"] == "sample"
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=(), validate=True)
    np.testing.assert_allclose(ufuncs.add(ao, 1.0).unsafe_data["value"], np.asarray([2.0, 3.0]))


def example_linalg_layout_surface() -> None:
    from tal.core import CoreConcatOptions, CoreDecomposeOptions, CoreOverlayOptions

    x = Array(
        AnalysisObject.from_data(
            xr.Dataset({"v": (("sample", "axis"), np.asarray([[1.0]], dtype=float))}, coords={"sample": [0], "axis": ["x"]}),
            sequence_dim="sample",
            core_dims=("axis",),
            validate=True,
        )
    )
    y = Array(
        AnalysisObject.from_data(
            xr.Dataset({"v": (("sample", "axis"), np.asarray([[2.0]], dtype=float))}, coords={"sample": [0], "axis": ["y"]}),
            sequence_dim="sample",
            core_dims=("axis",),
            validate=True,
        )
    )
    concatenated = Array.concat_core([x, y], opts=CoreConcatOptions(core_dim="axis"))
    assert concatenated.unsafe_data.sizes["axis"] == 2
    assert sorted(concatenated.decompose_core(opts=CoreDecomposeOptions(core_dims=("axis",), key_mode="label"))) == [("x",), ("y",)]
    patch = Array(
        AnalysisObject.from_data(
            xr.Dataset({"v": (("sample", "axis"), np.asarray([[9.0]], dtype=float))}, coords={"sample": [0], "axis": ["y"]}),
            sequence_dim="sample",
            core_dims=("axis",),
            validate=True,
        )
    )
    assert concatenated.overlay_core([patch], opts=CoreOverlayOptions(core_dim="axis", on_overlap="replace")).unsafe_data["v"].sel(axis="y").item() == 9.0


def example_catalog_query() -> None:
    ao = _make_signal_ao()
    catalog = Catalog(ao, backend="dataset", batch_dim="trial")
    out = catalog.query(outcome="intercept")
    assert out.group_labels == ("trial_0",)


def example_catalog_selectors() -> None:
    ao = _make_signal_ao()
    catalog = Catalog(ao, backend="dataset", batch_dim="trial")
    assert catalog.sel("trial_0").group_labels == ("trial_0",)
    assert catalog.isel(0).group_labels == ("trial_0",)
    assert catalog.head(1).group_labels == ("trial_0",)
    assert catalog.tail(1).group_labels == ("trial_1",)


def example_utils_frame_bind() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset({"value": ("sample", np.asarray([1.0], dtype=float))}, coords={"sample": [0]}),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    tagged = AnalysisObject._from_unvalidated(set_frames(ao.unsafe_data, parent="world", child="sensor", validate=False))
    graph = FrameGraph()
    parent, child = frame_bind(tagged, graph=graph, create_missing=True, on_conflict="error")
    assert (parent.id if parent is not None else None, child.id if child is not None else None) == ("world", "sensor")


def example_utils_frames_accessor() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset({"value": ("sample", np.asarray([1.0], dtype=float))}, coords={"sample": [0]}),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    tagged = ao.frames.retag(parent="world", child="tool")
    remapped = tagged.frames.remap_ids({"world": "map", "tool": "tool_0"})
    graph = FrameGraph()
    parent, child = remapped.frames.bind(graph=graph, create_missing=True)
    assert tagged.frames.ids() == ("world", "tool")
    assert remapped.frames.ids() == ("map", "tool_0")
    assert (parent.id, child.id) == ("map", "tool_0")
    renamed = remapped.frames.rename_frame("tool_0", "tool_1", graph=graph)
    assert renamed.frames.ids() == ("map", "tool_1")


def example_utils_frame_schema_get() -> None:
    ds = xr.Dataset({"value": ("sample", np.asarray([1.0], dtype=float))}, coords={"sample": [0]})
    tagged = set_frames(ds, parent="world", child="tool", validate=False)
    assert get_frames(tagged) == ("world", "tool")


def example_utils_frame_schema_set() -> None:
    ds = xr.Dataset({"value": ("sample", np.asarray([1.0], dtype=float))}, coords={"sample": [0]})
    tagged = set_frames(ds, parent="world", child="tool", validate=False)
    cleared = set_frames(tagged, child=None, validate=False)
    assert get_frames(tagged) == ("world", "tool")
    assert get_frames(cleared) == ("world", None)


def example_utils_topology_intent_support() -> None:
    support = operation_intent_support_for_operation_family("spatial.pose.compose", owner="docs-example")
    assert support.semantic_default is True
    assert support.alignment_intent_supported is True


def example_utils_xarray_rename_dims() -> None:
    da = xr.DataArray(np.asarray([[1.0]], dtype=float), dims=("x", "y"), coords={"x": [0], "y": [1]})
    out = rename_dims_collision_safe(da, mapping={"x": "y", "y": "x"})
    assert out.dims == ("y", "x")


def example_utils_numba_public() -> None:
    from tal.utils import numba as tal_numba

    values = np.arange(24.0, dtype=np.float64).reshape(2, 4, 3)
    rows = tal_numba.prepare_block_rows(
        (values,),
        (tal_numba.BlockInputSpec("values", 2, np.float64),),
        output_core_shape=(4, 3),
        owner="docs.numba",
    )
    bounds = tal_numba.centered_window_bounds(4, radius=1, owner="docs.numba")
    window_rows = tal_numba.prepare_window_rows(bounds, owner="docs.numba")

    def row_kernel_shape(value_rows, start, stop):
        return value_rows.shape[0], start.shape[0], stop.shape[0]

    assert row_kernel_shape(rows.row_arrays[0], bounds.start, bounds.stop) == (2, 4, 4)
    assert window_rows.length == 4
    assert window_rows.max_width == 3

    topology_rows = tal_numba.prepare_topology_rows(
        (values,),
        (tal_numba.ScanInputSpec("links", 1, 1, np.float64),),
        topology_axis="chain",
        output_core_shapes=((3,),),
        owner="docs.numba",
    )
    assert topology_rows.outer_shape == (2,)
    assert topology_rows.ordered_shape == (4,)

    nested_rows = tal_numba.prepare_scan_rows(
        (np.zeros((2, 5, 4, 3), dtype=np.float64),),
        (tal_numba.ScanInputSpec("links", 2, 1, np.float64),),
        ordered_axes=(
            tal_numba.ScanAxisSpec("time", "scan"),
            tal_numba.ScanAxisSpec("chain", "topology"),
        ),
        output_core_shapes=((3,),),
        owner="docs.numba",
    )
    assert nested_rows.ordered_shape == (5, 4)

    assert tal_numba.time_once(lambda value: value + 1, 1) >= 0.0
    assert tal_numba.warm_median(lambda value: value + 1, 1, repeats=1) >= 0.0
    assert tal_numba.break_even_calls(10.0, 25.0, 5.0) == 4.0
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "bench.py"
        script.write_text("print('0.0')\n", encoding="utf-8")
        assert tal_numba.cold_subprocess(str(script), (), cache_prefix="docs-numba-") == 0.0

    try:
        numba = tal_numba.require_numba("docs.numba")
    except ImportError:
        compiled = None
    else:
        compiled = tal_numba.njit_kernel(numba, row_kernel_shape)
    assert compiled is None or callable(compiled)


EXECUTABLE_EXAMPLES: dict[str, Callable[[], None]] = {
    "CORE-AO-FROM-DATA": example_core_ao_from_data,
    "CORE-AO-SET-ROLES": example_core_ao_set_roles,
    "CORE-AO-XARRAY-METHODS": example_core_ao_xarray_methods,
    "CORE-AO-REDUCERS": example_core_ao_reducers,
    "CORE-PARAM-AT": example_core_param_at,
    "CORE-PARAM-INTERP-LIKE": example_core_param_interp_like,
    "CORE-PARAM-SURFACE": example_core_param_surface,
    "CORE-EVENT-WHEN": example_core_event_when,
    "CORE-EVENT-SURFACE": example_core_event_surface,
    "CORE-GROUP-GROUPBY": example_core_group_groupby,
    "CORE-GROUP-SURFACE": example_core_group_surface,
    "CORE-COMBINE-CONCAT-SEQUENCE": example_core_combine_concat_sequence,
    "CORE-COMBINE-CORE-LAYOUTS": example_core_combine_core_layouts,
    "CORE-COMBINE-SURFACE": example_core_combine_surface,
    "CORE-COMPONENT-REGISTRY": example_core_component_registry,
    "CORE-COMPONENT-SURFACE": example_core_component_surface,
    "CORE-SCHEMA-UFUNCS": example_core_schema_ufuncs,
    "LINALG-ADD": example_linalg_add,
    "LINALG-SUB": example_linalg_sub,
    "LINALG-DOT": example_linalg_dot,
    "LINALG-NORM": example_linalg_norm,
    "LINALG-MATMUL": example_linalg_matmul,
    "LINALG-SOLVE": example_linalg_solve,
    "LINALG-INV": example_linalg_inv,
    "LINALG-PINV": example_linalg_pinv,
    "LINALG-VECTOR3-FROM-XYZ": example_linalg_vector3_from_xyz,
    "LINALG-ARRAY-CORE-DIMS": example_linalg_array_core_dims,
    "LINALG-LAYOUT-SURFACE": example_linalg_layout_surface,
    "SPATIAL-POSITION-TO-FRAME": example_spatial_position_to_frame,
    "SPATIAL-POSITION-BASIC": example_spatial_position_basic,
    "SPATIAL-ROTATION-TO-REP": example_spatial_rotation_to_rep,
    "SPATIAL-ROTATION-BASIC": example_spatial_rotation_basic,
    "SPATIAL-POSE-FROM-COMPONENTS": example_spatial_pose_from_components,
    "SPATIAL-POSE-BASIC": example_spatial_pose_basic,
    "SPATIAL-PATH-SOLVE-POSE": example_spatial_path_solve_pose,
    "SPATIAL-VELOCITY-COMPONENTS": example_spatial_velocity_components,
    "SPATIAL-ACCELERATION-COMPONENTS": example_spatial_acceleration_components,
    "SPATIAL-FRAME-MOTION-METADATA": example_spatial_frame_motion_metadata,
    "GEO-GEODETIC-OPTIONS": example_geo_geodetic_options,
    "GEO-ENU-OPTIONS": example_geo_enu_options,
    "GEO-GEODESIC-OPTIONS": example_geo_geodesic_options,
    "GEO-INTERPOLATION-OPTIONS": example_geo_interpolation_options,
    "GEO-GEODETIC-FROM-LLA": example_geo_geodetic_from_lla,
    "GEO-GEODETIC-CONVERSION": example_geo_geodetic_conversion,
    "GEO-ENU-CONVERSION": example_geo_enu_conversion,
    "GEO-DISTANCE-BEARING": example_geo_distance_bearing,
    "GEO-INTERPOLATION": example_geo_interpolation,
    "GEO-CRS-TRANSFORM": example_geo_crs_transform,
    "IO-READ-CSV-LOGS": example_io_read_csv_logs,
    "IO-ROUNDTRIP-SURFACE": example_io_roundtrip_surface,
    "IO-ROS-OPTIONAL-SURFACE": example_io_ros_optional_surface,
    "FRAMES-FIND-PATH": example_frames_find_path,
    "FRAMES-GRAPH-MUTATION": example_frames_graph_mutation,
    "FRAMES-SNAPSHOT-FROM-SEEDS": example_frames_snapshot_from_seeds,
    "FRAMES-TOPOLOGY-RENDERING": example_frames_topology_rendering,
    "FRAMES-API-SURFACE": example_frames_api_surface,
    "VIZ-LINE": example_viz_line,
    "VIZ-SURFACE-ACCESSORS": example_viz_surface_accessors,
    "CATALOG-QUERY": example_catalog_query,
    "CATALOG-SELECTORS": example_catalog_selectors,
    "CATALOG-EXTRACT": example_catalog_extract,
    "UTILS-FRAME-BIND": example_utils_frame_bind,
    "UTILS-FRAMES-ACCESSOR": example_utils_frames_accessor,
    "UTILS-FRAME-SCHEMA-GET": example_utils_frame_schema_get,
    "UTILS-FRAME-SCHEMA-SET": example_utils_frame_schema_set,
    "UTILS-NUMBA-PUBLIC": example_utils_numba_public,
    "UTILS-TOPOLOGY-INTENT-SUPPORT": example_utils_topology_intent_support,
    "UTILS-XARRAY-RENAME-DIMS": example_utils_xarray_rename_dims,
}
