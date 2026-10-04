"""Contracts 005/014/061/077: positional work retains native index identity."""

import itertools

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from scipy.spatial.transform import Rotation as ScipyRotation

from tal.core import (
    AnalysisLayoutSpec,
    AnalysisObject,
    ComponentExtractOptions,
    read_components,
)
from tal.core.event_ops import AroundOptions, ConditionEvalOptions
from tal.spatial import LinearVelocity, Pose, Position, Rotation
from tests.core.test_spatial_validity_topology import _position, _rotation


class _PairTransform(xr.indexes.CoordinateTransform):
    def __init__(self, size, calls, dim):
        self.calls = calls
        self.dim = dim
        super().__init__((dim, "alias"), {dim: size})

    def __deepcopy__(self, memo):
        return type(self)(self.dim_size[self.dim], self.calls, self.dim)

    def forward(self, dim_positions):
        self.calls.append("forward")
        positions = dim_positions[self.dim]
        return {self.dim: positions + 10., "alias": positions + 100.}

    def reverse(self, coord_labels):
        self.calls.append("reverse")
        return {self.dim: coord_labels[self.dim] - 10.}

    def equals(self, other, **kwargs):
        return isinstance(other, _PairTransform) and self.dim == other.dim and self.dim_size == other.dim_size


def _indexed(ds, *, dim, kind, calls):
    out = ds.drop_vars(dim, errors="ignore")
    if kind == "unindexed":
        return out
    index = (xr.indexes.RangeIndex.arange(ds.sizes[dim], dim=dim) if kind == "range"
             else xr.indexes.CoordinateTransformIndex(_PairTransform(ds.sizes[dim], calls, dim)))
    return out.assign_coords(xr.Coordinates.from_xindex(index))


def _assert_indexes(source, result, dim):
    names = tuple(source.xindexes.get_all_coords(dim)) if dim in source.xindexes else ()
    if not names:
        assert dim not in result.xindexes
    for name in names:
        assert type(result.xindexes[name]) is type(source.xindexes[name])
        assert result.xindexes[name].equals(source.xindexes[name])
    if names:
        assert tuple(result.xindexes.get_all_coords(dim)) == names


@pytest.mark.parametrize("kind", ["range", "transform", "unindexed"])
@pytest.mark.parametrize("lazy,validate,empty", [(False, True, False), (True, False, False), (True, True, True)])
@pytest.mark.parametrize("operation", ["inverse", "matrix", "compose", "apply", "pose_inverse", "pose_compose"])
def test_native_sequence_indexes_survive_spatial_validity(kind, lazy, validate, empty, operation):
    """ID: NATIVE_NUMERICAL_TOPOLOGY_001; native sequence labels never become positions."""
    n = 0 if empty else 3
    calls = []
    raw = _rotation(batched=True, count=[n, min(n, 1)], lazy=lazy, n=n)
    source = Rotation(_indexed(raw.as_dataset(), dim="sample", kind=kind, calls=calls))
    position = Position(_indexed(_position(raw).as_dataset(), dim="sample", kind=kind, calls=calls))
    pose = Pose.from_components(source, position) if operation.startswith("pose") else None
    before = source.as_dataset()
    calls.clear()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        if operation == "inverse":
            result = source.inverse(validate=validate).as_matrix()
        elif operation == "matrix":
            result = source.as_matrix(validate=validate)
        elif operation == "compose":
            result = source.compose(source, validate=validate).as_matrix()
        elif operation == "apply":
            result = source.apply(LinearVelocity(position), validate=validate)
        elif operation == "pose_inverse":
            result = pose.inverse(validate=validate).as_matrix()
        else:
            result = pose.compose(pose, validate=validate).as_matrix()
    assert not tasks and not calls
    ds = result.as_dataset()
    _assert_indexes(before, ds, "sample")
    _assert_indexes(before, source.as_dataset(), "sample")
    np.testing.assert_array_equal(ds["count"], [n, min(n, 1)])
    data = ds[next(iter(ds.data_vars))].transpose("trial", "sample", ...).data
    actual = data.compute() if lazy else data
    valid = np.arange(n)[None, :] < np.array([n, min(n, 1)])[:, None]
    angle = -.3 if "inverse" in operation else .6 if "compose" in operation else .3
    matrix = ScipyRotation.from_euler("z", angle).as_matrix()
    expected = matrix
    if operation == "apply":
        expected = matrix @ [1., 0., 0.]
    if operation.startswith("pose"):
        expected = np.eye(4)
        expected[:3, :3] = matrix
        expected[:3, 3] = -matrix @ [1., 0., 0.] if operation == "pose_inverse" else np.array([1., 0., 0.]) + ScipyRotation.from_euler("z", .3).apply([1., 0., 0.])
    np.testing.assert_allclose(actual[valid], np.broadcast_to(expected, actual[valid].shape), atol=1e-12)
    assert np.isnan(actual[~valid]).all()
    assert not calls


@pytest.mark.parametrize("kind", ["range", "transform", "unindexed"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("family", ["generic", "rotation"])
@pytest.mark.parametrize("reverse", [False, True])
def test_semantic_batch_expansion_carries_complete_indexes(kind, lazy, family, reverse):
    """ID: NATIVE_NUMERICAL_TOPOLOGY_002; semantic broadcast shares actual index groups."""
    calls = []
    if family == "rotation":
        left = _rotation(batched=False, lazy=lazy)
        ds = _indexed(_rotation(batched=True, lazy=lazy).as_dataset(), dim="trial", kind=kind, calls=calls)
        right = Rotation(ds)
    else:
        ds = _indexed(xr.Dataset({"v": (("trial", "sample"), [[1., 2.], [3., 4.]])}, coords={"trial": [0, 1], "sample": [0, 1]}), dim="trial", kind=kind, calls=calls)
        if lazy:
            ds["v"] = ds.v.chunk({"sample": 1}).variable
        right = AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",)).wrap(ds)
        left = AnalysisLayoutSpec(sequence_dim="sample").wrap(xr.Dataset({"v": ("sample", [10., 20.])}, coords={"sample": [0, 1]}))
    if reverse:
        left, right = right, left
    calls.clear()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = left.compose(right).as_matrix() if family == "rotation" else left + right
    assert not calls and not tasks
    out = result.as_dataset()
    _assert_indexes(ds, out, "trial")
    data = out[next(iter(out.data_vars))].transpose("trial", "sample", ...).data
    actual = data.compute() if lazy else data
    expected = np.broadcast_to(ScipyRotation.from_euler("z", .6).as_matrix(), (2, 3, 3, 3)) if family == "rotation" else [[11., 22.], [13., 24.]]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert not calls


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("kind", ["range", "transform", "unindexed"])
def test_native_structural_masks_support_reducers_and_parameter_queries(lazy, kind):
    """ID: NATIVE_NUMERICAL_TOPOLOGY_003; shared masks serve core and typed consumers."""
    calls = []
    ds = xr.Dataset({"v": (("trial", "sample"), [[1., 2., 90.], [4., 90., 90.]])}, coords={"trial": [0, 1], "sample": [0, 1, 2], "time": ("sample", [0., 1., 2.]), "count": ("trial", [2, 1])})
    ds = _indexed(ds, dim="trial", kind=kind, calls=calls)
    if lazy:
        ds["v"] = ds.v.chunk({"sample": 2}).variable
    source = AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",), param_coord="time", sequence_size_coord="count").wrap(ds)
    calls.clear()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        reduced = source.sum(dim="sample")
        sampled = source.param.at([0., .5])
    assert not calls and not tasks
    _assert_indexes(ds, sampled.as_dataset(), "trial")
    np.testing.assert_allclose(reduced.as_dataset().v.data.compute() if lazy else reduced.as_dataset().v.data, [3., 4.])
    np.testing.assert_allclose(sampled.as_dataset().v.data.compute() if lazy else sampled.as_dataset().v.data, [[1., 1.5], [4., np.nan]], equal_nan=True)


@pytest.mark.parametrize("kind", ["range", "transform", "unindexed"])
@pytest.mark.parametrize("lazy,empty", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("validate", [False, True])
def test_component_projection_preserves_native_validity(kind, lazy, empty, validate):
    """ID: NATIVE_NUMERICAL_TOPOLOGY_004; core projection never changes sequence rows."""
    n = 0 if empty else 3
    calls = []
    raw = _rotation(batched=True, count=[n, min(n, 1)], lazy=lazy, n=n)
    rotation = Rotation(_indexed(raw.as_dataset(), dim="sample", kind=kind, calls=calls))
    position = Position(_indexed(_position(raw).as_dataset(), dim="sample", kind=kind, calls=calls))
    source = Pose.from_components(rotation, position)
    tasks = []
    calls.clear()
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        part = source.components.extract(
            opts=ComponentExtractOptions(names=("position",), output_var="selected"), validate=validate,
        )["position"]
    assert not tasks and not calls
    ds = part.as_dataset()
    assert type(part) is AnalysisObject
    assert set(ds.data_vars) == {"selected"}
    assert read_components(part)["position"].var == "selected"
    assert ds.attrs["tal"]["core"]["validity"]["sequence_size_coord"] == "count"
    assert ds.attrs["tal"]["core"]["param_coord"]["name"] == "time"
    np.testing.assert_array_equal(ds["count"], [n, min(n, 1)])
    _assert_indexes(source.as_dataset(), ds, "sample")
    assert not calls


def test_unindexed_unknown_slice_still_prunes_validity():
    """ID: NATIVE_NUMERICAL_TOPOLOGY_005; projection intent does not relax slicing."""
    raw = _rotation(batched=False, count=1)
    source = Rotation(raw.as_dataset().drop_vars("sample"))
    result = source.isel(sample=[2, 1])
    assert result.as_dataset().attrs["tal"]["core"].get("validity") is None


@pytest.mark.parametrize("kind", ["range", "pandas", "unindexed"])
@pytest.mark.parametrize("lanes,lazy,limit", [(2, False, None), (2, True, 2), (0, False, None), (0, True, 2)])
@pytest.mark.parametrize("operation", ["events", "intervals", "boundaries", "segments", "stream", "windows", "stacked"])
def test_tut_006_007_event_native_batch_consumers(kind, lanes, lazy, limit, operation):
    """TUT-006/007: complete native groups survive tables, selection, windows and zero-lane stacking."""
    from tal.core.event_ops import (
        AroundOptions,
        AtBoundariesOptions,
        Condition,
        EventExtractOptions,
        IntervalExtractOptions,
        WhenOptions,
    )

    ds = xr.Dataset({"value": (("trial", "sample"), np.tile([0., 3., 4., 0.], (lanes, 1)))},
                    coords={"time": ("sample", [0., .2, .7, 1.5]), "sample": [10, 20, 30, 40]})
    if kind == "range":
        ds = ds.assign_coords(xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(10, 10 + lanes * 10, 10, dim="trial")))
    elif kind == "pandas":
        ds = ds.assign_coords(trial=np.arange(lanes) + 10, trial_tag=("trial", np.arange(lanes) + 100)).set_xindex("trial_tag")
    if lazy:
        ds = ds.chunk({"sample": 2})
    source = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=("trial",), param_coord="time")
    snapshot = source.as_dataset(copy="deep")
    condition = Condition.compare(Condition.var("value"), "gt", 2.)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        if operation == "events":
            result = source.events.events(condition, opts=EventExtractOptions(max_events=limit))
        elif operation == "intervals":
            result = source.events.intervals(condition, opts=IntervalExtractOptions(max_segments=limit))
        elif operation == "boundaries":
            result = source.events.at_boundaries(condition, opts=AtBoundariesOptions(max_events=limit))
        elif operation in ("segments", "stream"):
            result = source.events.when(condition, opts=WhenOptions(layout=operation, max_segments=limit))
        else:
            anchors = xr.DataArray([.2], dims="anchor")
            result = source.events.around(anchors, opts=AroundOptions(pre=0., post=0., dt=.1,
                                          layout="stacked" if operation == "stacked" else "segments"))
    assert tasks == []
    actual = result if isinstance(result, xr.Dataset) else result.as_dataset()
    assert actual.sizes["trial"] == lanes
    if kind == "unindexed":
        assert "trial" not in actual.coords and "trial" not in actual.xindexes
    else:
        for name in snapshot.xindexes.get_all_coords("trial"):
            assert type(actual.xindexes[name]) is type(snapshot.xindexes[name])
            assert actual.xindexes[name].equals(snapshot.xindexes[name])
        if kind == "pandas":
            assert actual.xindexes["trial_tag"].equals(snapshot.xindexes["trial_tag"])
    eager = actual.compute(scheduler="synchronous")
    if operation == "events":
        assert set(eager.data_vars) == {"time", "edge_code", "sample_index_before", "sample_index_after"}
        np.testing.assert_allclose(eager.time, np.tile([.2, .7] if lanes or limit else [], (lanes, 1)))
        np.testing.assert_array_equal(eager.edge_code, np.tile([1, 2] if lanes or limit else [], (lanes, 1)))
    elif operation == "intervals" and lanes:
        segment = next(dim for dim in eager.valid_segment.dims if dim != "trial")
        np.testing.assert_allclose(eager.time.isel({segment: 0}), np.tile([.2, .7] if lanes or limit else [], (lanes, 1)))
    elif operation == "boundaries":
        np.testing.assert_allclose(eager.value, np.tile([3., 4.] if lanes or limit else [], (lanes, 1)))
    elif operation in ("windows", "stacked"):
        np.testing.assert_allclose(eager.value.data.reshape(lanes), np.full(lanes, 3.))
        assert set(eager.data_vars) == {"value"}
    elif lanes:
        np.testing.assert_array_equal((eager.orig_index >= 0).sum(axis=-1),
            [[2, 0], [2, 0]] if operation == "segments" and limit else [[2], [2]] if operation == "segments" else [2, 2])
    xr.testing.assert_identical(source.as_dataset(), snapshot)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("name", ["time", "edge_code", "sample_index_before", "sample_index_after",
                                  "sample_index", "is_trigger", "valid_segment"])
@pytest.mark.parametrize("operation", ["events", "intervals"])
def test_tut_010_table_columns_protect_surviving_batch_indexes(tutorial_audit_source, name, operation, lazy):
    """TUT-010: fixed data columns cannot discard or replace a surviving batch index."""
    from tal.core.event_ops import (
        Condition,
        ConditionEvalOptions,
        EventExtractOptions,
        IntervalExtractOptions,
    )

    source = tutorial_audit_source(lazy=lazy, index_name=name)
    snapshot = source.as_dataset(copy="deep")
    condition = Condition.compare(Condition.var("value"), "gt", .5)
    options = (EventExtractOptions(eval=ConditionEvalOptions(coord_name="clock"), max_events=4)
               if operation == "events" else IntervalExtractOptions(
                   eval=ConditionEvalOptions(coord_name="clock"), max_segments=3))
    required = ({"time", "edge_code", "sample_index_before", "sample_index_after"} if operation == "events"
                else {"time", "sample_index", "is_trigger", "valid_segment"})
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        if name in required:
            with pytest.raises(ValueError, match=rf"^events.{operation}: .*conflict.*") as error:
                getattr(source.events, operation)(condition, opts=options)
            assert "rename" in str(error.value)
        else:
            out = getattr(source.events, operation)(condition, opts=options)
            assert required == set(out.data_vars)
            for coord in snapshot.xindexes:
                if "sample" not in snapshot[coord].dims:
                    assert type(out.xindexes[coord]) is type(snapshot.xindexes[coord])
                    assert out.xindexes[coord].equals(snapshot.xindexes[coord])
    assert not tasks
    xr.testing.assert_identical(source.as_dataset(), snapshot)


@pytest.mark.parametrize("operation", ["events", "intervals", "boundaries", "segments", "stream", "windows"])
@pytest.mark.parametrize("case", ["unbounded", "no_episodes", "empty_sequence", "zero_batch"])
def test_tut_010_table_preflight_covers_adjacent_empty_consumers(tutorial_audit_source, operation, case):
    """TUT-010: reject known source conflicts before eager discovery or empty assembly."""
    from tal.core.event_ops import (
        AroundOptions,
        AtBoundariesOptions,
        Condition,
        ConditionEvalOptions,
        EventExtractOptions,
        IntervalExtractOptions,
        WhenOptions,
    )

    source = tutorial_audit_source(index_name="time", n=0 if case == "empty_sequence" else 6,
                                  batch=(0,) if case == "zero_batch" else (2,))
    snapshot = source.as_dataset(copy="deep")
    condition = Condition.compare(Condition.var("value"), "gt", 100. if case == "no_episodes" else .5)
    limit = None if case == "unbounded" else 3
    evaluation = ConditionEvalOptions(coord_name="clock")
    with pytest.raises(ValueError, match="^events.*conflict"):
        if operation == "events":
            source.events.events(condition, opts=EventExtractOptions(eval=evaluation, max_events=limit))
        elif operation == "intervals":
            source.events.intervals(condition, opts=IntervalExtractOptions(eval=evaluation, max_segments=limit))
        elif operation == "boundaries":
            source.events.at_boundaries(condition, opts=AtBoundariesOptions(eval=evaluation, max_events=limit))
        elif operation in ("segments", "stream"):
            source.events.when(condition, opts=WhenOptions(eval=evaluation, layout=operation, max_segments=limit))
        else:
            source.events.around(condition, opts=AroundOptions(eval=evaluation, grid=np.array([0.])))
    xr.testing.assert_identical(source.as_dataset(), snapshot)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("name", ["time", "edge_code", "sample_index_before", "sample_index_after"])
def test_tut_010_explicit_anchor_coordinates_cannot_claim_table_columns(tutorial_audit_source, lazy, name):
    """TUT-010: incoming surviving coordinate/index claims obey the same column ownership."""
    from tal.core.event_ops import AroundOptions, ConditionEvalOptions

    source = tutorial_audit_source(lazy=lazy, batch=())
    anchors = xr.DataArray([1., 4.], dims="anchor", coords={name: ("anchor", [20, 21])}).set_xindex(name)
    if lazy:
        anchors = anchors.chunk(anchor=1)
    snapshot, original = source.as_dataset(copy="deep"), anchors.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)), pytest.raises(
            ValueError, match="^events.around: .*conflict"):
        source.events.around(anchors, opts=AroundOptions(eval=ConditionEvalOptions(coord_name="clock"),
                                                        grid=np.array([0.])))
    assert not tasks
    xr.testing.assert_identical(source.as_dataset(), snapshot)
    xr.testing.assert_identical(anchors, original)


NATIVE_ANCHOR_CASES = [
    (lazy, kind, n, ())
    for lazy, kind, n in itertools.product(
        [False, True], ["range", "unindexed", "pandas"], [0, 2]
    )
]
NATIVE_ANCHOR_CASES += [
    (lazy, kind, 2, (0,))
    for lazy, kind in itertools.product([False, True], ["range", "unindexed"])
]


@pytest.mark.parametrize("lazy,kind,n,batch", NATIVE_ANCHOR_CASES)
def test_tut_018_explicit_anchor_native_axis(
    tutorial_audit_source, lazy, kind, n, batch
):
    """TUT-018: public ownership regression and controls."""
    a = tutorial_audit_source(
        clock_values=[0.0, 1.0, 3.0, 5.0, 8.0, 10.0], lazy=lazy, batch=batch
    )
    if kind == "range":
        coordinates = (
            xr.Dataset(
                coords=xr.Coordinates.from_xindex(
                    xr.indexes.RangeIndex.arange(max(n, 1), dim="anchor")
                )
            )
            .isel(anchor=slice(0, n))
            .coords
        )
    elif kind == "unindexed":
        coordinates = xr.Coordinates(
            {"anchor": xr.Variable("anchor", np.arange(n) + 20)}, indexes={}
        )
    else:
        coordinates = xr.Coordinates({"anchor": np.arange(n) + 20})
    q = xr.DataArray(np.array([1.0, 8.0])[:n], dims="anchor", coords=coordinates)
    if lazy:
        q = q.chunk(anchor=1)
    before = q.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        out = a.events.around(
            q,
            opts=AroundOptions(
                eval=ConditionEvalOptions(coord_name="clock"), grid=np.array([0.0])
            ),
        ).as_dataset()
    assert not tasks
    dim = out.attrs["tal"]["core"]["roles"]["batch_dims"][-1]
    if kind in ("range", "pandas"):
        expected = q.xindexes["anchor"].rename({"anchor": dim}, {"anchor": dim})
        assert type(out.xindexes[dim]) is type(expected) and out.xindexes[dim].equals(
            expected
        )
    else:
        assert dim not in out.xindexes
    xr.testing.assert_identical(q, before)


@pytest.mark.parametrize('layout', ['segments', 'stacked'])
@pytest.mark.parametrize('batch', [(2,), (0,), (2, 0)])
def test_tut_017_018_lazy_unindexed_batch_labels(tutorial_audit_source, layout, batch):
    """TUT-017/018: batch metadata must not force planning even in zero lanes."""
    source = tutorial_audit_source(lazy=True, labels='lazy', batch=batch)
    ds = source.as_dataset()
    dims = tuple(f'b{i}' for i in range(len(batch)))
    ds = ds.drop_indexes(dims)
    variables = {dim: ds[dim].chunk({dim: max(1, size)}).variable for dim, size in zip(dims, batch, strict=True)}
    ds = ds.assign_coords(xr.Coordinates(variables, indexes={}))
    source = AnalysisObject.from_data(ds, sequence_dim='sample', batch_dims=dims, param_coord='clock')
    anchors = xr.DataArray([1., 4.], dims='anchor', coords=xr.Coordinates({'anchor':xr.Variable('anchor',['alpha','beta'])}, indexes={})).chunk(anchor=1)
    before, original = source.as_dataset(copy='deep'), anchors.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda key, *_:tasks.append(key)):
        mask = source.events.mask(source > .5, opts=ConditionEvalOptions(coord_name='clock'))
        result = source.events.around(anchors, opts=AroundOptions(eval=ConditionEvalOptions(coord_name='clock'), grid=np.array([0.,.5]), layout=layout)).as_dataset()
    assert not tasks
    assert not mask.xindexes and not result.xindexes
    for dim in dims:
        xr.testing.assert_identical(xr.DataArray(result[dim].variable).compute(scheduler='synchronous'), xr.DataArray(before[dim].variable).compute(scheduler='synchronous'))
    xr.testing.assert_identical(source.as_dataset(), before)
    xr.testing.assert_identical(anchors, original)


@pytest.mark.parametrize("batch", [(), (2,), (2, 3), (0, 3)])
@pytest.mark.parametrize("layout", ["segments", "stacked"])
@pytest.mark.parametrize("empty", [False, True])
def test_tut_020_reviewed_empty_window_native_groups(tutorial_audit_source, batch, layout, empty):
    """Shared anchors restore complete native batch groups for every empty product."""
    source = tutorial_audit_source(lazy=True, batch=batch, labels="lazy")
    ds = source.as_dataset()
    if batch:
        ds = ds.drop_vars("b0").assign_coords(xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(batch[0], dim="b0")))
        ds = ds.assign_coords(alias=("b0", np.arange(batch[0]) + 100)).set_xindex("alias")
    source = AnalysisObject(ds)
    n = 0 if empty else 2
    anchors = xr.DataArray(np.array([1., 3.])[:n], dims="anchor", coords=xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(n, dim="anchor")))
    before, original = source.as_dataset(), anchors.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = source.events.around(anchors, opts=AroundOptions(eval=ConditionEvalOptions(coord_name="clock"), grid=np.array([0., .5]), layout=layout)).as_dataset()
    assert not tasks
    roles = result.attrs["tal"]["core"]["roles"]
    assert roles["core_dims"] == []
    assert roles["batch_dims"][:len(batch)] == [f"b{i}" for i in range(len(batch))]
    assert result.clock.dims == result.value.dims == result.valid.dims
    assert result.valid.dtype == np.dtype(bool)
    for name, index in ds.xindexes.items():
        assert type(result.xindexes[name]) is type(index)
        assert result.xindexes[name].equals(index)
    if layout == "segments":
        event = roles["batch_dims"][-1]
        expected = anchors.xindexes["anchor"].rename({"anchor": event}, {"anchor": event})
        assert result.xindexes[event].equals(expected)
    assert set(result.data_vars) == {"value"}
    xr.testing.assert_identical(source.as_dataset(), before)
    xr.testing.assert_identical(anchors, original)
