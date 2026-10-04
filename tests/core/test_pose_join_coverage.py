"""Contracts 061/062/081: composite joins share correspondence and coverage."""

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from scipy.spatial.transform import Rotation as SciRotation

from tal.core import AnalysisLayoutSpec
from tal.core.schema_read import read_param_coord_name
from tal.frames import FrameGraph
from tal.spatial import LinearVelocity, Pose, Position, Rotation


def _reference(trial, sample, offset):
    matrix = np.eye(4)
    matrix[:3, :3] = SciRotation.from_euler("zy", [offset + trial * .01, sample * .1]).as_matrix()
    matrix[:3, 3] = [offset + sample, trial * .1, sample * .2]
    return matrix


def _pose(trials, samples, *, offset, counts=True, lazy=False, rep="components", graph=None):
    matrices = np.array([[_reference(t, s, offset) for s in samples] for t in trials])
    coords = {"trial": trials, "sample": samples, "axis": list("xyz"), "quat": list("xyzw")}
    if counts:
        coords["count"] = ("trial", [len(samples), max(0, len(samples) - 1)])
    ds = xr.Dataset({
        "p": (("trial", "sample", "axis"), matrices[..., :3, 3]),
        "q": (("trial", "sample", "quat"), SciRotation.from_matrix(matrices[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(len(trials), len(samples), 4)),
    }, coords=coords)
    ds.attrs["description"] = {"source": [offset]}
    ds["p"].attrs["units"] = "m"
    if lazy:
        for name in ds.data_vars:
            ds[name] = ds[name].chunk({"trial": 1, "sample": 2})
    kwargs = {"sequence_dim": "sample", "batch_dims": ("trial",), "sequence_size_coord": "count" if counts else None}
    position = Position(AnalysisLayoutSpec(core_dims=("axis",), **kwargs).wrap(ds[["p"]]))
    rotation = Rotation(AnalysisLayoutSpec(core_dims=("quat",), **kwargs).wrap(ds[["q"]]))
    return Pose.from_components(rotation, position, graph=graph).to_rep(rep)


def _labels(left, right, join):
    if join == "inner":
        return sorted(set(left) & set(right))
    if join == "outer":
        return sorted(set(left) | set(right))
    return left if join == "left" else right


def _expected(trials, samples, left_axes, right_axes, *, counts):
    expected = np.full((len(trials), len(samples), 4, 4), np.nan)
    valid = np.zeros((len(trials), len(samples)), dtype=bool)
    for i, trial in enumerate(trials):
        for j, sample in enumerate(samples):
            present = all(trial in ts and sample in ss for ts, ss in (left_axes, right_axes))
            covered = present and all(not counts or ss.index(sample) < len(ss) - ts.index(trial) for ts, ss in (left_axes, right_axes))
            if covered:
                expected[i, j] = _reference(trial, sample, .7) @ _reference(trial, sample, .2)
                valid[i, j] = True
    return expected, valid


def _assert_coverage(ds, expected):
    validity = ds.attrs["tal"]["core"].get("validity", {})
    size_name = validity.get("sequence_size_coord")
    if size_name is not None:
        actual = np.arange(ds.sizes["sample"])[None, :] < ds[size_name].data[:, None]
    else:
        actual = ds.valid.broadcast_like(ds[next(iter(ds.data_vars))].isel({dim: 0 for dim in ds.attrs["tal"]["core"]["roles"]["core_dims"]}, drop=True, missing_dims="ignore")).transpose("trial", "sample").data
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("axis", ["trial", "sample"])
@pytest.mark.parametrize("join", ["inner", "outer", "left", "right"])
@pytest.mark.parametrize("lazy,validate,rep,counts", [
    (False, True, "components", True), (False, False, "matrix", False),
    (True, False, "components", False), (True, True, "matrix", True),
])
def test_pose_join_correspondence_and_coverage(axis, join, lazy, validate, rep, counts):
    """ID: POSE_JOIN_COVERAGE_001; one correspondence for both components."""
    left_axes = ([10, 20], [0, 1, 2])
    right_axes = ([20, 30], [0, 1, 2]) if axis == "trial" else ([10, 20], [1, 2, 3])
    graph = FrameGraph()
    left = _pose(*left_axes, offset=.2, counts=counts, lazy=lazy, rep=rep, graph=graph)
    right = _pose(*right_axes, offset=.7, counts=counts, lazy=lazy, rep="components", graph=graph)
    left = Pose(left, parent="world", child="deck", graph=graph)
    right = Pose(right, parent="deck", child="body", graph=graph)
    before = [obj.as_dataset() for obj in (left, right)]
    policy = {"batch_join" if axis == "trial" else "sequence_join": join}
    operands = (left.a(**policy), right) if validate else (left, right.a(**policy))
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        output = operands[0].compose(operands[1], validate=validate)
    assert not tasks
    assert output.graph is graph and output.frames.ids() == ("world", "body")
    ds = output.as_dataset().compute(scheduler="synchronous")
    trials = _labels(left_axes[0], right_axes[0], join) if axis == "trial" else left_axes[0]
    samples = _labels(left_axes[1], right_axes[1], join) if axis == "sample" else left_axes[1]
    np.testing.assert_array_equal(ds.trial, trials)
    np.testing.assert_array_equal(ds["sample"], samples)
    expected, valid = _expected(trials, samples, left_axes, right_axes, counts=counts)
    matrix = output.as_matrix().to_dataarray().compute(scheduler="synchronous").transpose("trial", "sample", ...)
    np.testing.assert_allclose(matrix, expected, atol=1e-12)
    if counts or join != "inner":
        _assert_coverage(ds, valid)
    assert ds.attrs["tal"]["core"]["roles"]["batch_dims"] == ["trial"]
    assert len(ds.data_vars) == (2 if rep == "components" else 1)
    for obj, original in zip((left, right), before, strict=True):
        xr.testing.assert_identical(obj.as_dataset(), original)


@pytest.mark.parametrize("axis", ["trial", "sample"])
@pytest.mark.parametrize("lazy,validate", [(False, True), (True, False)])
def test_pose_disjoint_inner_join_is_empty(axis, lazy, validate):
    """ID: POSE_JOIN_COVERAGE_002; an empty intersection has truthful topology."""
    left = _pose([10, 20], [0, 1, 2], offset=.2, lazy=lazy)
    axes = ([30, 40], [0, 1, 2]) if axis == "trial" else ([10, 20], [3, 4, 5])
    right = _pose(*axes, offset=.7, lazy=lazy)
    policy = {"batch_join" if axis == "trial" else "sequence_join": "inner"}
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        output = left.a(**policy).compose(right, validate=validate)
    assert not tasks
    ds = output.as_dataset().compute(scheduler="synchronous")
    assert ds.sizes[axis] == 0
    assert set(ds.data_vars) == {"p", "q"}
    assert not ds["count"].data.any()


@pytest.mark.parametrize("lazy,validate", [(False, True), (True, False)])
def test_pose_parameter_key_reaches_both_components(lazy, validate):
    """ID: POSE_JOIN_COVERAGE_003; parameter labels, not sample labels, are the key."""
    left = _pose([10, 20], [0, 1, 2], offset=.2, lazy=lazy)
    right = _pose([10, 20], [0, 1, 2], offset=.7, lazy=lazy)
    left = Pose(left.as_dataset().assign_coords(time=("sample", [0., 1., 2.]))).set_param_coord(name="time")
    right = Pose(right.as_dataset().assign_coords(time=("sample", [0., 1., 2.]), sample=[10, 11, 12])).set_param_coord(name="time")
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        output = left.a(on="param", sequence_join=None).compose(right, validate=validate)
    assert not tasks
    expected, _ = _expected([10, 20], [0, 1, 2], ([10, 20], [0, 1, 2]), ([10, 20], [0, 1, 2]), counts=True)
    np.testing.assert_allclose(output.as_matrix().to_dataarray().compute().transpose("trial", "sample", ...), expected, atol=1e-12)
    np.testing.assert_array_equal(output.as_dataset().coords["sample"], [0, 1, 2])


@pytest.mark.parametrize("lazy,validate", [(False, False), (True, True)])
@pytest.mark.parametrize("reached", [False, True])
def test_pose_intersection_controls_quaternion_reachability(lazy, validate, reached):
    """ID: POSE_JOIN_COVERAGE_004; a sibling's coverage applies before translation."""
    left = _pose([10, 20], [0, 1, 2], offset=.2).set_validity(sequence_size_coord="count")
    left = Pose(left.as_dataset().assign_coords(count=("trial", [1, 1])))
    right = _pose([10, 20], [0, 1, 2], offset=.7)
    ds = right.as_dataset()
    values = ds.q.data.copy()
    values[:, 0 if reached else 1] = 0.
    ds["q"] = (ds.q.dims, values)
    if lazy:
        for name in ds.data_vars:
            ds[name] = ds[name].chunk({"sample": 1})
    right = Pose(ds)
    if reached:
        with pytest.raises(ValueError, match="spatial.pose.compose"):
            left.compose(right, validate=validate).as_dataset().compute(scheduler="synchronous")
        return
    output = left.compose(right, validate=validate).as_dataset().compute(scheduler="synchronous")
    np.testing.assert_array_equal(output["count"], [1, 1])
    assert np.isnan(output.p[:, 1:]).all() and np.isnan(output.q[:, 1:]).all()


@pytest.mark.parametrize("family", ["rotation", "position", "linear", "pose_apply"])
@pytest.mark.parametrize("lazy,validate", [(False, True), (True, False)])
def test_implicit_coverage_shared_by_adjacent_operations(family, lazy, validate):
    """ID: POSE_JOIN_COVERAGE_005; join membership does not require size metadata."""
    left = _pose([10, 20], [0, 1, 2], offset=.2, counts=False, lazy=lazy)
    right = _pose([20, 30], [0, 1, 2], offset=.7, counts=False, lazy=lazy)
    _, lr = left.decompose()
    rp, rr = right.decompose()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        if family == "rotation":
            output = lr.a(batch_join="outer").compose(rr, validate=validate)
        else:
            target = LinearVelocity(rp) if family == "linear" else rp
            transform = left if family == "pose_apply" else lr
            output = transform.a(batch_join="outer").apply(target, validate=validate)
    assert not tasks
    ds = output.as_dataset().compute(scheduler="synchronous")
    np.testing.assert_array_equal(ds.valid.transpose("trial", "sample"), [[False]*3, [True]*3, [False]*3])
    if family == "rotation":
        actual = output.as_matrix().to_dataarray().compute().transpose("trial", "sample", ...).data[1]
        expected = [_reference(20, s, .7)[:3, :3] @ _reference(20, s, .2)[:3, :3] for s in range(3)]
    else:
        actual = output.to_dataarray().compute().transpose("trial", "sample", ...).data[1]
        expected = [_reference(20, s, .2)[:3, :3] @ _reference(20, s, .7)[:3, 3] + (_reference(20, s, .2)[:3, 3] if family == "pose_apply" else 0) for s in range(3)]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("lazy,empty", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("collision", ["payload", "dimension", "index"])
def test_implicit_coverage_protects_output_namespace(lazy, empty, collision):
    """ID: POSE_JOIN_COVERAGE_006; generated validity never replaces an output owner."""
    left = _pose([10, 20], [0, 1, 2], offset=.2, counts=False, lazy=lazy)
    right = _pose([20, 30], [0, 1, 2], offset=.7, counts=False, lazy=lazy)
    if collision == "payload":
        left = left.rename({"p": "valid"})
    elif collision == "dimension":
        left, right = left.rename({"trial": "valid"}), right.rename({"trial": "valid"})
    else:
        left = Pose(left.as_dataset().assign_coords(valid=("trial", [10, 20])).set_xindex("valid"))
        right = Pose(right.as_dataset().assign_coords(valid=("trial", [20, 30])).set_xindex("valid"))
    if empty:
        left, right = left.isel(sample=slice(0, 0)), right.isel(sample=slice(0, 0))
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])), pytest.raises(ValueError, match=r"spatial.pose.compose.*valid"):
        left.a(batch_join="outer").compose(right, validate=False)
    assert not tasks


@pytest.mark.parametrize("lazy", [False, True])
def test_join_retains_shared_and_one_sided_coordinates(lazy):
    """ID: POSE_JOIN_COVERAGE_007; generated metadata leaves ordinary ownership intact."""
    left = _pose([10, 20], [0, 1, 2], offset=.2, counts=False, lazy=lazy)
    right = _pose([20, 30], [0, 1, 2], offset=.7, counts=False, lazy=lazy)
    shared = xr.DataArray([0., 1., 2.], dims="sample")
    if lazy:
        shared = shared.chunk({"sample": 1})
    left = Pose(left.as_dataset().assign_coords(tag=shared, note="left", valid=False))
    right = Pose(right.as_dataset().assign_coords(tag=("sample", [0., 1., 2.])))
    before = [value.as_dataset() for value in (left, right)]
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        output = left.a(batch_join="outer").compose(right)
    assert not tasks
    ds = output.as_dataset().compute(scheduler="synchronous")
    assert ds.note.data == "left"
    np.testing.assert_array_equal(ds.tag, [0., 1., 2.])
    np.testing.assert_array_equal(ds.valid.transpose("trial", "sample"), [[False]*3, [True]*3, [False]*3])
    for value, original in zip((left, right), before, strict=True):
        xr.testing.assert_identical(value.as_dataset(), original)


@pytest.mark.parametrize("family", ["rotation", "rotation_apply", "pose_apply"])
@pytest.mark.parametrize("lazy", [False, True])
def test_implicit_index_collision_precedes_adjacent_payload_alignment(family, lazy):
    """ID: POSE_JOIN_COVERAGE_008; sibling operations share namespace preflight."""
    left = _pose([10, 20], [0, 1, 2], offset=.2, counts=False, lazy=lazy)
    right = _pose([20, 30], [0, 1, 2], offset=.7, counts=False, lazy=lazy)
    left = Pose(left.as_dataset().assign_coords(valid=("trial", [10, 20])).set_xindex("valid"))
    right = Pose(right.as_dataset().assign_coords(valid=("trial", [20, 30])).set_xindex("valid"))
    _, lr = left.decompose()
    rp, rr = right.decompose()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])), pytest.raises(ValueError, match=r"spatial\.(pose|rotation)\.(apply|compose).*valid"):
        if family == "rotation":
            lr.a(batch_join="outer").compose(rr)
        elif family == "rotation_apply":
            lr.a(batch_join="outer").apply(rp)
        else:
            left.a(batch_join="outer").apply(rp)
    assert not tasks


def _with_parameter(pose, *, per_trial, empty):
    ds = pose.as_dataset()
    time = xr.DataArray([0., 1., 2.], dims="sample")
    if per_trial:
        time = (ds.trial * .01 + time).transpose("trial", "sample")
    output = Pose(ds.assign_coords(time=time, note="recorded")).set_param_coord(name="time")
    return output.isel(sample=slice(0, 0)) if empty else output


def _joined_parameter_result(family, left, right, join, validate):
    if family == "pose":
        return left.a(batch_join=join).compose(right, validate=validate)
    _, rotation = left.decompose()
    target, other_rotation = right.decompose()
    if family == "rotation":
        return rotation.a(batch_join=join).compose(other_rotation, validate=validate)
    transform = left if family == "pose_apply" else rotation
    return transform.a(batch_join=join).apply(target, validate=validate)


def _assert_parameter_join_values(output, family, trials, *, empty):
    matrices, valid = _expected(trials, [] if empty else [0, 1, 2], ([10, 20], [0, 1, 2]), ([20, 30], [0, 1, 2]), counts=True)
    if family in {"pose", "rotation"}:
        actual = output.as_matrix().to_dataarray().compute().transpose("trial", "sample", ...)
        expected = matrices if family == "pose" else matrices[..., :3, :3]
    else:
        actual = output.to_dataarray().compute().transpose("trial", "sample", ...)
        expected = np.full((len(trials), 0 if empty else 3, 3), np.nan)
        for i, j in np.argwhere(valid):
            left, right = _reference(trials[i], j, .2), _reference(trials[i], j, .7)
            expected[i, j] = left[:3, :3] @ right[:3, 3] + (left[:3, 3] if family == "pose_apply" else 0)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    _assert_coverage(output.as_dataset().compute(), valid)


@pytest.mark.parametrize("family,join", [
    (family, join)
    for family in ("pose", "rotation", "pose_apply", "rotation_apply")
    for join in (("inner", "outer") if family.endswith("apply") else ("inner", "outer", "left", "right"))
])
@pytest.mark.parametrize("lazy,validate,rep,per_trial,empty", [
    (False, True, "components", True, False),
    (True, False, "matrix", True, False),
    (False, False, "matrix", False, False),
    (True, True, "components", False, False),
    (True, True, "components", True, True),
])
def test_join_finalizes_only_surviving_parameter_declarations(family, join, lazy, validate, rep, per_trial, empty):
    """ID: POSE_JOIN_COVERAGE_009; eager auxiliary merging cannot leave stale schema."""
    left = _with_parameter(_pose([10, 20], [0, 1, 2], offset=.2, lazy=lazy, rep=rep), per_trial=per_trial, empty=empty)
    right = _with_parameter(_pose([20, 30], [0, 1, 2], offset=.7, lazy=lazy), per_trial=per_trial, empty=empty)
    before = [value.as_dataset() for value in (left, right)]
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        output = _joined_parameter_result(family, left, right, join, validate)
    assert not tasks
    ds = output.as_dataset().compute()
    trials = _labels([10, 20], [20, 30], join)
    np.testing.assert_array_equal(ds.trial, trials)
    np.testing.assert_array_equal(ds["sample"], [] if empty else [0, 1, 2])
    retained = join == "inner" or not per_trial or empty
    assert ("time" in ds.coords) == retained
    assert read_param_coord_name(ds) == ("time" if retained else None)
    if retained:
        expected_time = np.empty((len(trials), 0)) if empty else np.array(trials)[:, None] * .01 + np.arange(3)
        np.testing.assert_allclose(ds.time, expected_time if per_trial else np.arange(3))
    assert ds.note.data == "recorded"
    _assert_parameter_join_values(output, family, trials, empty=empty)
    for value, original in zip((left, right), before, strict=True):
        xr.testing.assert_identical(value.as_dataset(), original)
