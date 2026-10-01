"""Execute the published tutorials and check independently derived results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation, Slerp

from tal import AnalysisObject
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.spatial import Position

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "examples/tutorial"
DATA = TUTORIAL / "data/flight_telemetry"


def _execute(number):
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    plt = pytest.importorskip("matplotlib.pyplot")
    pytest.importorskip("IPython")
    path = next(TUTORIAL.glob(f"{number:02d}_*.ipynb"))
    namespace, snapshots = {}, {}
    try:
        for index, cell in enumerate(json.loads(path.read_text())["cells"]):
            if cell["cell_type"] != "code":
                continue
            exec(compile("".join(cell["source"]), f"{path}:cell-{index}", "exec"), namespace)  # noqa: S102
            for name in (
                "observations", "ship_log", "drone_log", "source", "declared", "ship_pose", "position",
                "selection_demo", "event_demo", "ragged_demo", "memory_source", "plot_demo", "association_point",
                "batch_axis_samples", "lazy_event_demo", "orientation_source",
            ):
                value = namespace.get(name)
                if isinstance(value, AnalysisObject) and name not in snapshots:
                    snapshots[name] = value.as_dataset(copy="shallow").copy(deep=True)
        for name, snapshot in snapshots.items():
            xr.testing.assert_identical(namespace[name].as_dataset(copy="shallow"), snapshot)
        return namespace
    finally:
        plt.close("all")


def _reference_trial(trial):
    observations = pd.read_csv(DATA / "observations" / f"{trial}.csv")
    ship = pd.read_csv(DATA / "ship_poses" / f"{trial}.csv")
    time = observations["time"].to_numpy()
    xyz = observations[[f"drone.position.{axis}" for axis in "xyz"]].to_numpy()
    origin = np.column_stack([np.interp(time, ship["time"], ship[f"ship.position.{axis}"]) for axis in "xyz"])
    rotations = Rotation.from_quat(ship[[f"ship.rotation.{axis}" for axis in "xyzw"]].to_numpy())
    relative = Slerp(ship["time"], rotations)(time).inv().apply(xyz - origin)
    crossing = np.flatnonzero((relative[1:, 2] <= 0) & (relative[:-1, 2] > 0)) + 1
    return time, relative, crossing


def _check_landing(namespace):
    """ID: DOC_CAPSTONE_WORKFLOW_001; observed geometry, topology, frames, and censoring."""
    relative = namespace["relative"].as_dataset(copy="shallow")
    windows = namespace["windows"].as_dataset(copy="shallow")
    report = namespace["report"]
    lengths, eligible = [], np.zeros(3, dtype=int)
    sensitivity = np.zeros((3, 3), dtype=int)
    for trial in report.index:
        time, expected, crossing = _reference_trial(trial)
        lengths.append(len(time))
        actual = relative["position"].sel(trial=trial)
        np.testing.assert_allclose(actual[:len(time)], expected, atol=1e-10)
        assert np.isnan(actual[len(time):]).all()
        if not crossing.size:
            assert not report.loc[trial, "observed"]
            assert np.isnan(windows["position"].sel(trial=trial)).all()
            continue
        index = crossing[0]
        point = expected[index]
        np.testing.assert_allclose(report.loc[trial, ["along_m", "across_m", "height_m"]].astype(float), point, atol=1e-10)
        eligible += (abs(point[1]) <= np.array([1., 1.5, 2.])) & (abs(point[0]) <= 2.)
        sensitivity += (abs(point[0]) <= np.array([1., 2., 3.])[:, None]) & (abs(point[1]) <= np.array([1., 1.5, 2.])[None, :])
        np.testing.assert_allclose(windows["event_time"].sel(trial=trial), [time[index]])
        query = time[index] + windows["tau"].to_numpy()
        expected_window = np.column_stack([np.interp(query, time, expected[:, axis]) for axis in range(3)])
        np.testing.assert_allclose(windows["position"].sel(trial=trial).isel(event=0).transpose("tau", "axis"), expected_window, atol=1e-10)
    np.testing.assert_array_equal(relative["sequence_size"], lengths)
    np.testing.assert_array_equal(namespace["width_comparison"]["eligible"], eligible)
    np.testing.assert_array_equal(namespace["corridor_sensitivity"], sensitivity)
    assert read_roles(relative) == (True, "sample", ("trial",), ("axis",))
    assert read_param_coord_name(relative) == "time"
    assert read_sequence_size_coord_name(relative) == "sequence_size"
    assert read_roles(windows) == (True, "tau", ("trial", "event"), ("axis",))
    assert namespace["relative"].frames.ids() == ("ship", "drone")
    assert namespace["relative"].graph is namespace["graph"]


def _check_tabular(number, namespace):
    if number == 1:
        expected = [pd.read_csv(path)["altitude"].min() for path in sorted((DATA / "observations").glob("*.csv"))]
        np.testing.assert_allclose(namespace["lowest"].to_dataarray(copy="shallow"), expected)
    if number == 5:
        for path in sorted((DATA / "observations").glob("*.csv")):
            log = pd.read_csv(path)
            entered = np.flatnonzero((log["altitude"].to_numpy()[1:] < 2) & (log["altitude"].to_numpy()[:-1] >= 2)) + 1
            actual = namespace["entries"].as_dataset(copy="shallow")["time"].sel(trial=path.stem)
            expected = [log["time"].iloc[entered[0]]] if entered.size else [np.nan]
            np.testing.assert_allclose(actual, expected)
    if number == 6:
        expected = [pd.read_csv(path)["altitude"].mean() for path in sorted((DATA / "observations").glob("*.csv"))]
        np.testing.assert_allclose(namespace["means"].to_dataarray(copy="shallow"), expected)
        np.testing.assert_allclose(namespace["by_guidance"].to_dataarray(copy="shallow"), [np.mean(expected[:6]), np.mean(expected[6:])])


def _check_frame_expression(namespace):
    """Ragged observations preserve validity through independent basis rotation."""
    output = namespace["camera_basis"].to_dataarray(copy="shallow")
    restored = namespace["back_in_world"].as_dataset(copy="shallow")
    for index in range(12):
        trial = f"flight_{index + 1:02d}"
        ship = pd.read_csv(DATA / "ship_poses" / f"{trial}.csv")
        drone = pd.read_csv(DATA / "observations" / f"{trial}.csv")
        xyz = drone[[f"drone.position.{axis}" for axis in "xyz"]].to_numpy()
        rotation = Rotation.from_quat(ship[[f"ship.rotation.{axis}" for axis in "xyzw"]].to_numpy())
        expected = Slerp(ship["time"], rotation)(drone["time"]).inv().apply(xyz)
        actual = output.sel(trial=trial)
        np.testing.assert_allclose(actual[:len(drone)], expected, atol=1e-10)
        assert np.isnan(actual[len(drone):]).all()
        recovered = restored["position"].sel(trial=trial)
        np.testing.assert_allclose(recovered[:len(drone)], xyz, atol=1e-10)
        assert np.isnan(recovered[len(drone):]).all()
        np.testing.assert_allclose(restored.time.sel(trial=trial)[:len(drone)], drone["time"])
        assert restored.sequence_size.sel(trial=trial).item() == len(drone)
    assert namespace["camera_basis"].frames.ids() == ("world", "drone")
    assert namespace["in_camera"].frames.ids() == ("camera", "drone")
    assert namespace["camera_basis"].graph is namespace["graph"]
    assert namespace["back_in_world"].frames.ids() == ("world", "drone")
    assert namespace["back_in_world"].graph is namespace["graph"]
    assert read_roles(restored) == (True, "sample", ("trial",), ("axis",))
    xr.testing.assert_identical(restored.coords.to_dataset(), namespace["position"].as_dataset(copy="shallow").coords.to_dataset())


def _check_spatial_components_and_alignment(namespace):
    """ID: DOC_TUTORIAL_SPATIAL_001; component interfaces and explicit population joins."""
    generic, typed = namespace["generic_position"], namespace["translation"]
    assert type(generic) is AnalysisObject and type(typed) is Position
    xr.testing.assert_identical(generic.to_dataarray(), typed.to_dataarray())
    assert read_roles(generic.as_dataset()) == (True, "sample", ("trial",), ("axis",))
    for name, trials in (("shared", ["flight_02"]), ("population", ["flight_01", "flight_02", "flight_03"])):
        output = namespace[name]
        ds = output.as_dataset(copy="shallow")
        np.testing.assert_array_equal(ds.trial, trials)
        assert set(ds.data_vars) == {"position", "rotation"}
        assert read_roles(ds) == (True, "sample", ("trial",), ("axis", "quat"))
        assert output.frames.ids() == ("world", "world")
        expected = np.full((len(trials), 25, 4, 4), np.nan)
        expected[trials.index("flight_02")] = np.eye(4)
        np.testing.assert_allclose(output.as_matrix().to_dataarray(), expected, atol=1e-12)
    shared = namespace["shared"].as_dataset(copy="shallow")
    population = namespace["population"].as_dataset(copy="shallow")
    assert read_param_coord_name(shared) == "time"
    np.testing.assert_allclose(shared.time, np.arange(25)[None, :] / 2)
    np.testing.assert_array_equal(shared.sequence_size, [25])
    assert read_param_coord_name(population) is None and "time" not in population.coords
    np.testing.assert_array_equal(population.valid, np.repeat([[False], [True], [False]], 25, axis=1))


def _check_lazy_lifecycle(namespace):
    """ID: DOC_TUTORIAL_LIFECYCLE_001; lazy field assembly through resource cleanup."""
    assert namespace["planning_tasks"] == []
    assert not namespace["output_dir"].exists()
    assert isinstance(namespace["lazy_position"], Position)
    assert namespace["planned"].to_dataarray(copy="shallow").chunks is not None
    expected = []
    for path in sorted((DATA / "observations").glob("*.csv")):
        recorded = pd.read_csv(path)[[f"drone.position.{axis}" for axis in "xyz"]].to_numpy()
        expected.append(np.linalg.norm(recorded, axis=1).mean())
    result = namespace["result"]
    np.testing.assert_allclose(result["datavar"], expected)
    np.testing.assert_array_equal(result.trial, [f"flight_{i:02d}" for i in range(1, 13)])
    assert dict(result.sizes) == {"trial": 12}
    assert read_roles(result) == (True, None, ("trial",), ())
    xr.testing.assert_identical(namespace["decoded"], namespace["observations"].as_dataset(copy="shallow"))


def _check_construction_and_selection(number, ns):
    """ID: DOC_TUTORIAL_TEACHING_001; roles, ownership, and labeled query decisions."""
    if number == 2:
        assert read_roles(ns["core_only"].as_dataset()) == (True, None, (), ("axis",))
        assert read_roles(ns["batch_core"].as_dataset()) == (True, None, ("run",), ("axis",))
        np.testing.assert_allclose(ns["valid_mean"].to_dataarray(), [[1., 2., 3.]])
        np.testing.assert_allclose(ns["missing_skip"].to_dataarray(), 1.)
        assert np.isnan(ns["missing_keep"].to_dataarray())
        assert read_roles(ns["renamed_demo"].as_dataset()) == (True, "sample", ("run",), ("axis",))
        np.testing.assert_allclose(ns["declared"].to_dataarray(), [[[1., 2., 3.], [99., 99., 99.]]])
    if number == 3:
        assert ns["one_run"].as_dataset().sizes == {"step": 3}
        assert ns["kept_run"].as_dataset().sizes == {"run": 1, "step": 3}
        np.testing.assert_array_equal(ns["sample_map"], [1])
        np.testing.assert_allclose(ns["nearest_demo"].to_dataarray(), [10.])
        np.testing.assert_allclose(ns["nearest_demo"].as_dataset()["clock"], [1.])
        np.testing.assert_allclose(ns["axis_selected"].as_dataset()["sample"], [1.])
        np.testing.assert_allclose(ns["axis_evaluated"].as_dataset()["sample"], [1.4])
        np.testing.assert_allclose(ns["axis_selected"].to_dataarray(), [10.])
        np.testing.assert_allclose(ns["axis_evaluated"].to_dataarray(), [14.])
        assert "sample" not in ns["axis_selected"].as_dataset().xindexes
        np.testing.assert_allclose(ns["batch_axis_selected"]["sample"], [[1., 3.], [2., 2.]])
        np.testing.assert_allclose(ns["batch_axis_evaluated"].signal, [[12., 25.], [12., 25.]])
        np.testing.assert_allclose(ns["axis_grid_selected"]["sample"], [[0., 1.], [3., 3.]])
        np.testing.assert_allclose(ns["axis_grid_evaluated"].signal, [[2.5, 12.], [25., 28.]])
        assert ns["axis_grid_evaluated"]["sample"].dims == ("sample", "column")
        assert "sample" not in ns["axis_grid_evaluated"].xindexes
        assert read_param_coord_name(ns["axis_grid_evaluated"]) is None
        np.testing.assert_allclose(ns["auxiliary_grid_selected"].clock, [[1., 1.], [1., 3.]])
        np.testing.assert_allclose(ns["auxiliary_grid_evaluated"].signal, [[6., 14.], [16., 28.]])
        assert "clock" not in ns["auxiliary_grid_evaluated"].xindexes
        assert read_param_coord_name(ns["auxiliary_grid_evaluated"]) is None
        assert ns["typed_label_tasks"] == []
        np.testing.assert_array_equal(ns["lazy_typed_display"].row, [100, 100, 200, 200])
        np.testing.assert_allclose(ns["lazy_typed_display"].time, [.2, .3, 1.2, 1.3])
        np.testing.assert_allclose(ns["lazy_typed_result"].as_matrix().to_dataarray().compute(), np.tile(np.eye(3), (4, 1, 1)))
        assert "row" not in ns["lazy_typed_display"].xindexes
        assert ns["masked_demo"].as_dataset().sizes == ns["selection_demo"].as_dataset().sizes
        np.testing.assert_allclose(ns["evaluated_demo"].to_dataarray(), [14.])
        np.testing.assert_allclose(ns["evaluated_demo"].as_dataset()["clock"], [1.4])
        np.testing.assert_array_equal(ns["inner_demo"].as_dataset().step, [20])
        np.testing.assert_allclose(ns["inner_demo"].to_dataarray(), [20.])
        np.testing.assert_allclose(ns["outer_demo"].to_dataarray(), [np.nan, 20., np.nan])
        xr.testing.assert_identical(ns["broadcast_demo"].as_dataset(), ns["scalar_demo"].as_dataset())
    if number == 4:
        np.testing.assert_allclose(ns["linear_demo"].to_dataarray(), [np.nan, 1.1, 2.6, np.nan])
        np.testing.assert_allclose(ns["domain_difference"].to_dataarray(), 0.1)
        np.testing.assert_allclose(ns["distance_at"].to_dataarray(), [[5., 20.], [10., 30.]])
        np.testing.assert_array_equal(ns["distance_at"].as_dataset().location, ["near", "far"])
        np.testing.assert_allclose(ns["calendar_at"].to_dataarray(), [1.])


def _check_events_and_statistics(number, ns):
    """ID: DOC_TUTORIAL_TEACHING_002; event topology and explicit statistical populations."""
    if number == 5:
        assert ns["window_planning_tasks"] == []
        empty = ns["empty_windows"]
        assert empty.sizes[ns["empty_event_dim"]] == 0
        assert empty.time.dims == empty.signal.dims == empty.valid.dims
        assert empty.valid.dtype == np.dtype(bool)
        packed = ns["packing_display"]
        np.testing.assert_allclose(packed.time, [[.5, 1., 1.5, 4.5, 5., 5.5, np.nan, np.nan, np.nan]] * 2, equal_nan=True)
        np.testing.assert_array_equal(packed.window_event_index, [[0, 0, 0, 2, 2, 2, -1, -1, -1]] * 2)
        np.testing.assert_array_equal(packed.window_size, [6, 6])
        np.testing.assert_array_equal(packed.valid, [[True] * 6 + [False] * 3] * 2)
        assert packed.window_event_index.dtype == np.dtype("int64")
        assert packed.valid.dtype == np.dtype(bool)
        assert not ns["label_planning_tasks"]
        assert "sample" not in ns["unindexed_lazy_mask"].xindexes
        np.testing.assert_array_equal(ns["unindexed_lazy_mask"].compute(), ns["event_values"] > 2)
        np.testing.assert_array_equal(ns["labeled_stack"].window_event_index, [[0, 0, 0, 1, 1, 1]] * 2)
        np.testing.assert_allclose(ns["labeled_stack"].time, [[.5, 1., 1.5, 4.5, 5., 5.5]] * 2)
        assert isinstance(ns["labeled_windows"].xindexes[ns["labeled_event_dim"]], xr.indexes.RangeIndex)
        assert ns["condition_planning_tasks"] == []
        np.testing.assert_array_equal(ns["lazy_mask_values"], [[True, True, True], [False, False, False]])
        assert ns["lazy_reused_mask"].dims == ("run", "sample")
        np.testing.assert_array_equal(ns["lazy_reused_mask"].coords["time"].compute(), [1., 2., 5.])
        np.testing.assert_allclose(ns["initial_events_demo"].time, [1., 2.])
        np.testing.assert_allclose(ns["irregular_stream"].as_dataset().time.isel(run=0), [.2, .7, 3., 4.])
        np.testing.assert_allclose(ns["irregular_segments"].as_dataset().time.isel(run=0, sample=slice(0, 2)), [[.2, .7], [3., 4.]])
        np.testing.assert_allclose(ns["axis_segments"]["sample"][:, :2], [[.2, .7], [3., 4.]])
        np.testing.assert_allclose(ns["axis_stream"]["sample"][:4], [.2, .7, 3., 4.])
        assert "sample" not in ns["axis_segments"].xindexes
        np.testing.assert_array_equal(ns["axis_stream"].orig_index[4:], [-1] * 12)
        assert np.isnan(ns["axis_stream"]["sample"][4:]).all()
        entries = ns["episode_entries"].as_dataset()
        np.testing.assert_allclose(entries.time, [[1., 5.], [np.nan, np.nan]])
        np.testing.assert_allclose(entries.signal, [[3., 3.], [np.nan, np.nan]])
        stream = ns["episode_stream"].as_dataset()
        np.testing.assert_allclose(stream.signal, [[3., 4., 3., 4.], [np.nan] * 4])
        np.testing.assert_array_equal(stream.stream_size, [4, 0])
        np.testing.assert_array_equal(stream.orig_index.isel(run=0), [1, 2, 5, 6])
        segments = ns["episode_segments"].as_dataset()
        np.testing.assert_array_equal(segments.segment_size, [[2, 2], [0, 0]])
        np.testing.assert_allclose(segments.signal.isel(run=0, sample=slice(0, 2)), [[3., 4.], [3., 4.]])
        windows = ns["episode_windows"].as_dataset()
        assert read_roles(windows)[1] == "tau"
        assert ns["window_event_dim"] != "event"  # Inherited anchor coordinate cannot be replaced.
        np.testing.assert_allclose(windows.event_time.isel(run=0), [1., 5.])
        np.testing.assert_allclose(windows.signal.sel(run="episodes"), [[1.5, 3., 3.5]] * 2)
        assert np.isnan(windows.signal.sel(run="none")).all()
    if number == 6:
        np.testing.assert_allclose(ns["per_trial_demo"].to_dataarray(), [2., 6.])
        np.testing.assert_allclose(ns["equal_trial_demo"].to_dataarray(), 4.)
        np.testing.assert_allclose(ns["pooled_demo"].to_dataarray(), 4.4)
        np.testing.assert_allclose(ns["weighted_demo"].to_dataarray(), 3.)
        sequence = ns["sequence_demo"].as_dataset()
        np.testing.assert_allclose(sequence.signal, [1., 2., 3., 4.])
        np.testing.assert_allclose(sequence.time, [0., 1., 2., 3.])
        assert ns["batch_demo"].as_dataset().sizes == {"recording": 2, "sample": 2}
        assert set(ns["merged_demo"].as_dataset().data_vars) == {"signal", "other"}
        np.testing.assert_allclose(ns["padded_group_demo"].sum().to_dataarray(), 22.)
        np.testing.assert_allclose(ns["padded_group_demo"].count().to_dataarray(), 5.)


def _check_numeric_and_frame_examples(number, ns):
    """ID: DOC_TUTORIAL_TEACHING_003; labeled kernels, representations, and frame meaning."""
    if number == 7:
        np.testing.assert_allclose(ns["assembled_demo"].to_dataarray().transpose("sample", "channel_axis"), [[1., 2.], [3., 4.]])
        np.testing.assert_allclose(ns["xyz_demo"].to_dataarray().transpose("sample", "axis"), [[1., 0., 1.], [3., 0., 1.]])
        np.testing.assert_allclose(ns["least_squares_demo"].to_dataarray(), [4/3, 7/3])
        np.testing.assert_allclose(ns["fitted_rhs"].to_dataarray(), [4/3, 7/3, 11/3])
    if number == 8:
        np.testing.assert_allclose(ns["raw_recipe_pose"].as_matrix().to_dataarray(), ns["ship_pose"].as_matrix().to_dataarray())
        np.testing.assert_allclose(ns["negative_turn"].as_matrix().to_dataarray(), ns["turn"].as_matrix().to_dataarray(), atol=1e-12)
        result = ns["typed_query_demo"].as_dataset()
        assert result.sizes == {"trial": 12, "sample": 4, "quat": 4}
        np.testing.assert_array_equal(result["case"], ["first", "first", "second", "second"])
        np.testing.assert_array_equal(result["when"], ["early", "late", "early", "late"])
    if number == 9:
        expected = np.column_stack((2 * ns["irregular_time"], np.full(5, 2.), np.zeros(5)))
        np.testing.assert_allclose(ns["irregular_velocity"].to_dataarray(), expected, atol=1e-10)
        t = ns["time"][35:46]
        expected_smooth = np.mean(t**2 + 0.02 * np.sin(30*t))
        np.testing.assert_allclose(ns["moving_average_demo"].to_dataarray().sel(axis="x").isel(sample=40), expected_smooth)
    if number == 10:
        assert ns["reverse_total"] == 5. and ns["identity_total"] == 0.
        assert ns["tagged_value"].frames.ids() == ("ship", "camera")
        assert ns["remapped_value"].frames.ids() == ("ship", "camera_renamed")
    if number == 11:
        np.testing.assert_allclose(ns["relative_demo"].to_dataarray(), [1., -1., 0.], atol=1e-12)
        np.testing.assert_allclose(ns["basis_demo"].to_dataarray(), [1., -3., 0.], atol=1e-12)
        assert ns["relative_demo"].frames.ids() == ("sensor", "target")
        assert ns["basis_demo"].frames.ids() == ("lab", "target")
        assert ns["detached_demo"].graph is None
        assert ns["coverage_rejected"]


def _check_execution_and_presentation(number, ns):
    """ID: DOC_TUTORIAL_TEACHING_004; lazy planning and plotting validity."""
    if number == 12:
        assert ns["memory_tasks"] == []
        assert ns["memory_plan"].to_dataarray(copy="shallow").chunks is not None
        np.testing.assert_allclose(ns["computed_memory"].signal, 5.5)
        assert ns["dataset_position"].graph is None
        assert ns["reattached_position"].graph is ns["runtime_graph"]
    if number == 13:
        assert np.nanmax(ns["valid_plot"].data["signal"]) == 2.
        assert np.nanmax(ns["storage_plot"].data["signal"]) == 99.


@pytest.mark.parametrize("cwd", [ROOT, TUTORIAL], ids=["root", "tutorial"])
@pytest.mark.parametrize("number", range(1, 15))
def test_doc_tutorials_001_public_notebooks_execute(number, cwd, monkeypatch):
    """ID: DOC_TUTORIALS_001; actual cells, independent references, and immutable sources."""
    monkeypatch.chdir(cwd)
    namespace = _execute(number)
    _check_tabular(number, namespace)
    _check_construction_and_selection(number, namespace)
    _check_events_and_statistics(number, namespace)
    _check_numeric_and_frame_examples(number, namespace)
    _check_execution_and_presentation(number, namespace)
    if number == 8:
        _check_spatial_components_and_alignment(namespace)
    if number == 11:
        _check_frame_expression(namespace)
    if number == 12:
        _check_lazy_lifecycle(namespace)
    if number == 14:
        _check_landing(namespace)
