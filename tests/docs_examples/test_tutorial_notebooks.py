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
            for name in ("observations", "ship_log", "drone_log", "source", "declared", "ship_pose", "position"):
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
        np.testing.assert_allclose(windows["event_time"].sel(trial=trial), [time[index]])
        query = time[index] + windows["tau"].to_numpy()
        expected_window = np.column_stack([np.interp(query, time, expected[:, axis]) for axis in range(3)])
        np.testing.assert_allclose(windows["position"].sel(trial=trial).isel(event=0).transpose("tau", "axis"), expected_window, atol=1e-10)
    np.testing.assert_array_equal(relative["sequence_size"], lengths)
    np.testing.assert_array_equal(namespace["width_comparison"]["eligible"], eligible)
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


@pytest.mark.parametrize("cwd", [ROOT, TUTORIAL], ids=["root", "tutorial"])
@pytest.mark.parametrize("number", range(1, 15))
def test_doc_tutorials_001_public_notebooks_execute(number, cwd, monkeypatch):
    """ID: DOC_TUTORIALS_001; actual cells, independent references, and immutable sources."""
    monkeypatch.chdir(cwd)
    namespace = _execute(number)
    _check_tabular(number, namespace)
    if number == 8:
        _check_spatial_components_and_alignment(namespace)
    if number == 11:
        _check_frame_expression(namespace)
    if number == 12:
        _check_lazy_lifecycle(namespace)
    if number == 14:
        _check_landing(namespace)
