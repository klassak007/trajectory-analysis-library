"""Execute the alternate capstone and compare it with independent log math."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation, Slerp

from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "examples/tutorial/15_capstone_robot_catch.ipynb"
DATA = NOTEBOOK.parent / "data/robot_catch"


def _execute_notebook():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    plt = pytest.importorskip("matplotlib.pyplot")
    pytest.importorskip("IPython")
    namespace, snapshots = {}, {}
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            # Execute only the repository-owned notebook, without duplicating its workflow.
            exec(compile("".join(cell["source"]), f"{NOTEBOOK}:cell-{index}", "exec"), namespace)  # noqa: S102
            for name in ("ball_logs", "robot_logs"):
                if name in namespace and name not in snapshots:
                    snapshots[name] = namespace[name].as_dataset(copy="shallow").copy(deep=True)
        return namespace, snapshots
    finally:
        plt.close("all")


def _reference_trial(trial):
    camera = pd.read_csv(DATA / "camera" / f"{trial}.csv")
    robot = pd.read_csv(DATA / "robot" / f"{trial}.csv")
    time = camera["time"].to_numpy()
    xyz = camera[[f"ball.position.{axis}" for axis in "xyz"]].to_numpy()
    translation = np.column_stack([
        np.interp(time, robot["time"], robot[f"gripper.position.{axis}"])
        for axis in "xyz"
    ])
    quaternion = robot[[f"gripper.rotation.{axis}" for axis in "xyzw"]].to_numpy()
    orientation = Slerp(robot["time"], Rotation.from_quat(quaternion))(time)
    relative = orientation.inv().apply(xyz - translation)
    return time, relative, np.linalg.norm(relative, axis=-1)


def _check_paths_and_windows(namespace):
    relative = namespace["ball_in_gripper"].as_dataset(copy="shallow")
    distance = namespace["distance"].to_dataarray(copy="shallow")
    windows = namespace["approach"].as_dataset(copy="shallow")
    expected_minima = []
    for trial in namespace["catalog"].index:
        time, expected_xyz, expected_distance = _reference_trial(trial)
        actual_xyz = relative["position"].sel(trial=trial)
        np.testing.assert_allclose(actual_xyz[:len(time)], expected_xyz, atol=1e-10)
        assert np.isnan(actual_xyz[len(time):]).all()
        np.testing.assert_allclose(distance.sel(trial=trial)[:len(time)], expected_distance, atol=1e-10)
        crossing = np.flatnonzero(expected_distance < 0.35)[0]
        event_time = time[crossing]
        np.testing.assert_allclose(windows["event_time"].sel(trial=trial), [event_time])
        tau = windows["tau"].to_numpy()
        np.testing.assert_allclose(tau, np.linspace(-0.12, 0.32, 45), atol=1e-14)
        expected_window = np.interp(event_time + tau, time, expected_distance)
        np.testing.assert_allclose(
            windows["distance"].sel(trial=trial).isel(event=0), expected_window, atol=1e-10,
        )
        expected_minima.append(expected_distance.min())
    return np.asarray(expected_minima)


def _check_topology(namespace, snapshots):
    relative = namespace["ball_in_gripper"]
    ds = relative.as_dataset(copy="shallow")
    assert relative.frames.ids() == ("gripper", "ball")
    assert relative.graph is namespace["graph"]
    assert read_roles(ds) == (True, "sample", ("trial",), ("axis",))
    assert read_param_coord_name(ds) == "time"
    assert read_sequence_size_coord_name(ds) == "sequence_size"
    assert set(ds.data_vars) == {"position"}
    assert list(ds["axis"].values) == ["x", "y", "z"]
    for name in ("trial", "time", "sequence_size"):
        xr.testing.assert_identical(ds[name].variable, snapshots["ball_logs"][name].variable)
    window = namespace["approach"].as_dataset(copy="shallow")
    assert read_roles(window) == (True, "tau", ("trial", "event"), ())
    assert read_param_coord_name(window) == "tau"
    assert read_sequence_size_coord_name(window) == "tau_len"
    assert window.sizes == {"trial": 12, "event": 1, "tau": 45}
    assert bool(window["valid"].all())
    np.testing.assert_array_equal(window["tau_len"], np.full((12, 1), 45))
    for name, snapshot in snapshots.items():
        xr.testing.assert_identical(namespace[name].as_dataset(copy="shallow"), snapshot)


@pytest.mark.parametrize("working_directory", [ROOT, NOTEBOOK.parent], ids=["repository", "notebook"])
def test_robot_catch_notebook_executes_and_matches_independent_reference(monkeypatch, working_directory):
    """ID: DOC_ROBOT_CATCH_WORKFLOW_001_public_notebook_matches_log_reference."""
    monkeypatch.chdir(working_directory)
    namespace, snapshots = _execute_notebook()
    expected_minima = _check_paths_and_windows(namespace)
    _check_topology(namespace, snapshots)
    np.testing.assert_allclose(namespace["closest"].to_dataarray(copy="shallow"), expected_minima, atol=1e-10)
    controllers = namespace["catalog"]["controller"].to_numpy()
    for label, count in (("predictive", 5), ("reactive", 1)):
        selected = expected_minima[controllers == label]
        assert (selected < 0.08).sum() == count
        row = namespace["scorecard"].loc[label]
        np.testing.assert_allclose(row["Mean closest distance (cm)"], 100 * selected.mean())
        np.testing.assert_allclose(row["Capture-eligible (%)"], 100 * count / 6)
    for throw in range(1, 7):
        predictive = pd.read_csv(DATA / "camera" / f"predictive_{throw:02d}.csv")
        reactive = pd.read_csv(DATA / "camera" / f"reactive_{throw:02d}.csv")
        pd.testing.assert_frame_equal(predictive, reactive)
