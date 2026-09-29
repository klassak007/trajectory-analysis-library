"""Behavioral coverage for the published AnalysisObject diagram example."""

import re
from pathlib import Path

import numpy as np
import xarray as xr

from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.spatial import Pose, Position, Rotation

PAGE = Path(__file__).resolve().parents[2] / "docs/user-guide/illustrated_example.md"


def _check_source_and_components(namespace):
    times = np.array([
        [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
        [0.02, 0.17, 0.31, 0.48, np.nan, np.nan],
        [0.00, 0.08, 0.23, 0.39, 0.57, np.nan],
    ])
    zero = np.where(np.isfinite(times), 0.0, np.nan)
    expected_position = np.stack([times, 2 * times, zero], axis=-1)
    expected_rotation = np.stack([zero, zero, zero, zero + 1], axis=-1)
    source = namespace["source"]
    np.testing.assert_allclose(source["time"], times)
    np.testing.assert_array_equal(source["sequence_size"], [6, 4, 5])
    assert source.attrs == {}
    assert set(source.data_vars) == {f"position.{c}" for c in "xyz"} | {f"rotation.{c}" for c in "xyzw"}
    for index, label in enumerate("xyz"):
        np.testing.assert_allclose(source[f"position.{label}"], expected_position[..., index])
    for index, label in enumerate("xyzw"):
        np.testing.assert_allclose(source[f"rotation.{label}"], expected_rotation[..., index])
    poses = namespace["poses"].as_dataset(copy="shallow")
    assert poses.sizes == {"trial": 3, "sample": 6, "axis": 3, "quat": 4}
    assert set(poses.data_vars) == {"position", "rotation"}
    assert read_roles(poses) == (True, "sample", ("trial",), ("axis", "quat"))
    assert read_param_coord_name(poses) == "time"
    assert read_sequence_size_coord_name(poses) == "sequence_size"
    np.testing.assert_allclose(poses["time"], times)
    np.testing.assert_array_equal(poses["sequence_size"], [6, 4, 5])
    np.testing.assert_array_equal(poses["trial"], ["A", "B", "C"])
    np.testing.assert_array_equal(poses["axis"], list("xyz"))
    np.testing.assert_array_equal(poses["quat"], list("xyzw"))
    np.testing.assert_allclose(poses["position"].transpose("trial", "sample", "axis"), expected_position)
    np.testing.assert_allclose(poses["rotation"].transpose("trial", "sample", "quat"), expected_rotation)
    assert isinstance(namespace["poses"], Pose)
    assert isinstance(namespace["positions"], Position)
    assert isinstance(namespace["rotations"], Rotation)
    np.testing.assert_allclose(namespace["positions"].to_dataarray(copy="shallow"), expected_position)
    np.testing.assert_allclose(namespace["rotations"].to_dataarray(copy="shallow"), expected_rotation)


def _check_operations(namespace):
    source = namespace["poses"].as_dataset(copy="shallow")
    trial_b = namespace["trial_b"].as_dataset(copy="shallow")
    assert read_roles(trial_b) == (True, "sample", (), ("axis", "quat"))
    xr.testing.assert_identical(trial_b["position"], source["position"].sel(trial="B"))
    first_two = namespace["first_two"].as_dataset(copy="shallow")
    np.testing.assert_array_equal(first_two["sequence_size"], [2, 2, 2])
    xr.testing.assert_equal(first_two["time"].reset_coords(drop=True), source["time"].isel(sample=slice(0, 2)).reset_coords(drop=True))
    slot = namespace["slot_two"].to_dataarray(copy="shallow")
    np.testing.assert_allclose(slot["time"], [0.20, 0.31, 0.23])
    np.testing.assert_allclose(slot.sel(axis="x"), [0.20, 0.31, 0.23])
    result = namespace["at_time"].as_dataset(copy="shallow")
    assert result.sizes == {"trial": 3, "sample": 1, "axis": 3, "quat": 4}
    np.testing.assert_allclose(result["position"].transpose("trial", "sample", "axis"), np.tile([0.2, 0.4, 0.0], (3, 1, 1)))
    np.testing.assert_allclose(result["rotation"].transpose("trial", "sample", "quat"), np.tile([0.0, 0.0, 0.0, 1.0], (3, 1, 1)))
    np.testing.assert_allclose(result["time"], 0.2)
    mean = namespace["mean_position"].as_dataset(copy="shallow")
    assert read_roles(mean) == (True, None, ("trial",), ("axis",))
    np.testing.assert_allclose(mean["position"], [[.25, .5, 0.], [.245, .49, 0.], [.254, .508, 0.]])


def example_guide_illustrated_workflow():
    """ID: UG-ILLUSTRATED-WORKFLOW; execute actual page code and check its claims."""
    namespace = {}
    blocks = re.findall(r"^```python\n(.*?)^```", PAGE.read_text(), re.MULTILINE | re.DOTALL)
    assert blocks
    for index, code in enumerate(blocks):
        exec(compile(code, f"{PAGE}:block-{index}", "exec"), namespace)  # noqa: S102
    _check_source_and_components(namespace)
    _check_operations(namespace)
