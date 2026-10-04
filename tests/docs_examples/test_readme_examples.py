"""Execute the README's sequential examples against independent expectations."""

import re
from pathlib import Path

import numpy as np
import xarray as xr

README = Path(__file__).resolve().parents[2] / "README.md"


def _execute_examples():
    blocks = re.findall(r"^```python\n(.*?)^```", README.read_text(), re.MULTILINE | re.DOTALL)
    assert blocks, "The README must contain its executable quick example."
    namespace = {}
    source = None
    for index, code in enumerate(blocks):
        exec(compile(code, f"{README}:block-{index + 1}", "exec"), namespace)  # noqa: S102
        if source is None:
            source = namespace["ds"].copy(deep=True)
    xr.testing.assert_identical(namespace["ds"], source)
    return namespace


def _check_event_windows(namespace, time, expected_speed):
    high = expected_speed > 1.2
    entries = np.flatnonzero(high[1:] & ~high[:-1]) + 1
    events = namespace["events"]
    np.testing.assert_allclose(events["time"].where(events["edge_code"] == 1, drop=True), time[entries][None, :])
    windows = namespace["windows"].as_dataset(copy="shallow")
    np.testing.assert_allclose(windows["event_time"], time[entries][None, :])
    np.testing.assert_allclose(windows["tau"], np.arange(-2, 5) * 0.05, atol=1e-15)
    expected = np.interp(windows["time"].to_numpy(), time, expected_speed)
    np.testing.assert_allclose(namespace["windows"].to_dataarray(copy="shallow"), expected)
    assert windows["valid"].all()


def test_readme_examples_execute_with_current_api(tmp_path, monkeypatch):
    """README examples preserve values, selections, source data, and persistence."""
    monkeypatch.chdir(tmp_path)
    namespace = _execute_examples()
    time = namespace["time"]
    expected_speed = np.sqrt(1 + np.sin(time)**2)
    np.testing.assert_allclose(namespace["speed"].to_dataarray(copy="shallow"), expected_speed[None, :])
    expected_query = np.sqrt(1 + np.sin([0.5, 1.0, 1.5])**2)
    np.testing.assert_allclose(namespace["resampled"].to_dataarray(copy="shallow"), expected_query)
    selected = namespace["first_second"].as_dataset(copy="shallow")
    np.testing.assert_array_equal(selected["valid"], time <= 1.0)
    assert namespace["single_run"].as_dataset(copy="shallow").sizes == {"sample": 101, "axis": 3}
    source = namespace["ds"]["velocity"].to_numpy()
    expected_aligned = np.column_stack([np.interp(time, time[::10], source[::10, axis]) for axis in range(3)])
    aligned = namespace["aligned"].to_dataarray(copy="shallow").transpose("run", "sample", "axis")
    np.testing.assert_allclose(aligned, expected_aligned[None, ...])
    expected_error = np.linalg.norm(expected_aligned - source, axis=-1)
    np.testing.assert_allclose(namespace["tracking_error"].to_dataarray(copy="shallow"), expected_error[None, :])
    actual_velocity = namespace["linear_velocity"].to_dataarray(copy="shallow").transpose("sample", "axis")
    np.testing.assert_allclose(actual_velocity, np.tile([1.0, 2.0, 0.0], (time.size, 1)), atol=1e-12)
    _check_event_windows(namespace, time, expected_speed)
    xr.testing.assert_identical(namespace["restored"], namespace["ao"].as_dataset(copy="shallow"))
