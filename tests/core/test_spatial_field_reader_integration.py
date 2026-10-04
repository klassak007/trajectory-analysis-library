from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisLayoutSpec
from tal.core.component_ops import read_components
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.io import CsvIngestOptions, RosIngestOptions, read_csv_logs, read_ros_logs
from tal.io import ros_logs as ros_logs_module
from tal.io import ros_payload as ros_payload_module
from tal.io import ros_reader as ros_reader_module
from tal.spatial import Pose, Position, Rotation
from tal.utils.frame_schema import get_frames

_CSV_FIELDS = (
    "camera.position.x",
    "camera.position.y",
    "camera.position.z",
    "camera.rotation.x",
    "camera.rotation.y",
    "camera.rotation.z",
    "camera.rotation.w",
)


def _write_pose_csv(path: Path, rows: tuple[tuple[float, ...], ...]) -> None:
    header = ",".join(("time", *_CSV_FIELDS))
    records = [header, *(",".join(str(value) for value in row) for row in rows)]
    path.write_text("\n".join(records) + "\n", encoding="utf-8")


def _csv_reader_result(tmp_path: Path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_pose_csv(
        left,
        (
            (0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0),
        ),
    )
    _write_pose_csv(
        right,
        ((0.5, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0),),
    )
    source = read_csv_logs(
        {"left": str(left), "right": str(right)},
        opts=CsvIngestOptions(
            time_col="time",
            value_columns=_CSV_FIELDS,
            batch_dim="run",
            sequence_dim="tick",
            sequence_size_coord="count",
            param_coord="stamp",
        ),
    )
    return source, (left, right)


def _assert_reader_layout(value: object) -> None:
    ds = value.as_dataset(copy="none")
    declared, sequence_dim, batch_dims, core_dims = read_roles(ds)
    assert declared
    assert sequence_dim == "tick"
    assert batch_dims == ("run",)
    assert core_dims == ("axis", "quat")
    assert read_param_coord_name(ds) == "stamp"
    assert read_sequence_size_coord_name(ds) == "count"
    assert tuple(ds.coords["stamp"].dims) == ("run", "tick")
    assert tuple(ds.coords["count"].dims) == ("run",)
    assert ds.coords["run"].values.tolist() == ["left", "right"]
    assert ds.coords["count"].values.tolist() == [2, 1]
    assert set(read_components(value)) == {"position", "rotation"}


def test_spatial_core_field_csv_reader_001(tmp_path: Path) -> None:
    """ID: SPATIAL_CORE_FIELD_CSV_READER_001."""
    source, paths = _csv_reader_result(tmp_path)
    before = source.as_dataset(copy="none").copy(deep=True)
    source_run_index = source.as_dataset(copy="none").xindexes["run"]
    for path in paths:
        path.unlink()

    position = Position.from_fields(source, "camera.position.{x,y,z}")
    rotation = Rotation.from_fields(source, "camera.rotation.{x,y,z,w}")
    one_shot = Pose.from_fields(
        source,
        position="camera.position.{x,y,z}",
        rotation="camera.rotation.{x,y,z,w}",
    )
    recipe = Pose.fields(
        position="position.{x,y,z}",
        rotation="rotation.{x,y,z,w}",
    )
    reused = recipe.build(source, prefix="camera.")

    expected_position = np.asarray(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [np.nan, np.nan, np.nan]],
        ]
    )
    expected_rotation = np.asarray(
        [
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0, 1.0], [np.nan, np.nan, np.nan, np.nan]],
        ]
    )
    np.testing.assert_allclose(
        position.as_dataset(copy="none")["position"],
        expected_position,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        rotation.as_dataset(copy="none")["rotation"],
        expected_rotation,
        equal_nan=True,
    )
    for pose in (one_shot, reused):
        _assert_reader_layout(pose)
        result_run_index = pose.as_dataset(copy="none").xindexes["run"]
        assert type(result_run_index) is type(source_run_index)
        assert result_run_index.equals(source_run_index)
        pos, rot = pose.decompose()
        np.testing.assert_allclose(
            pos.as_dataset(copy="none")["position"],
            expected_position,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            rot.as_dataset(copy="none")["rotation"],
            expected_rotation,
            equal_nan=True,
        )
        ordinary = {
            name: value
            for name, value in pose.as_dataset(copy="none").attrs.items()
            if name != "tal"
        }
        source_ordinary = {
            name: value
            for name, value in before.attrs.items()
            if name != "tal"
        }
        assert ordinary == source_ordinary
    xr.testing.assert_identical(one_shot.as_dataset(), reused.as_dataset())
    xr.testing.assert_identical(source.as_dataset(), before)


def _pose_stamped(index: int) -> object:
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=index, nanosec=0),
            frame_id="map",
        ),
        pose=SimpleNamespace(
            position=SimpleNamespace(
                x=float(index),
                y=float(index + 1),
                z=float(index + 2),
            ),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


def _ros_message(index: int) -> ros_reader_module.RosMessage:
    msgtype = "geometry_msgs/msg/PoseStamped"
    return ros_reader_module.RosMessage(
        topic="/pose",
        msgtype=msgtype,
        family=ros_payload_module.require_message_family(
            msgtype,
            owner="test ROS reader fixture",
        ),
        msg=_pose_stamped(index),
        receive_ns=index * 1_000_000_000,
    )


def _ros_reader_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
):
    path = tmp_path / "robot.bag"
    path.write_text("fixture", encoding="utf-8")

    def messages(_path: str, *, opts: RosIngestOptions, owner: str):
        _ = opts, owner
        events.append("opened")
        try:
            yield _ros_message(1)
            yield _ros_message(2)
        finally:
            events.append("closed")

    monkeypatch.setattr(ros_logs_module, "_iter_ros_messages", messages)
    return read_ros_logs(str(path), opts=RosIngestOptions(topic="/pose"))


def test_spatial_core_field_ros_reader_001(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_CORE_FIELD_ROS_READER_001."""
    events: list[str] = []
    source = _ros_reader_result(tmp_path, monkeypatch, events)
    before = source.as_dataset(copy="none").copy(deep=True)
    source_trial_index = source.as_dataset(copy="none").xindexes["trial"]
    assert events == ["opened", "closed"]

    one_shot = Pose.from_fields(
        source,
        position="translation_{x,y,z}",
        rotation="quaternion_{x,y,z,w}",
    )
    recipe = Pose.fields(
        position="translation_{x,y,z}",
        rotation="quaternion_{x,y,z,w}",
    )
    reused = recipe.build(source)

    expected_position = np.asarray([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]])
    expected_rotation = np.asarray(
        [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    for pose in (one_shot, reused):
        ds = pose.as_dataset(copy="none")
        declared, sequence_dim, batch_dims, core_dims = read_roles(ds)
        assert (declared, sequence_dim, batch_dims, core_dims) == (
            True,
            "sample",
            ("trial",),
            ("axis", "quat"),
        )
        assert read_param_coord_name(ds) == "time"
        assert read_sequence_size_coord_name(ds) == "sequence_size"
        assert ds.coords["time"].dtype == np.dtype("int64")
        assert ds.coords["time"].values.tolist() == [
            [1_000_000_000, 2_000_000_000]
        ]
        assert ds.attrs["ros_topic"] == "/pose"
        assert ds.attrs["ros_msgtype"] == "geometry_msgs/msg/PoseStamped"
        assert ds.attrs["ros_parent_frame"] == "map"
        assert get_frames(ds) == (None, None)
        result_trial_index = ds.xindexes["trial"]
        assert type(result_trial_index) is type(source_trial_index)
        assert result_trial_index.equals(source_trial_index)
        position, rotation = pose.decompose()
        np.testing.assert_array_equal(
            position.as_dataset(copy="none")["position"],
            expected_position,
        )
        np.testing.assert_array_equal(
            rotation.as_dataset(copy="none")["rotation"],
            expected_rotation,
        )
    assert events == ["opened", "closed"]
    xr.testing.assert_identical(one_shot.as_dataset(), reused.as_dataset())
    xr.testing.assert_identical(source.as_dataset(), before)


def test_spatial_hard_field_reader_boundary_001(tmp_path: Path) -> None:
    """ID: SPATIAL_HARD_FIELD_READER_BOUNDARY_001."""
    path = tmp_path / "selected.csv"
    _write_pose_csv(
        path,
        ((0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),),
    )
    retained = tuple(name for name in _CSV_FIELDS if name != "camera.position.z")
    source = read_csv_logs(
        str(path),
        opts=CsvIngestOptions(time_col="time", value_columns=retained),
    )
    before = source.as_dataset(copy="none").copy(deep=True)
    path.unlink()

    with pytest.raises(
        ValueError,
        match=r"Pose\.from_fields: position fields .*camera\.position\.z.*not source data variables",
    ):
        Pose.from_fields(
            source,
            position="camera.position.{x,y,z}",
            rotation="camera.rotation.{x,y,z,w}",
        )
    with pytest.raises(ValueError, match="source_layout cannot accompany"):
        Rotation.from_fields(
            source,
            "camera.rotation.{x,y,z,w}",
            source_layout=AnalysisLayoutSpec(
                sequence_dim="sample",
                batch_dims=("trial",),
                param_coord="time",
                sequence_size_coord="sequence_size",
            ),
        )
    xr.testing.assert_identical(source.as_dataset(), before)


def test_spatial_ownership_field_reader_001(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_OWNERSHIP_FIELD_READER_001."""
    events: list[str] = []
    source = _ros_reader_result(tmp_path, monkeypatch, events)
    recipe = Pose.fields(
        position="translation_{x,y,z}",
        rotation="quaternion_{x,y,z,w}",
    )
    first = recipe.build(source)
    second = recipe.build(source)
    assert events == ["opened", "closed"]
    xr.testing.assert_identical(first.as_dataset(), second.as_dataset())
    first.close()
    second.close()
    source.close()
    assert events == ["opened", "closed"]
