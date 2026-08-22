from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tal.io import RosIngestOptions, read_ros_logs
from tal.io import ros_logs as ros_logs_module


def _pose_stamped(sec: int, nsec: int, *, frame_id: str = "map") -> object:
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=sec, nanosec=nsec),
            frame_id=frame_id,
        ),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


def _odometry(sec: int, nsec: int) -> object:
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=sec, nanosec=nsec),
            frame_id="map",
        ),
        child_frame_id="base",
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        ),
    )


def test_io_core_p10b_007_ros_timestamp_source_precedence_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_CORE_P10B_007_ros_timestamp_source_precedence_is_deterministic."""
    path = tmp_path / "one.bag"
    path.write_text("stub", encoding="utf-8")

    def _fake_iter_ros_messages(_path: str, *, owner: str):
        _ = owner
        return [
            ros_logs_module._RosMessage(
                topic="/pose",
                msgtype="geometry_msgs/msg/PoseStamped",
                msg=_pose_stamped(10, 0),
                receive_ns=20_000_000_000,
            ),
            ros_logs_module._RosMessage(
                topic="/pose",
                msgtype="geometry_msgs/msg/PoseStamped",
                msg=_pose_stamped(0, 0),
                receive_ns=30_000_000_000,
            ),
        ]

    monkeypatch.setattr(ros_logs_module, "_iter_ros_messages", _fake_iter_ros_messages)
    ao = read_ros_logs([str(path)], opts=RosIngestOptions(topic="/pose"))
    times = ao.unsafe_data.coords["time"].isel(trial=0).values.tolist()
    assert times[:2] == pytest.approx([10.0, 30.0])


def test_io_core_p10b_003_ros_ingest_timestamp_fallback_and_bad_time_policy_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_CORE_P10B_003_ros_ingest_timestamp_fallback_and_bad_time_policy_are_deterministic."""
    path = tmp_path / "policy.bag"
    path.write_text("stub", encoding="utf-8")

    def _messages(_path: str, *, owner: str):
        _ = owner
        return [
            ros_logs_module._RosMessage("/pose", "geometry_msgs/msg/PoseStamped", _pose_stamped(1, 0), 1_000_000_000),
            ros_logs_module._RosMessage("/pose", "geometry_msgs/msg/PoseStamped", _pose_stamped(0, 0), None),
        ]

    monkeypatch.setattr(ros_logs_module, "_iter_ros_messages", _messages)
    with pytest.raises(ValueError, match="invalid/non-finite"):
        read_ros_logs(
            [str(path)],
            opts=RosIngestOptions(topic="/pose", timestamp_source="header", invalid_time="fail"),
        )
    ao = read_ros_logs(
        [str(path)],
        opts=RosIngestOptions(topic="/pose", timestamp_source="header", invalid_time="drop"),
    )
    times = ao.unsafe_data.coords["time"].isel(trial=0).values.tolist()
    assert times[:1] == [1.0]


def test_io_hard_p10b_002_topic_or_message_type_ambiguity_fails_closed_with_owner_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_002_topic_or_message_type_ambiguity_fails_closed_with_owner_prefix."""
    path = tmp_path / "ambiguous.bag"
    path.write_text("stub", encoding="utf-8")

    def _topic_ambiguous(_path: str, *, owner: str):
        _ = owner
        return [
            ros_logs_module._RosMessage("/a", "geometry_msgs/msg/PoseStamped", _pose_stamped(1, 0), 1_000_000_000),
            ros_logs_module._RosMessage("/b", "geometry_msgs/msg/PoseStamped", _pose_stamped(2, 0), 2_000_000_000),
        ]

    monkeypatch.setattr(ros_logs_module, "_iter_ros_messages", _topic_ambiguous)
    with pytest.raises(ValueError, match="topic ambiguity"):
        read_ros_logs([str(path)], opts=RosIngestOptions())

    def _type_ambiguous(_path: str, *, owner: str):
        _ = owner
        return [
            ros_logs_module._RosMessage("/pose", "geometry_msgs/msg/PoseStamped", _pose_stamped(1, 0), 1_000_000_000),
            ros_logs_module._RosMessage("/pose", "nav_msgs/msg/Odometry", _odometry(2, 0), 2_000_000_000),
        ]

    monkeypatch.setattr(ros_logs_module, "_iter_ros_messages", _type_ambiguous)
    with pytest.raises(ValueError, match="message-type ambiguity"):
        read_ros_logs([str(path)], opts=RosIngestOptions(topic="/pose"))


def test_io_hard_p10b_003_ingest_no_valid_rows_or_messages_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_003_ingest_no_valid_rows_or_messages_fails_closed."""
    path = tmp_path / "invalid.bag"
    path.write_text("stub", encoding="utf-8")

    def _all_invalid(_path: str, *, owner: str):
        _ = owner
        return [
            ros_logs_module._RosMessage(
                topic="/pose",
                msgtype="geometry_msgs/msg/PoseStamped",
                msg=_pose_stamped(0, 0),
                receive_ns=None,
            )
        ]

    monkeypatch.setattr(ros_logs_module, "_iter_ros_messages", _all_invalid)
    with pytest.raises(ValueError, match="no valid ROS messages remained"):
        read_ros_logs(
            [str(path)],
            opts=RosIngestOptions(topic="/pose", timestamp_source="header", invalid_time="drop"),
        )
