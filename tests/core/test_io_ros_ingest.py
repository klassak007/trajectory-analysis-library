from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from tal.io import AdapterMetadataPromotionOptions, RosIngestOptions, read_ros_logs
from tal.io import adapter_temp as adapter_temp_module
from tal.io import ros_logs as ros_logs_module
from tal.io import ros_payload as ros_payload_module
from tal.io import ros_reader as ros_reader_module
from tests._io_helpers import cleanup_failing_temporary_directory


def _pose_stamped(sec: int, nsec: int, *, frame_id: str | None = "map") -> object:
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


def _ros_message(
    topic: str,
    msgtype: str,
    msg: object,
    receive_ns: object | None,
) -> ros_reader_module.RosMessage:
    family = ros_payload_module.require_message_family(msgtype, owner="test ROS message")
    return ros_reader_module.RosMessage(topic, msgtype, family, msg, receive_ns)


class _TwoMessagePoseReader:
    def __init__(self, _paths: object, *, connection: object) -> None:
        self._connection = connection

    @property
    def connections(self) -> tuple[object, ...]:
        return (self._connection,)

    def __enter__(self) -> "_TwoMessagePoseReader":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def messages(self, *, connections: tuple[object, ...]):
        assert connections == self.connections
        for index in range(2):
            yield (
                self._connection,
                (index + 1) * 1_000_000_000,
                _pose_stamped(index + 1, 0),
            )

    def deserialize(self, rawdata: object, _msgtype: str) -> object:
        return rawdata


class _MalformedPoseReader:
    def __init__(
        self,
        _paths: object,
        *,
        connection: object,
        events: list[str],
        close_error: BaseException | None,
    ) -> None:
        self.connections = (connection,)
        self._events = events
        self._close_error = close_error

    def __enter__(self) -> "_MalformedPoseReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self._events.append("closed")
        if self._close_error is not None:
            raise self._close_error

    def messages(self, *, connections: tuple[object, ...]):
        assert connections == self.connections
        yield self.connections[0], 1_000_000_000, object()

    def deserialize(self, _rawdata: object, _msgtype: str) -> object:
        return SimpleNamespace(pose=SimpleNamespace())


def test_io_core_p10b_007_ros_timestamp_source_precedence_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_CORE_P10B_007_ros_timestamp_source_precedence_is_deterministic."""
    path = tmp_path / "one.bag"
    path.write_text("stub", encoding="utf-8")

    def _fake_iter_ros_messages(_path: str, *, opts: RosIngestOptions, owner: str):
        _ = opts, owner
        return [
            _ros_message(
                topic="/pose",
                msgtype="geometry_msgs/msg/PoseStamped",
                msg=_pose_stamped(10, 0),
                receive_ns=20_000_000_000,
            ),
            _ros_message(
                topic="/pose",
                msgtype="geometry_msgs/msg/PoseStamped",
                msg=_pose_stamped(0, 0),
                receive_ns=30_000_000_000,
            ),
        ]

    monkeypatch.setattr(ros_logs_module, "_iter_ros_messages", _fake_iter_ros_messages)
    ao = read_ros_logs([str(path)], opts=RosIngestOptions(topic="/pose"))
    times = ao.unsafe_data.coords["time"].isel(trial=0).values.tolist()
    assert times[:2] == [10_000_000_000, 30_000_000_000]


def test_io_core_p10b_023_ros_ingest_accepts_rosbag2_directory_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_CORE_P10B_023_ros_ingest_accepts_rosbag2_directory_paths."""
    bag_dir = tmp_path / "rosbag2_run"
    bag_dir.mkdir()
    observed_paths: list[str] = []

    def _messages(path: str, *, opts: RosIngestOptions, owner: str):
        _ = opts, owner
        observed_paths.append(path)
        return [
            _ros_message(
                "/pose",
                "geometry_msgs/msg/PoseStamped",
                _pose_stamped(1, 0),
                1_000_000_000,
            )
        ]

    monkeypatch.setattr(ros_logs_module, "_iter_ros_messages", _messages)
    ao = read_ros_logs(str(bag_dir), opts=RosIngestOptions(topic="/pose"))

    assert observed_paths == [str(bag_dir.resolve())]
    assert ao.unsafe_data.sizes["sample"] == 1


def test_io_hard_p10b_039_ros_backend_lifecycle_errors_retain_public_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_039_ros_backend_lifecycle_errors_retain_public_owner."""
    path = tmp_path / "backend_failure.bag"
    path.write_text("stub", encoding="utf-8")

    class OpenFailureReader:
        def __init__(self, _paths: object) -> None:
            raise FileNotFoundError("backend open exploded")

    monkeypatch.setattr(
        ros_logs_module,
        "_load_any_reader_class",
        lambda *, owner: OpenFailureReader,
    )
    with pytest.raises(
        ValueError,
        match="tal.io.read_ros_logs: failed opening ROS recording",
    ) as open_error:
        read_ros_logs(str(path))
    assert isinstance(open_error.value.__cause__, FileNotFoundError)

    connection = SimpleNamespace(
        topic="/pose",
        msgtype="geometry_msgs/msg/PoseStamped",
        msgcount=1,
    )

    class IterationFailureReader:
        connections = (connection,)

        def __init__(self, _paths: object) -> None:
            pass

        def __enter__(self) -> "IterationFailureReader":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def messages(self, *, connections: tuple[object, ...]):
            _ = connections
            raise OSError("backend iteration exploded")
            yield

    monkeypatch.setattr(
        ros_logs_module,
        "_load_any_reader_class",
        lambda *, owner: IterationFailureReader,
    )
    with pytest.raises(
        ValueError,
        match="tal.io.read_ros_logs: failed reading ROS messages",
    ) as iteration_error:
        read_ros_logs(str(path), opts=RosIngestOptions(topic="/pose"))
    assert isinstance(iteration_error.value.__cause__, OSError)


def test_io_hard_p10b_048_ros_primary_failure_survives_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_048_ros_primary_failure_survives_close_failure."""
    path = tmp_path / "primary_and_close_failure.bag"
    path.write_text("stub", encoding="utf-8")
    connection = SimpleNamespace(
        topic="/pose",
        msgtype="geometry_msgs/msg/PoseStamped",
        msgcount=1,
    )

    class PrimaryAndCloseFailureReader:
        connections = (connection,)

        def __init__(self, _paths: object) -> None:
            pass

        def __enter__(self) -> "PrimaryAndCloseFailureReader":
            return self

        def __exit__(self, *_args: object) -> None:
            raise OSError("close exploded")

        def messages(self, *, connections: tuple[object, ...]):
            _ = connections
            raise RuntimeError("iteration exploded")
            yield

    monkeypatch.setattr(
        ros_logs_module,
        "_load_any_reader_class",
        lambda *, owner: PrimaryAndCloseFailureReader,
    )

    with pytest.raises(
        ValueError,
        match="tal.io.read_ros_logs: failed reading ROS messages",
    ) as error:
        read_ros_logs(str(path), opts=RosIngestOptions(topic="/pose"))

    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "iteration exploded"


def test_io_hard_p10b_064_ros_cleanup_interrupt_preserves_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_064_ros_cleanup_interrupt_preserves_primary."""
    path = tmp_path / "primary_and_cleanup_interrupt.bag"
    path.write_text("stub", encoding="utf-8")
    connection = SimpleNamespace(
        topic="/pose",
        msgtype="geometry_msgs/msg/PoseStamped",
        msgcount=1,
    )

    class PrimaryAndInterruptingCloseReader:
        connections = (connection,)

        def __init__(self, _paths: object) -> None:
            pass

        def __enter__(self) -> "PrimaryAndInterruptingCloseReader":
            return self

        def __exit__(self, *_args: object) -> None:
            raise KeyboardInterrupt("close interrupted")

        def messages(self, *, connections: tuple[object, ...]):
            _ = connections
            raise RuntimeError("iteration exploded")
            yield

    monkeypatch.setattr(
        ros_logs_module,
        "_load_any_reader_class",
        lambda *, owner: PrimaryAndInterruptingCloseReader,
    )

    with pytest.raises(
        ValueError,
        match="tal.io.read_ros_logs: failed reading ROS messages",
    ) as error:
        read_ros_logs(str(path), opts=RosIngestOptions(topic="/pose"))

    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "iteration exploded"


def test_io_hard_p10b_108_ros_close_only_failure_retains_public_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_108_ros_close_only_failure_retains_public_owner."""
    path = tmp_path / "close_failure.bag"
    path.write_text("stub", encoding="utf-8")
    connection = SimpleNamespace(
        topic="/pose",
        msgtype="geometry_msgs/msg/PoseStamped",
        msgcount=1,
    )

    class CloseFailureReader:
        connections = (connection,)

        def __init__(self, _paths: object) -> None:
            pass

        def __enter__(self) -> "CloseFailureReader":
            return self

        def __exit__(self, *_args: object) -> None:
            raise OSError("close exploded")

        def messages(self, *, connections: tuple[object, ...]):
            _ = connections
            yield connection, 1_000_000_000, object()

        def deserialize(self, _rawdata: object, _msgtype: str) -> object:
            return _pose_stamped(1, 0)

    monkeypatch.setattr(
        ros_logs_module,
        "_load_any_reader_class",
        lambda *, owner: CloseFailureReader,
    )

    with pytest.raises(
        ValueError,
        match="tal.io.read_ros_logs: failed closing ROS recording",
    ) as error:
        read_ros_logs(str(path), opts=RosIngestOptions(topic="/pose"))

    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.parametrize(
    "close_error",
    [None, OSError("close exploded"), KeyboardInterrupt("close interrupted")],
    ids=("clean-close", "close-error", "close-interrupt"),
)
def test_io_hard_p10b_112_ros_consumer_failure_closes_reader_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    close_error: BaseException | None,
) -> None:
    """ID: IO_HARD_P10B_112_ros_consumer_failure_closes_reader_immediately."""
    path = tmp_path / "consumer_failure.bag"
    path.write_text("stub", encoding="utf-8")
    connection = SimpleNamespace(
        topic="/pose",
        msgtype="geometry_msgs/msg/PoseStamped",
        msgcount=1,
    )
    events: list[str] = []

    monkeypatch.setattr(
        ros_logs_module,
        "_load_any_reader_class",
        lambda *, owner: lambda paths: _MalformedPoseReader(
            paths,
            connection=connection,
            events=events,
            close_error=close_error,
        ),
    )

    with pytest.raises(
        ValueError,
        match="tal.io.read_ros_logs: malformed ROS message payload",
    ) as error:
        read_ros_logs(str(path), opts=RosIngestOptions(topic="/pose"))

    assert isinstance(error.value.__cause__, AttributeError)
    assert events == ["closed"]


@pytest.mark.parametrize("source", ["header", "receive"])
def test_io_core_p10b_022_ros_timestamps_preserve_exact_int64_nanoseconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """ID: IO_CORE_P10B_022_ros_timestamps_preserve_exact_int64_nanoseconds."""
    path = tmp_path / f"exact_{source}.bag"
    path.write_text("stub", encoding="utf-8")
    expected = [1_700_000_000_000_000_001, 1_700_000_000_000_000_002]
    messages = [
        _ros_message(
            "/pose",
            "geometry_msgs/msg/PoseStamped",
            _pose_stamped(1_700_000_000, offset + 1)
            if source == "header"
            else _pose_stamped("ignored", 0),  # type: ignore[arg-type]
            object() if source == "header" else expected[offset],
        )
        for offset in range(2)
    ]
    monkeypatch.setattr(
        ros_logs_module,
        "_iter_ros_messages",
        lambda *_args, **_kwargs: messages,
    )

    ao = read_ros_logs(
        str(path),
        opts=RosIngestOptions(
            topic="/pose",
            timestamp_source=source,  # type: ignore[arg-type]
            monotonic_order="strict",
        ),
    )

    time = ao.unsafe_data.coords["time"]
    assert time.dtype == np.dtype("int64")
    assert time.isel(trial=0).values.tolist() == expected
    assert time.attrs == {"units": "ns", "epoch": "unix"}


def test_io_core_p10b_003_ros_ingest_timestamp_fallback_and_bad_time_policy_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_CORE_P10B_003_ros_ingest_timestamp_fallback_and_bad_time_policy_are_deterministic."""
    path = tmp_path / "policy.bag"
    path.write_text("stub", encoding="utf-8")

    def _messages(_path: str, *, opts: RosIngestOptions, owner: str):
        _ = opts, owner
        return [
            _ros_message("/pose", "geometry_msgs/msg/PoseStamped", _pose_stamped(1, 0), 1_000_000_000),
            _ros_message("/pose", "geometry_msgs/msg/PoseStamped", _pose_stamped(0, 0), None),
        ]

    monkeypatch.setattr(ros_logs_module, "_iter_ros_messages", _messages)
    with pytest.raises(ValueError, match="missing ROS timestamp"):
        read_ros_logs(
            [str(path)],
            opts=RosIngestOptions(topic="/pose", timestamp_source="header", invalid_time="fail"),
        )
    ao = read_ros_logs(
        [str(path)],
        opts=RosIngestOptions(topic="/pose", timestamp_source="header", invalid_time="drop"),
    )
    times = ao.unsafe_data.coords["time"].isel(trial=0).values.tolist()
    assert times[:1] == [1_000_000_000]


def test_io_hard_p10b_032_missing_timestamp_drop_precedes_pose_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_032_missing_timestamp_drop_precedes_pose_extraction."""
    path = tmp_path / "drop_before_pose.bag"
    path.write_text("stub", encoding="utf-8")
    messages = [
        _ros_message(
            "/pose",
            "geometry_msgs/msg/PoseStamped",
            _pose_stamped(1, 0),
            1_000_000_000,
        ),
        _ros_message(
            "/pose",
            "geometry_msgs/msg/PoseStamped",
            SimpleNamespace(pose=SimpleNamespace()),
            None,
        ),
    ]
    monkeypatch.setattr(
        ros_logs_module,
        "_iter_ros_messages",
        lambda *_args, **_kwargs: messages,
    )

    ao = read_ros_logs(
        str(path),
        opts=RosIngestOptions(
            topic="/pose",
            timestamp_source="receive",
            invalid_time="drop",
        ),
    )

    assert ao.unsafe_data.coords["time"].isel(trial=0).values.tolist() == [1_000_000_000]
    assert ao.unsafe_data["translation_x"].isel(trial=0).values.tolist() == [1.0]


def test_io_hard_p10b_002_topic_or_message_type_ambiguity_fails_closed_with_owner_prefix(
) -> None:
    """ID: IO_HARD_P10B_002_topic_or_message_type_ambiguity_fails_closed_with_owner_prefix."""
    topic_ambiguous = (
        SimpleNamespace(topic="/a", msgtype="geometry_msgs/msg/PoseStamped"),
        SimpleNamespace(topic="/b", msgtype="geometry_msgs/msg/PoseStamped"),
    )
    with pytest.raises(ValueError, match="tal.io.read_ros_logs: topic ambiguity"):
        ros_reader_module._select_ros_connections(
            topic_ambiguous,
            opts=RosIngestOptions(),
            owner="tal.io.read_ros_logs",
            path="ambiguous.bag",
        )

    type_ambiguous = (
        SimpleNamespace(topic="/pose", msgtype="geometry_msgs/msg/PoseStamped"),
        SimpleNamespace(topic="/pose", msgtype="nav_msgs/msg/Odometry"),
    )
    with pytest.raises(ValueError, match="tal.io.read_ros_logs: message-type ambiguity"):
        ros_reader_module._select_ros_connections(
            type_ambiguous,
            opts=RosIngestOptions(topic="/pose"),
            owner="tal.io.read_ros_logs",
            path="ambiguous.bag",
        )

    for msgtype in ("sensor_msgs/msg/Image", "custom_msgs/msg/NotPoseStamped"):
        unsupported = (SimpleNamespace(topic="/image", msgtype=msgtype),)
        with pytest.raises(ValueError, match="tal.io.read_ros_logs: unsupported ROS message type"):
            ros_reader_module._select_ros_connections(
                unsupported,
                opts=RosIngestOptions(),
                owner="tal.io.read_ros_logs",
                path="unsupported.bag",
            )

    selected = SimpleNamespace(
        topic="/pose",
        msgtype="geometry_msgs/msg/PoseStamped",
        msgcount=1,
    )
    empty = SimpleNamespace(topic="/camera", msgtype="sensor_msgs/msg/Image", msgcount=0)
    selected_contexts = ros_reader_module._select_ros_connections(
        (selected, empty),
        opts=RosIngestOptions(),
        owner="tal.io.read_ros_logs",
        path="empty_connection.bag",
    )
    assert tuple(context.connection for context in selected_contexts) == (selected,)


def test_io_hard_p10b_003_ingest_no_valid_rows_or_messages_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_003_ingest_no_valid_rows_or_messages_fails_closed."""
    path = tmp_path / "invalid.bag"
    path.write_text("stub", encoding="utf-8")

    def _all_invalid(_path: str, *, opts: RosIngestOptions, owner: str):
        _ = opts, owner
        return [
            _ros_message(
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


def test_io_perf_p10b_005_ros_filters_connections_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10B_005_ros_filters_connections_before_deserialization."""
    path = tmp_path / "filtered.bag"
    path.write_text("stub", encoding="utf-8")
    selected = SimpleNamespace(topic="/pose", msgtype="geometry_msgs/msg/PoseStamped")
    ignored = SimpleNamespace(topic="/camera", msgtype="sensor_msgs/msg/Image")
    deserialized_topics: list[str] = []
    selected_topics: list[str] = []

    class FakeReader:
        connections = (selected, ignored)

        def __enter__(self) -> "FakeReader":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def messages(self, *, connections: tuple[object, ...]):
            selected_topics.extend(str(conn.topic) for conn in connections)
            for conn in connections:
                yield conn, 1_000_000_000, _pose_stamped(1, 0)

        def deserialize(self, rawdata: object, _msgtype: str) -> object:
            topic = selected.topic if rawdata is not None else ignored.topic
            deserialized_topics.append(topic)
            return rawdata

    monkeypatch.setattr(
        ros_logs_module,
        "_load_any_reader_class",
        lambda *, owner: lambda _paths: FakeReader(),
    )

    ao = read_ros_logs(
        str(path),
        opts=RosIngestOptions(
            topic="/pose",
            message_type="geometry_msgs/msg/PoseStamped",
        ),
    )

    assert selected_topics == ["/pose"]
    assert deserialized_topics == ["/pose"]
    assert ao.unsafe_data.sizes["sample"] == 1


def test_io_perf_p10b_009_ros_connection_metadata_is_normalized_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10B_009_ros_connection_metadata_is_normalized_once."""
    path = tmp_path / "connection_context.bag"
    path.write_text("stub", encoding="utf-8")

    class CountingConnection:
        msgcount = 1

        def __init__(self, topic: str, msgtype: str) -> None:
            self._topic = topic
            self._msgtype = msgtype
            self.topic_reads = 0
            self.msgtype_reads = 0

        @property
        def topic(self) -> str:
            self.topic_reads += 1
            return self._topic

        @property
        def msgtype(self) -> str:
            self.msgtype_reads += 1
            return self._msgtype

    selected = CountingConnection("/pose", "geometry_msgs/msg/PoseStamped")
    ignored = CountingConnection("/camera", "sensor_msgs/msg/Image")

    class FakeReader:
        connections = (selected, ignored)

        def __enter__(self) -> "FakeReader":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def messages(self, *, connections: tuple[object, ...]):
            assert connections == (selected,)
            for offset in range(2):
                yield selected, 1_000_000_000 + offset, _pose_stamped(1, offset)

        def deserialize(self, rawdata: object, _msgtype: str) -> object:
            return rawdata

    monkeypatch.setattr(
        ros_logs_module,
        "_load_any_reader_class",
        lambda *, owner: lambda _paths: FakeReader(),
    )

    read_ros_logs(
        str(path),
        opts=RosIngestOptions(
            topic="/pose",
            message_type="geometry_msgs/msg/PoseStamped",
        ),
    )

    assert (selected.topic_reads, selected.msgtype_reads) == (1, 1)
    assert (ignored.topic_reads, ignored.msgtype_reads) == (1, 0)


def test_io_hard_p10b_025_ros_frame_metadata_follows_timestamp_sort_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_025_ros_frame_metadata_follows_timestamp_sort_order."""
    path = tmp_path / "frame_order.bag"
    path.write_text("stub", encoding="utf-8")

    def _messages(_path: str, *, opts: RosIngestOptions, owner: str):
        _ = opts, owner
        return [
            _ros_message(
                "/pose",
                "geometry_msgs/msg/PoseStamped",
                _pose_stamped(2, 0, frame_id="late"),
                2_000_000_000,
            ),
            _ros_message(
                "/pose",
                "geometry_msgs/msg/PoseStamped",
                _pose_stamped(1, 0, frame_id="early"),
                1_000_000_000,
            ),
        ]

    monkeypatch.setattr(ros_logs_module, "_iter_ros_messages", _messages)
    options = RosIngestOptions(
        topic="/pose",
        metadata_promotion=AdapterMetadataPromotionOptions(nonscalar_target="attrs"),
    )

    ao = read_ros_logs(str(path), opts=options)

    assert ao.unsafe_data.coords["time"].isel(trial=0).values.tolist() == [
        1_000_000_000,
        2_000_000_000,
    ]
    assert ao.unsafe_data.attrs["ros_parent_frame"] == ["early", "late"]


def test_io_core_p10b_028_ros_all_missing_optional_frames_are_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_CORE_P10B_028_ros_all_missing_optional_frames_are_omitted."""
    path = tmp_path / "missing_child_frame.bag"
    path.write_text("stub", encoding="utf-8")
    message = _ros_message(
        "/pose",
        "geometry_msgs/msg/PoseStamped",
        _pose_stamped(1, 0),
        1_000_000_000,
    )
    monkeypatch.setattr(
        ros_logs_module,
        "_iter_ros_messages",
        lambda *_args, **_kwargs: [message],
    )

    ao = read_ros_logs(str(path), opts=RosIngestOptions(topic="/pose"))

    assert ao.unsafe_data.attrs["ros_parent_frame"] == "map"
    assert "ros_child_frame" not in ao.unsafe_data.attrs
    assert "ros_child_frame" not in ao.unsafe_data.coords


def test_io_core_p10b_029_ros_retained_mixed_frames_preserve_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_CORE_P10B_029_ros_retained_mixed_frames_preserve_alignment."""
    path = tmp_path / "mixed_parent_frame.bag"
    path.write_text("stub", encoding="utf-8")
    messages = [
        _ros_message(
            "/pose",
            "geometry_msgs/msg/PoseStamped",
            _pose_stamped(2, 0, frame_id=None),
            2_000_000_000,
        ),
        _ros_message(
            "/pose",
            "geometry_msgs/msg/PoseStamped",
            _pose_stamped(1, 0, frame_id="map"),
            1_000_000_000,
        ),
    ]
    monkeypatch.setattr(
        ros_logs_module,
        "_iter_ros_messages",
        lambda *_args, **_kwargs: messages,
    )
    metadata = AdapterMetadataPromotionOptions(nonscalar_target="attrs")

    ao = read_ros_logs(
        str(path),
        opts=RosIngestOptions(topic="/pose", metadata_promotion=metadata),
    )

    assert ao.unsafe_data.coords["time"].values.tolist() == [
        [1_000_000_000, 2_000_000_000]
    ]
    assert ao.unsafe_data.attrs["ros_parent_frame"] == ["map", None]
    assert "ros_child_frame" not in ao.unsafe_data.attrs


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        (field_name, bad_value)
        for field_name in ("sort_time", "allow_nonmonotonic_normalize")
        for bad_value in ("false", 0)
    ],
)
def test_io_hard_p10b_017_ros_boolean_options_are_strict(
    tmp_path: Path,
    field_name: str,
    bad_value: object,
) -> None:
    """ID: IO_HARD_P10B_017_ros_boolean_options_are_strict."""
    path = tmp_path / "strict_bool.bag"
    path.write_text("stub", encoding="utf-8")
    kwargs: dict[str, object] = {field_name: bad_value}
    opts = RosIngestOptions(**kwargs)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=rf"{field_name} must be bool"):
        read_ros_logs(path, opts=opts)


@pytest.mark.parametrize(
    "field_name",
    ["batch_dim", "sequence_dim", "sequence_size_coord", "param_coord"],
)
def test_io_hard_p10b_061_ros_layout_names_reject_payload_collisions_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    """ID: IO_HARD_P10B_061_ros_layout_names_reject_payload_collisions_preflight."""
    opts = RosIngestOptions(**{field_name: "translation_x"})  # type: ignore[arg-type]

    def unexpected_path_resolution(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        raise AssertionError("ROS layout collision reached path resolution")

    monkeypatch.setattr(
        ros_logs_module,
        "resolve_ingest_inputs",
        unexpected_path_resolution,
    )
    with pytest.raises(
        ValueError,
        match=rf"tal.io.read_ros_logs: {field_name} 'translation_x' collides with a ROS payload field",
    ):
        read_ros_logs(str(tmp_path / "missing.bag"), opts=opts)


def test_io_core_p10b_026_generated_metadata_attrs_can_share_ros_layout_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_CORE_P10B_026_generated_metadata_attrs_can_share_ros_layout_names."""
    path = tmp_path / "metadata_namespace.bag"
    path.write_text("stub", encoding="utf-8")

    def _messages(_path: str, *, opts: RosIngestOptions, owner: str):
        _ = opts, owner
        return [
            _ros_message(
                "/pose",
                "geometry_msgs/msg/PoseStamped",
                _pose_stamped(1, 0),
                1_000_000_000,
            )
        ]

    monkeypatch.setattr(ros_logs_module, "_iter_ros_messages", _messages)

    ao = read_ros_logs(
        str(path),
        opts=RosIngestOptions(topic="/pose", batch_dim="ros_timestamp_unit"),
    )

    assert ao.unsafe_data.sizes["ros_timestamp_unit"] == 1
    assert ao.unsafe_data.attrs["ros_timestamp_unit"] == "ns"


def test_io_hard_p10b_066_ros_generated_batch_coord_collision_preflights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_066_ros_generated_batch_coord_collision_preflights."""
    options = RosIngestOptions(
        batch_dim="ros_timestamp_unit",
        metadata_promotion=AdapterMetadataPromotionOptions(
            scalar_target="batch_coord"
        ),
    )

    def unexpected_path_resolution(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        raise AssertionError("ROS metadata collision reached path resolution")

    monkeypatch.setattr(
        ros_logs_module,
        "resolve_ingest_inputs",
        unexpected_path_resolution,
    )

    with pytest.raises(ValueError, match="generated adapter metadata"):
        read_ros_logs(str(tmp_path / "missing.bag"), opts=options)


@pytest.mark.parametrize(
    ("field_name", "opts"),
    [
        ("timestamp_source", RosIngestOptions(timestamp_source=[])),  # type: ignore[arg-type]
        ("monotonic_order", RosIngestOptions(monotonic_order={})),  # type: ignore[arg-type]
        ("invalid_time", RosIngestOptions(invalid_time=[])),  # type: ignore[arg-type]
    ],
)
def test_io_hard_p10b_028_ros_choice_options_reject_non_strings(
    tmp_path: Path,
    field_name: str,
    opts: RosIngestOptions,
) -> None:
    """ID: IO_HARD_P10B_028_ros_choice_options_reject_non_strings."""
    path = tmp_path / "choice_option.bag"
    path.write_text("stub", encoding="utf-8")

    with pytest.raises(TypeError, match=rf"tal.io.read_ros_logs: {field_name} must be a string"):
        read_ros_logs(str(path), opts=opts)


def test_io_hard_p10b_029_malformed_supported_ros_messages_retain_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_029_malformed_supported_ros_messages_retain_owner."""
    path = tmp_path / "malformed.bag"
    path.write_text("stub", encoding="utf-8")

    class FailingFloat:
        def __float__(self) -> float:
            raise RuntimeError("float conversion exploded")

    class FailingIndex:
        def __index__(self) -> int:
            raise RuntimeError("integer conversion exploded")

    malformed_payload = _ros_message(
        "/pose",
        "geometry_msgs/msg/PoseStamped",
        SimpleNamespace(pose=SimpleNamespace()),
        1_000_000_000,
    )
    malformed_timestamp = _ros_message(
        "/pose",
        "geometry_msgs/msg/PoseStamped",
        _pose_stamped("bad", 0),  # type: ignore[arg-type]
        1_000_000_000,
    )
    out_of_range_timestamp = _ros_message(
        "/pose",
        "geometry_msgs/msg/PoseStamped",
        _pose_stamped(1, 1_000_000_000),
        1_000_000_000,
    )
    overflowing_timestamp = _ros_message(
        "/pose",
        "geometry_msgs/msg/PoseStamped",
        _pose_stamped(int(np.iinfo(np.int64).max), 0),
        1_000_000_000,
    )
    runtime_payload = _ros_message(
        "/pose",
        "geometry_msgs/msg/PoseStamped",
        _pose_stamped(1, 0),
        1_000_000_000,
    )
    runtime_payload.msg.pose.position.x = FailingFloat()
    runtime_timestamp = _ros_message(
        "/pose",
        "geometry_msgs/msg/PoseStamped",
        SimpleNamespace(),
        FailingIndex(),
    )

    for message, detail in (
        (malformed_payload, "malformed ROS message payload"),
        (malformed_timestamp, "malformed ROS timestamp payload"),
        (out_of_range_timestamp, "malformed ROS timestamp payload"),
        (overflowing_timestamp, "malformed ROS timestamp payload"),
        (runtime_payload, "malformed ROS message payload"),
        (runtime_timestamp, "malformed ROS timestamp payload"),
    ):
        monkeypatch.setattr(
            ros_logs_module,
            "_iter_ros_messages",
            lambda *_args, **_kwargs: [message],
        )
        with pytest.raises(ValueError) as excinfo:
            read_ros_logs(str(path), opts=RosIngestOptions(topic="/pose"))
        assert str(excinfo.value).startswith(f"tal.io.read_ros_logs: {detail}")


def test_io_hard_p10b_052_ros_canonical_stamp_ignores_failing_legacy_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_052_ros_runtime_conversion_and_alias_access_retain_owner."""
    path = tmp_path / "canonical_stamp.bag"
    path.write_text("stub", encoding="utf-8")

    class CanonicalStamp:
        sec = 1
        nanosec = 2

        @property
        def secs(self) -> object:
            raise RuntimeError("legacy seconds alias accessed")

        @property
        def nsecs(self) -> object:
            raise RuntimeError("legacy nanoseconds alias accessed")

    message = _ros_message(
        "/pose",
        "geometry_msgs/msg/PoseStamped",
        _pose_stamped(1, 2),
        1_000_000_002,
    )
    message.msg.header.stamp = CanonicalStamp()
    monkeypatch.setattr(
        ros_logs_module,
        "_iter_ros_messages",
        lambda *_args, **_kwargs: [message],
    )

    ao = read_ros_logs(
        str(path),
        opts=RosIngestOptions(topic="/pose", timestamp_source="header"),
    )

    assert ao.unsafe_data.coords["time"].values.tolist() == [[1_000_000_002]]


def test_io_hard_p10b_050_ros_frame_text_conversion_retains_public_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_050_ros_frame_text_conversion_retains_public_owner."""
    path = tmp_path / "bad_frame.bag"
    path.write_text("stub", encoding="utf-8")

    class FailingFrame:
        def __str__(self) -> str:
            raise RuntimeError("frame conversion exploded")

    message = _ros_message(
        "/pose",
        "geometry_msgs/msg/PoseStamped",
        _pose_stamped(1, 0, frame_id=FailingFrame()),  # type: ignore[arg-type]
        1_000_000_000,
    )
    monkeypatch.setattr(
        ros_logs_module,
        "_iter_ros_messages",
        lambda *_args, **_kwargs: [message],
    )

    with pytest.raises(
        ValueError,
        match="tal.io.read_ros_logs: failed normalizing ROS parent frame",
    ) as error:
        read_ros_logs(str(path), opts=RosIngestOptions(topic="/pose"))

    assert isinstance(error.value.__cause__, RuntimeError)


def test_io_hard_p10b_070_disabled_ros_metadata_does_not_read_frame_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_070_disabled_ros_metadata_does_not_read_frame_payload."""
    path = tmp_path / "disabled_frame_metadata.bag"
    path.write_text("stub", encoding="utf-8")

    class FailingFrameHeader:
        @property
        def frame_id(self) -> object:
            raise RuntimeError("disabled frame metadata was evaluated")

    message = _ros_message(
        "/pose",
        "geometry_msgs/msg/PoseStamped",
        _pose_stamped(1, 0),
        1_000_000_000,
    )
    message.msg.header = FailingFrameHeader()
    monkeypatch.setattr(
        ros_logs_module,
        "_iter_ros_messages",
        lambda *_args, **_kwargs: [message],
    )
    metadata = AdapterMetadataPromotionOptions(
        scalar_target="none",
        nonscalar_target="none",
    )

    ao = read_ros_logs(
        str(path),
        opts=RosIngestOptions(
            topic="/pose",
            timestamp_source="receive",
            metadata_promotion=metadata,
        ),
    )

    assert ao.unsafe_data["translation_x"].values.tolist() == [[1.0]]
    assert "ros_parent_frame" not in ao.unsafe_data.attrs


def test_io_perf_p10b_008_ros_default_frame_policy_uses_constant_memory() -> None:
    """ID: IO_PERF_P10B_008_ros_default_frame_policy_uses_constant_memory."""
    messages = [
        _ros_message(
            "/pose",
            "geometry_msgs/msg/PoseStamped",
            _pose_stamped(index + 1, 0, frame_id=f"frame-{index}"),
            (index + 1) * 1_000_000_000,
        )
        for index in range(100)
    ]

    samples = ros_logs_module._collect_ros_samples(
        messages,
        opts=RosIngestOptions(topic="/pose"),
        owner="tal.io.read_ros_logs",
        path="memory.bag",
    )

    assert samples.parent_frames.varied
    assert samples.parent_frames.count == 100
    assert samples.parent_frames.values is None


def test_io_perf_p10b_013_ros_message_family_is_resolved_once_per_connection_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10B_013_ros_message_family_is_resolved_once_per_connection_plan."""
    calls: list[str] = []
    original = ros_reader_module.require_message_family

    def tracked(msgtype: str, *, owner: str) -> str:
        calls.append(msgtype)
        return original(msgtype, owner=owner)

    monkeypatch.setattr(ros_reader_module, "require_message_family", tracked)
    connection = SimpleNamespace(
        topic="/pose",
        msgtype="geometry_msgs/msg/PoseStamped",
        msgcount=2,
    )

    messages = tuple(
        ros_reader_module.iter_ros_messages(
            "family-once.bag",
            opts=RosIngestOptions(),
            owner="tal.io.read_ros_logs",
            reader_loader=lambda *, owner: lambda paths: _TwoMessagePoseReader(
                paths,
                connection=connection,
            ),
        )
    )

    samples = ros_logs_module._collect_ros_samples(
        messages,
        opts=RosIngestOptions(),
        owner="tal.io.read_ros_logs",
        path="family-once.bag",
    )

    assert len(samples.times) == 2
    assert calls == ["geometry_msgs/msg/PoseStamped"]


def test_io_hard_p10b_109_ros_ingest_cleanup_failure_retains_public_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_109_ros_ingest_cleanup_failure_retains_public_owner."""
    path = tmp_path / "cleanup.bag"
    path.write_text("stub", encoding="utf-8")
    message = _ros_message(
        "/pose",
        "geometry_msgs/msg/PoseStamped",
        _pose_stamped(1, 0),
        1_000_000_000,
    )
    monkeypatch.setattr(
        ros_logs_module,
        "_iter_ros_messages",
        lambda *_args, **_kwargs: [message],
    )
    failing_directory = cleanup_failing_temporary_directory(
        adapter_temp_module.TemporaryDirectory,
        message="cleanup exploded",
    )
    monkeypatch.setattr(adapter_temp_module, "TemporaryDirectory", failing_directory)

    with pytest.raises(
        ValueError,
        match="tal.io.read_ros_logs: failed cleaning ROS ingest spool directory",
    ) as error:
        read_ros_logs(str(path), opts=RosIngestOptions(topic="/pose"))

    assert isinstance(error.value.__cause__, OSError)


def test_io_perf_p10b_007_ros_ingest_releases_records_before_grid_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10B_007_ros_ingest_releases_records_before_grid_allocation."""
    paths = [tmp_path / "first.bag", tmp_path / "second.bag"]
    for path in paths:
        path.write_text("stub", encoding="utf-8")
    previous_refs: list[weakref.ReferenceType[np.ndarray]] = []
    call_count = 0

    def _record(path_info: object, *, opts: RosIngestOptions, owner: str):
        nonlocal call_count, previous_refs
        _ = opts, owner
        gc.collect()
        assert all(reference() is None for reference in previous_refs)
        call_count += 1
        times = np.asarray([call_count], dtype=np.int64)
        values = {
            name: np.asarray([float(call_count)], dtype=float)
            for name in ros_logs_module._ROS_COLUMNS
        }
        previous_refs = [weakref.ref(times), *(weakref.ref(value) for value in values.values())]
        return ros_logs_module._RosRecord(
            label=path_info.label,
            resolved_path=path_info.resolved_path,
            times=times,
            values=values,
            metadata={},
        )

    monkeypatch.setattr(ros_logs_module, "_collect_ros_record", _record)
    ao = read_ros_logs([str(path) for path in paths], opts=RosIngestOptions(topic="/pose"))
    gc.collect()

    assert all(reference() is None for reference in previous_refs)
    assert ao.unsafe_data.sizes == {"trial": 2, "sample": 1}
