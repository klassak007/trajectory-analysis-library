from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .adapter_finalize import finalize_adapter_dataset
from .adapter_metadata import aggregate_batch_metadata, promote_adapter_metadata
from .adapter_paths import resolve_ingest_inputs
from .csv_logs import _is_monotonic
from .options import RosIngestOptions, coerce_ros_ingest_options

_ROS_COLUMNS = (
    "translation_x",
    "translation_y",
    "translation_z",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "quaternion_w",
)


@dataclass(frozen=True)
class _RosMessage:
    topic: str
    msgtype: str
    msg: object
    receive_ns: int | None


@dataclass(frozen=True)
class _RosRecord:
    label: str
    resolved_path: str
    times: np.ndarray
    values: dict[str, np.ndarray]
    metadata: dict[str, object]


def _iter_ros_messages(path: str, *, owner: str) -> list[_RosMessage]:
    try:
        from rosbags.highlevel import AnyReader  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency envelope.
        raise ImportError(
            f"{owner}: read_ros_logs requires optional dependency 'rosbags'."
        ) from exc
    out: list[_RosMessage] = []
    with AnyReader([Path(path)]) as reader:
        for conn, timestamp_ns, rawdata in reader.messages():
            msg = reader.deserialize(rawdata, conn.msgtype)
            out.append(
                _RosMessage(
                    topic=str(conn.topic),
                    msgtype=str(conn.msgtype),
                    msg=msg,
                    receive_ns=int(timestamp_ns),
                )
            )
    return out


def _select_topic(messages: Sequence[_RosMessage], *, opts: RosIngestOptions, owner: str, path: str) -> tuple[str, tuple[_RosMessage, ...]]:
    topics = sorted({message.topic for message in messages})
    if opts.topic is not None:
        selected = tuple(message for message in messages if message.topic == opts.topic)
        if not selected:
            raise ValueError(f"{owner}: topic {opts.topic!r} was not found in {path!r}.")
        return opts.topic, selected
    if len(topics) != 1:
        raise ValueError(
            f"{owner}: topic ambiguity in {path!r}; candidates={tuple(topics)!r}. "
            "Provide topic explicitly."
        )
    topic = topics[0]
    return topic, tuple(message for message in messages if message.topic == topic)


def _select_msgtype(messages: Sequence[_RosMessage], *, opts: RosIngestOptions, owner: str, path: str) -> str:
    msgtypes = sorted({message.msgtype for message in messages})
    if opts.message_type is not None:
        if opts.message_type not in msgtypes:
            raise ValueError(
                f"{owner}: message_type {opts.message_type!r} was not found in {path!r}; "
                f"available={tuple(msgtypes)!r}."
            )
        return opts.message_type
    if len(msgtypes) != 1:
        raise ValueError(
            f"{owner}: message-type ambiguity in {path!r}; candidates={tuple(msgtypes)!r}. "
            "Provide message_type explicitly."
        )
    return msgtypes[0]


def _header_stamp_seconds(msg: object) -> float | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", getattr(stamp, "secs", None))
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", None))
    if sec is None or nsec is None:
        return None
    sec_i = int(sec)
    nsec_i = int(nsec)
    if sec_i == 0 and nsec_i == 0:
        return None
    return float(sec_i) + float(nsec_i) * 1e-9


def _timestamp_seconds(
    msg: object,
    *,
    receive_ns: int | None,
    source: str,
) -> float:
    header_seconds = _header_stamp_seconds(msg)
    receive_seconds = None if receive_ns is None else float(receive_ns) * 1e-9
    if source == "header":
        return float("nan") if header_seconds is None else header_seconds
    if source == "receive":
        return float("nan") if receive_seconds is None else receive_seconds
    if header_seconds is not None:
        return header_seconds
    return float("nan") if receive_seconds is None else receive_seconds


def _extract_pose_values(msg: object, *, msgtype: str, owner: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float], str | None, str | None]:
    if msgtype.endswith("PoseStamped"):
        pos = msg.pose.position
        ori = msg.pose.orientation
        parent = getattr(getattr(msg, "header", None), "frame_id", None)
        return (float(pos.x), float(pos.y), float(pos.z)), (float(ori.x), float(ori.y), float(ori.z), float(ori.w)), parent, None
    if msgtype.endswith("Odometry"):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        parent = getattr(getattr(msg, "header", None), "frame_id", None)
        child = getattr(msg, "child_frame_id", None)
        return (float(pos.x), float(pos.y), float(pos.z)), (float(ori.x), float(ori.y), float(ori.z), float(ori.w)), parent, child
    if msgtype.endswith("TransformStamped"):
        trans = msg.transform.translation
        rot = msg.transform.rotation
        parent = getattr(getattr(msg, "header", None), "frame_id", None)
        child = getattr(msg, "child_frame_id", None)
        return (float(trans.x), float(trans.y), float(trans.z)), (float(rot.x), float(rot.y), float(rot.z), float(rot.w)), parent, child
    raise ValueError(f"{owner}: unsupported ROS message type {msgtype!r}.")


def _apply_ros_monotonic_policy(
    times: np.ndarray,
    values: dict[str, np.ndarray],
    *,
    opts: RosIngestOptions,
    owner: str,
    path: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if opts.sort_time:
        order = np.argsort(times, kind="stable")
        times = times[order]
        values = {name: arr[order] for name, arr in values.items()}
    if _is_monotonic(times, order=opts.monotonic_order):
        return times, values
    if not opts.allow_nonmonotonic_normalize:
        raise ValueError(
            f"{owner}: ROS timestamps in {path!r} violate monotonic_order={opts.monotonic_order!r}."
        )
    order = np.argsort(times, kind="stable")
    times_sorted = times[order]
    values_sorted = {name: arr[order] for name, arr in values.items()}
    if not _is_monotonic(times_sorted, order=opts.monotonic_order):
        raise ValueError(
            f"{owner}: ROS timestamp normalization could not satisfy "
            f"monotonic_order={opts.monotonic_order!r} in {path!r}."
        )
    return times_sorted, values_sorted


def _collect_ros_record(path_info: Any, *, opts: RosIngestOptions, owner: str) -> _RosRecord:
    messages = _iter_ros_messages(str(path_info.path), owner=owner)
    if not messages:
        raise ValueError(f"{owner}: no messages were found in {path_info.path!r}.")
    topic, topic_messages = _select_topic(messages, opts=opts, owner=owner, path=str(path_info.path))
    msgtype = _select_msgtype(topic_messages, opts=opts, owner=owner, path=str(path_info.path))
    filtered = [message for message in topic_messages if message.msgtype == msgtype]
    times, columns, parent_values, child_values = _collect_ros_samples(
        filtered,
        opts=opts,
        owner=owner,
        path=str(path_info.path),
    )
    if not times:
        raise ValueError(f"{owner}: no valid ROS messages remained after filtering for {path_info.path!r}.")
    time_arr = np.asarray(times, dtype=float)
    value_arrays = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    time_arr, value_arrays = _apply_ros_monotonic_policy(
        time_arr,
        value_arrays,
        opts=opts,
        owner=owner,
        path=str(path_info.path),
    )
    metadata = {
        "ros_topic": topic,
        "ros_msgtype": msgtype,
        "ros_parent_frame": parent_values[0] if parent_values and all(parent_values[0] == v for v in parent_values[1:]) else parent_values,
        "ros_child_frame": child_values[0] if child_values and all(child_values[0] == v for v in child_values[1:]) else child_values,
    }
    return _RosRecord(
        label=path_info.label,
        resolved_path=path_info.resolved_path,
        times=time_arr,
        values=value_arrays,
        metadata=metadata,
    )


def _collect_ros_samples(
    messages: Sequence[_RosMessage],
    *,
    opts: RosIngestOptions,
    owner: str,
    path: str,
) -> tuple[list[float], dict[str, list[float]], list[str | None], list[str | None]]:
    times: list[float] = []
    columns = {name: [] for name in _ROS_COLUMNS}
    parent_values: list[str | None] = []
    child_values: list[str | None] = []
    for message in messages:
        translation, quaternion, parent, child = _extract_pose_values(
            message.msg,
            msgtype=message.msgtype,
            owner=owner,
        )
        timestamp = _timestamp_seconds(
            message.msg,
            receive_ns=message.receive_ns,
            source=opts.timestamp_source,
        )
        if not np.isfinite(timestamp):
            if opts.invalid_time == "fail":
                raise ValueError(f"{owner}: invalid/non-finite ROS timestamp encountered in {path!r}.")
            continue
        times.append(float(timestamp))
        columns["translation_x"].append(float(translation[0]))
        columns["translation_y"].append(float(translation[1]))
        columns["translation_z"].append(float(translation[2]))
        columns["quaternion_x"].append(float(quaternion[0]))
        columns["quaternion_y"].append(float(quaternion[1]))
        columns["quaternion_z"].append(float(quaternion[2]))
        columns["quaternion_w"].append(float(quaternion[3]))
        parent_values.append(None if parent is None else str(parent))
        child_values.append(None if child is None else str(child))
    return times, columns, parent_values, child_values


def _records_to_dataset(
    records: Sequence[_RosRecord],
    *,
    opts: RosIngestOptions,
    owner: str,
) -> xr.Dataset:
    batch = len(records)
    width = max(record.times.size for record in records)
    labels = [record.label for record in records]
    sizes = np.asarray([record.times.size for record in records], dtype=np.int64)
    time_grid = np.full((batch, width), np.nan, dtype=float)
    values = {name: np.full((batch, width), np.nan, dtype=float) for name in _ROS_COLUMNS}
    for row, record in enumerate(records):
        size = int(record.times.size)
        time_grid[row, :size] = record.times
        for name in _ROS_COLUMNS:
            values[name][row, :size] = record.values[name]
    ds = xr.Dataset(
        data_vars={
            name: ((opts.batch_dim, opts.sequence_dim), values[name])
            for name in _ROS_COLUMNS
        },
        coords={
            opts.batch_dim: np.asarray(labels, dtype=object),
            opts.sequence_dim: np.arange(width, dtype=np.int64),
            opts.param_coord: ((opts.batch_dim, opts.sequence_dim), time_grid),
            opts.sequence_size_coord: (opts.batch_dim, sizes),
        },
    )
    metadata = aggregate_batch_metadata([record.metadata for record in records])
    metadata["io_source_paths"] = [record.resolved_path for record in records]
    return promote_adapter_metadata(
        ds,
        batch_dim=opts.batch_dim,
        metadata=metadata,
        options=opts.metadata_promotion,
        owner=owner,
    )


def read_ros_logs(
    inputs: str | Sequence[str] | Mapping[str, str],
    *,
    opts: RosIngestOptions | None = None,
    validate: bool = True,
):
    """Read ROS log files into a schema-annotated TAL AnalysisObject.

    Parameters
    ----------
    inputs : str | Sequence[str] | Mapping[str, str]
        Input paths/mapping consumed by ingestion.
    opts : RosIngestOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``RosIngestOptions`` key fields: ``topic`` (default None), ``message_type`` (default None), ``timestamp_source`` (default 'auto'), ``sort_time`` (default True).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    object
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> from tal.io import RosIngestOptions, read_ros_logs
    >>> opts = RosIngestOptions(topic="/robot/pose", message_type="geometry_msgs/msg/PoseStamped")
    >>> try:
    ...     ao = read_ros_logs("robot_run.mcap", opts=opts)
    ... except (ImportError, ValueError, FileNotFoundError):
    ...     ao = None
    >>> ao is None or "translation_x" in ao.unsafe_data.data_vars
    True
    """
    owner = "tal.io.read_ros_logs"
    options = coerce_ros_ingest_options(opts, owner=owner)
    path_infos = resolve_ingest_inputs(inputs, owner=owner)
    records = [_collect_ros_record(path_info, opts=options, owner=owner) for path_info in path_infos]
    ds = _records_to_dataset(records, opts=options, owner=owner)
    return finalize_adapter_dataset(
        ds,
        batch_dim=options.batch_dim,
        sequence_dim=options.sequence_dim,
        size_name=options.sequence_size_coord,
        param_name=options.param_coord,
        validate=validate,
        owner=owner,
    )


def read_ros_logs_catalog(
    inputs: str | Sequence[str] | Mapping[str, str],
    *,
    opts: RosIngestOptions | None = None,
    validate: bool = True,
):
    """Read ROS logs and expose them through the browse-only ``Catalog`` surface.

    Parameters
    ----------
    inputs : str | Sequence[str] | Mapping[str, str]
        Input paths/mapping consumed by ingestion.
    opts : RosIngestOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``RosIngestOptions`` key fields: ``topic`` (default None), ``message_type`` (default None), ``timestamp_source`` (default 'auto'), ``sort_time`` (default True).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    object
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> from tal.io import RosIngestOptions, read_ros_logs_catalog
    >>> opts = RosIngestOptions(topic="/robot/pose", message_type="geometry_msgs/msg/PoseStamped")
    >>> try:
    ...     catalog = read_ros_logs_catalog("robot_run.mcap", opts=opts)
    ... except (ImportError, ValueError, FileNotFoundError):
    ...     catalog = None
    >>> catalog is None or catalog.backend == "dataset"
    True
    """
    from tal.catalog import Catalog

    ao = read_ros_logs(inputs, opts=opts, validate=validate)
    options = coerce_ros_ingest_options(opts, owner="tal.io.read_ros_logs_catalog")
    return Catalog(ao, backend="dataset", batch_dim=options.batch_dim)


__all__ = ["read_ros_logs", "read_ros_logs_catalog"]
