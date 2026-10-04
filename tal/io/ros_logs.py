from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from .adapter_finalize import _finalize_owned_adapter_dataset
from .adapter_metadata import (
    aggregate_batch_metadata,
    promote_adapter_metadata,
    require_generated_metadata_preflight,
)
from .adapter_paths import ResolvedIngestInput, resolve_ingest_inputs
from .adapter_spool import AdapterArraySpool, fill_spooled_adapter_row, spool_adapter_arrays
from .adapter_temp import owned_temporary_directory
from .adapter_time import is_monotonic
from .options import RosIngestOptions, coerce_ros_ingest_options
from .ros_metadata import RosFrameMetadata, normalize_ros_text
from .ros_payload import ROS_COLUMNS as _ROS_COLUMNS
from .ros_payload import extract_frame_values as _extract_frame_values
from .ros_payload import extract_pose_values as _extract_pose_values
from .ros_reader import RosMessage as _RosMessage
from .ros_reader import iter_ros_messages, owned_ros_message_stream
from .ros_time import normalize_paired_ros_time_ns, normalize_receive_time_ns

_MISSING_ROS_TIME_FIELD = object()
_ROS_GENERATED_METADATA_NAMES = (
    "ros_topic",
    "ros_msgtype",
    "ros_timestamp_epoch",
    "ros_timestamp_unit",
    "ros_parent_frame",
    "ros_child_frame",
    "io_source_paths",
)
_ROS_SCALAR_GENERATED_METADATA_NAMES = _ROS_GENERATED_METADATA_NAMES[:-1]


@dataclass(frozen=True)
class _RosRecord:
    label: str
    resolved_path: str
    times: np.ndarray
    values: dict[str, np.ndarray]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _RosSpoolRecord:
    label: str
    resolved_path: str
    metadata: dict[str, object]
    arrays: AdapterArraySpool


@dataclass(frozen=True)
class _CollectedRosSamples:
    topic: str | None
    msgtype: str | None
    times: list[np.int64]
    columns: dict[str, list[float]]
    parent_frames: RosFrameMetadata
    child_frames: RosFrameMetadata


def _load_any_reader_class(*, owner: str) -> type:
    try:
        from rosbags.highlevel import AnyReader  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency envelope.
        raise ImportError(
            f"{owner}: read_ros_logs requires optional dependency 'rosbags'."
        ) from exc
    return AnyReader


def _iter_ros_messages(
    path: str,
    *,
    opts: RosIngestOptions,
    owner: str,
) -> Iterable[_RosMessage]:
    yield from iter_ros_messages(
        path,
        opts=opts,
        owner=owner,
        reader_loader=_load_any_reader_class,
    )


def _require_ros_layout_names(opts: RosIngestOptions, *, owner: str) -> None:
    """Reject semantic names owned by the fixed ROS pose payload."""
    layout_names = (
        ("batch_dim", opts.batch_dim),
        ("sequence_dim", opts.sequence_dim),
        ("sequence_size_coord", opts.sequence_size_coord),
        ("param_coord", opts.param_coord),
    )
    for field_name, name in layout_names:
        if name in _ROS_COLUMNS:
            raise ValueError(
                f"{owner}: {field_name} {name!r} collides with a ROS payload field."
            )


def _require_ros_metadata_preflight(opts: RosIngestOptions, *, owner: str) -> None:
    occupied = (
        opts.batch_dim,
        opts.sequence_dim,
        opts.sequence_size_coord,
        opts.param_coord,
        *_ROS_COLUMNS,
    )
    require_generated_metadata_preflight(
        options=opts.metadata_promotion,
        generated_names=_ROS_GENERATED_METADATA_NAMES,
        scalar_generated_names=_ROS_SCALAR_GENERATED_METADATA_NAMES,
        user_metadata_names=(),
        occupied_names=occupied,
        owner=owner,
    )


def _ros_time_field(stamp: object, primary: str, legacy: str) -> object | None:
    value = getattr(stamp, primary, _MISSING_ROS_TIME_FIELD)
    if value is _MISSING_ROS_TIME_FIELD:
        return getattr(stamp, legacy, None)
    return value


def _header_stamp_ns(msg: object) -> np.int64 | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = _ros_time_field(stamp, "sec", "secs")
    nsec = _ros_time_field(stamp, "nanosec", "nsecs")
    if sec is None or nsec is None:
        return None
    normalized = normalize_paired_ros_time_ns(sec, nsec)
    if normalized == 0:
        return None
    return normalized


def _timestamp_ns(
    msg: object,
    *,
    receive_ns: object | None,
    source: str,
) -> np.int64 | None:
    if source == "receive":
        return None if receive_ns is None else normalize_receive_time_ns(receive_ns)
    header_ns = _header_stamp_ns(msg)
    if source == "header":
        return header_ns
    if header_ns is not None:
        return header_ns
    return None if receive_ns is None else normalize_receive_time_ns(receive_ns)


def _resolve_ros_timestamp(
    message: _RosMessage,
    *,
    opts: RosIngestOptions,
    owner: str,
    path: str,
) -> np.int64 | None:
    try:
        return _timestamp_ns(
            message.msg,
            receive_ns=message.receive_ns,
            source=opts.timestamp_source,
        )
    except Exception as exc:
        raise ValueError(
            f"{owner}: malformed ROS timestamp payload in {path!r}."
        ) from exc


def _resolve_ros_sample_order(
    times: np.ndarray,
    *,
    opts: RosIngestOptions,
    owner: str,
    path: str,
) -> np.ndarray:
    order = np.arange(times.size)
    if opts.sort_time:
        order = np.argsort(times, kind="stable")
    ordered_times = times[order]
    if is_monotonic(ordered_times, order=opts.monotonic_order):
        return order
    if not opts.allow_nonmonotonic_normalize:
        raise ValueError(
            f"{owner}: ROS timestamps in {path!r} violate monotonic_order={opts.monotonic_order!r}."
        )
    order = np.argsort(times, kind="stable")
    if not is_monotonic(times[order], order=opts.monotonic_order):
        raise ValueError(
            f"{owner}: ROS timestamp normalization could not satisfy "
            f"monotonic_order={opts.monotonic_order!r} in {path!r}."
        )
    return order


def _collect_ros_samples_from_reader(
    path: str,
    *,
    opts: RosIngestOptions,
    owner: str,
) -> _CollectedRosSamples:
    with owned_ros_message_stream(
        _iter_ros_messages(path, opts=opts, owner=owner)
    ) as messages:
        return _collect_ros_samples(
            messages,
            opts=opts,
            owner=owner,
            path=path,
        )


def _collect_ros_record(
    path_info: ResolvedIngestInput,
    *,
    opts: RosIngestOptions,
    owner: str,
) -> _RosRecord:
    path = str(path_info.resolved_path)
    samples = _collect_ros_samples_from_reader(path, opts=opts, owner=owner)
    if samples.topic is None or samples.msgtype is None:
        raise ValueError(f"{owner}: no messages were found in {path!r}.")
    if not samples.times:
        raise ValueError(f"{owner}: no valid ROS messages remained after filtering for {path!r}.")
    time_arr = np.asarray(samples.times, dtype=np.int64)
    value_arrays = {
        name: np.asarray(values, dtype=float)
        for name, values in samples.columns.items()
    }
    order = _resolve_ros_sample_order(
        time_arr,
        opts=opts,
        owner=owner,
        path=path,
    )
    time_arr = time_arr[order]
    value_arrays = {name: values[order] for name, values in value_arrays.items()}
    parent_present, parent_value = samples.parent_frames.collapse(order)
    child_present, child_value = samples.child_frames.collapse(order)
    metadata = {
        "ros_topic": samples.topic,
        "ros_msgtype": samples.msgtype,
        "ros_timestamp_epoch": "unix",
        "ros_timestamp_unit": "ns",
    }
    if parent_present:
        metadata["ros_parent_frame"] = parent_value
    if child_present:
        metadata["ros_child_frame"] = child_value
    return _RosRecord(
        label=path_info.label,
        resolved_path=path_info.resolved_path,
        times=time_arr,
        values=value_arrays,
        metadata=metadata,
    )


def _append_pose_columns(
    columns: dict[str, list[float]],
    translation: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> None:
    for name, value in zip(_ROS_COLUMNS[:3], translation, strict=True):
        columns[name].append(value)
    for name, value in zip(_ROS_COLUMNS[3:], quaternion, strict=True):
        columns[name].append(value)


def _append_frame_metadata(
    frames: RosFrameMetadata,
    value: object,
    *,
    field: str,
    owner: str,
    path: str,
) -> None:
    frames.append(
        None
        if value is None
        else normalize_ros_text(value, field=field, owner=owner, path=path)
    )


def _append_ros_sample(
    message: _RosMessage,
    *,
    opts: RosIngestOptions,
    owner: str,
    path: str,
    times: list[np.int64],
    columns: dict[str, list[float]],
    parent_frames: RosFrameMetadata,
    child_frames: RosFrameMetadata,
    collect_frames: bool,
) -> None:
    timestamp = _resolve_ros_timestamp(message, opts=opts, owner=owner, path=path)
    if timestamp is None and opts.invalid_time == "fail":
        raise ValueError(f"{owner}: missing ROS timestamp encountered in {path!r}.")
    if timestamp is None:
        return
    translation, quaternion = _extract_pose_values(
        message.msg,
        family=message.family,
        msgtype=message.msgtype,
        owner=owner,
    )
    times.append(timestamp)
    _append_pose_columns(columns, translation, quaternion)
    if not collect_frames:
        return
    parent, child = _extract_frame_values(
        message.msg,
        family=message.family,
        msgtype=message.msgtype,
        owner=owner,
    )
    _append_frame_metadata(
        parent_frames,
        parent,
        field="parent frame",
        owner=owner,
        path=path,
    )
    _append_frame_metadata(
        child_frames,
        child,
        field="child frame",
        owner=owner,
        path=path,
    )


def _collect_ros_samples(
    messages: Iterable[_RosMessage],
    *,
    opts: RosIngestOptions,
    owner: str,
    path: str,
) -> _CollectedRosSamples:
    topic: str | None = None
    msgtype: str | None = None
    times: list[np.int64] = []
    columns = {name: [] for name in _ROS_COLUMNS}
    collect_frames = (
        opts.metadata_promotion.scalar_target != "none"
        or opts.metadata_promotion.nonscalar_target != "none"
    )
    retain_frames = opts.metadata_promotion.nonscalar_target != "none"
    parent_frames = RosFrameMetadata(retain_nonscalar=retain_frames)
    child_frames = RosFrameMetadata(retain_nonscalar=retain_frames)
    for message in messages:
        topic = message.topic
        msgtype = message.msgtype
        _append_ros_sample(
            message,
            opts=opts,
            owner=owner,
            path=path,
            times=times,
            columns=columns,
            parent_frames=parent_frames,
            child_frames=child_frames,
            collect_frames=collect_frames,
        )
    return _CollectedRosSamples(
        topic=topic,
        msgtype=msgtype,
        times=times,
        columns=columns,
        parent_frames=parent_frames,
        child_frames=child_frames,
    )


def _spool_ros_records(
    path_infos: Sequence[ResolvedIngestInput],
    *,
    spool_dir: str,
    opts: RosIngestOptions,
    owner: str,
) -> tuple[_RosSpoolRecord, ...]:
    plans: list[_RosSpoolRecord] = []
    for row, path_info in enumerate(path_infos):
        record = _collect_ros_record(path_info, opts=opts, owner=owner)
        arrays = spool_adapter_arrays(
            spool_dir,
            row=row,
            time_values=record.times,
            field_values=tuple(record.values[name] for name in _ROS_COLUMNS),
            owner=owner,
        )
        plans.append(
            _RosSpoolRecord(
                label=record.label,
                resolved_path=record.resolved_path,
                metadata=record.metadata,
                arrays=arrays,
            )
        )
        del record
    return tuple(plans)


def _spooled_ros_records_to_dataset(
    records: Sequence[_RosSpoolRecord],
    *,
    opts: RosIngestOptions,
    owner: str,
) -> xr.Dataset:
    batch = len(records)
    width = max(record.arrays.size for record in records)
    labels = [record.label for record in records]
    sizes = np.asarray([record.arrays.size for record in records], dtype=np.int64)
    time_grid = np.full((batch, width), np.iinfo(np.int64).min, dtype=np.int64)
    values = {name: np.full((batch, width), np.nan, dtype=float) for name in _ROS_COLUMNS}
    for row, record in enumerate(records):
        fill_spooled_adapter_row(
            record.arrays,
            row=row,
            time_target=time_grid,
            field_targets=tuple(values[name] for name in _ROS_COLUMNS),
            owner=owner,
        )
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
    ds.coords[opts.param_coord].attrs = {"units": "ns", "epoch": "unix"}
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
        ROS 1 file or ROS 2 bag-directory paths consumed by ingestion.
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
    ROS header and receive timestamps are normalized exactly to signed int64
    Unix nanoseconds. Normalized records are temporarily spooled so final
    padded grids do not coexist with every source record in memory.

    Examples
    --------
    >>> from tal.io import RosIngestOptions, read_ros_logs
    >>> opts = RosIngestOptions(topic="/robot/pose", message_type="geometry_msgs/msg/PoseStamped")
    >>> try:
    ...     ao = read_ros_logs("robot_run.mcap", opts=opts)
    ... except (ImportError, ValueError, FileNotFoundError):
    ...     ao = None
    >>> ao is None or "translation_x" in ao.as_dataset().data_vars
    True
    """
    owner = "tal.io.read_ros_logs"
    options = coerce_ros_ingest_options(opts, owner=owner)
    _require_ros_layout_names(options, owner=owner)
    _require_ros_metadata_preflight(options, owner=owner)
    path_infos = resolve_ingest_inputs(
        inputs,
        owner=owner,
        allow_directories=True,
    )
    with owned_temporary_directory(
        prefix="tal-ros-ingest-",
        owner=owner,
        purpose="ROS ingest spool directory",
    ) as spool_dir:
        records = _spool_ros_records(
            path_infos,
            spool_dir=spool_dir,
            opts=options,
            owner=owner,
        )
        ds = _spooled_ros_records_to_dataset(records, opts=options, owner=owner)
    return _finalize_owned_adapter_dataset(
        ds,
        batch_dim=options.batch_dim,
        sequence_dim=options.sequence_dim,
        size_name=options.sequence_size_coord,
        param_name=options.param_coord,
        validate=validate,
        owner=owner,
    )


__all__ = ["read_ros_logs"]
