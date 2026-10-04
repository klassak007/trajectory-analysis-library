from __future__ import annotations

import operator

import numpy as np

_NANOSECONDS_PER_SECOND = 1_000_000_000
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


def _exact_integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer.")
    try:
        result = operator.index(value)
    except Exception as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    return int(result)


def _require_int64(value: int, *, label: str) -> np.int64:
    if value < _INT64_MIN or value > _INT64_MAX:
        raise ValueError(f"{label} is outside signed int64 range.")
    return np.int64(value)


def normalize_paired_ros_time_ns(sec: object, nsec: object) -> np.int64:
    """Normalize paired ROS seconds/nanoseconds without float conversion."""
    seconds = _exact_integer(sec, label="ROS timestamp seconds")
    nanoseconds = _exact_integer(nsec, label="ROS timestamp nanoseconds")
    if nanoseconds < 0 or nanoseconds >= _NANOSECONDS_PER_SECOND:
        raise ValueError("ROS timestamp nanoseconds must be in [0, 1000000000).")
    return _require_int64(
        seconds * _NANOSECONDS_PER_SECOND + nanoseconds,
        label="ROS timestamp",
    )


def normalize_receive_time_ns(value: object) -> np.int64:
    """Normalize an exact backend receive timestamp to signed int64 nanoseconds."""
    timestamp = _exact_integer(value, label="ROS receive timestamp")
    return _require_int64(timestamp, label="ROS receive timestamp")


__all__ = ["normalize_paired_ros_time_ns", "normalize_receive_time_ns"]
