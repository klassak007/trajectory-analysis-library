from __future__ import annotations

ROS_COLUMNS = (
    "translation_x",
    "translation_y",
    "translation_z",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "quaternion_w",
)
_SUPPORTED_FAMILIES = ("PoseStamped", "Odometry", "TransformStamped")


def message_family(msgtype: str) -> str | None:
    """Return the exact supported terminal ROS message family, if any."""
    family = msgtype.rsplit("/", maxsplit=1)[-1]
    if family not in _SUPPORTED_FAMILIES:
        return None
    return family


def require_message_family(msgtype: str, *, owner: str) -> str:
    """Return a supported terminal ROS message family or fail closed."""
    family = message_family(msgtype)
    if family is None:
        raise ValueError(f"{owner}: unsupported ROS message type {msgtype!r}.")
    return family


def _resolve_pose_components(
    msg: object,
    *,
    family: str,
) -> tuple[object, object]:
    if family == "PoseStamped":
        return msg.pose.position, msg.pose.orientation
    if family == "Odometry":
        return msg.pose.pose.position, msg.pose.pose.orientation
    if family == "TransformStamped":
        return msg.transform.translation, msg.transform.rotation
    raise AssertionError(f"unhandled supported ROS message family {family!r}")


def _resolve_frame_components(msg: object, *, family: str) -> tuple[object, object]:
    parent = getattr(getattr(msg, "header", None), "frame_id", None)
    if family == "PoseStamped":
        return parent, None
    if family in {"Odometry", "TransformStamped"}:
        return parent, msg.child_frame_id
    raise AssertionError(f"unhandled supported ROS message family {family!r}")


def extract_pose_values(
    msg: object,
    *,
    family: str,
    msgtype: str,
    owner: str,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]:
    """Decode one supported ROS pose payload without orchestration state."""
    try:
        translation, quaternion = _resolve_pose_components(
            msg,
            family=family,
        )
        translation_values = (
            float(translation.x),
            float(translation.y),
            float(translation.z),
        )
        quaternion_values = (
            float(quaternion.x),
            float(quaternion.y),
            float(quaternion.z),
            float(quaternion.w),
        )
    except Exception as exc:
        raise ValueError(f"{owner}: malformed ROS message payload for type {msgtype!r}.") from exc
    return translation_values, quaternion_values


def extract_frame_values(
    msg: object,
    *,
    family: str,
    msgtype: str,
    owner: str,
) -> tuple[object, object]:
    """Decode optional frame metadata without coupling it to pose values."""
    try:
        return _resolve_frame_components(msg, family=family)
    except Exception as exc:
        raise ValueError(f"{owner}: malformed ROS frame payload for type {msgtype!r}.") from exc


__all__ = [
    "ROS_COLUMNS",
    "extract_frame_values",
    "extract_pose_values",
    "message_family",
    "require_message_family",
]
