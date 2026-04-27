from __future__ import annotations

from tal.frames import Frame, FrameGraph
from tal.frames.registry import (
    get_edge_to_parent_runtime_ext,
    get_frame_runtime_ext,
    set_edge_to_parent_runtime_ext,
    set_frame_runtime_ext,
)

EDGE_MOTION_CLASS_VALUES: tuple[str, ...] = ("static", "galilean", "dynamic", "unknown")
FRAME_INERTIAL_STATUS_VALUES: tuple[str, ...] = ("inertial", "non_inertial", "unknown")

_EDGE_MOTION_CLASS_KEY = "spatial.edge_motion_class"
_FRAME_INERTIAL_STATUS_KEY = "spatial.frame_inertial_status"


def _require_frame(value: object, *, owner: str, arg: str) -> Frame:
    if isinstance(value, Frame):
        return value
    raise TypeError(f"{owner}: {arg} must be Frame, got {type(value).__name__}.")


def _require_graph(frame: Frame, *, owner: str, arg: str) -> FrameGraph:
    graph = frame._graph
    if isinstance(graph, FrameGraph):
        return graph
    raise ValueError(f"{owner}: {arg} frame is not bound to a valid FrameGraph.")


def _require_registered(frame: Frame, *, owner: str, arg: str) -> Frame:
    graph = _require_graph(frame, owner=owner, arg=arg)
    if graph._frames.get(frame.id) is frame:
        return frame
    raise ValueError(f"{owner}: {arg} frame {frame.id!r} is not registered in graph.")


def _require_same_graph(left: Frame, right: Frame, *, owner: str, left_arg: str, right_arg: str) -> None:
    left_graph = _require_graph(left, owner=owner, arg=left_arg)
    right_graph = _require_graph(right, owner=owner, arg=right_arg)
    if left_graph is not right_graph:
        raise ValueError(f"{owner}: {left_arg} and {right_arg} belong to different FrameGraph instances.")


def _require_value(value: object, *, allowed: tuple[str, ...], owner: str, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{owner}: {what} must be one of {allowed!r}, got {type(value).__name__}.")
    cleaned = value.strip()
    if cleaned in allowed:
        return cleaned
    raise ValueError(f"{owner}: {what} must be one of {allowed!r}, got {cleaned!r}.")


def _resolve_parent(child: Frame, parent: Frame | None, *, owner: str) -> Frame:
    child = _require_registered(child, owner=owner, arg="child")
    if parent is None:
        resolved = child.parent
        if resolved is None:
            raise ValueError(f"{owner}: child frame {child.id!r} has no parent edge.")
        return _require_registered(resolved, owner=owner, arg="parent")
    parent = _require_registered(parent, owner=owner, arg="parent")
    _require_same_graph(child, parent, owner=owner, left_arg="child", right_arg="parent")
    if child.parent is parent:
        return parent
    raise ValueError(f"{owner}: no registered edge for (child={child.id!r}, parent={parent.id!r}).")


def get_edge_motion_class(
    child: Frame,
    parent: Frame | None = None,
    *,
    owner: str = "spatial.frame_motion.get_edge_motion_class",
) -> str:
    """Return motion-class metadata for the edge ``parent -> child``.

    Parameters
    ----------
    child : Frame
        Child frame whose edge-to-parent metadata should be read.
    parent : Frame | None, optional
        Expected parent frame. When omitted, the child's current parent is used.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    str
        One of ``"static"``, ``"galilean"``, ``"dynamic"``, or ``"unknown"``.

    Notes
    -----
    Missing metadata is reported as ``"unknown"``. Child and parent must be
    registered in the same graph.

    Examples
    --------
    >>> from tal.frames import FrameGraph
    >>> from tal.spatial.metadata.frame_motion import get_edge_motion_class, set_edge_motion_class
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> base = graph.get_or_create_frame("base", parent=world)
    >>> set_edge_motion_class(base, "static", parent=world)
    >>> get_edge_motion_class(base, parent=world)
    'static'
    """
    child = _require_frame(child, owner=owner, arg="child")
    _ = _resolve_parent(child, parent, owner=owner)
    value = get_edge_to_parent_runtime_ext(child, _EDGE_MOTION_CLASS_KEY, owner=owner)
    return _require_value(
        "unknown" if value is None else value,
        allowed=EDGE_MOTION_CLASS_VALUES,
        owner=owner,
        what="edge motion class",
    )


def set_edge_motion_class(
    child: Frame,
    motion_class: str,
    parent: Frame | None = None,
    *,
    owner: str = "spatial.frame_motion.set_edge_motion_class",
) -> None:
    """Set motion-class metadata on the edge ``parent -> child``.

    Parameters
    ----------
    child : Frame
        Child frame whose edge-to-parent metadata should be written.
    motion_class : str
        Edge motion-class tag: ``"static"``, ``"galilean"``, ``"dynamic"``,
        or ``"unknown"``.
    parent : Frame | None, optional
        Expected parent frame. When omitted, the child's current parent is used.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    None
        Returns ``None``; side effects are applied through owned state/metadata updates.

    Notes
    -----
    The function mutates runtime frame metadata; it does not write AO schema.

    Examples
    --------
    >>> from tal.frames import FrameGraph
    >>> from tal.spatial.metadata.frame_motion import get_edge_motion_class, set_edge_motion_class
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> base = graph.get_or_create_frame("base", parent=world)
    >>> set_edge_motion_class(base, "static", parent=world)
    >>> get_edge_motion_class(base, parent=world)
    'static'
    """
    child = _require_frame(child, owner=owner, arg="child")
    _ = _resolve_parent(child, parent, owner=owner)
    set_edge_to_parent_runtime_ext(
        child,
        _EDGE_MOTION_CLASS_KEY,
        _require_value(motion_class, allowed=EDGE_MOTION_CLASS_VALUES, owner=owner, what="edge motion class"),
        owner=owner,
    )


def get_frame_inertial_status(
    frame: Frame,
    *,
    owner: str = "spatial.frame_motion.get_frame_inertial_status",
) -> str:
    """Return inertial-status metadata for a frame.

    Parameters
    ----------
    frame : Frame
        Registered frame whose inertial status should be read.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    str
        One of ``"inertial"``, ``"non_inertial"``, or ``"unknown"``.

    Notes
    -----
    Missing metadata is reported as ``"unknown"``.

    Examples
    --------
    >>> from tal.frames import FrameGraph
    >>> from tal.spatial.metadata.frame_motion import get_frame_inertial_status, set_frame_inertial_status
    >>> frame = FrameGraph().get_or_create_frame("world")
    >>> set_frame_inertial_status(frame, "inertial")
    >>> get_frame_inertial_status(frame)
    'inertial'
    """
    frame = _require_registered(_require_frame(frame, owner=owner, arg="frame"), owner=owner, arg="frame")
    value = get_frame_runtime_ext(frame, _FRAME_INERTIAL_STATUS_KEY, owner=owner)
    return _require_value(
        "unknown" if value is None else value,
        allowed=FRAME_INERTIAL_STATUS_VALUES,
        owner=owner,
        what="frame inertial status",
    )


def set_frame_inertial_status(
    frame: Frame,
    status: str,
    *,
    owner: str = "spatial.frame_motion.set_frame_inertial_status",
) -> None:
    """Set inertial-status metadata for a frame.

    Parameters
    ----------
    frame : Frame
        Registered frame whose inertial status should be written.
    status : str
        Status value: ``"inertial"``, ``"non_inertial"``, or ``"unknown"``.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    None
        Returns ``None``; side effects are applied through owned state/metadata updates.

    Notes
    -----
    The function mutates runtime frame metadata; it does not write AO schema.

    Examples
    --------
    >>> from tal.frames import FrameGraph
    >>> from tal.spatial.metadata.frame_motion import get_frame_inertial_status, set_frame_inertial_status
    >>> frame = FrameGraph().get_or_create_frame("world")
    >>> set_frame_inertial_status(frame, "inertial")
    >>> get_frame_inertial_status(frame)
    'inertial'
    """
    frame = _require_registered(_require_frame(frame, owner=owner, arg="frame"), owner=owner, arg="frame")
    set_frame_runtime_ext(
        frame,
        _FRAME_INERTIAL_STATUS_KEY,
        _require_value(status, allowed=FRAME_INERTIAL_STATUS_VALUES, owner=owner, what="frame inertial status"),
        owner=owner,
    )


def propagate_inertial_status(
    child: Frame,
    parent: Frame | None = None,
    *,
    owner: str = "spatial.frame_motion.propagate_inertial_status",
) -> str:
    """Infer and write a child's inertial status from parent status and edge class.

    Parameters
    ----------
    child : Frame
        Child frame whose status should be inferred.
    parent : Frame | None, optional
        Expected parent frame. When omitted, the child's current parent is used.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    str
        Inferred child inertial status after it has been written.

    Notes
    -----
    Static and Galilean edges from an inertial parent propagate inertial status.
    Dynamic edges from an inertial parent mark the child as non-inertial.

    Examples
    --------
    >>> from tal.frames import FrameGraph
    >>> from tal.spatial.metadata.frame_motion import (
    ...     get_frame_inertial_status,
    ...     propagate_inertial_status,
    ...     set_edge_motion_class,
    ...     set_frame_inertial_status,
    ... )
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> base = graph.get_or_create_frame("base", parent=world)
    >>> set_frame_inertial_status(world, "inertial")
    >>> set_edge_motion_class(base, "static", parent=world)
    >>> propagate_inertial_status(base, parent=world)
    'inertial'
    >>> get_frame_inertial_status(base)
    'inertial'
    """
    child = _require_frame(child, owner=owner, arg="child")
    parent = _resolve_parent(child, parent, owner=owner)
    parent_status = get_frame_inertial_status(parent, owner=owner)
    edge_class = get_edge_motion_class(child, parent, owner=owner)
    if parent_status != "inertial":
        next_status = "unknown"
    elif edge_class in {"static", "galilean"}:
        next_status = "inertial"
    elif edge_class == "dynamic":
        next_status = "non_inertial"
    else:
        next_status = "unknown"
    set_frame_inertial_status(child, next_status, owner=owner)
    return next_status


__all__ = [
    "EDGE_MOTION_CLASS_VALUES",
    "FRAME_INERTIAL_STATUS_VALUES",
    "get_edge_motion_class",
    "get_frame_inertial_status",
    "propagate_inertial_status",
    "set_edge_motion_class",
    "set_frame_inertial_status",
]
