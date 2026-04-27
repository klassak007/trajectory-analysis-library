from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .registry import Frame, FrameGraph


def _require_frame_endpoint(value: Any, *, owner: str, arg: str) -> Frame:
    if isinstance(value, Frame):
        return value
    raise TypeError(f"{owner}: {arg} must be Frame, got {type(value).__name__}.")


def _require_frame_graph_binding(frame: Frame, *, owner: str, arg: str) -> FrameGraph:
    graph = frame._graph
    if isinstance(graph, FrameGraph):
        return graph
    raise ValueError(f"{owner}: {arg} frame is not bound to a valid FrameGraph.")


def _require_registered(frame: Frame, *, owner: str, arg: str) -> None:
    graph = _require_frame_graph_binding(frame, owner=owner, arg=arg)
    if graph._frames.get(frame.id) is frame:
        return
    raise ValueError(f"{owner}: {arg} {frame!r} is not registered in its FrameGraph.")


def _require_same_graph(src: Frame, dst: Frame, *, owner: str) -> None:
    if src._graph is dst._graph:
        return
    raise ValueError(f"{owner}: src and dst belong to different FrameGraph instances.")


def _ancestors_with_depth(node: Frame) -> dict[Frame, int]:
    out: dict[Frame, int] = {}
    depth = 0
    cur: Frame | None = node
    while cur is not None:
        out[cur] = depth
        depth += 1
        cur = cur.parent
    return out


def _build_path_nodes(src: Frame, lca: Frame, up_from_dst: list[Frame]) -> tuple[Frame, ...]:
    up_from_src: list[Frame] = []
    cur: Frame | None = src
    while cur is not lca:
        up_from_src.append(cur)
        cur = cur.parent
    up_from_src.append(lca)
    down_to_dst = list(reversed(up_from_dst))
    return tuple(up_from_src + down_to_dst)


def _build_oriented_steps(nodes: tuple[Frame, ...], lca_index: int) -> tuple["PathStep", ...]:
    steps: list[PathStep] = []
    for i in range(0, lca_index):
        steps.append(PathStep(child=nodes[i], parent=nodes[i + 1], invert=False))
    for i in range(lca_index, len(nodes) - 1):
        steps.append(PathStep(child=nodes[i + 1], parent=nodes[i], invert=True))
    return tuple(steps)


@dataclass(frozen=True)
class PathStep:
    """One oriented edge traversal within a resolved frame path.

    Notes
    -----
    ``invert=False`` means traversal follows the child-to-parent edge.
    ``invert=True`` means traversal uses the edge in the reverse direction.

    Examples
    --------
    >>> from tal.frames import FrameGraph, find_path
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> base = graph.get_or_create_frame("base", parent=world)
    >>> find_path(base, world).steps[0].invert
    False
    """

    child: Frame
    parent: Frame
    invert: bool


@dataclass(frozen=True)
class FramePath:
    """Resolved path between two frames in a shared ``FrameGraph``.

    Notes
    -----
    ``nodes`` is ordered from source to destination. ``steps`` stores the edge
    traversals needed to fold transforms or other edge-attached values along
    that route.

    Examples
    --------
    >>> from tal.frames import FrameGraph, find_path
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> base = graph.get_or_create_frame("base", parent=world)
    >>> [node.id for node in find_path(base, world).nodes]
    ['base', 'world']
    """

    nodes: tuple[Frame, ...]
    lca: Frame
    lca_index: int
    steps: tuple[PathStep, ...]


def _find_path_impl(src: Frame, dst: Frame, *, owner: str) -> FramePath:
    src_frame = _require_frame_endpoint(src, owner=owner, arg="src")
    dst_frame = _require_frame_endpoint(dst, owner=owner, arg="dst")
    _require_registered(src_frame, owner=owner, arg="src")
    _require_registered(dst_frame, owner=owner, arg="dst")
    _require_same_graph(src_frame, dst_frame, owner=owner)
    if src_frame is dst_frame:
        return FramePath(nodes=(src_frame,), lca=src_frame, lca_index=0, steps=())
    src_ancestors = _ancestors_with_depth(src_frame)
    up_from_dst: list[Frame] = []
    cur: Frame | None = dst_frame
    lca: Frame | None = None
    while cur is not None:
        if cur in src_ancestors:
            lca = cur
            break
        up_from_dst.append(cur)
        cur = cur.parent
    if lca is None:
        raise ValueError(f"{owner}: no path between src and dst in this graph component.")
    nodes = _build_path_nodes(src_frame, lca, up_from_dst)
    lca_index = nodes.index(lca)
    steps = _build_oriented_steps(nodes, lca_index)
    return FramePath(nodes=nodes, lca=lca, lca_index=lca_index, steps=steps)


def find_path(src: Frame, dst: Frame) -> FramePath:
    """Find the oriented frame path between two nodes.

    Parameters
    ----------
    src : Frame
    dst : Frame

    Returns
    -------
    FramePath

    Notes
    -----
    Inputs must be registered in the same graph; mismatches fail closed.

    Examples
    --------
    >>> from tal.frames import FrameGraph, find_path
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> base = graph.get_or_create_frame("base", parent=world)
    >>> [node.id for node in find_path(base, world).nodes]
    ['base', 'world']
    """
    return _find_path_impl(src, dst, owner="find_path")


def fold_path(
    path: FramePath,
    *,
    edge_value_fn: Callable[[Frame, Frame], Any],
    compose: Callable[[Any, Any], Any],
    inverse: Callable[[Any], Any],
    identity: Callable[[], Any],
) -> Any:
    """Accumulate edge values across an oriented frame path.

    Parameters
    ----------
    path : FramePath
        Path returned by :func:`find_path`.
    edge_value_fn : callable
        Called as ``edge_value_fn(child, parent)`` for each path step.
    compose : callable
        Binary accumulator function ``compose(acc, value)``.
    inverse : callable
        Maps an edge value to its inverse when traversing opposite edge direction.
    identity : callable
        Zero-argument factory returning the initial accumulator value.

    Returns
    -------
    Any
        Folded accumulator result.

    Notes
    -----
    Steps marked ``invert`` receive ``inverse(edge_value)`` before composition.
    The caller controls the edge value type and composition algebra.

    Examples
    --------
    >>> from tal.frames import FrameGraph, find_path, fold_path
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> base = graph.get_or_create_frame("base", parent=world)
    >>> path = find_path(base, world)
    >>> fold_path(
    ...     path,
    ...     edge_value_fn=lambda child, parent: [(child.id, parent.id)],
    ...     compose=lambda acc, value: acc + value,
    ...     inverse=lambda value: [(value[0][1], value[0][0])],
    ...     identity=lambda: [],
    ... )
    [('base', 'world')]
    """
    owner = "fold_path"
    if not isinstance(path, FramePath):
        raise TypeError(f"{owner}: path must be FramePath, got {type(path).__name__}.")
    acc = identity()
    for step in path.steps:
        value = edge_value_fn(step.child, step.parent)
        if step.invert:
            value = inverse(value)
        acc = compose(acc, value)
    return acc
