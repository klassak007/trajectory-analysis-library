from __future__ import annotations

import importlib
from collections import deque
from dataclasses import dataclass
from typing import Any

from .registry import Frame, FrameGraph, get_active_frame_graph

_SNAPSHOT_ISSUE_CODES: tuple[str, ...] = (
    "unknown_child_ref",
    "parent_child_mismatch",
    "duplicate_parent_claim",
    "cycle_detected",
    "unreachable_registered_frame",
)


@dataclass(frozen=True)
class SnapshotIssue:
    """Structural issue discovered while snapshotting a frame graph.

    Notes
    -----
    Snapshot issues are diagnostic rows. They make inconsistent graph state
    explicit without hiding the partial topology that could still be read.

    Examples
    --------
    >>> from tal.frames.snapshot import SnapshotIssue
    >>> SnapshotIssue(code="unreachable_registered_frame", frame_id="camera").frame_id
    'camera'
    """

    code: str
    frame_id: str | None = None
    related_frame_id: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class FrameSnapshot:
    """Immutable diagnostic view of a ``FrameGraph`` topology.

    Notes
    -----
    A snapshot stores frame ids, parent edges, child lookup rows, roots, seeds,
    and any structural issues observed while traversing the graph.

    Examples
    --------
    >>> from tal.frames import FrameGraph, snapshot_from_seeds
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> _ = graph.get_or_create_frame("base", parent=world)
    >>> snapshot_from_seeds(("world",), graph=graph).root_ids
    ('world',)
    """

    seed_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    root_ids: tuple[str, ...]
    parent_edges: tuple[tuple[str, str], ...]
    parent_by_node: tuple[tuple[str, str | None], ...]
    children_by_node: tuple[tuple[str, tuple[str, ...]], ...]
    issues: tuple[SnapshotIssue, ...]


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<unrepr:{type(value).__name__}>"


def _issue(
    code: str,
    *,
    frame_id: str | None = None,
    related_frame_id: str | None = None,
    detail: str = "",
) -> SnapshotIssue:
    if code not in _SNAPSHOT_ISSUE_CODES:
        raise ValueError(f"snapshot_from_seeds: unsupported issue code {code!r}.")
    return SnapshotIssue(code=code, frame_id=frame_id, related_frame_id=related_frame_id, detail=detail)


def _require_graph(graph: FrameGraph | None, *, owner: str) -> FrameGraph:
    if graph is None:
        return get_active_frame_graph()
    if isinstance(graph, FrameGraph):
        return graph
    raise TypeError(f"{owner}: graph must be FrameGraph or None, got {type(graph).__name__}.")


def _normalize_seed_name(seed: str, *, owner: str) -> str:
    if not isinstance(seed, str):
        raise TypeError(f"{owner}: seed names must be non-empty strings.")
    cleaned = seed.strip()
    if cleaned:
        return cleaned
    raise ValueError(f"{owner}: seed names must be non-empty strings.")


def _normalize_node_id(key: Any, frame: Frame) -> str:
    if isinstance(key, str):
        cleaned = key.strip()
        if cleaned:
            return cleaned
    if isinstance(frame.id, str):
        cleaned = frame.id.strip()
        if cleaned:
            return cleaned
    return f"invalid_id:{_safe_repr(key)}"


def _graph_nodes(graph: FrameGraph) -> tuple[tuple[str, Frame], ...]:
    items: list[tuple[str, Frame]] = []
    for key, value in graph._frames.items():
        if not isinstance(value, Frame):
            continue
        items.append((_normalize_node_id(key, value), value))
    items.sort(key=lambda item: item[0])
    return tuple(items)


def _build_obj_index(nodes: tuple[tuple[str, Frame], ...]) -> dict[int, str]:
    out: dict[int, str] = {}
    for node_id, frame in nodes:
        out.setdefault(id(frame), node_id)
    return out


def _require_registered_seed_name(seed_id: str, *, graph: FrameGraph, owner: str) -> str:
    resolved = graph._frames.get(seed_id)
    if resolved is None:
        raise ValueError(f"{owner}: seed frame {seed_id!r} not found.")
    if not isinstance(resolved, Frame):
        raise ValueError(f"{owner}: seed frame {seed_id!r} is not a registered Frame object.")
    if resolved._graph is not graph:
        raise ValueError(f"{owner}: seed frame {seed_id!r} is not bound to the target FrameGraph.")
    if not isinstance(resolved.id, str) or resolved.id.strip() != seed_id:
        raise ValueError(f"{owner}: seed frame {seed_id!r} is not registered in the target FrameGraph.")
    if graph._frames.get(seed_id) is not resolved:
        raise ValueError(f"{owner}: seed frame {seed_id!r} is not registered in the target FrameGraph.")
    return seed_id


def _resolve_default_seed_ids(graph: FrameGraph, *, owner: str) -> tuple[str, ...]:
    out: list[str] = []
    for key, value in graph._frames.items():
        if not isinstance(key, str):
            raise ValueError(f"{owner}: frame registry contains invalid key {_safe_repr(key)}.")
        if key.strip() != key or not key:
            raise ValueError(f"{owner}: frame registry contains invalid key {_safe_repr(key)}.")
        if not isinstance(value, Frame):
            raise ValueError(f"{owner}: frame registry entry {key!r} is not a Frame.")
        if value._graph is not graph:
            raise ValueError(f"{owner}: frame registry entry {key!r} is not bound to this FrameGraph.")
        if not isinstance(value.id, str) or value.id.strip() != key:
            raise ValueError(f"{owner}: frame registry entry {key!r} has inconsistent frame id.")
        if graph._frames.get(key) is not value:
            raise ValueError(f"{owner}: frame registry entry {key!r} is not registered consistently.")
        out.append(key)
    return tuple(sorted(out))


def _resolve_seed_ids(
    seeds: tuple[Frame | str, ...] | None,
    *,
    graph: FrameGraph,
    obj_index: dict[int, str],
    owner: str,
) -> tuple[str, ...]:
    if seeds is None:
        return _resolve_default_seed_ids(graph, owner=owner)
    if not isinstance(seeds, tuple):
        raise TypeError(f"{owner}: seeds must be a tuple[Frame | str, ...] or None.")
    out: set[str] = set()
    for seed in seeds:
        if isinstance(seed, str):
            seed_id = _normalize_seed_name(seed, owner=owner)
            out.add(_require_registered_seed_name(seed_id, graph=graph, owner=owner))
            continue
        if isinstance(seed, Frame):
            if not isinstance(seed._graph, FrameGraph):
                raise ValueError(f"{owner}: seed frame is not bound to a valid FrameGraph.")
            if seed._graph is not graph:
                raise ValueError(f"{owner}: seed frame {seed!r} belongs to a different FrameGraph.")
            node_id = obj_index.get(id(seed))
            if node_id is None or graph._frames.get(node_id) is not seed:
                raise ValueError(f"{owner}: seed frame {seed!r} is not registered in the target FrameGraph.")
            out.add(node_id)
            continue
        raise TypeError(f"{owner}: seeds must contain only Frame or str, got {type(seed).__name__}.")
    return tuple(sorted(out))


def _build_parent_by_node(
    nodes: tuple[tuple[str, Frame], ...],
    *,
    obj_index: dict[int, str],
    issues: list[SnapshotIssue],
) -> dict[str, str | None]:
    parent_by_node: dict[str, str | None] = {}
    for node_id, frame in nodes:
        parent = frame.parent
        if parent is None:
            parent_by_node[node_id] = None
            continue
        parent_id = obj_index.get(id(parent))
        if parent_id is None:
            issues.append(
                _issue(
                    "parent_child_mismatch",
                    frame_id=node_id,
                    related_frame_id=parent.id if isinstance(parent.id, str) else None,
                    detail="parent reference is not registered in this graph",
                )
            )
            parent_by_node[node_id] = None
            continue
        parent_by_node[node_id] = parent_id
    return parent_by_node


def _build_children_and_claims(
    nodes: tuple[tuple[str, Frame], ...],
    *,
    obj_index: dict[int, str],
    parent_by_node: dict[str, str | None],
    issues: list[SnapshotIssue],
) -> tuple[dict[str, tuple[str, ...]], dict[str, set[str]]]:
    children_by_node: dict[str, tuple[str, ...]] = {}
    parent_claims: dict[str, set[str]] = {}
    for parent_id, frame in nodes:
        child_ids: set[str] = set()
        for child in frame.children:
            child_id = obj_index.get(id(child))
            if child_id is None:
                issues.append(
                    _issue(
                        "unknown_child_ref",
                        frame_id=parent_id,
                        related_frame_id=child.id if isinstance(child.id, str) else None,
                        detail="child reference is not registered in this graph",
                    )
                )
                continue
            child_ids.add(child_id)
            parent_claims.setdefault(child_id, set()).add(parent_id)
            if parent_by_node.get(child_id) != parent_id:
                issues.append(
                    _issue(
                        "parent_child_mismatch",
                        frame_id=parent_id,
                        related_frame_id=child_id,
                        detail=f"child parent points to {parent_by_node.get(child_id)!r}",
                    )
                )
        children_by_node[parent_id] = tuple(sorted(child_ids))
    return children_by_node, parent_claims


def _append_duplicate_parent_claims(parent_claims: dict[str, set[str]], *, issues: list[SnapshotIssue]) -> None:
    for child_id, claims in sorted(parent_claims.items()):
        if len(claims) <= 1:
            continue
        parents = ", ".join(sorted(claims))
        issues.append(
            _issue(
                "duplicate_parent_claim",
                frame_id=child_id,
                detail=f"claimed by multiple parents: {parents}",
            )
        )


def _append_cycle_issues(parent_by_node: dict[str, str | None], *, issues: list[SnapshotIssue]) -> None:
    visited: set[str] = set()
    for node_id in sorted(parent_by_node):
        if node_id in visited:
            continue
        trail: list[str] = []
        seen: dict[str, int] = {}
        current: str | None = node_id
        while current is not None and current not in visited:
            if current in seen:
                for cycle_node in trail[seen[current] :]:
                    issues.append(_issue("cycle_detected", frame_id=cycle_node))
                break
            seen[current] = len(trail)
            trail.append(current)
            current = parent_by_node.get(current)
        visited.update(trail)


def _collect_reachable(
    seed_ids: tuple[str, ...],
    *,
    parent_by_node: dict[str, str | None],
    children_by_node: dict[str, tuple[str, ...]],
) -> set[str]:
    queue: deque[str] = deque(seed_ids)
    reachable: set[str] = set(seed_ids)
    while queue:
        node_id = queue.popleft()
        parent = parent_by_node.get(node_id)
        if parent is not None and parent not in reachable:
            reachable.add(parent)
            queue.append(parent)
        for child in children_by_node.get(node_id, ()):
            if child in reachable:
                continue
            reachable.add(child)
            queue.append(child)
    return reachable


def _append_unreachable_issues(
    *,
    all_node_ids: tuple[str, ...],
    reachable: set[str],
    issues: list[SnapshotIssue],
) -> None:
    for node_id in all_node_ids:
        if node_id not in reachable:
            issues.append(_issue("unreachable_registered_frame", frame_id=node_id))


def _select_domain_ids(
    *,
    seeds: tuple[Frame | str, ...] | None,
    include_unreachable: bool,
    all_node_ids: tuple[str, ...],
    seed_ids: tuple[str, ...],
    parent_by_node: dict[str, str | None],
    children_by_node: dict[str, tuple[str, ...]],
    issues: list[SnapshotIssue],
) -> tuple[str, ...]:
    if seeds is None:
        return all_node_ids
    reachable = _collect_reachable(seed_ids, parent_by_node=parent_by_node, children_by_node=children_by_node)
    if include_unreachable:
        _append_unreachable_issues(all_node_ids=all_node_ids, reachable=reachable, issues=issues)
        return all_node_ids
    return tuple(node_id for node_id in all_node_ids if node_id in reachable)


def _filter_domain_rows(
    *,
    domain_ids: tuple[str, ...],
    parent_by_node: dict[str, str | None],
    children_by_node: dict[str, tuple[str, ...]],
    issues: list[SnapshotIssue],
) -> tuple[dict[str, str | None], dict[str, tuple[str, ...]], list[SnapshotIssue]]:
    domain_set = set(domain_ids)
    filtered_parent = {
        node_id: (parent if parent in domain_set else None)
        for node_id, parent in parent_by_node.items()
        if node_id in domain_set
    }
    filtered_children = {
        node_id: tuple(child for child in children_by_node.get(node_id, ()) if child in domain_set)
        for node_id in domain_ids
    }
    filtered_issues = [
        row
        for row in issues
        if row.frame_id is None or row.frame_id in domain_set or row.code == "unreachable_registered_frame"
    ]
    return filtered_parent, filtered_children, filtered_issues


def _finalize_snapshot(
    *,
    seed_ids: tuple[str, ...],
    node_ids: tuple[str, ...],
    parent_by_node: dict[str, str | None],
    children_by_node: dict[str, tuple[str, ...]],
    issues: list[SnapshotIssue],
) -> FrameSnapshot:
    parent_rows = tuple((node_id, parent_by_node.get(node_id)) for node_id in node_ids)
    children_rows = tuple((node_id, tuple(sorted(children_by_node.get(node_id, ())))) for node_id in node_ids)
    roots = tuple(node_id for node_id, parent in parent_rows if parent is None)
    edges = tuple(sorted((child, parent) for child, parent in parent_rows if parent is not None))
    issue_rows = tuple(
        sorted(
            issues,
            key=lambda row: (
                row.code,
                row.frame_id or "",
                row.related_frame_id or "",
                row.detail,
            ),
        )
    )
    return FrameSnapshot(
        seed_ids=seed_ids,
        node_ids=node_ids,
        root_ids=roots,
        parent_edges=edges,
        parent_by_node=parent_rows,
        children_by_node=children_rows,
        issues=issue_rows,
    )


def _snapshot_from_seeds_impl(
    *,
    seeds: tuple[Frame | str, ...] | None,
    resolved_graph: FrameGraph,
    include_unreachable: bool,
    owner: str,
) -> FrameSnapshot:
    nodes = _graph_nodes(resolved_graph)
    obj_index = _build_obj_index(nodes)
    seed_ids = _resolve_seed_ids(seeds, graph=resolved_graph, obj_index=obj_index, owner=owner)
    issues: list[SnapshotIssue] = []
    parent_by_node = _build_parent_by_node(nodes, obj_index=obj_index, issues=issues)
    children_by_node, parent_claims = _build_children_and_claims(
        nodes,
        obj_index=obj_index,
        parent_by_node=parent_by_node,
        issues=issues,
    )
    _append_duplicate_parent_claims(parent_claims, issues=issues)
    _append_cycle_issues(parent_by_node, issues=issues)
    all_node_ids = tuple(node_id for node_id, _ in nodes)
    domain_ids = _select_domain_ids(
        seeds=seeds,
        include_unreachable=include_unreachable,
        all_node_ids=all_node_ids,
        seed_ids=seed_ids,
        parent_by_node=parent_by_node,
        children_by_node=children_by_node,
        issues=issues,
    )
    filtered_parent, filtered_children, filtered_issues = _filter_domain_rows(
        domain_ids=domain_ids,
        parent_by_node=parent_by_node,
        children_by_node=children_by_node,
        issues=issues,
    )
    return _finalize_snapshot(
        seed_ids=seed_ids,
        node_ids=domain_ids,
        parent_by_node=filtered_parent,
        children_by_node=filtered_children,
        issues=filtered_issues,
    )


def snapshot_from_seeds(
    seeds: tuple[Frame | str, ...] | None = None,
    *,
    graph: FrameGraph | None = None,
    include_unreachable: bool = True,
) -> FrameSnapshot:
    """Build a frame-graph topology snapshot.

    Parameters
    ----------
    seeds : object
        Optional seed list of frame ids or Frame objects.
    graph : FrameGraph
        Optional graph override.
    include_unreachable : bool
        Include unreachable nodes when True.

    Returns
    -------
    FrameSnapshot

    Notes
    -----
    Collects structural issues (cycles, mismatches, and unreachable nodes) fail-closed.

    Examples
    --------
    >>> from tal.frames import FrameGraph, snapshot_from_seeds
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> _ = graph.get_or_create_frame("base", parent=world)
    >>> snapshot_from_seeds(("world",), graph=graph).node_ids
    ('base', 'world')
    """
    owner = "snapshot_from_seeds"
    resolved_graph = _require_graph(graph, owner=owner)
    return _snapshot_from_seeds_impl(
        seeds=seeds,
        resolved_graph=resolved_graph,
        include_unreachable=include_unreachable,
        owner=owner,
    )


def render_snapshot_ascii(snapshot: FrameSnapshot) -> str:
    """Render a human-readable ASCII summary of a ``FrameSnapshot``.

    Parameters
    ----------
    snapshot : FrameSnapshot
        Snapshot to render.

    Returns
    -------
    str
        Multi-line ASCII report containing seeds, nodes, edges, and issues.

    Notes
    -----
    The output is intended for diagnostics and tests, not as a stable storage
    format.

    Examples
    --------
    >>> from tal.frames import FrameGraph, render_snapshot_ascii, snapshot_from_seeds
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> _ = graph.get_or_create_frame("base", parent=world)
    >>> "base" in render_snapshot_ascii(snapshot_from_seeds(("world",), graph=graph))
    True
    """
    owner = "render_snapshot_ascii"
    if not isinstance(snapshot, FrameSnapshot):
        raise TypeError(f"{owner}: snapshot must be FrameSnapshot, got {type(snapshot).__name__}.")
    children_map = dict(snapshot.children_by_node)
    lines = [
        "FrameSnapshot",
        f"Seeds: {', '.join(snapshot.seed_ids) if snapshot.seed_ids else '(none)'}",
        "Nodes:",
    ]
    for node_id, parent_id in snapshot.parent_by_node:
        children = ", ".join(children_map.get(node_id, ()))
        parent_label = parent_id if parent_id is not None else "-"
        lines.append(f"- {node_id} parent={parent_label} children=[{children}]")
    lines.append("Edges:")
    if snapshot.parent_edges:
        for child_id, parent_id in snapshot.parent_edges:
            lines.append(f"- {parent_id} -> {child_id}")
    else:
        lines.append("- (none)")
    lines.append("Issues:")
    if snapshot.issues:
        for issue in snapshot.issues:
            lines.append(
                f"- {issue.code} frame={issue.frame_id or '-'} related={issue.related_frame_id or '-'} detail={issue.detail}"
            )
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def snapshot_to_networkx(snapshot: FrameSnapshot) -> Any:
    """Convert a ``FrameSnapshot`` to a ``networkx.DiGraph``.

    Parameters
    ----------
    snapshot : FrameSnapshot
        Snapshot to convert.

    Returns
    -------
    Any
        ``networkx.DiGraph`` with frame ids as nodes and parent-to-child edges.

    Notes
    -----
    ``networkx`` is imported lazily. The graph stores ``seed_ids`` and
    serialized issue rows in ``graph.graph`` metadata.

    Examples
    --------
    >>> from tal.frames import FrameGraph, snapshot_from_seeds, snapshot_to_networkx
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> _ = graph.get_or_create_frame("base", parent=world)
    >>> nx_graph = snapshot_to_networkx(snapshot_from_seeds(("world",), graph=graph))
    >>> "base" in nx_graph.nodes
    True
    """
    owner = "snapshot_to_networkx"
    if not isinstance(snapshot, FrameSnapshot):
        raise TypeError(f"{owner}: snapshot must be FrameSnapshot, got {type(snapshot).__name__}.")
    try:
        nx = importlib.import_module("networkx")
    except ImportError as exc:
        raise ImportError("snapshot_to_networkx: networkx is required") from exc
    graph = nx.DiGraph()
    for node_id in snapshot.node_ids:
        graph.add_node(node_id)
    for child_id, parent_id in snapshot.parent_edges:
        graph.add_edge(parent_id, child_id)
    graph.graph["seed_ids"] = snapshot.seed_ids
    graph.graph["issues"] = tuple(
        {
            "code": row.code,
            "frame_id": row.frame_id,
            "related_frame_id": row.related_frame_id,
            "detail": row.detail,
        }
        for row in snapshot.issues
    )
    return graph
