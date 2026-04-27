from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from .registry import Frame, FrameGraph, get_active_frame_graph
from .snapshot import FrameSnapshot, snapshot_from_seeds, snapshot_to_networkx
from .topology import FramePath, find_path

_VALID_LAYOUTS: tuple[str, ...] = ("spring", "kamada_kawai", "circular", "shell", "planar")


@dataclass(frozen=True)
class FrameGraphDrawOptions:
    """Rendering options for :func:`tal.frames.draw_frame_graph`.

    Notes
    -----
    Options configure layout selection and matplotlib/networkx styling only;
    they do not change the frame graph or snapshot topology.

    Examples
    --------
    >>> from tal.frames.visualization import FrameGraphDrawOptions
    >>> opts = FrameGraphDrawOptions(layout="circular", with_labels=False)
    >>> (opts.layout, opts.with_labels)
    ('circular', False)
    """

    layout: str = "spring"
    with_labels: bool = True
    include_legend: bool = True
    node_size: int = 900
    font_size: int = 9
    edge_width: float = 1.4
    path_edge_width: float = 2.8
    arrowsize: int = 18
    arrowstyle: str = "-|>"
    node_color: str = "#D1D5DB"
    node_edge_color: str = "#4B5563"
    edge_color: str = "#9CA3AF"
    path_forward_color: str = "#2E7D32"
    path_reverse_color: str = "#EF6C00"


def _coerce_draw_options(opts: FrameGraphDrawOptions | None, *, owner: str) -> FrameGraphDrawOptions:
    if opts is None:
        return FrameGraphDrawOptions()
    if isinstance(opts, FrameGraphDrawOptions):
        return opts
    raise TypeError(f"{owner}: opts must be FrameGraphDrawOptions or None, got {type(opts).__name__}.")


def _resolve_graph(graph: FrameGraph | None, *, owner: str) -> FrameGraph:
    if graph is None:
        return get_active_frame_graph()
    if isinstance(graph, FrameGraph):
        return graph
    raise TypeError(f"{owner}: graph must be FrameGraph or None, got {type(graph).__name__}.")


def _import_networkx(*, owner: str) -> Any:
    try:
        return importlib.import_module("networkx")
    except ImportError as exc:
        raise ImportError(f"{owner}: networkx is required.") from exc


def _import_matplotlib_pyplot(*, owner: str) -> Any:
    try:
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError(f"{owner}: matplotlib is required.") from exc


def _resolve_endpoint(value: Frame | str, *, graph: FrameGraph, owner: str, arg: str) -> Frame:
    if isinstance(value, Frame):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{owner}: {arg} must be Frame or str, got {type(value).__name__}.")
    frame_id = value.strip()
    if not frame_id:
        raise ValueError(f"{owner}: {arg} must be non-empty frame id string.")
    frame = graph.get_frame(frame_id)
    if frame is None:
        raise ValueError(f"{owner}: {arg} frame {frame_id!r} not found.")
    return frame


def _resolve_highlight_path(
    *,
    graph: FrameGraph,
    path: FramePath | None,
    src: Frame | str | None,
    dst: Frame | str | None,
    owner: str,
) -> FramePath | None:
    if path is not None and (src is not None or dst is not None):
        raise ValueError(f"{owner}: path and src/dst are mutually exclusive.")
    if path is not None:
        if isinstance(path, FramePath):
            return path
        raise TypeError(f"{owner}: path must be FramePath, got {type(path).__name__}.")
    if src is None and dst is None:
        return None
    if src is None or dst is None:
        raise ValueError(f"{owner}: src and dst must be provided together.")
    src_frame = _resolve_endpoint(src, graph=graph, owner=owner, arg="src")
    dst_frame = _resolve_endpoint(dst, graph=graph, owner=owner, arg="dst")
    return find_path(src_frame, dst_frame)


def _require_path_within_snapshot(path: FramePath, *, snapshot: FrameSnapshot, owner: str) -> None:
    domain = set(snapshot.node_ids)
    for node in path.nodes:
        if node.id not in domain:
            raise ValueError(f"{owner}: path node {node.id!r} is outside rendered snapshot domain.")


def _split_path_edges(path: FramePath | None) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    if path is None:
        return (), ()
    forward: list[tuple[str, str]] = []
    reverse: list[tuple[str, str]] = []
    for step in path.steps:
        edge = (step.parent.id, step.child.id)
        if step.invert:
            forward.append(edge)
            continue
        reverse.append(edge)
    return tuple(forward), tuple(reverse)


def _resolve_layout_positions(*, nx: Any, nx_graph: Any, layout: str, owner: str) -> dict[str, Any]:
    if layout == "spring":
        return nx.spring_layout(nx_graph)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(nx_graph)
    if layout == "circular":
        return nx.circular_layout(nx_graph)
    if layout == "shell":
        return nx.shell_layout(nx_graph)
    if layout == "planar":
        try:
            return nx.planar_layout(nx_graph)
        except Exception:
            return nx.spring_layout(nx_graph)
    expected = ", ".join(_VALID_LAYOUTS)
    raise ValueError(f"{owner}: opts.layout must be one of ({expected}); got {layout!r}.")


def _resolve_axes(ax: Any | None, *, plt: Any) -> Any:
    if ax is not None:
        return ax
    _, axis = plt.subplots()
    return axis


def _draw_edges(
    *,
    nx: Any,
    nx_graph: Any,
    pos: dict[str, Any],
    all_edges: list[tuple[str, str]],
    forward_edges: tuple[tuple[str, str], ...],
    reverse_edges: tuple[tuple[str, str], ...],
    ax: Any,
    opts: FrameGraphDrawOptions,
) -> None:
    base = dict(ax=ax, arrows=True, arrowstyle=opts.arrowstyle, arrowsize=opts.arrowsize, connectionstyle="arc3,rad=0.0")
    if all_edges:
        nx.draw_networkx_edges(
            nx_graph,
            pos,
            edgelist=all_edges,
            edge_color=opts.edge_color,
            width=opts.edge_width,
            **base,
        )
    if forward_edges:
        nx.draw_networkx_edges(
            nx_graph,
            pos,
            edgelist=list(forward_edges),
            edge_color=opts.path_forward_color,
            width=opts.path_edge_width,
            **base,
        )
    if reverse_edges:
        nx.draw_networkx_edges(
            nx_graph,
            pos,
            edgelist=list(reverse_edges),
            edge_color=opts.path_reverse_color,
            width=opts.path_edge_width,
            style="dashed",
            **base,
        )


def _draw_nodes_and_labels(
    *,
    nx: Any,
    nx_graph: Any,
    pos: dict[str, Any],
    snapshot: FrameSnapshot,
    ax: Any,
    opts: FrameGraphDrawOptions,
) -> None:
    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        nodelist=list(snapshot.node_ids),
        node_color=opts.node_color,
        edgecolors=opts.node_edge_color,
        node_size=opts.node_size,
        linewidths=1.4,
        ax=ax,
    )
    if opts.with_labels:
        labels = {node_id: node_id for node_id in snapshot.node_ids}
        nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=opts.font_size, ax=ax)


def _draw_legend(
    *,
    ax: Any,
    opts: FrameGraphDrawOptions,
    forward_edges: tuple[tuple[str, str], ...],
    reverse_edges: tuple[tuple[str, str], ...],
) -> None:
    if not opts.include_legend:
        return
    if not forward_edges and not reverse_edges:
        return
    if forward_edges:
        ax.plot([], [], color=opts.path_forward_color, linewidth=opts.path_edge_width, label="path forward (parent->child)")
    if reverse_edges:
        ax.plot(
            [],
            [],
            color=opts.path_reverse_color,
            linewidth=opts.path_edge_width,
            linestyle="dashed",
            label="path reverse (child->parent)",
        )
    ax.legend(loc="best", frameon=False)


def draw_frame_graph(
    *,
    graph: FrameGraph | None = None,
    seeds: tuple[Frame | str, ...] | None = None,
    include_unreachable: bool = True,
    path: FramePath | None = None,
    src: Frame | str | None = None,
    dst: Frame | str | None = None,
    opts: FrameGraphDrawOptions | None = None,
    ax: Any | None = None,
) -> Any:
    """Draw a frame-graph snapshot and optionally highlight a resolved path.

    Parameters
    ----------
    graph : FrameGraph | None, optional
        Optional frame graph to draw. When omitted, uses the active
        context-local frame graph.
    seeds : tuple[Frame | str, ...] | None, optional
        Optional root frame ids used to limit snapshot traversal.
    include_unreachable : bool, optional
        When ``True``, include disconnected frames in the rendered snapshot.
    path : FramePath | None, optional
        Filesystem path used by this IO operation.
    src : Frame | str | None, optional
        Source frame id/object.
    dst : Frame | str | None, optional
        Destination frame id/object.
    opts : FrameGraphDrawOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``FrameGraphDrawOptions`` key fields: ``layout`` (default 'spring'), ``with_labels`` (default True), ``include_legend`` (default True), ``node_size`` (default 900).
    ax : Any | None, optional
        Optional matplotlib axis used for rendering output.

    Returns
    -------
    Any
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
    >>> from tal.frames import FrameGraph
    >>> from tal.frames.visualization import FrameGraphDrawOptions, draw_frame_graph
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> _ = graph.get_or_create_frame("base", parent=world)
    >>> try:
    ...     ax = draw_frame_graph(graph=graph, seeds=("world",), opts=FrameGraphDrawOptions(layout="circular", include_legend=False))
    ... except ImportError:
    ...     ax = None
    >>> ax is None or hasattr(ax, "plot")
    True
    """
    owner = "draw_frame_graph"
    options = _coerce_draw_options(opts, owner=owner)
    resolved_graph = _resolve_graph(graph, owner=owner)
    snapshot = snapshot_from_seeds(seeds, graph=resolved_graph, include_unreachable=include_unreachable)
    highlight_path = _resolve_highlight_path(graph=resolved_graph, path=path, src=src, dst=dst, owner=owner)
    if highlight_path is not None:
        _require_path_within_snapshot(highlight_path, snapshot=snapshot, owner=owner)
    forward_edges, reverse_edges = _split_path_edges(highlight_path)
    nx = _import_networkx(owner=owner)
    plt = _import_matplotlib_pyplot(owner=owner)
    nx_graph = snapshot_to_networkx(snapshot)
    positions = _resolve_layout_positions(nx=nx, nx_graph=nx_graph, layout=options.layout, owner=owner)
    axis = _resolve_axes(ax, plt=plt)
    all_edges = list(nx_graph.edges())
    _draw_edges(
        nx=nx,
        nx_graph=nx_graph,
        pos=positions,
        all_edges=all_edges,
        forward_edges=forward_edges,
        reverse_edges=reverse_edges,
        ax=axis,
        opts=options,
    )
    _draw_nodes_and_labels(nx=nx, nx_graph=nx_graph, pos=positions, snapshot=snapshot, ax=axis, opts=options)
    _draw_legend(ax=axis, opts=options, forward_edges=forward_edges, reverse_edges=reverse_edges)
    axis.set_axis_off()
    return axis
