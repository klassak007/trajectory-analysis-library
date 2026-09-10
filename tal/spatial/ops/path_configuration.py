from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tal.frames import Frame, FrameGraph, get_active_frame_graph

from ..association import SpatialAssociationPlan, resolve_passive_association

if TYPE_CHECKING:
    from ..path_solve import PathSolveOptions


@dataclass(frozen=True)
class PathConfiguration:
    options: PathSolveOptions
    graph: FrameGraph | None
    association: SpatialAssociationPlan = field(
        default_factory=lambda: SpatialAssociationPlan(None),
    )


@dataclass(frozen=True)
class SelectedPathConfiguration:
    options: PathSolveOptions
    graph: FrameGraph


@dataclass(frozen=True)
class ResolvedPathEndpointPlan:
    """Selected graph and registered endpoints for one path request."""

    configuration: SelectedPathConfiguration
    association: SpatialAssociationPlan
    source: Frame
    destination: Frame

    @property
    def is_identity(self) -> bool:
        return self.source is self.destination


def resolve_path_configuration(
    opts: object | None,
    *,
    graph: FrameGraph | None,
    owner: str,
    participants: tuple[object, ...] = (),
) -> PathConfiguration:
    """Resolve graph sugar without changing the options instance or its policies."""
    from ..path_solve import _coerce_path_solve_options

    options = _coerce_path_solve_options(opts, owner=owner)
    if graph is not None:
        if not isinstance(graph, FrameGraph):
            raise TypeError(f"{owner}: graph must be FrameGraph or None.")
        if options.graph is not None:
            raise ValueError(f"{owner}: graph and opts.graph cannot both be supplied.")
    if options.graph is not None and not isinstance(options.graph, FrameGraph):
        raise TypeError(f"{owner}: opts.graph must be FrameGraph or None.")
    selected = options.graph if graph is None else graph
    association = (
        SpatialAssociationPlan(selected)
        if selected is not None
        else resolve_passive_association(participants, owner=owner)
    )
    return PathConfiguration(options, selected, association)


def _require_graph_binding(frame: Frame, *, owner: str, arg: str) -> FrameGraph:
    graph = frame._graph
    if isinstance(graph, FrameGraph):
        return graph
    raise ValueError(f"{owner}: {arg} frame is not bound to a valid FrameGraph.")


def require_strict_path_policy(strict: object, *, owner: str) -> None:
    """Validate the strict path policy at its consuming boundary."""
    if not isinstance(strict, bool):
        raise TypeError(f"{owner}: opts.strict must be bool, got {type(strict).__name__}.")
    if not strict:
        raise ValueError(f"{owner}: opts.strict=False is not supported.")


def _require_frame_id(value: object, *, owner: str, arg: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{owner}: {arg} must be Frame or non-empty string frame id.")
    cleaned = value.strip()
    if cleaned:
        return cleaned
    raise ValueError(f"{owner}: {arg} must be Frame or non-empty string frame id.")


def resolve_path_endpoint(value: object, *, graph: FrameGraph, owner: str, arg: str) -> Frame:
    """Resolve one strict registered path endpoint in ``graph``."""
    if isinstance(value, Frame):
        if _require_graph_binding(value, owner=owner, arg=arg) is not graph:
            raise ValueError(f"{owner}: {arg} frame belongs to a different FrameGraph.")
        if graph.get_frame(value.id) is value:
            return value
        raise ValueError(f"{owner}: {arg} frame {value.id!r} is not registered in graph.")
    frame_id = _require_frame_id(value, owner=owner, arg=arg)
    resolved = graph.get_frame(frame_id)
    if isinstance(resolved, Frame) and graph.get_frame(resolved.id) is resolved:
        return resolved
    raise ValueError(f"{owner}: {arg} frame {frame_id!r} is not registered in graph.")


def _require_endpoint_graph(value: object, graph: FrameGraph, *, owner: str, arg: str) -> None:
    if isinstance(value, Frame) and _require_graph_binding(value, owner=owner, arg=arg) is not graph:
        raise ValueError(f"{owner}: {arg} frame belongs to a different FrameGraph.")


def resolve_path_graph(
    src: object,
    dst: object,
    *,
    graph: FrameGraph | None,
    remembered_graph: FrameGraph | None = None,
    owner: str,
) -> FrameGraph:
    """Resolve one graph source and enforce Frame endpoint ownership."""
    if graph is not None:
        if not isinstance(graph, FrameGraph):
            raise TypeError(f"{owner}: opts.graph must be FrameGraph or None, got {type(graph).__name__}.")
        _require_endpoint_graph(src, graph, owner=owner, arg="src")
        _require_endpoint_graph(dst, graph, owner=owner, arg="dst")
        return graph
    src_graph = _require_graph_binding(src, owner=owner, arg="src") if isinstance(src, Frame) else None
    dst_graph = _require_graph_binding(dst, owner=owner, arg="dst") if isinstance(dst, Frame) else None
    candidates = tuple(
        candidate
        for candidate in (remembered_graph, src_graph, dst_graph)
        if candidate is not None
    )
    if any(candidate is not candidates[0] for candidate in candidates[1:]):
        raise ValueError(f"{owner}: object and endpoint graphs identify different FrameGraph instances.")
    return candidates[0] if candidates else get_active_frame_graph()


def select_path_graph(
    configuration: PathConfiguration | SelectedPathConfiguration,
    *,
    src,
    dst,
    owner: str,
) -> SelectedPathConfiguration:
    """Pin endpoint/active graph selection for a non-identity orchestration request."""
    if isinstance(configuration, SelectedPathConfiguration):
        return configuration
    graph = resolve_path_graph(
        src,
        dst,
        graph=configuration.graph,
        remembered_graph=configuration.association.graph,
        owner=owner,
    )
    return SelectedPathConfiguration(configuration.options, graph)


def resolve_path_endpoint_plan(
    configuration: PathConfiguration | SelectedPathConfiguration,
    *,
    src: object,
    dst: object,
    owner: str,
) -> ResolvedPathEndpointPlan:
    """Select one graph and resolve both endpoints before path-only policy work."""
    selected = select_path_graph(configuration, src=src, dst=dst, owner=owner)
    source = resolve_path_endpoint(src, graph=selected.graph, owner=owner, arg="src")
    destination = resolve_path_endpoint(dst, graph=selected.graph, owner=owner, arg="dst")
    return ResolvedPathEndpointPlan(
        selected,
        SpatialAssociationPlan(selected.graph),
        source,
        destination,
    )


def select_identity_graph(
    configuration: PathConfiguration | SelectedPathConfiguration,
    *,
    src: str,
    dst: Frame | str,
    owner: str,
) -> FrameGraph | None:
    """Apply the object-centered identity graph-consistency matrix."""
    if isinstance(configuration, SelectedPathConfiguration):
        resolve_path_endpoint(src, graph=configuration.graph, owner=owner, arg="src")
        resolve_path_endpoint(dst, graph=configuration.graph, owner=owner, arg="dst")
        return configuration.graph
    if configuration.graph is not None:
        resolve_path_endpoint(src, graph=configuration.graph, owner=owner, arg="src")
        resolve_path_endpoint(dst, graph=configuration.graph, owner=owner, arg="dst")
        return configuration.graph
    if not isinstance(dst, Frame):
        return configuration.association.graph
    endpoint_graph = _require_graph_binding(dst, owner=owner, arg="dst")
    remembered = configuration.association.graph
    if remembered is not None and remembered is not endpoint_graph:
        raise ValueError(f"{owner}: object and dst belong to different FrameGraph instances.")
    resolve_path_endpoint(dst, graph=endpoint_graph, owner=owner, arg="dst")
    return endpoint_graph


__all__ = [
    "PathConfiguration",
    "ResolvedPathEndpointPlan",
    "SelectedPathConfiguration",
    "require_strict_path_policy",
    "resolve_path_configuration",
    "resolve_path_endpoint",
    "resolve_path_endpoint_plan",
    "resolve_path_graph",
    "select_identity_graph",
    "select_path_graph",
]
