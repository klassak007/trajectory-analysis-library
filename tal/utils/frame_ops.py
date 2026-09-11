from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.core.schema import UNSET, UnsetType, validate_schema
from tal.frames import Frame, FrameGraph, get_active_frame_graph

from .frame_schema import get_frames, set_frames

if TYPE_CHECKING:
    from tal.core.analysis_object import AnalysisObject


def _rewrap_like(source: "AnalysisObject", ds, *, validate: bool) -> "AnalysisObject":
    candidate = validate_schema(ds) if validate else ds
    return source._rewrap_dataset(candidate, validate=validate)


def _normalize_frame_name(value: object, *, owner: str, arg: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{owner}: {arg} must be a non-empty string frame id.")
    cleaned = value.strip()
    if cleaned:
        return cleaned
    raise ValueError(f"{owner}: {arg} must be a non-empty string frame id.")


def _normalize_remap_mapping(mapping: Mapping[object, object], *, owner: str) -> dict[str, str]:
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{owner}: mapping must be a mapping[str, str].")
    out: dict[str, str] = {}
    for raw_src, raw_dst in mapping.items():
        src = _normalize_frame_name(raw_src, owner=owner, arg="mapping key")
        dst = _normalize_frame_name(raw_dst, owner=owner, arg="mapping value")
        existing = out.get(src)
        if existing is not None and existing != dst:
            raise ValueError(f"{owner}: conflicting mapping for source frame {src!r}.")
        out[src] = dst
    if len(set(out.values())) != len(out):
        raise ValueError(f"{owner}: mapping values must be injective.")
    return out


def _remembered_graph(source: AnalysisObject, *, owner: str) -> FrameGraph | None:
    candidate = getattr(source, "graph", None)
    if candidate is None or isinstance(candidate, FrameGraph):
        return candidate
    raise TypeError(f"{owner}: source graph association must be FrameGraph or None.")


def _require_graph_argument(graph: object, *, owner: str) -> FrameGraph | None:
    if graph is None or isinstance(graph, FrameGraph):
        return graph
    raise TypeError(f"{owner}: graph must be FrameGraph or None, got {type(graph).__name__}.")


def _resolve_read_graph(
    source: AnalysisObject,
    graph: FrameGraph | None,
    *,
    owner: str,
) -> FrameGraph:
    if graph is not None:
        return graph
    remembered = _remembered_graph(source, owner=owner)
    return remembered if remembered is not None else get_active_frame_graph()


def _require_registered_frame_object(
    value: object,
    *,
    graph: FrameGraph,
    frame_id: str,
    owner: str,
    role: str,
) -> Frame:
    if not isinstance(value, Frame):
        raise TypeError(f"{owner}: {role} frame {frame_id!r} is not a registered Frame object in graph.")
    if value.id != frame_id:
        raise ValueError(f"{owner}: {role} frame {frame_id!r} is not a registered Frame object in graph.")
    graph._assert_owned(value, context=owner)
    return value


def _resolve_frame_id(
    graph: FrameGraph,
    frame_id: str | None,
    *,
    owner: str,
    role: str,
) -> Frame | None:
    if frame_id is None:
        return None
    resolved = graph.get_frame(frame_id)
    if resolved is None:
        raise ValueError(f"{owner}: {role} frame {frame_id!r} not found in graph.")
    return _require_registered_frame_object(
        resolved,
        graph=graph,
        frame_id=frame_id,
        owner=owner,
        role=role,
    )


def frame_ids(ao: object) -> tuple[str | None, str | None]:
    """Return ``(parent, child)`` frame ids from an AnalysisObject schema.

    Parameters
    ----------
    ao : object
        AnalysisObject-like input value.

    Returns
    -------
    tuple[str | None, str | None]
        Tuple of output values produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.utils.frame_ops import frame_ids, frame_retag
    >>> ao = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> frame_ids(frame_retag(ao, parent="world", child="tool"))
    ('world', 'tool')
    """
    owner = "frames.ids"
    source = coerce_analysis_object_input(ao, owner=owner)
    return get_frames(analysis_object_dataset(source))


def frame_retag(
    ao: object,
    *,
    parent: str | None | UnsetType = UNSET,
    child: str | None | UnsetType = UNSET,
    validate: bool = True,
) -> "AnalysisObject":
    """Return a copy with updated parent/child frame metadata tags.

    Parameters
    ----------
    ao : object
        AnalysisObject-like input value.
    parent : str | None | UnsetType, optional
        Parent frame identifier or ``None`` to clear.
    child : str | None | UnsetType, optional
        Child frame identifier or ``None`` to clear.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.utils.frame_ops import frame_retag
    >>> ao = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> frame_retag(ao, parent="world", child="tool").frames.ids()
    ('world', 'tool')
    """
    owner = "frames.retag"
    source = coerce_analysis_object_input(ao, owner=owner)
    updated = set_frames(analysis_object_dataset(source), parent=parent, child=child, validate=False)
    return _rewrap_like(source, updated, validate=validate)


def frame_remap_ids(
    ao: object,
    mapping: Mapping[str, str],
    *,
    validate: bool = True,
) -> "AnalysisObject":
    """Return a copy with frame ids remapped according to ``mapping``.

    Parameters
    ----------
    ao : object
        AnalysisObject-like input value.
    mapping : Mapping[str, str]
        Injective mapping from source identifiers to replacement identifiers.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.utils.frame_ops import frame_remap_ids, frame_retag
    >>> ao = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> tagged = frame_retag(ao, parent="world", child="tool")
    >>> frame_remap_ids(tagged, {"world": "map", "tool": "tool_0"}).frames.ids()
    ('map', 'tool_0')
    """
    owner = "frames.remap_ids"
    source = coerce_analysis_object_input(ao, owner=owner)
    normalized = _normalize_remap_mapping(mapping, owner=owner)
    source_ds = analysis_object_dataset(source)
    current_parent, current_child = get_frames(source_ds)
    next_parent = normalized.get(current_parent, current_parent) if current_parent is not None else None
    next_child = normalized.get(current_child, current_child) if current_child is not None else None
    updated = set_frames(source_ds, parent=next_parent, child=next_child, validate=False)
    return _rewrap_like(source, updated, validate=validate)


class FramesAccessor:
    """Frame utility accessor mounted on ``AnalysisObject`` as ``ao.frames``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    def __init__(self, ao: "AnalysisObject") -> None:
        self._ao = ao

    def ids(self) -> tuple[str | None, str | None]:
        """Return ``(parent, child)`` frame ids.

        Returns
        -------
        tuple[str | None, str | None]
            Parent and child frame ids read from AO schema metadata.

        Notes
        -----
        Missing ids are returned as ``None``.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> tagged = ao.frames.retag(parent="world", child="tool")
        >>> tagged.frames.ids()
        ('world', 'tool')
        """
        return frame_ids(self._ao)

    def retag(
        self,
        *,
        parent: str | None | UnsetType = UNSET,
        child: str | None | UnsetType = UNSET,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Set or clear ``parent``/``child`` frame ids on this object.

        Parameters
        ----------
        parent : str | None | UnsetType, optional
            Parent frame identifier or ``None`` to clear.
        child : str | None | UnsetType, optional
            Child frame identifier or ``None`` to clear.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AnalysisObject
            AO with updated frame metadata.

        Notes
        -----
        Retagging changes metadata only. It does not transform spatial payload
        values.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> tagged = ao.frames.retag(parent="world", child="tool")
        >>> tagged.frames.ids()
        ('world', 'tool')
        """
        return frame_retag(self._ao, parent=parent, child=child, validate=validate)

    def remap_ids(
        self,
        mapping: Mapping[str, str],
        *,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Remap frame ids using a one-to-one mapping.

        Parameters
        ----------
        mapping : Mapping[str, str]
            Injective mapping from source identifiers to replacement identifiers.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AnalysisObject
            AO with matching frame ids rewritten according to ``mapping``.

        Notes
        -----
        The mapping must be injective. Frame ids absent from ``mapping`` are
        preserved.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> tagged = ao.frames.retag(parent="world", child="tool")
        >>> tagged.frames.remap_ids({"world": "map", "tool": "tool_0"}).frames.ids()
        ('map', 'tool_0')
        """
        return frame_remap_ids(self._ao, mapping, validate=validate)

    def resolve(
        self,
        graph: FrameGraph | None = None,
    ) -> tuple[Frame | None, Frame | None]:
        """Resolve present frame IDs without mutating graph topology.

        Parameters
        ----------
        graph : FrameGraph or None, optional
            Explicit graph override. Otherwise a remembered spatial graph is
            preferred before the active graph.

        Returns
        -------
        tuple[Frame | None, Frame | None]
            Registered parent and child frames. An absent metadata ID produces
            ``None`` in the corresponding position.

        Raises
        ------
        TypeError
            If ``graph`` is neither a ``FrameGraph`` nor ``None``.
        ValueError
            If a present frame ID is not registered in the selected graph.

        Notes
        -----
        Resolution never creates, attaches, reparents, renames, or removes a
        frame. An explicit graph overrides a remembered wrapper association.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.frames import FrameGraph
        >>> graph = FrameGraph()
        >>> world = graph.get_or_create_frame("world")
        >>> tool = graph.get_or_create_frame("tool", parent=world)
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample", core_dims=(), validate=True,
        ... ).frames.retag(parent="world", child="tool")
        >>> ao.frames.resolve(graph) == (world, tool)
        True
        """
        owner = "frames.resolve"
        selected_arg = _require_graph_argument(graph, owner=owner)
        parent_id, child_id = get_frames(analysis_object_dataset(self._ao))
        if parent_id is None and child_id is None:
            return None, None
        selected = _resolve_read_graph(self._ao, selected_arg, owner=owner)
        parent = _resolve_frame_id(selected, parent_id, owner=owner, role="parent")
        child = _resolve_frame_id(selected, child_id, owner=owner, role="child")
        return parent, child


def install_analysis_object_frames_accessor() -> None:
    """Install the ``ao.frames`` accessor on ``AnalysisObject``.

    Parameters
    ----------
    None
        This callable does not accept user-facing parameters.

    Returns
    -------
    None
        Returns ``None``; side effects are applied through owned state/metadata updates.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    from tal.core.analysis_object import AnalysisObject

    existing = getattr(AnalysisObject, "frames", None)
    if isinstance(existing, property):
        fget = existing.fget
        if fget is not None and getattr(fget, "__module__", "") == __name__:
            return
        raise ValueError(
            "install_analysis_object_frames_accessor: AnalysisObject.frames is already owned by another accessor."
        )
    if existing is not None:
        raise ValueError(
            "install_analysis_object_frames_accessor: AnalysisObject.frames already exists and is not a property."
        )

    def _frames_accessor(self: "AnalysisObject") -> FramesAccessor:
        """Return the ``ao.frames`` accessor bound to this AnalysisObject.

        Returns
        -------
        FramesAccessor
            Frame metadata accessor for this AO.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> ao.frames.retag(parent="world", child="tool").frames.ids()
        ('world', 'tool')
        """
        return FramesAccessor(self)

    AnalysisObject.frames = property(_frames_accessor)  # type: ignore[assignment]


__all__ = [
    "FramesAccessor",
    "frame_ids",
    "frame_remap_ids",
    "frame_retag",
    "install_analysis_object_frames_accessor",
]
