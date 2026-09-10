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


def _validate_on_conflict(value: str, *, owner: str) -> str:
    if value in {"error", "replace"}:
        return value
    raise ValueError(f"{owner}: on_conflict must be 'error' or 'replace'.")


def _normalize_create_missing(value: object, *, owner: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{owner}: create_missing must be bool, got {type(value).__name__}.")


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


def _resolve_graph(graph: FrameGraph | None, *, owner: str) -> FrameGraph:
    if graph is None:
        return get_active_frame_graph()
    if isinstance(graph, FrameGraph):
        return graph
    raise TypeError(f"{owner}: graph must be FrameGraph or None, got {type(graph).__name__}.")


def _resolve_or_create_frame(
    graph: FrameGraph,
    frame_id: str,
    *,
    create_missing: bool,
    owner: str,
    role: str,
) -> Frame:
    resolved: object = graph.get_frame(frame_id)
    if resolved is None and create_missing:
        resolved = graph.get_or_create_frame(frame_id)
    if resolved is None:
        raise ValueError(f"{owner}: {role} frame {frame_id!r} not found in graph.")
    return _require_registered_frame_object(
        resolved,
        graph=graph,
        frame_id=frame_id,
        owner=owner,
        role=role,
    )


def _require_registered_frame_object(
    value: object,
    *,
    graph: FrameGraph,
    frame_id: str,
    owner: str,
    role: str,
) -> Frame:
    if not isinstance(value, Frame):
        raise ValueError(f"{owner}: {role} frame {frame_id!r} is not a registered Frame object in graph.")
    if value._graph is not graph or value.id != frame_id or graph._frames.get(frame_id) is not value:
        raise ValueError(f"{owner}: {role} frame {frame_id!r} is not a registered Frame object in graph.")
    return value


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


def frame_bind(
    ao: object,
    *,
    graph: FrameGraph | None = None,
    create_missing: bool = True,
    on_conflict: str = "error",
) -> tuple[Frame | None, Frame | None]:
    """Bind AO frame ids to concrete ``Frame`` objects.

    Parameters
    ----------
    ao : object
    graph : FrameGraph
        Optional graph override.
    create_missing : bool
        Create missing frames when True.
    on_conflict : str
        Conflict policy ("error" or "replace").

    Returns
    -------
    tuple[Frame | None, Frame | None]

    Notes
    -----
    Uses fail-closed frame id validation and graph registration checks.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.frames import FrameGraph
    >>> ao = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> tagged = ao.frames.retag(parent="world", child="tool")
    >>> parent, child = tagged.frames.bind(graph=FrameGraph(), create_missing=True)
    >>> (parent.id, child.parent.id)
    ('world', 'world')
    """
    owner = "frames.bind"
    source = coerce_analysis_object_input(ao, owner=owner)
    policy = _validate_on_conflict(on_conflict, owner=owner)
    create_flag = _normalize_create_missing(create_missing, owner=owner)
    resolved_graph = _resolve_graph(graph, owner=owner)
    return _frame_bind_impl(
        source=source,
        policy=policy,
        create_flag=create_flag,
        resolved_graph=resolved_graph,
        owner=owner,
    )


def _frame_bind_impl(
    *,
    source: "AnalysisObject",
    policy: str,
    create_flag: bool,
    resolved_graph: FrameGraph,
    owner: str,
) -> tuple[Frame | None, Frame | None]:
    parent_id, child_id = get_frames(analysis_object_dataset(source))
    parent = None
    child = None
    if parent_id is not None:
        parent = _resolve_or_create_frame(
            resolved_graph,
            parent_id,
            create_missing=create_flag,
            owner=owner,
            role="parent",
        )
    if child_id is not None:
        child = _resolve_or_create_frame(
            resolved_graph,
            child_id,
            create_missing=create_flag,
            owner=owner,
            role="child",
        )
    if parent is not None and child is not None and child.parent is not parent:
        try:
            resolved_graph.reparent_frame(child, parent, on_conflict=policy)
        except ValueError as exc:
            raise ValueError(f"{owner}: {exc}") from exc
    return parent, child


def frame_rename(
    ao: object,
    old: str,
    new: str,
    *,
    graph: FrameGraph | None = None,
    on_conflict: str = "error",
    validate: bool = True,
) -> "AnalysisObject":
    """Rename a frame id in AO metadata and optionally in a backing graph.

    Parameters
    ----------
    ao : object
        AnalysisObject-like input value.
    old : str
        Frame/schema rewrite selector used by this operation.
    new : str
        Frame/schema rewrite selector used by this operation.
    graph : FrameGraph | None, optional
        Optional ``FrameGraph`` override used for path resolution.
    on_conflict : str, optional
        Conflict policy controlling behavior when target ids already exist.
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
    >>> from tal.frames import FrameGraph
    >>> from tal.utils.frame_ops import frame_rename
    >>> graph = FrameGraph()
    >>> _ = graph.get_or_create_frame("world")
    >>> _ = graph.get_or_create_frame("tool")
    >>> ao = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> tagged = ao.frames.retag(parent="world", child="tool")
    >>> frame_rename(tagged, "tool", "tool_0", graph=graph).frames.ids()
    ('world', 'tool_0')
    """
    owner = "frames.rename_frame"
    source = coerce_analysis_object_input(ao, owner=owner)
    old_id = _normalize_frame_name(old, owner=owner, arg="old")
    new_id = _normalize_frame_name(new, owner=owner, arg="new")
    policy = _validate_on_conflict(on_conflict, owner=owner)
    resolved_graph = _resolve_graph(graph, owner=owner)
    remapped = frame_remap_ids(source, {old_id: new_id}, validate=validate)
    try:
        resolved_graph.rename_frame(old_id, new_id, on_conflict=policy)
    except ValueError as exc:
        raise ValueError(f"{owner}: {exc}") from exc
    return remapped


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

    def bind(
        self,
        *,
        graph: FrameGraph | None = None,
        create_missing: bool = True,
        on_conflict: str = "error",
    ) -> tuple[Frame | None, Frame | None]:
        """Bind AO frame ids to registered ``Frame`` objects in a ``FrameGraph``.

        Parameters
        ----------
        graph : FrameGraph | None, optional
            Optional ``FrameGraph`` override used for path resolution.
        create_missing : bool, optional
            Whether missing frame identifiers should be created in the target graph.
        on_conflict : str, optional
            Conflict policy controlling behavior when identifiers already exist.

        Returns
        -------
        tuple[Frame | None, Frame | None]
            Bound parent and child frames. Missing ids return ``None`` in their
            corresponding position.

        Notes
        -----
        When both ids are present, the child is parented under the parent in the
        target graph according to ``on_conflict``.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.frames import FrameGraph
        >>> ao = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> tagged = ao.frames.retag(parent="world", child="tool")
        >>> parent, child = tagged.frames.bind(graph=FrameGraph(), create_missing=True)
        >>> (parent.id, child.id)
        ('world', 'tool')
        """
        return frame_bind(
            self._ao,
            graph=graph,
            create_missing=create_missing,
            on_conflict=on_conflict,
        )

    def rename_frame(
        self,
        old: str,
        new: str,
        *,
        graph: FrameGraph | None = None,
        on_conflict: str = "error",
        validate: bool = True,
    ) -> "AnalysisObject":
        """Rename a frame id in this object (and graph when provided).

        Parameters
        ----------
        old : str
            Existing frame id to replace.
        new : str
            Replacement frame id.
        graph : FrameGraph | None, optional
            Optional ``FrameGraph`` override used for path resolution.
        on_conflict : str, optional
            Conflict policy controlling behavior when target ids already exist.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AnalysisObject
            AO with matching frame metadata renamed.

        Notes
        -----
        When ``graph`` is provided, the runtime graph frame is renamed as well.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.frames import FrameGraph
        >>> graph = FrameGraph()
        >>> _ = graph.get_or_create_frame("world")
        >>> _ = graph.get_or_create_frame("tool")
        >>> ao = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> tagged = ao.frames.retag(parent="world", child="tool")
        >>> tagged.frames.rename_frame("tool", "tool_0", graph=graph).frames.ids()
        ('world', 'tool_0')
        """
        return frame_rename(
            self._ao,
            old,
            new,
            graph=graph,
            on_conflict=on_conflict,
            validate=validate,
        )


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
    "frame_bind",
    "frame_ids",
    "frame_remap_ids",
    "frame_rename",
    "frame_retag",
    "install_analysis_object_frames_accessor",
]
