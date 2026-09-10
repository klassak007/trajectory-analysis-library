from __future__ import annotations

import contextvars
from typing import Any


def _require_frame_name(name: Any, *, owner: str, arg: str = "name") -> str:
    if not isinstance(name, str):
        raise ValueError(f"{owner}: {arg} must be a non-empty string frame id.")
    cleaned = name.strip()
    if not cleaned:
        raise ValueError(f"{owner}: {arg} must be a non-empty string frame id.")
    return cleaned


def _iter_subtree(root: "Frame") -> list["Frame"]:
    out: list[Frame] = []
    stack: list[Frame] = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        for child in reversed(list(node._children.values())):
            stack.append(child)
    return out


class Frame:
    """Runtime frame node owned by a FrameGraph.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.

    Examples
    --------
    >>> from tal.frames import FrameGraph
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> base = graph.get_or_create_frame("base", parent=world)
    >>> (base.id, base.parent.id)
    ('base', 'world')
    """

    __slots__ = (
        "id",
        "_graph",
        "_parent",
        "_children",
        "_frame_ext",
        "_edge_to_parent_ext",
    )

    def __init__(self, *, name: str, graph: "FrameGraph") -> None:
        self.id = name
        self._graph = graph
        self._parent: Frame | None = None
        self._children: dict[str, Frame] = {}
        self._frame_ext: dict[str, Any] = {}
        self._edge_to_parent_ext: dict[str, Any] = {}

    @property
    def parent(self) -> "Frame | None":
        """Return the parent frame, or ``None`` for a root node.

        Returns
        -------
        Frame | None
            Resolved property value.
        """
        return self._parent

    @property
    def children(self) -> tuple["Frame", ...]:
        """Return direct child frames in insertion order.

        Returns
        -------
        tuple['Frame', ...]
            Resolved property value.
        """
        return tuple(self._children.values())

    def child(self, name: str) -> "Frame":
        """Get or create a child frame under this frame.

        Parameters
        ----------
        name : str
            Child frame id.

        Returns
        -------
        Frame
            Existing or newly-created child frame.

        Notes
        -----
        This delegates to the owning graph and refuses implicit reparenting.

        Examples
        --------
        >>> from tal.frames import FrameGraph
        >>> world = FrameGraph().get_or_create_frame("world")
        >>> world.child("base").parent is world
        True
        """
        return self._graph.get_or_create_frame(name, parent=self, allow_reparent=False)

    def reparent(
        self,
        new_parent: "Frame | None",
        *,
        on_conflict: str = "error",
    ) -> None:
        """Move this frame under ``new_parent`` in the same graph.

        Parameters
        ----------
        new_parent : Frame | None
            New parent frame, or ``None`` to make this frame a root.
        on_conflict : str, optional
            Conflict policy controlling behavior when target ids already exist.

        Returns
        -------
        None
            Returns ``None``; side effects are applied through owned state/metadata updates.

        Notes
        -----
        Reparenting mutates the owning graph and rejects cycles.

        Examples
        --------
        >>> from tal.frames import FrameGraph
        >>> graph = FrameGraph()
        >>> world = graph.get_or_create_frame("world")
        >>> base = graph.get_or_create_frame("base")
        >>> base.reparent(world)
        >>> base.parent is world
        True
        """
        self._graph.reparent_frame(self, new_parent, on_conflict=on_conflict)

    def rename(self, new_name: str, *, on_conflict: str = "error") -> "Frame":
        """Rename this frame id inside its owning graph.

        Parameters
        ----------
        new_name : str
            Replacement frame id.
        on_conflict : str, optional
            Conflict policy controlling behavior when target ids already exist.

        Returns
        -------
        Frame
            This frame after the registry id has been updated.

        Notes
        -----
        Child references from the parent and graph lookup keys are updated
        together.

        Examples
        --------
        >>> from tal.frames import FrameGraph
        >>> frame = FrameGraph().get_or_create_frame("base")
        >>> frame.rename("base_link").id
        'base_link'
        """
        return self._graph.rename_frame(self, new_name, on_conflict=on_conflict)

    def remove(self, *, subtree: bool = True) -> None:
        """Remove this frame from the graph.

        Parameters
        ----------
        subtree : bool, optional
            Whether descendants should also be removed.

        Returns
        -------
        None
            Returns ``None``; side effects are applied through owned state/metadata updates.

        Notes
        -----
        With ``subtree=True``, descendants are removed from the graph registry.

        Examples
        --------
        >>> from tal.frames import FrameGraph
        >>> graph = FrameGraph()
        >>> frame = graph.get_or_create_frame("base")
        >>> frame.remove()
        >>> graph.get_frame("base") is None
        True
        """
        self._graph.remove_frame(self, subtree=subtree)

    def __repr__(self) -> str:
        parent_id = self._parent.id if self._parent is not None else None
        return f"Frame(id={self.id!r}, parent={parent_id!r})"


class FrameGraph:
    """Context-scoped runtime frame registry.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.

    Examples
    --------
    >>> from tal.frames import FrameGraph
    >>> graph = FrameGraph()
    >>> world = graph.get_or_create_frame("world")
    >>> base = graph.get_or_create_frame("base", parent=world)
    >>> [child.id for child in world.children]
    ['base']
    """

    def __init__(self, *, frozen: bool = False) -> None:
        self._frames: dict[str, Frame] = {}
        self._frozen = bool(frozen)

    @property
    def frozen(self) -> bool:
        """Whether this graph is currently immutable.

        Returns
        -------
        bool
            Resolved property value.
        """
        return bool(self._frozen)

    def freeze(self) -> "FrameGraph":
        """Freeze this graph and return ``self`` for chaining.

        Returns
        -------
        FrameGraph
            This graph after it has been marked immutable.

        Notes
        -----
        Frozen graphs reject creation, removal, reparenting, and rename
        operations.

        Examples
        --------
        >>> from tal.frames import FrameGraph
        >>> graph = FrameGraph().freeze()
        >>> graph.frozen
        True
        """
        self._frozen = True
        return self

    def _assert_mutable(self, *, context: str) -> None:
        if self._frozen:
            raise ValueError(f"{context}: FrameGraph is frozen; mutation is not allowed.")

    def _assert_owned(self, frame: object, *, context: str) -> None:
        if not isinstance(frame, Frame):
            raise TypeError(f"{context}: frame must be Frame, got {type(frame).__name__}.")
        if frame._graph is not self:
            raise ValueError(f"{context}: frame belongs to a different FrameGraph.")
        if self._frames.get(frame.id) is not frame:
            raise ValueError(f"{context}: frame is not registered in this FrameGraph.")

    def _is_ancestor(self, node: Frame, candidate_ancestor: Frame) -> bool:
        cur: Frame | None = node
        while cur is not None:
            if cur is candidate_ancestor:
                return True
            cur = cur._parent
        return False

    def _detach_from_parent(self, frame: Frame) -> None:
        parent = frame._parent
        if parent is not None:
            parent._children.pop(frame.id, None)
        frame._parent = None
        frame._edge_to_parent_ext.clear()

    def _require_attachable(
        self,
        frame: Frame | None,
        parent: Frame | None,
        *,
        frame_id: str,
        parent_id: str,
        allow_reparent: bool,
        on_conflict: str,
        context: str,
    ) -> Frame | None:
        if frame is not None:
            self._assert_owned(frame, context=context)
        if parent is not None:
            self._assert_owned(parent, context=context)
        if on_conflict not in {"error", "replace"}:
            raise ValueError(f"{context}: on_conflict must be 'error' or 'replace'.")
        if frame_id == parent_id or (frame is not None and parent is not None and self._is_ancestor(parent, frame)):
            raise ValueError(f"{context}: reparent would create a cycle.")
        current_parent = None if frame is None else frame._parent
        if current_parent is not None and current_parent is not parent and not allow_reparent:
            raise ValueError(
                f"{context}: frame {frame_id!r} already has parent {current_parent.id!r}; "
                f"refusing reparent to {parent_id!r}."
            )
        conflict = None if parent is None else parent._children.get(frame_id)
        if conflict is None or conflict is frame:
            return None
        if on_conflict == "error":
            raise ValueError(f"{context}: child id collision under parent {parent_id!r}: {frame_id!r}.")
        if frame is not None and self._is_replace_conflict_unsafe(frame, conflict):
            raise ValueError(
                f"{context}: on_conflict='replace' cannot replace ancestor/descendant frame {conflict.id!r}."
            )
        return conflict

    def _attach_to_parent(
        self,
        frame: Frame,
        parent: Frame,
        *,
        context: str,
        on_conflict: str = "error",
    ) -> None:
        conflict = self._require_attachable(
            frame,
            parent,
            frame_id=frame.id,
            parent_id=parent.id,
            allow_reparent=True,
            on_conflict=on_conflict,
            context=context,
        )
        if conflict is not None:
            self.remove_frame(conflict, subtree=True)
        self._detach_from_parent(frame)
        parent._children[frame.id] = frame
        frame._parent = parent

    def _reparent_frame(
        self,
        frame: Frame,
        new_parent: Frame | None,
        *,
        context: str,
        on_conflict: str = "error",
    ) -> None:
        self._assert_owned(frame, context=context)
        if new_parent is not None:
            self._assert_owned(new_parent, context=context)
        if on_conflict not in {"error", "replace"}:
            raise ValueError(f"{context}: on_conflict must be 'error' or 'replace'.")
        if frame._parent is new_parent:
            return
        if frame._parent is not None and new_parent is not None and on_conflict == "error":
            old_parent = frame._parent.id if frame._parent is not None else None
            new_parent_id = new_parent.id if new_parent is not None else None
            raise ValueError(
                f"{context}: frame {frame.id!r} already has parent {old_parent!r}; "
                f"refusing reparent to {new_parent_id!r}."
            )
        self._assert_mutable(context=context)
        if new_parent is None:
            self._detach_from_parent(frame)
            return
        self._attach_to_parent(frame, new_parent, context=context, on_conflict=on_conflict)

    def get_frame(self, name: str) -> Frame | None:
        """Return a frame by id, or ``None`` when it does not exist.

        Parameters
        ----------
        name : str
            Frame id to look up.

        Returns
        -------
        Frame | None
            Registered frame, or ``None`` when the id is unknown.

        Notes
        -----
        Lookup is exact after whitespace validation of ``name``.

        Examples
        --------
        >>> from tal.frames import FrameGraph
        >>> graph = FrameGraph()
        >>> _ = graph.get_or_create_frame("world")
        >>> graph.get_frame("world").id
        'world'
        """
        key = _require_frame_name(name, owner="FrameGraph.get_frame")
        return self._frames.get(key)

    def get_or_create_frame(
        self,
        name: str,
        *,
        parent: Frame | None = None,
        allow_reparent: bool = False,
    ) -> Frame:
        """Get an existing frame by id or create a new one.

        Parameters
        ----------
        name : str
            Frame id to retrieve or create.
        parent : Frame | None, optional
            Optional parent for a newly-created frame.
        allow_reparent : bool, optional
            Whether an existing frame may be moved under ``parent``.

        Returns
        -------
        Frame
            Existing or newly-created frame.

        Notes
        -----
        Existing frames are returned unchanged unless ``allow_reparent=True`` or
        the requested parent already matches.

        Examples
        --------
        >>> from tal.frames import FrameGraph
        >>> graph = FrameGraph()
        >>> world = graph.get_or_create_frame("world")
        >>> graph.get_or_create_frame("base", parent=world).parent is world
        True
        """
        key = _require_frame_name(name, owner="FrameGraph.get_or_create_frame")
        if parent is not None:
            self._assert_owned(parent, context="FrameGraph.get_or_create_frame")
        frame = self._frames.get(key)
        if frame is None:
            self._assert_mutable(context="FrameGraph.get_or_create_frame")
            frame = Frame(name=key, graph=self)
            self._frames[key] = frame
            if parent is not None:
                self._attach_to_parent(frame, parent, context="FrameGraph.get_or_create_frame")
            return frame
        if parent is None or frame._parent is parent:
            return frame
        if frame._parent is None or allow_reparent:
            self._assert_mutable(context="FrameGraph.get_or_create_frame")
            self._attach_to_parent(frame, parent, context="FrameGraph.get_or_create_frame")
            return frame
        old_parent = frame._parent.id if frame._parent is not None else None
        new_parent = parent.id if parent is not None else None
        raise ValueError(
            "FrameGraph.get_or_create_frame: "
            f"frame {key!r} already has parent {old_parent!r}; refusing reparent to {new_parent!r}."
        )

    def _is_replace_conflict_unsafe(self, target: Frame, conflict: Frame) -> bool:
        return self._is_ancestor(target, conflict) or self._is_ancestor(conflict, target)

    def reparent_frame(
        self,
        frame: Frame | str,
        new_parent: Frame | str | None,
        *,
        on_conflict: str = "error",
    ) -> None:
        """Reparent an existing frame within this graph.

        Parameters
        ----------
        frame : Frame | str
            Existing frame or frame id to move.
        new_parent : Frame | str | None
            New parent frame/id, or ``None`` to make ``frame`` a root.
        on_conflict : str, optional
            Conflict policy controlling behavior when target ids already exist.

        Returns
        -------
        None
            Returns ``None``; side effects are applied through owned state/metadata updates.

        Notes
        -----
        Reparenting rejects missing frames, cycles, and parent conflicts unless
        the conflict policy explicitly allows replacement.

        Examples
        --------
        >>> from tal.frames import FrameGraph
        >>> graph = FrameGraph()
        >>> world = graph.get_or_create_frame("world")
        >>> base = graph.get_or_create_frame("base")
        >>> graph.reparent_frame(base, world)
        >>> base.parent is world
        True
        """
        owner = "FrameGraph.reparent_frame"
        target = self.get_frame(frame) if isinstance(frame, str) else frame
        if target is None:
            raise ValueError(f"{owner}: frame {frame!r} not found.")
        parent_target = self.get_frame(new_parent) if isinstance(new_parent, str) else new_parent
        if isinstance(new_parent, str) and parent_target is None:
            raise ValueError(f"{owner}: new_parent {new_parent!r} not found.")
        self._reparent_frame(target, parent_target, context=owner, on_conflict=on_conflict)

    def rename_frame(
        self,
        frame: Frame | str,
        new_name: str,
        *,
        on_conflict: str = "error",
    ) -> Frame:
        """Rename an existing frame id in this graph.

        Parameters
        ----------
        frame : Frame | str
            Existing frame or frame id to rename.
        new_name : str
            Replacement frame id.
        on_conflict : str, optional
            Conflict policy controlling behavior when target ids already exist.

        Returns
        -------
        Frame
            Renamed frame.

        Notes
        -----
        The graph registry and parent-child lookup table are updated
        atomically.

        Examples
        --------
        >>> from tal.frames import FrameGraph
        >>> graph = FrameGraph()
        >>> _ = graph.get_or_create_frame("base")
        >>> graph.rename_frame("base", "base_link").id
        'base_link'
        """
        owner = "FrameGraph.rename_frame"
        if on_conflict not in {"error", "replace"}:
            raise ValueError(f"{owner}: on_conflict must be 'error' or 'replace'.")
        target = self.get_frame(frame) if isinstance(frame, str) else frame
        if target is None:
            raise ValueError(f"{owner}: frame {frame!r} not found.")
        self._assert_owned(target, context=owner)
        new_id = _require_frame_name(new_name, owner=owner, arg="new_name")
        if target.id == new_id:
            return target
        self._assert_mutable(context=owner)
        existing = self._frames.get(new_id)
        if existing is not None and existing is not target:
            if on_conflict == "error":
                raise ValueError(f"{owner}: target name {new_id!r} already exists.")
            if self._is_replace_conflict_unsafe(target, existing):
                raise ValueError(
                    f"{owner}: on_conflict='replace' cannot replace ancestor/descendant frame {new_id!r}."
                )
            self.remove_frame(existing, subtree=True)
        parent = target._parent
        if parent is not None:
            sibling = parent._children.get(new_id)
            if sibling is not None and sibling is not target:
                raise ValueError(f"{owner}: child id collision under parent {parent.id!r}: {new_id!r}.")
            parent._children.pop(target.id, None)
            parent._children[new_id] = target
        self._frames.pop(target.id, None)
        target.id = new_id
        self._frames[new_id] = target
        return target

    def remove_frame(self, frame: Frame | str, *, subtree: bool = True) -> None:
        """Remove a frame by object or id.

        Parameters
        ----------
        frame : Frame | str
            Existing frame or frame id to remove.
        subtree : bool, optional
            Whether descendants should also be removed.

        Returns
        -------
        None
            Returns ``None``; side effects are applied through owned state/metadata updates.

        Notes
        -----
        Missing frame ids are ignored. Registered frames from other graphs are
        rejected.

        Examples
        --------
        >>> from tal.frames import FrameGraph
        >>> graph = FrameGraph()
        >>> _ = graph.get_or_create_frame("base")
        >>> graph.remove_frame("base")
        >>> graph.get_frame("base") is None
        True
        """
        owner = "FrameGraph.remove_frame"
        target = self.get_frame(frame) if isinstance(frame, str) else frame
        if target is None:
            return
        self._assert_owned(target, context=owner)
        self._assert_mutable(context=owner)
        if subtree:
            nodes = _iter_subtree(target)
            for node in nodes:
                self._frames.pop(node.id, None)
            for node in nodes:
                self._detach_from_parent(node)
                node._children.clear()
            return
        self._frames.pop(target.id, None)
        self._detach_from_parent(target)
        for child in list(target._children.values()):
            self._detach_from_parent(child)
        target._children.clear()

    def __enter__(self) -> "FrameGraph":
        token = _ACTIVE_FRAME_GRAPH.set(self)
        stack = _FRAME_GRAPH_CONTEXT_STACK.get()
        _FRAME_GRAPH_CONTEXT_STACK.set((*stack, (self, token)))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        stack = _FRAME_GRAPH_CONTEXT_STACK.get()
        if not stack or stack[-1][0] is not self:
            raise RuntimeError(
                "FrameGraph.__exit__: no matching context-local entry for this graph."
            )
        _, token = stack[-1]
        try:
            _ACTIVE_FRAME_GRAPH.reset(token)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "FrameGraph.__exit__: no matching context-local entry for this graph."
            ) from exc
        _FRAME_GRAPH_CONTEXT_STACK.set(stack[:-1])


_FRAME_GRAPH_CONTEXT_STACK: contextvars.ContextVar[
    tuple[tuple[FrameGraph, contextvars.Token[FrameGraph]], ...]
] = contextvars.ContextVar("tal_frame_graph_context_stack", default=())
_DEFAULT_FRAME_GRAPH = FrameGraph()
_ACTIVE_FRAME_GRAPH: contextvars.ContextVar[FrameGraph] = contextvars.ContextVar(
    "tal_active_frame_graph",
    default=_DEFAULT_FRAME_GRAPH,
)


def get_active_frame_graph() -> FrameGraph:
    """Return the currently active context-local frame graph.

    Parameters
    ----------
    None
        This callable does not accept user-facing parameters.

    Returns
    -------
    FrameGraph
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> from tal.frames import FrameGraph, get_active_frame_graph
    >>> graph = FrameGraph()
    >>> with graph:
    ...     get_active_frame_graph() is graph
    True
    """
    return _ACTIVE_FRAME_GRAPH.get()


def get_or_create_frame(name: str, parent: Frame | None = None, *, allow_reparent: bool = False) -> Frame:
    """Get or create a frame in the active frame graph.

    Parameters
    ----------
    name : str
        Identifier/name used for lookup or registration.
    parent : Frame | None, optional
        Parent frame identifier or ``None`` to clear.
    allow_reparent : bool, optional
        Behavior flag/policy controlling boundary semantics.

    Returns
    -------
    Frame
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> from tal.frames import FrameGraph, get_or_create_frame
    >>> graph = FrameGraph()
    >>> with graph:
    ...     world = get_or_create_frame("world")
    ...     base = get_or_create_frame("base", parent=world)
    >>> base.parent is world
    True
    """
    return get_active_frame_graph().get_or_create_frame(
        name=name,
        parent=parent,
        allow_reparent=allow_reparent,
    )


def _require_extension_key(key: object, *, owner: str, arg: str = "key") -> str:
    if not isinstance(key, str):
        raise TypeError(f"{owner}: {arg} must be non-empty string.")
    cleaned = key.strip()
    if cleaned:
        return cleaned
    raise ValueError(f"{owner}: {arg} must be non-empty string.")


def _require_registered_frame(frame: object, *, owner: str, arg: str = "frame") -> Frame:
    if not isinstance(frame, Frame):
        raise TypeError(f"{owner}: {arg} must be Frame, got {type(frame).__name__}.")
    graph = frame._graph
    if not isinstance(graph, FrameGraph):
        raise ValueError(f"{owner}: {arg} is not bound to a valid FrameGraph.")
    if graph._frames.get(frame.id) is frame:
        return frame
    raise ValueError(f"{owner}: {arg} frame {frame.id!r} is not registered in graph.")


def get_frame_runtime_ext(frame: Frame, key: str, *, owner: str = "frames.registry.get_frame_runtime_ext") -> Any:
    target = _require_registered_frame(frame, owner=owner)
    return target._frame_ext.get(_require_extension_key(key, owner=owner))


def set_frame_runtime_ext(
    frame: Frame,
    key: str,
    value: Any,
    *,
    owner: str = "frames.registry.set_frame_runtime_ext",
) -> None:
    target = _require_registered_frame(frame, owner=owner)
    target._frame_ext[_require_extension_key(key, owner=owner)] = value


def get_edge_to_parent_runtime_ext(
    child: Frame,
    key: str,
    *,
    owner: str = "frames.registry.get_edge_to_parent_runtime_ext",
) -> Any:
    target = _require_registered_frame(child, owner=owner, arg="child")
    if target.parent is None:
        raise ValueError(f"{owner}: child frame {target.id!r} has no parent edge.")
    return target._edge_to_parent_ext.get(_require_extension_key(key, owner=owner))


def set_edge_to_parent_runtime_ext(
    child: Frame,
    key: str,
    value: Any,
    *,
    owner: str = "frames.registry.set_edge_to_parent_runtime_ext",
) -> None:
    target = _require_registered_frame(child, owner=owner, arg="child")
    target._graph._assert_mutable(context=owner)
    if target.parent is None:
        raise ValueError(f"{owner}: child frame {target.id!r} has no parent edge.")
    target._edge_to_parent_ext[_require_extension_key(key, owner=owner)] = value
