from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.frames import Frame, FrameGraph

from .ops.edge_resolver_ops import (
    PreparedEdgeResolver,
    _PathCallbackSignatureError,
    _raise_public_signature_error,
)
from .ops.path_configuration import (
    PathConfiguration,
    SelectedPathConfiguration,
    resolve_path_configuration,
)
from .ops.path_solve_ops import (
    solve_pose_path_transform_impl,
    solve_rotation_path_transform_impl,
)
from .temporal.options import PoseTemporalOptions

if TYPE_CHECKING:
    from .pose import Pose
    from .rotation import Rotation

PathQuery = xr.DataArray | np.ndarray | Sequence[float] | float


@dataclass(frozen=True)
class KinematicsPathSupportOptions:
    edge_motion_class_fn: Callable[[Frame, Frame], str] | None = None
    frame_inertial_status_fn: Callable[[Frame], str] | None = None
    edge_velocity_fn: Callable[[Frame, Frame], object] | None = None
    edge_acceleration_fn: Callable[[Frame, Frame], object] | None = None


@dataclass(frozen=True)
class PathSolveOptions:
    """Options for frame-path transform solving.

    ``graph`` selects the topology source, ``strict`` controls path policy,
    ``kinematics_support`` supplies velocity/acceleration transport support,
    and ``temporal`` controls evaluation of native-rate providers. Temporal
    ``on`` selects provider coordinates only; its shared nested ``query_dim``
    names the direct-query sequence axis and remains internal for object
    transforms whose caller topology is restored.

    Notes
    -----
    Path APIs accept an instance of this class (including subclasses) or
    ``None``. Only ``None`` selects defaults. Other types raise an
    operation-prefixed ``TypeError`` before graph/resolver work, including
    identity shortcuts and calls with ``validate=False``. Individual policy
    fields are checked by the operation that consumes them.
    """

    graph: FrameGraph | None = None
    strict: bool = True
    kinematics_support: KinematicsPathSupportOptions | None = None
    temporal: PoseTemporalOptions | None = None


def _coerce_path_solve_options(opts: object | None, *, owner: str) -> PathSolveOptions:
    """Normalize the outer options type without inspecting policy or graph state."""
    if opts is None:
        return PathSolveOptions()
    if isinstance(opts, PathSolveOptions):
        return opts
    raise TypeError(f"{owner}: opts must be PathSolveOptions or None.")


def solve_rotation_path_transform(
    src: Frame | str,
    dst: Frame | str,
    *,
    edge_rotation_fn=None,
    graph: FrameGraph | None = None,
    query: PathQuery | None = None,
    opts: PathSolveOptions | None = None,
) -> Rotation:
    """Solve a composed rotation transform from ``src`` to ``dst`` frames.

    Parameters
    ----------
    src : Frame | str
        Source frame id/object.
    dst : Frame | str
        Destination frame id/object.
    edge_rotation_fn : object
        Optional explicit rotation resolver. Returned edge values must be
        represented in the edge parent. When omitted, use bound Pose rotations.
    graph : FrameGraph or None
        Graph shorthand. May accompany options only when ``opts.graph`` is unset.
    query : object, optional
        Explicit scalar, one-dimensional, or labeled batched direct query
        grid. For a multidimensional DataArray, leading dimensions own output
        batch topology and the final dimension is the query axis. Required
        when any edge provider is dynamic.
    opts : PathSolveOptions or None
        Only ``None`` selects defaults. Otherwise a ``PathSolveOptions``
        instance is required. Key fields are ``graph`` (default None),
        ``strict`` (default True), ``kinematics_support`` (default None), and
        ``temporal`` (default None).

    Returns
    -------
    Rotation
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If ``opts`` is neither ``PathSolveOptions`` nor ``None``, before graph
        or resolver resolution, or if a consumed option field has the wrong type.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> from tal.spatial.path_solve import PathSolveOptions
    >>> opts = PathSolveOptions(strict=False)
    >>> isinstance(opts, PathSolveOptions)
    True
    """
    return _solve_rotation_path_transform_with_owner(
        src,
        dst,
        edge_rotation_fn=edge_rotation_fn,
        graph=graph,
        query=query,
        opts=opts,
        owner="spatial.path_solve.rotation",
    )


def _solve_rotation_path_transform_with_owner(
    src: Frame | str,
    dst: Frame | str,
    *,
    edge_rotation_fn=None,
    graph: FrameGraph | None = None,
    query: PathQuery | None = None,
    opts: PathSolveOptions | None = None,
    configuration: PathConfiguration | SelectedPathConfiguration | None = None,
    prepared_resolver: PreparedEdgeResolver | None = None,
    caller=None,
    owner: str,
) -> Rotation:
    if configuration is None:
        configuration = resolve_path_configuration(opts, graph=graph, owner=owner)
    try:
        return solve_rotation_path_transform_impl(
            src,
            dst,
            edge_rotation_fn=edge_rotation_fn,
            configuration=configuration,
            query=query,
            caller=caller,
            owner=owner,
            prepared_resolver=prepared_resolver,
        )
    except _PathCallbackSignatureError as exc:
        if prepared_resolver is not None and prepared_resolver.propagate_signature_error:
            raise
        _raise_public_signature_error(exc)


def solve_pose_path_transform(
    src: Frame | str,
    dst: Frame | str,
    *,
    edge_pose_fn=None,
    graph: FrameGraph | None = None,
    query: PathQuery | None = None,
    opts: PathSolveOptions | None = None,
) -> Pose:
    """Solve a pose transform between two frames.

    Parameters
    ----------
    src : Frame | str
        Source frame id/object.
    dst : Frame | str
        Destination frame id/object.
    edge_pose_fn : object
        Optional explicit pose resolver. Returned edge values must be
        represented in the edge parent. When omitted, use bound Pose providers.
    graph : FrameGraph or None
        Graph shorthand. May accompany options only when ``opts.graph`` is unset.
    query : object, optional
        Explicit scalar, one-dimensional, or labeled batched direct query
        grid. For a multidimensional DataArray, leading dimensions own output
        batch topology and the final dimension is the query axis. Required
        when any edge provider is dynamic.
    opts : PathSolveOptions or None
        Only ``None`` selects defaults. Otherwise a ``PathSolveOptions``
        instance is required. Key fields are ``graph`` (default None),
        ``strict`` (default True), ``kinematics_support`` (default None), and
        ``temporal`` (default None).

    Returns
    -------
    Pose
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If ``opts`` is neither ``PathSolveOptions`` nor ``None``, before graph
        or resolver resolution, or if a consumed option field has the wrong type.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> from tal.spatial.path_solve import PathSolveOptions
    >>> opts = PathSolveOptions(strict=False)
    >>> isinstance(opts, PathSolveOptions)
    True
    """
    return _solve_pose_path_transform_with_owner(
        src,
        dst,
        edge_pose_fn=edge_pose_fn,
        graph=graph,
        query=query,
        opts=opts,
        owner="spatial.path_solve.pose",
    )


def _solve_pose_path_transform_with_owner(
    src: Frame | str,
    dst: Frame | str,
    *,
    edge_pose_fn=None,
    graph: FrameGraph | None = None,
    query: PathQuery | None = None,
    opts: PathSolveOptions | None = None,
    configuration: PathConfiguration | SelectedPathConfiguration | None = None,
    prepared_resolver: PreparedEdgeResolver | None = None,
    caller=None,
    owner: str,
) -> Pose:
    if configuration is None:
        configuration = resolve_path_configuration(opts, graph=graph, owner=owner)
    try:
        return solve_pose_path_transform_impl(
            src,
            dst,
            edge_pose_fn=edge_pose_fn,
            configuration=configuration,
            query=query,
            caller=caller,
            owner=owner,
            prepared_resolver=prepared_resolver,
        )
    except _PathCallbackSignatureError as exc:
        if prepared_resolver is not None and prepared_resolver.propagate_signature_error:
            raise
        _raise_public_signature_error(exc)


__all__ = [
    "KinematicsPathSupportOptions",
    "PathSolveOptions",
    "solve_pose_path_transform",
    "solve_rotation_path_transform",
]
