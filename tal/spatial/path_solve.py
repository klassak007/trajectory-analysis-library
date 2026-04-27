from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tal.frames import Frame, FrameGraph

from .ops.path_solve_ops import solve_pose_path_transform_impl, solve_rotation_path_transform_impl

if TYPE_CHECKING:
    from .pose import Pose
    from .rotation import Rotation


@dataclass(frozen=True)
class KinematicsPathSupportOptions:
    edge_motion_class_fn: Callable[[Frame, Frame], str] | None = None
    frame_inertial_status_fn: Callable[[Frame], str] | None = None
    edge_velocity_fn: Callable[[Frame, Frame], object] | None = None
    edge_acceleration_fn: Callable[[Frame, Frame], object] | None = None


@dataclass(frozen=True)
class PathSolveOptions:
    """Options for frame-path transform solving.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    graph: FrameGraph | None = None
    strict: bool = True
    kinematics_support: KinematicsPathSupportOptions | None = None


def solve_rotation_path_transform(
    src: Frame | str,
    dst: Frame | str,
    *,
    edge_rotation_fn,
    opts: PathSolveOptions | None = None,
) -> "Rotation":
    """Solve a composed rotation transform from ``src`` to ``dst`` frames.

    Parameters
    ----------
    src : Frame | str
        Source frame id/object.
    dst : Frame | str
        Destination frame id/object.
    edge_rotation_fn : object
        Callable resolving rotation edges for frame-path traversal.
    opts : PathSolveOptions or None
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``PathSolveOptions`` key fields: ``graph`` (default None), ``strict`` (default True), ``kinematics_support`` (default None).

    Returns
    -------
    Rotation
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
    >>> from tal.spatial.path_solve import PathSolveOptions
    >>> opts = PathSolveOptions(strict=False)
    >>> isinstance(opts, PathSolveOptions)
    True
    """
    return _solve_rotation_path_transform_with_owner(
        src,
        dst,
        edge_rotation_fn=edge_rotation_fn,
        opts=opts,
        owner="spatial.path_solve.rotation",
    )


def _solve_rotation_path_transform_with_owner(
    src: Frame | str,
    dst: Frame | str,
    *,
    edge_rotation_fn,
    opts: PathSolveOptions | None,
    owner: str,
) -> "Rotation":
    options = opts or PathSolveOptions()
    return solve_rotation_path_transform_impl(
        src,
        dst,
        edge_rotation_fn=edge_rotation_fn,
        graph=options.graph,
        strict=options.strict,
        owner=owner,
    )


def solve_pose_path_transform(
    src: Frame | str,
    dst: Frame | str,
    *,
    edge_pose_fn,
    opts: PathSolveOptions | None = None,
) -> "Pose":
    """Solve a pose transform between two frames.

    Parameters
    ----------
    src : Frame | str
        Source frame id/object.
    dst : Frame | str
        Destination frame id/object.
    edge_pose_fn : object
        Callable resolving pose edges for frame-path traversal.
    opts : PathSolveOptions or None
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``PathSolveOptions`` key fields: ``graph`` (default None), ``strict`` (default True), ``kinematics_support`` (default None).

    Returns
    -------
    Pose
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
    >>> from tal.spatial.path_solve import PathSolveOptions
    >>> opts = PathSolveOptions(strict=False)
    >>> isinstance(opts, PathSolveOptions)
    True
    """
    return _solve_pose_path_transform_with_owner(
        src,
        dst,
        edge_pose_fn=edge_pose_fn,
        opts=opts,
        owner="spatial.path_solve.pose",
    )


def _solve_pose_path_transform_with_owner(
    src: Frame | str,
    dst: Frame | str,
    *,
    edge_pose_fn,
    opts: PathSolveOptions | None,
    owner: str,
) -> "Pose":
    options = opts or PathSolveOptions()
    return solve_pose_path_transform_impl(
        src,
        dst,
        edge_pose_fn=edge_pose_fn,
        graph=options.graph,
        strict=options.strict,
        owner=owner,
    )


__all__ = [
    "KinematicsPathSupportOptions",
    "PathSolveOptions",
    "solve_pose_path_transform",
    "solve_rotation_path_transform",
]
