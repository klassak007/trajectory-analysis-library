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

    ``graph`` selects the topology source, ``strict`` controls path policy,
    and ``kinematics_support`` supplies velocity/acceleration transport support.

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
        Only ``None`` selects defaults. Otherwise a ``PathSolveOptions``
        instance is required. Key fields are ``graph`` (default None),
        ``strict`` (default True), and ``kinematics_support`` (default None).

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
    options = _coerce_path_solve_options(opts, owner=owner)
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
        Only ``None`` selects defaults. Otherwise a ``PathSolveOptions``
        instance is required. Key fields are ``graph`` (default None),
        ``strict`` (default True), and ``kinematics_support`` (default None).

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
    options = _coerce_path_solve_options(opts, owner=owner)
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
