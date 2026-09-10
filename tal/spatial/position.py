from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.core.orchestration.runtime_checks import select_single_numeric_var
from tal.core.schema_read import read_roles, validate_schema_if_needed
from tal.linalg import add as linalg_add
from tal.utils.frame_schema import get_frames, set_frames

from .association import finalize_spatial_as, resolve_passive_association
from .construction import SpatialConfigurationConstructionMixin
from .metadata import (
    get_position_intent,
    get_position_rep,
    normalize_configuration_relation_semantics,
    set_position_intent,
    set_position_rep,
)
from .ops.frame_api_ops import position_to_frame
from .policies.intent import PositionAddPlan, resolve_position_add_intent
from .policies.runtime_checks import require_xyz_core_labels

_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")

if TYPE_CHECKING:
    from tal.frames import Frame, FrameGraph

    from .path_solve import PathSolveOptions
    from .temporal.options import (
        KinematicsDerivativeOptions,
        KinematicsSmoothingOptions,
    )
    from .velocity import LinearVelocity


def _coerce_position_source(value: object, *, owner: str) -> AnalysisObject:
    return coerce_analysis_object_input(value, owner=owner)


def _require_single_core_dim(ds: xr.Dataset, *, owner: str) -> str:
    declared, _, _, core_dims = read_roles(ds)
    if not declared:
        raise ValueError(f"{owner}: Position requires declared roles.")
    if len(core_dims) != 1:
        raise ValueError(f"{owner}: Position requires exactly one core dim; got {core_dims!r}.")
    axis = core_dims[0]
    if int(ds.sizes.get(axis, -1)) != 3:
        raise ValueError(f"{owner}: Position core dim {axis!r} must have length 3.")
    return axis


def _enforce_position_dataset_invariants(ds: xr.Dataset, *, owner: str) -> None:
    candidate = validate_schema_if_needed(ds)
    _ = get_frames(candidate)
    select_single_numeric_var(candidate, owner=owner, what="Position")
    axis = _require_single_core_dim(candidate, owner=owner)
    require_xyz_core_labels(candidate, core_dim=axis, owner=owner, what="Position")
    rep = get_position_rep(candidate, owner=owner)
    if rep != "cart":
        raise ValueError(f"{owner}: Position supports only cart representation; got {rep!r}.")
    _ = get_position_intent(candidate, owner=owner)


def _normalize_position_metadata(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    rep = get_position_rep(ds, owner=owner)
    out = set_position_rep(ds, rep=rep, validate=False, owner=owner)
    return normalize_configuration_relation_semantics(
        out,
        validate=False,
        owner=owner,
    )


def _coerce_position_operand(value: object, *, owner: str, side: str) -> "Position":
    if isinstance(value, Position):
        return value
    try:
        return Position(value)
    except TypeError as exc:
        raise TypeError(
            f"{owner}: {side} operand must be Position, AnalysisObject, xr.Dataset, or xr.DataArray."
        ) from exc
    except ValueError as exc:
        raise ValueError(f"{owner}: {side} operand is not a valid Position: {exc}") from exc


def _finalize_position_addition(
    ds: xr.Dataset,
    *,
    plan: PositionAddPlan,
    owner: str,
) -> xr.Dataset:
    out = set_position_rep(ds, rep="cart", validate=False, owner=owner)
    out = set_position_intent(out, intent=None, validate=False, owner=owner)
    out = set_frames(out, parent=plan.output_parent, child=plan.output_child, validate=False)
    return out


def _add_positions(left_input: object, right_input: object, *, owner: str) -> "Position":
    left = _coerce_position_operand(left_input, owner=owner, side="left")
    right = _coerce_position_operand(right_input, owner=owner, side="right")
    association = resolve_passive_association((left, right), owner=owner)
    plan = resolve_position_add_intent(left, right, owner=owner)
    numeric = linalg_add(left, right)
    finalized = _finalize_position_addition(analysis_object_dataset(numeric), plan=plan, owner=owner)
    return finalize_spatial_as(
        Position,
        finalized,
        validate=True,
        association=association,
    )


class Position(SpatialConfigurationConstructionMixin, AnalysisObject):
    """Cartesian 3D position with frame-aware spatial operations.

    Parameters
    ----------
    data : AnalysisObject, xarray.Dataset, or xarray.DataArray
        Position payload accepted by the typed ownership boundary.
    parent, child, expressed_in : str or None, optional
        Frame declarations to inherit, confirm, add, or explicitly clear. Omitting a
        declaration inherits it from ``data``.
    graph : FrameGraph or None, optional
        Passive wrapper association. Omission inherits any association from ``data``;
        this does not create graph topology.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    XYZ_LABELS: tuple[str, str, str] = _XYZ_LABELS
    SPATIAL_CONSTRUCTION_OWNER = "spatial.position.__init__"
    SPATIAL_SOURCE_COERCER = staticmethod(_coerce_position_source)

    @classmethod
    def _from_validated(cls, ds: xr.Dataset | xr.DataArray) -> "Position":
        obj = super()._from_validated(ds)
        obj._normalize_metadata(owner=f"{cls.__name__}._from_validated")
        obj._enforce_invariants(owner=f"{cls.__name__}._from_validated")
        return obj

    @classmethod
    def _from_unvalidated(cls, ds: xr.Dataset | xr.DataArray, *, schema_prepared: bool = False) -> "Position":
        obj = super()._from_unvalidated(ds, schema_prepared=schema_prepared)
        obj._normalize_metadata(owner=f"{cls.__name__}._from_unvalidated")
        obj._enforce_invariants(owner=f"{cls.__name__}._from_unvalidated")
        return obj

    def _normalize_metadata(self, *, owner: str) -> None:
        normalized = _normalize_position_metadata(analysis_object_dataset(self), owner=owner)
        self._bind_dataset(normalized)

    def _enforce_invariants(self, *, owner: str) -> None:
        _enforce_position_dataset_invariants(analysis_object_dataset(self), owner=owner)

    def as_delta(self, *, validate: bool = True) -> "Position":
        """Return this position re-tagged with ``intent="delta"`` semantics.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Position
            Position with the same numeric values and delta-position intent
            metadata.

        Notes
        -----
        Delta positions represent displacement vectors. The operation changes
        TAL spatial metadata only; it does not transform coordinates or frame
        ids.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.spatial import Position
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"position": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... )
        >>> isinstance(Position(ao).as_delta(), Position)
        True
        """
        ds = set_position_intent(
            analysis_object_dataset(self),
            intent="delta",
            validate=False,
            owner="spatial.position.as_delta",
        )
        return self._rewrap_dataset(ds, validate=validate)

    def to_frame(
        self,
        dst: "Frame | str",
        *,
        edge_pose_fn=None,
        graph: FrameGraph | None = None,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "Position":
        """Express this position in a destination frame.

        Parameters
        ----------
        dst : Frame | str
            Destination frame id/object.
        edge_pose_fn : object, optional
            Optional explicit parent-basis pose resolver; omitted calls use bound Pose providers.
        graph : FrameGraph or None, optional
            Use this graph with bound providers; cannot accompany non-None ``opts.graph``.
        opts : PathSolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``graph`` (override graph source), ``strict`` (strict path checks), and ``kinematics_support`` for velocity/acceleration transport metadata.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Position
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
        >>> opts = PathSolveOptions(strict=False)
        >>> isinstance(opts, PathSolveOptions)
        True
        """
        return position_to_frame(
            self,
            dst,
            edge_pose_fn=edge_pose_fn,
            graph=graph,
            opts=opts,
            validate=validate,
        )

    def express_in(
        self,
        dst: "Frame | str",
        *,
        edge_rotation_fn=None,
        graph: FrameGraph | None = None,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "Position":
        """Express this position in another frame using a rotation-only path solve.

        Parameters
        ----------
        dst : Frame | str
            Destination frame id/object.
        edge_rotation_fn : object, optional
            Optional explicit parent-basis rotation resolver; omitted calls use bound Pose rotations.
        graph : FrameGraph or None, optional
            Use this graph with bound providers; cannot accompany non-None ``opts.graph``.
        opts : PathSolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``graph`` (override graph source), ``strict`` (strict path checks), and ``kinematics_support`` for velocity/acceleration transport metadata.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Position
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
        >>> opts = PathSolveOptions(strict=False)
        >>> isinstance(opts, PathSolveOptions)
        True
        """
        from .ops.frame_expression_ops import position_express_in

        return position_express_in(
            self,
            dst,
            edge_rotation_fn=edge_rotation_fn,
            graph=graph,
            opts=opts,
            validate=validate,
        )

    def differentiate(
        self,
        *,
        on: str | None = None,
        opts: "KinematicsDerivativeOptions | None" = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: tuple[str, ...] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "LinearVelocity":
        """Differentiate position along the sequence axis into ``LinearVelocity``.

        Parameters
        ----------
        on : str | None, optional
            Coordinate/dimension name used as the operation domain.
        opts : KinematicsDerivativeOptions | None, optional
            When ``None``, defaults are used. Key fields are ``method`` (finite difference or local polynomial), ``order``, and ``window``/``poly_order`` for local polynomial fitting.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.
        sequence_dim : str | None, optional
            Optional override for the sequence dimension used by temporal semantics.
        batch_dims : tuple[str, ...] | None, optional
            Optional override for batch dimensions used by temporal semantics.
        sequence_size_coord : str | None, optional
            Optional sequence-size coordinate used for ragged validity handling.

        Returns
        -------
        LinearVelocity
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
        >>> opts = KinematicsDerivativeOptions(method='local_poly')
        >>> isinstance(opts, KinematicsDerivativeOptions)
        True
        """
        from .ops.kinematics_temporal_ops import (
            differentiate_position_to_linear_velocity,
        )

        return differentiate_position_to_linear_velocity(
            self,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.position.differentiate",
        )

    def smooth(
        self,
        *,
        on: str | None = None,
        opts: "KinematicsSmoothingOptions | None" = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: tuple[str, ...] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "Position":
        """Apply kinematics-aware smoothing while preserving position semantics.

        Parameters
        ----------
        on : str | None, optional
            Coordinate/dimension name used as the operation domain.
        opts : KinematicsSmoothingOptions | None, optional
            When ``None``, defaults are used. Key fields are ``method`` (moving average, local polynomial, or gaussian), ``window``, ``poly_order``, and ``sigma``.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.
        sequence_dim : str | None, optional
            Optional override for the sequence dimension used by temporal semantics.
        batch_dims : tuple[str, ...] | None, optional
            Optional override for batch dimensions used by temporal semantics.
        sequence_size_coord : str | None, optional
            Optional sequence-size coordinate used for ragged validity handling.

        Returns
        -------
        Position
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
        >>> opts = KinematicsSmoothingOptions(method='gaussian')
        >>> isinstance(opts, KinematicsSmoothingOptions)
        True
        """
        from .ops.kinematics_smoothing_ops import smooth_kinematics_like

        return smooth_kinematics_like(
            self,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.position.smooth",
        )

    def __add__(self, other: object) -> "Position":
        return _add_positions(self, other, owner="spatial.position.__add__")

    def __radd__(self, other: object) -> "Position":
        return _add_positions(other, self, owner="spatial.position.__radd__")


__all__ = ["Position"]


from .ops.magnitude_ops import (
    install_position_magnitude_methods as _install_position_magnitude_methods,
)

_install_position_magnitude_methods(Position)
