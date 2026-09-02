from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.typed_lifecycle import TypedAnalysisObject

from .kinematics.family import (
    KinematicsClasses,
    KinematicsFamilyConfig,
    family_angular,
    family_as_components,
    family_as_vector6,
    family_from_linear_angular,
    family_from_vector6,
    family_linear,
    family_to_rep,
)
from .kinematics.lifecycle import make_kinematics_lifecycle_spec
from .kinematics.vector6_ops import VELOCITY_VECTOR6_OPTS
from .metadata import (
    get_angular_velocity_rep,
    get_linear_velocity_rep,
    get_velocity_rep,
    normalize_kinematics_kind,
    set_angular_velocity_rep,
    set_kinematics_kind,
    set_linear_velocity_rep,
    set_velocity_rep,
    validate_spatial_roles,
)
from .kinematics.paired_components import PairAssemblyOptions

_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")

if TYPE_CHECKING:
    from tal.frames import Frame

    from .acceleration import Acceleration, AngularAcceleration, LinearAcceleration
    from .path_solve import PathSolveOptions
    from .position import Position
    from .temporal.options import (
        KinematicsDerivativeOptions,
        KinematicsIntegralOptions,
        KinematicsSmoothingOptions,
    )


def _set_linear_rep(ds: xr.Dataset, rep: str, validate: bool, owner: str) -> xr.Dataset:
    return set_linear_velocity_rep(ds, rep=rep, validate=validate, owner=owner)


def _set_angular_rep(ds: xr.Dataset, rep: str, validate: bool, owner: str) -> xr.Dataset:
    return set_angular_velocity_rep(ds, rep=rep, validate=validate, owner=owner)


def _set_family_rep(ds: xr.Dataset, rep: str, validate: bool, owner: str) -> xr.Dataset:
    return set_velocity_rep(ds, rep=rep, validate=validate, owner=owner)


def _get_linear_rep(ds: xr.Dataset, owner: str) -> str:
    return get_linear_velocity_rep(ds, owner=owner)


def _get_angular_rep(ds: xr.Dataset, owner: str) -> str:
    return get_angular_velocity_rep(ds, owner=owner)


def _get_family_rep(ds: xr.Dataset, owner: str) -> str:
    return get_velocity_rep(ds, owner=owner)


def _set_kind(ds: xr.Dataset, kind: str, validate: bool, owner: str) -> xr.Dataset:
    return set_kinematics_kind(ds, kind=kind, validate=validate, owner=owner)


def _normalize_kind(ds: xr.Dataset, expected_kind: str, validate: bool, owner: str) -> xr.Dataset:
    return normalize_kinematics_kind(ds, expected_kind=expected_kind, validate=validate, owner=owner)


_VELOCITY_CONFIG = KinematicsFamilyConfig(
    family_name="velocity",
    family_kind="velocity",
    linear_kind="linear_velocity",
    angular_kind="angular_velocity",
    linear_class_name="LinearVelocity",
    angular_class_name="AngularVelocity",
    family_class_name="Velocity",
    xyz_labels=_XYZ_LABELS,
    set_linear_rep=_set_linear_rep,
    set_angular_rep=_set_angular_rep,
    set_family_rep=_set_family_rep,
    get_linear_rep=_get_linear_rep,
    get_angular_rep=_get_angular_rep,
    get_family_rep=_get_family_rep,
    set_kinematics_kind=_set_kind,
    normalize_kinematics_kind=_normalize_kind,
    validate_spatial_roles=validate_spatial_roles,
    vector6_opts=VELOCITY_VECTOR6_OPTS,
    pair_opts=PairAssemblyOptions(
        left_what="LinearVelocity",
        right_what="AngularVelocity",
        pair_what="velocity",
        left_component_name="linear",
        right_component_name="angular",
        left_expected_labels=_XYZ_LABELS,
        right_expected_labels=_XYZ_LABELS,
    ),
)

_LINEAR_VELOCITY_LIFECYCLE = make_kinematics_lifecycle_spec(
    type_name="LinearVelocity",
    owner_prefix="spatial.linear_velocity",
    cfg=_VELOCITY_CONFIG,
    role="linear",
)
_ANGULAR_VELOCITY_LIFECYCLE = make_kinematics_lifecycle_spec(
    type_name="AngularVelocity",
    owner_prefix="spatial.angular_velocity",
    cfg=_VELOCITY_CONFIG,
    role="angular",
)
_VELOCITY_LIFECYCLE = make_kinematics_lifecycle_spec(
    type_name="Velocity",
    owner_prefix="spatial.velocity",
    cfg=_VELOCITY_CONFIG,
    role="family",
)


def _classes() -> KinematicsClasses:
    return KinematicsClasses(
        linear_cls=LinearVelocity,
        angular_cls=AngularVelocity,
        family_cls=Velocity,
    )


class LinearVelocity(TypedAnalysisObject):
    """Linear velocity vector type.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    XYZ_LABELS: tuple[str, str, str] = _XYZ_LABELS
    LIFECYCLE = _LINEAR_VELOCITY_LIFECYCLE

    def differentiate(
        self,
        *,
        on: str | None = None,
        opts: "KinematicsDerivativeOptions | None" = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: tuple[str, ...] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "LinearAcceleration":
        """Differentiate linear velocity into linear acceleration.

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
        LinearAcceleration
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
        from .ops.kinematics_temporal_ops import differentiate_linear_velocity_to_linear_acceleration

        return differentiate_linear_velocity_to_linear_acceleration(
            self,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.linear_velocity.differentiate",
        )
    
    def integrate(
        self,
        *,
        on: str | None = None,
        opts: "KinematicsIntegralOptions | None" = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: tuple[str, ...] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "Position":
        """Integrate linear velocity into position.

        Parameters
        ----------
        on : str | None, optional
            Coordinate/dimension name used as the operation domain.
        opts : KinematicsIntegralOptions | None, optional
            When ``None``, defaults are used. Key fields are ``method`` (``'cumulative_trapezoid'`` or ``'simpson'``) and ``initial_value``.
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
        >>> opts = KinematicsIntegralOptions(method='simpson')
        >>> isinstance(opts, KinematicsIntegralOptions)
        True
        """
        from .ops.kinematics_temporal_ops import integrate_linear_velocity_to_position

        return integrate_linear_velocity_to_position(
            self,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.linear_velocity.integrate",
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
    ) -> "LinearVelocity":
        """Apply kinematics-aware smoothing to linear velocity.

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
            owner="spatial.linear_velocity.smooth",
        )
    
    def to_frame(
        self,
        dst: "Frame | str",
        *,
        edge_pose_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "LinearVelocity":
        """Transform linear velocity into a destination frame using pose edges.

        Parameters
        ----------
        dst : Frame | str
            Destination frame id/object.
        edge_pose_fn : object
            Callable resolving pose edges for frame-path traversal.
        opts : PathSolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``graph`` (override graph source), ``strict`` (strict path checks), and ``kinematics_support`` for velocity/acceleration transport metadata.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

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
        >>> opts = PathSolveOptions(strict=False)
        >>> isinstance(opts, PathSolveOptions)
        True
        """
        from .ops.kinematics_frame_ops import to_frame_linear_velocity

        return to_frame_linear_velocity(
            self,
            dst=dst,
            edge_pose_fn=edge_pose_fn,
            opts=opts,
            validate=validate,
            owner="spatial.linear_velocity.to_frame",
        )
    
    def express_in(
        self,
        dst: "Frame | str",
        *,
        edge_rotation_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "LinearVelocity":
        """Express linear velocity in a destination frame using rotation edges.

        Parameters
        ----------
        dst : Frame | str
            Destination frame id/object.
        edge_rotation_fn : object
            Callable resolving rotation edges for frame-path traversal.
        opts : PathSolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``graph`` (override graph source), ``strict`` (strict path checks), and ``kinematics_support`` for velocity/acceleration transport metadata.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

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
        >>> opts = PathSolveOptions(strict=False)
        >>> isinstance(opts, PathSolveOptions)
        True
        """
        from .ops.kinematics_frame_ops import express_in_linear_velocity

        return express_in_linear_velocity(
            self,
            dst=dst,
            edge_rotation_fn=edge_rotation_fn,
            opts=opts,
            validate=validate,
            owner="spatial.linear_velocity.express_in",
        )


class AngularVelocity(TypedAnalysisObject):
    """Angular velocity vector type.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    XYZ_LABELS: tuple[str, str, str] = _XYZ_LABELS
    LIFECYCLE = _ANGULAR_VELOCITY_LIFECYCLE

    def differentiate(
        self,
        *,
        on: str | None = None,
        opts: "KinematicsDerivativeOptions | None" = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: tuple[str, ...] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "AngularAcceleration":
        """Differentiate angular velocity into angular acceleration.

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
        AngularAcceleration
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
        from .ops.kinematics_temporal_ops import differentiate_angular_velocity_to_angular_acceleration

        return differentiate_angular_velocity_to_angular_acceleration(
            self,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.angular_velocity.differentiate",
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
    ) -> "AngularVelocity":
        """Apply kinematics-aware smoothing to angular velocity.

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
        AngularVelocity
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
            owner="spatial.angular_velocity.smooth",
        )
    
    def to_frame(
        self,
        dst: "Frame | str",
        *,
        edge_pose_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "AngularVelocity":
        """Transform angular velocity into a destination frame using pose edges.

        Parameters
        ----------
        dst : Frame | str
            Destination frame id/object.
        edge_pose_fn : object
            Callable resolving pose edges for frame-path traversal.
        opts : PathSolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``graph`` (override graph source), ``strict`` (strict path checks), and ``kinematics_support`` for velocity/acceleration transport metadata.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AngularVelocity
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
        from .ops.kinematics_frame_ops import to_frame_angular_velocity

        return to_frame_angular_velocity(
            self,
            dst=dst,
            edge_pose_fn=edge_pose_fn,
            opts=opts,
            validate=validate,
            owner="spatial.angular_velocity.to_frame",
        )
    
    def express_in(
        self,
        dst: "Frame | str",
        *,
        edge_rotation_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "AngularVelocity":
        """Express angular velocity in a destination frame using rotation edges.

        Parameters
        ----------
        dst : Frame | str
            Destination frame id/object.
        edge_rotation_fn : object
            Callable resolving rotation edges for frame-path traversal.
        opts : PathSolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``graph`` (override graph source), ``strict`` (strict path checks), and ``kinematics_support`` for velocity/acceleration transport metadata.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AngularVelocity
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
        from .ops.kinematics_frame_ops import express_in_angular_velocity

        return express_in_angular_velocity(
            self,
            dst=dst,
            edge_rotation_fn=edge_rotation_fn,
            opts=opts,
            validate=validate,
            owner="spatial.angular_velocity.express_in",
        )


class Velocity(TypedAnalysisObject):
    """Spatial velocity family type (linear + angular).

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    XYZ_LABELS: tuple[str, str, str] = _XYZ_LABELS
    LIFECYCLE = _VELOCITY_LIFECYCLE

    @classmethod
    def from_linear_angular(cls, linear: object, angular: object, *, validate: bool = True) -> "Velocity":
        """Assemble a ``Velocity`` from linear and angular components.

        Parameters
        ----------
        linear : object
            Operand/component input consumed by this operation.
        angular : object
            Operand/component input consumed by this operation.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Velocity
            Result of applying this operation with TAL semantic constraints preserved.

        Notes
        -----
        Linear and angular component core dimensions must be distinct because
        xarray dimensions are name-addressed.

        Examples
        --------
        >>> velocity = Velocity.from_linear_angular(linear, angular)  # doctest: +SKIP
        >>> sorted(velocity.as_dataset().data_vars)  # doctest: +SKIP
        ['angular_velocity', 'linear_velocity']
        """
        return family_from_linear_angular(
            linear,
            angular,
            cfg=_VELOCITY_CONFIG,
            classes=_classes(),
            owner="spatial.velocity.from_linear_angular",
            validate=validate,
        )
    
    @classmethod
    def from_vector6(cls, data: "AnalysisObject | xr.Dataset | xr.DataArray", *, validate: bool = True) -> "Velocity":
        """Construct a ``Velocity`` from a 6-vector representation.

        Parameters
        ----------
        data : AnalysisObject | xr.Dataset | xr.DataArray
            Input data payload used to construct/derive an output object.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Velocity
            Result of applying this operation with TAL semantic constraints preserved.

        Notes
        -----
        The input must have a single six-label core dimension in TAL's velocity
        component order.

        Examples
        --------
        >>> velocity = Velocity.from_vector6(vector6_ao)  # doctest: +SKIP
        >>> velocity.as_components().linear()  # doctest: +SKIP
        """
        return family_from_vector6(
            data,
            cfg=_VELOCITY_CONFIG,
            classes=_classes(),
            owner="spatial.velocity.from_vector6",
            validate=validate,
        )
    
    def to_rep(self, rep: Literal["components", "vector6"], *, validate: bool = True) -> "Velocity":
        """Convert between ``components`` and ``vector6`` velocity representations.

        Parameters
        ----------
        rep : Literal['components', 'vector6']
            Target representation, either ``"components"`` or ``"vector6"``.
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Velocity
            Velocity stored in the requested representation.

        Notes
        -----
        Component representation stores separate linear and angular variables.
        ``vector6`` stores them along one six-component core dimension.

        Examples
        --------
        >>> velocity = Velocity.from_linear_angular(linear, angular)  # doctest: +SKIP
        >>> velocity.to_rep("vector6").as_dataset()["velocity"].shape[-1]  # doctest: +SKIP
        6
        """
        return family_to_rep(
            self,
            rep,
            cfg=_VELOCITY_CONFIG,
            classes=_classes(),
            owner="spatial.velocity.to_rep",
            validate=validate,
        )
    
    def as_components(self, *, validate: bool = True) -> "Velocity":
        """Return velocity in ``components`` representation.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Velocity
            Velocity with separate linear and angular variables.

        Notes
        -----
        This is equivalent to ``to_rep("components")``.

        Examples
        --------
        >>> velocity = Velocity.from_linear_angular(linear, angular)  # doctest: +SKIP
        >>> sorted(velocity.as_components().as_dataset().data_vars)  # doctest: +SKIP
        ['angular_velocity', 'linear_velocity']
        """
        return family_as_components(
            self,
            cfg=_VELOCITY_CONFIG,
            classes=_classes(),
            owner="spatial.velocity.to_rep",
            validate=validate,
        )
    
    def as_vector6(self, *, validate: bool = True) -> "Velocity":
        """Return velocity in 6-vector representation.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Velocity
            Velocity with one six-component core dimension.

        Notes
        -----
        This is equivalent to ``to_rep("vector6")``.

        Examples
        --------
        >>> velocity = Velocity.from_linear_angular(linear, angular)  # doctest: +SKIP
        >>> velocity.as_vector6().as_dataset()["velocity"].shape[-1]  # doctest: +SKIP
        6
        """
        return family_as_vector6(
            self,
            cfg=_VELOCITY_CONFIG,
            classes=_classes(),
            owner="spatial.velocity.to_rep",
            validate=validate,
        )
    
    def linear(self, *, validate: bool = True) -> LinearVelocity:
        """Extract the linear component as ``LinearVelocity``.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        LinearVelocity
            Linear velocity component.

        Notes
        -----
        Vector6 velocities are converted to component representation before
        extraction.

        Examples
        --------
        >>> velocity = Velocity.from_linear_angular(linear, angular)  # doctest: +SKIP
        >>> isinstance(velocity.linear(), LinearVelocity)  # doctest: +SKIP
        True
        """
        return family_linear(
            self,
            cfg=_VELOCITY_CONFIG,
            classes=_classes(),
            owner="spatial.velocity.linear",
            validate=validate,
        )
    
    def angular(self, *, validate: bool = True) -> AngularVelocity:
        """Extract the angular component as ``AngularVelocity``.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        AngularVelocity
            Angular velocity component.

        Notes
        -----
        Vector6 velocities are converted to component representation before
        extraction.

        Examples
        --------
        >>> velocity = Velocity.from_linear_angular(linear, angular)  # doctest: +SKIP
        >>> isinstance(velocity.angular(), AngularVelocity)  # doctest: +SKIP
        True
        """
        return family_angular(
            self,
            cfg=_VELOCITY_CONFIG,
            classes=_classes(),
            owner="spatial.velocity.angular",
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
    ) -> "Acceleration":
        """Differentiate velocity family into acceleration family.

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
        Acceleration
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
        from .ops.kinematics_family_temporal_ops import differentiate_velocity_family

        return differentiate_velocity_family(
            self,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.velocity.differentiate",
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
    ) -> "Velocity":
        """Apply family-preserving smoothing to velocity components.

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
        Velocity
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
        from .ops.kinematics_family_temporal_ops import smooth_velocity_family

        return smooth_velocity_family(
            self,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.velocity.smooth",
        )
    
    def to_frame(
        self,
        dst: "Frame | str",
        *,
        edge_pose_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "Velocity":
        """Transform velocity family into a destination frame using pose edges.

        Parameters
        ----------
        dst : Frame | str
            Destination frame id/object.
        edge_pose_fn : object
            Callable resolving pose edges for frame-path traversal.
        opts : PathSolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``graph`` (override graph source), ``strict`` (strict path checks), and ``kinematics_support`` for velocity/acceleration transport metadata.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Velocity
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
        from .ops.kinematics_frame_ops import to_frame_velocity_family

        return to_frame_velocity_family(
            self,
            dst=dst,
            edge_pose_fn=edge_pose_fn,
            opts=opts,
            validate=validate,
            owner="spatial.velocity.to_frame",
        )
    
    def express_in(
        self,
        dst: "Frame | str",
        *,
        edge_rotation_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "Velocity":
        """Express velocity family in a destination frame using rotation edges.

        Parameters
        ----------
        dst : Frame | str
            Destination frame id/object.
        edge_rotation_fn : object
            Callable resolving rotation edges for frame-path traversal.
        opts : PathSolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``graph`` (override graph source), ``strict`` (strict path checks), and ``kinematics_support`` for velocity/acceleration transport metadata.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Velocity
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
        from .ops.kinematics_frame_ops import express_in_velocity_family

        return express_in_velocity_family(
            self,
            dst=dst,
            edge_rotation_fn=edge_rotation_fn,
            opts=opts,
            validate=validate,
            owner="spatial.velocity.express_in",
        )


__all__ = [
    "AngularVelocity",
    "LinearVelocity",
    "Velocity",
]


from .ops.magnitude_ops import install_velocity_magnitude_methods as _install_velocity_magnitude_methods

_install_velocity_magnitude_methods(LinearVelocity, AngularVelocity)
