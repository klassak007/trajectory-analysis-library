from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import xarray as xr

from tal.core.analysis_object import AnalysisObject

from .kinematics.family import (
    KinematicsClasses,
    KinematicsFamilyConfig,
    coerce_source,
    enforce_angular_invariants,
    enforce_family_invariants,
    enforce_linear_invariants,
    family_angular,
    family_as_components,
    family_as_vector6,
    family_from_linear_angular,
    family_from_vector6,
    family_linear,
    family_to_rep,
    normalize_typed_metadata,
)
from .kinematics.vector6_ops import ACCELERATION_VECTOR6_OPTS
from .metadata import (
    get_acceleration_rep,
    get_angular_acceleration_rep,
    get_linear_acceleration_rep,
    normalize_kinematics_kind,
    set_acceleration_rep,
    set_angular_acceleration_rep,
    set_kinematics_kind,
    set_linear_acceleration_rep,
    validate_spatial_roles,
)
from .kinematics.paired_components import PairAssemblyOptions

_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")

if TYPE_CHECKING:
    from tal.frames import Frame

    from .path_solve import PathSolveOptions
    from .temporal.options import KinematicsIntegralOptions, KinematicsSmoothingOptions
    from .velocity import AngularVelocity, LinearVelocity, Velocity


def _set_linear_rep(ds: xr.Dataset, rep: str, validate: bool, owner: str) -> xr.Dataset:
    return set_linear_acceleration_rep(ds, rep=rep, validate=validate, owner=owner)


def _set_angular_rep(ds: xr.Dataset, rep: str, validate: bool, owner: str) -> xr.Dataset:
    return set_angular_acceleration_rep(ds, rep=rep, validate=validate, owner=owner)


def _set_family_rep(ds: xr.Dataset, rep: str, validate: bool, owner: str) -> xr.Dataset:
    return set_acceleration_rep(ds, rep=rep, validate=validate, owner=owner)


def _get_linear_rep(ds: xr.Dataset, owner: str) -> str:
    return get_linear_acceleration_rep(ds, owner=owner)


def _get_angular_rep(ds: xr.Dataset, owner: str) -> str:
    return get_angular_acceleration_rep(ds, owner=owner)


def _get_family_rep(ds: xr.Dataset, owner: str) -> str:
    return get_acceleration_rep(ds, owner=owner)


def _set_kind(ds: xr.Dataset, kind: str, validate: bool, owner: str) -> xr.Dataset:
    return set_kinematics_kind(ds, kind=kind, validate=validate, owner=owner)


def _normalize_kind(ds: xr.Dataset, expected_kind: str, validate: bool, owner: str) -> xr.Dataset:
    return normalize_kinematics_kind(ds, expected_kind=expected_kind, validate=validate, owner=owner)


_ACCELERATION_CONFIG = KinematicsFamilyConfig(
    family_name="acceleration",
    family_kind="acceleration",
    linear_kind="linear_acceleration",
    angular_kind="angular_acceleration",
    linear_class_name="LinearAcceleration",
    angular_class_name="AngularAcceleration",
    family_class_name="Acceleration",
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
    vector6_opts=ACCELERATION_VECTOR6_OPTS,
    pair_opts=PairAssemblyOptions(
        left_what="LinearAcceleration",
        right_what="AngularAcceleration",
        pair_what="acceleration",
        left_component_name="linear",
        right_component_name="angular",
        left_expected_labels=_XYZ_LABELS,
        right_expected_labels=_XYZ_LABELS,
    ),
)


def _classes() -> KinematicsClasses:
    return KinematicsClasses(
        linear_cls=LinearAcceleration,
        angular_cls=AngularAcceleration,
        family_cls=Acceleration,
    )


class LinearAcceleration(AnalysisObject):
    """Linear acceleration vector type.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    XYZ_LABELS: tuple[str, str, str] = _XYZ_LABELS

    def __init__(self, data: "AnalysisObject | xr.Dataset | xr.DataArray") -> None:
        source = coerce_source(data, owner="spatial.linear_acceleration.__init__")
        super().__init__(source.unsafe_data)
        self._normalize_metadata(owner="spatial.linear_acceleration.__init__")
        self._enforce_invariants(owner="spatial.linear_acceleration.__init__")

    @classmethod
    def _from_validated(cls, ds: xr.Dataset | xr.DataArray) -> "LinearAcceleration":
        obj = super()._from_validated(ds)
        obj._normalize_metadata(owner=f"{cls.__name__}._from_validated")
        obj._enforce_invariants(owner=f"{cls.__name__}._from_validated")
        return obj

    @classmethod
    def _from_unvalidated(cls, ds: xr.Dataset | xr.DataArray) -> "LinearAcceleration":
        obj = super()._from_unvalidated(ds)
        obj._normalize_metadata(owner=f"{cls.__name__}._from_unvalidated")
        obj._enforce_invariants(owner=f"{cls.__name__}._from_unvalidated")
        return obj

    def _normalize_metadata(self, *, owner: str) -> None:
        normalized = normalize_typed_metadata(
            self.unsafe_data,
            rep_getter=_ACCELERATION_CONFIG.get_linear_rep,
            rep_setter=_ACCELERATION_CONFIG.set_linear_rep,
            expected_kind=_ACCELERATION_CONFIG.linear_kind,
            cfg=_ACCELERATION_CONFIG,
            owner=owner,
        )
        self._bind_dataset(normalized)

    def _enforce_invariants(self, *, owner: str) -> None:
        enforce_linear_invariants(self.unsafe_data, owner=owner, cfg=_ACCELERATION_CONFIG)

    def integrate(
        self,
        *,
        on: str | None = None,
        opts: "KinematicsIntegralOptions | None" = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: tuple[str, ...] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "LinearVelocity":
        """Integrate linear acceleration into linear velocity.

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
        >>> opts = KinematicsIntegralOptions(method='simpson')
        >>> isinstance(opts, KinematicsIntegralOptions)
        True
        """
        from .ops.kinematics_temporal_ops import integrate_linear_acceleration_to_linear_velocity

        return integrate_linear_acceleration_to_linear_velocity(
            self,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.linear_acceleration.integrate",
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
    ) -> "LinearAcceleration":
        """Apply kinematics-aware smoothing to linear acceleration.

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
            owner="spatial.linear_acceleration.smooth",
        )

    def to_frame(
        self,
        dst: "Frame | str",
        *,
        edge_pose_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "LinearAcceleration":
        """Transform linear acceleration into a destination frame using pose edges.

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
        >>> opts = PathSolveOptions(strict=False)
        >>> isinstance(opts, PathSolveOptions)
        True
        """
        from .ops.kinematics_frame_ops import to_frame_linear_acceleration

        return to_frame_linear_acceleration(
            self,
            dst=dst,
            edge_pose_fn=edge_pose_fn,
            opts=opts,
            validate=validate,
            owner="spatial.linear_acceleration.to_frame",
        )

    def express_in(
        self,
        dst: "Frame | str",
        *,
        edge_rotation_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "LinearAcceleration":
        """Express linear acceleration in a destination frame using rotation edges.

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
        >>> opts = PathSolveOptions(strict=False)
        >>> isinstance(opts, PathSolveOptions)
        True
        """
        from .ops.kinematics_frame_ops import express_in_linear_acceleration

        return express_in_linear_acceleration(
            self,
            dst=dst,
            edge_rotation_fn=edge_rotation_fn,
            opts=opts,
            validate=validate,
            owner="spatial.linear_acceleration.express_in",
        )


class AngularAcceleration(AnalysisObject):
    """Angular acceleration vector type.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    XYZ_LABELS: tuple[str, str, str] = _XYZ_LABELS

    def __init__(self, data: "AnalysisObject | xr.Dataset | xr.DataArray") -> None:
        source = coerce_source(data, owner="spatial.angular_acceleration.__init__")
        super().__init__(source.unsafe_data)
        self._normalize_metadata(owner="spatial.angular_acceleration.__init__")
        self._enforce_invariants(owner="spatial.angular_acceleration.__init__")

    @classmethod
    def _from_validated(cls, ds: xr.Dataset | xr.DataArray) -> "AngularAcceleration":
        obj = super()._from_validated(ds)
        obj._normalize_metadata(owner=f"{cls.__name__}._from_validated")
        obj._enforce_invariants(owner=f"{cls.__name__}._from_validated")
        return obj

    @classmethod
    def _from_unvalidated(cls, ds: xr.Dataset | xr.DataArray) -> "AngularAcceleration":
        obj = super()._from_unvalidated(ds)
        obj._normalize_metadata(owner=f"{cls.__name__}._from_unvalidated")
        obj._enforce_invariants(owner=f"{cls.__name__}._from_unvalidated")
        return obj

    def _normalize_metadata(self, *, owner: str) -> None:
        normalized = normalize_typed_metadata(
            self.unsafe_data,
            rep_getter=_ACCELERATION_CONFIG.get_angular_rep,
            rep_setter=_ACCELERATION_CONFIG.set_angular_rep,
            expected_kind=_ACCELERATION_CONFIG.angular_kind,
            cfg=_ACCELERATION_CONFIG,
            owner=owner,
        )
        self._bind_dataset(normalized)

    def _enforce_invariants(self, *, owner: str) -> None:
        enforce_angular_invariants(self.unsafe_data, owner=owner, cfg=_ACCELERATION_CONFIG)

    def integrate(
        self,
        *,
        on: str | None = None,
        opts: "KinematicsIntegralOptions | None" = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: tuple[str, ...] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "AngularVelocity":
        """Integrate angular acceleration into angular velocity.

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
        >>> opts = KinematicsIntegralOptions(method='simpson')
        >>> isinstance(opts, KinematicsIntegralOptions)
        True
        """
        from .ops.kinematics_temporal_ops import integrate_angular_acceleration_to_angular_velocity

        return integrate_angular_acceleration_to_angular_velocity(
            self,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.angular_acceleration.integrate",
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
    ) -> "AngularAcceleration":
        """Apply kinematics-aware smoothing to angular acceleration.

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
            owner="spatial.angular_acceleration.smooth",
        )

    def to_frame(
        self,
        dst: "Frame | str",
        *,
        edge_pose_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "AngularAcceleration":
        """Transform angular acceleration into a destination frame using pose edges.

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
        >>> opts = PathSolveOptions(strict=False)
        >>> isinstance(opts, PathSolveOptions)
        True
        """
        from .ops.kinematics_frame_ops import to_frame_angular_acceleration

        return to_frame_angular_acceleration(
            self,
            dst=dst,
            edge_pose_fn=edge_pose_fn,
            opts=opts,
            validate=validate,
            owner="spatial.angular_acceleration.to_frame",
        )

    def express_in(
        self,
        dst: "Frame | str",
        *,
        edge_rotation_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "AngularAcceleration":
        """Express angular acceleration in a destination frame using rotation edges.

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
        >>> opts = PathSolveOptions(strict=False)
        >>> isinstance(opts, PathSolveOptions)
        True
        """
        from .ops.kinematics_frame_ops import express_in_angular_acceleration

        return express_in_angular_acceleration(
            self,
            dst=dst,
            edge_rotation_fn=edge_rotation_fn,
            opts=opts,
            validate=validate,
            owner="spatial.angular_acceleration.express_in",
        )


class Acceleration(AnalysisObject):
    """Spatial acceleration family type (linear + angular).

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    XYZ_LABELS: tuple[str, str, str] = _XYZ_LABELS

    def __init__(self, data: "AnalysisObject | xr.Dataset | xr.DataArray") -> None:
        source = coerce_source(data, owner="spatial.acceleration.__init__")
        super().__init__(source.unsafe_data)
        self._normalize_metadata(owner="spatial.acceleration.__init__")
        self._enforce_invariants(owner="spatial.acceleration.__init__")

    @classmethod
    def _from_validated(cls, ds: xr.Dataset | xr.DataArray) -> "Acceleration":
        obj = super()._from_validated(ds)
        obj._normalize_metadata(owner=f"{cls.__name__}._from_validated")
        obj._enforce_invariants(owner=f"{cls.__name__}._from_validated")
        return obj

    @classmethod
    def _from_unvalidated(cls, ds: xr.Dataset | xr.DataArray) -> "Acceleration":
        obj = super()._from_unvalidated(ds)
        obj._normalize_metadata(owner=f"{cls.__name__}._from_unvalidated")
        obj._enforce_invariants(owner=f"{cls.__name__}._from_unvalidated")
        return obj

    def _normalize_metadata(self, *, owner: str) -> None:
        normalized = normalize_typed_metadata(
            self.unsafe_data,
            rep_getter=_ACCELERATION_CONFIG.get_family_rep,
            rep_setter=_ACCELERATION_CONFIG.set_family_rep,
            expected_kind=_ACCELERATION_CONFIG.family_kind,
            cfg=_ACCELERATION_CONFIG,
            owner=owner,
        )
        self._bind_dataset(normalized)

    def _enforce_invariants(self, *, owner: str) -> None:
        enforce_family_invariants(self.unsafe_data, owner=owner, cfg=_ACCELERATION_CONFIG)

    @classmethod
    def from_linear_angular(cls, linear: object, angular: object, *, validate: bool = True) -> "Acceleration":
        """Assemble an ``Acceleration`` from linear and angular components.

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
        Acceleration
            Result of applying this operation with TAL semantic constraints preserved.

        Notes
        -----
        Linear and angular component core dimensions must be distinct because
        xarray dimensions are name-addressed.

        Examples
        --------
        >>> acceleration = Acceleration.from_linear_angular(linear, angular)  # doctest: +SKIP
        >>> sorted(acceleration.unsafe_data.data_vars)  # doctest: +SKIP
        ['angular_acceleration', 'linear_acceleration']
        """
        return family_from_linear_angular(
            linear,
            angular,
            cfg=_ACCELERATION_CONFIG,
            classes=_classes(),
            owner="spatial.acceleration.from_linear_angular",
            validate=validate,
        )

    @classmethod
    def from_vector6(
        cls,
        data: "AnalysisObject | xr.Dataset | xr.DataArray",
        *,
        validate: bool = True,
    ) -> "Acceleration":
        """Construct an ``Acceleration`` from a 6-vector representation.

        Parameters
        ----------
        data : AnalysisObject | xr.Dataset | xr.DataArray
            Input data payload used to construct/derive an output object.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Acceleration
            Result of applying this operation with TAL semantic constraints preserved.

        Notes
        -----
        The input must have a single six-label core dimension in TAL's
        acceleration component order.

        Examples
        --------
        >>> acceleration = Acceleration.from_vector6(vector6_ao)  # doctest: +SKIP
        >>> acceleration.as_components().linear()  # doctest: +SKIP
        """
        return family_from_vector6(
            data,
            cfg=_ACCELERATION_CONFIG,
            classes=_classes(),
            owner="spatial.acceleration.from_vector6",
            validate=validate,
        )

    def to_rep(self, rep: Literal["components", "vector6"], *, validate: bool = True) -> "Acceleration":
        """Convert between ``components`` and ``vector6`` acceleration representations.

        Parameters
        ----------
        rep : Literal['components', 'vector6']
            Target representation, either ``"components"`` or ``"vector6"``.
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Acceleration
            Acceleration stored in the requested representation.

        Notes
        -----
        Component representation stores separate linear and angular variables.
        ``vector6`` stores them along one six-component core dimension.

        Examples
        --------
        >>> acceleration = Acceleration.from_linear_angular(linear, angular)  # doctest: +SKIP
        >>> acceleration.to_rep("vector6").unsafe_data["acceleration"].shape[-1]  # doctest: +SKIP
        6
        """
        return family_to_rep(
            self,
            rep,
            cfg=_ACCELERATION_CONFIG,
            classes=_classes(),
            owner="spatial.acceleration.to_rep",
            validate=validate,
        )

    def as_components(self, *, validate: bool = True) -> "Acceleration":
        """Return acceleration in ``components`` representation.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Acceleration
            Acceleration with separate linear and angular variables.

        Notes
        -----
        This is equivalent to ``to_rep("components")``.

        Examples
        --------
        >>> acceleration = Acceleration.from_linear_angular(linear, angular)  # doctest: +SKIP
        >>> sorted(acceleration.as_components().unsafe_data.data_vars)  # doctest: +SKIP
        ['angular_acceleration', 'linear_acceleration']
        """
        return family_as_components(
            self,
            cfg=_ACCELERATION_CONFIG,
            classes=_classes(),
            owner="spatial.acceleration.to_rep",
            validate=validate,
        )

    def as_vector6(self, *, validate: bool = True) -> "Acceleration":
        """Return acceleration in 6-vector representation.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Acceleration
            Acceleration with one six-component core dimension.

        Notes
        -----
        This is equivalent to ``to_rep("vector6")``.

        Examples
        --------
        >>> acceleration = Acceleration.from_linear_angular(linear, angular)  # doctest: +SKIP
        >>> acceleration.as_vector6().unsafe_data["acceleration"].shape[-1]  # doctest: +SKIP
        6
        """
        return family_as_vector6(
            self,
            cfg=_ACCELERATION_CONFIG,
            classes=_classes(),
            owner="spatial.acceleration.to_rep",
            validate=validate,
        )

    def linear(self, *, validate: bool = True) -> LinearAcceleration:
        """Extract the linear component as ``LinearAcceleration``.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        LinearAcceleration
            Linear acceleration component.

        Notes
        -----
        Vector6 accelerations are converted to component representation before
        extraction.

        Examples
        --------
        >>> acceleration = Acceleration.from_linear_angular(linear, angular)  # doctest: +SKIP
        >>> isinstance(acceleration.linear(), LinearAcceleration)  # doctest: +SKIP
        True
        """
        return family_linear(
            self,
            cfg=_ACCELERATION_CONFIG,
            classes=_classes(),
            owner="spatial.acceleration.linear",
            validate=validate,
        )

    def angular(self, *, validate: bool = True) -> AngularAcceleration:
        """Extract the angular component as ``AngularAcceleration``.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        AngularAcceleration
            Angular acceleration component.

        Notes
        -----
        Vector6 accelerations are converted to component representation before
        extraction.

        Examples
        --------
        >>> acceleration = Acceleration.from_linear_angular(linear, angular)  # doctest: +SKIP
        >>> isinstance(acceleration.angular(), AngularAcceleration)  # doctest: +SKIP
        True
        """
        return family_angular(
            self,
            cfg=_ACCELERATION_CONFIG,
            classes=_classes(),
            owner="spatial.acceleration.angular",
            validate=validate,
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
    ) -> "Velocity":
        """Integrate acceleration family into velocity family.

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
        >>> opts = KinematicsIntegralOptions(method='simpson')
        >>> isinstance(opts, KinematicsIntegralOptions)
        True
        """
        from .ops.kinematics_family_temporal_ops import integrate_acceleration_family

        return integrate_acceleration_family(
            self,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.acceleration.integrate",
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
    ) -> "Acceleration":
        """Apply family-preserving smoothing to acceleration components.

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
        >>> opts = KinematicsSmoothingOptions(method='gaussian')
        >>> isinstance(opts, KinematicsSmoothingOptions)
        True
        """
        from .ops.kinematics_family_temporal_ops import smooth_acceleration_family

        return smooth_acceleration_family(
            self,
            on=on,
            opts=opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.acceleration.smooth",
        )

    def to_frame(
        self,
        dst: "Frame | str",
        *,
        edge_pose_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "Acceleration":
        """Transform acceleration family into a destination frame using pose edges.

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
        >>> opts = PathSolveOptions(strict=False)
        >>> isinstance(opts, PathSolveOptions)
        True
        """
        from .ops.kinematics_frame_ops import to_frame_acceleration_family

        return to_frame_acceleration_family(
            self,
            dst=dst,
            edge_pose_fn=edge_pose_fn,
            opts=opts,
            validate=validate,
            owner="spatial.acceleration.to_frame",
        )

    def express_in(
        self,
        dst: "Frame | str",
        *,
        edge_rotation_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "Acceleration":
        """Express acceleration family in a destination frame using rotation edges.

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
        >>> opts = PathSolveOptions(strict=False)
        >>> isinstance(opts, PathSolveOptions)
        True
        """
        from .ops.kinematics_frame_ops import express_in_acceleration_family

        return express_in_acceleration_family(
            self,
            dst=dst,
            edge_rotation_fn=edge_rotation_fn,
            opts=opts,
            validate=validate,
            owner="spatial.acceleration.express_in",
        )


__all__ = [
    "Acceleration",
    "AngularAcceleration",
    "LinearAcceleration",
]


from .ops.magnitude_ops import install_acceleration_magnitude_methods as _install_acceleration_magnitude_methods

_install_acceleration_magnitude_methods(LinearAcceleration, AngularAcceleration)
