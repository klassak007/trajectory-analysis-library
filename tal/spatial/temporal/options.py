from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from typing import Literal

from tal.core.param_ops.guards import validate_query_dim_name
from tal.core.param_ops.options import validate_eval_options
from tal.core.param_ops.types import ParamEvalOptions

_ROTATION_METHODS = {"preferred", "nearest", "linear", "slerp"}
_DUPLICATE_POLICIES = {"invalid", "left", "right", "raise"}
_DERIVATIVE_METHODS = {"finite_difference", "local_poly"}
_INTEGRAL_METHODS = {"cumulative_trapezoid", "simpson"}
_SMOOTHING_METHODS = {"moving_average", "local_poly", "gaussian"}


@dataclass(frozen=True)
class RotationTemporalOptions:
    """Typed temporal options for rotation interpolation.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    method: Literal["preferred", "nearest", "linear", "slerp"] = "preferred"
    duplicate_policy: Literal["invalid", "left", "right", "raise"] = "invalid"
    query_dim: str = "query"


@dataclass(frozen=True)
class PoseTemporalOptions:
    """Typed temporal options for pose interpolation.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    position_opts: ParamEvalOptions = ParamEvalOptions(method="linear")
    rotation_opts: RotationTemporalOptions = RotationTemporalOptions()
    on: str | None = None


@dataclass(frozen=True)
class KinematicsDerivativeOptions:
    """Typed derivative options for D3 kinematics temporal methods.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    method: Literal["finite_difference", "local_poly"] = "finite_difference"
    order: int = 1
    edge_mode: Literal["one_sided", "partial_renorm"] = "one_sided"
    window: int = 5
    poly_order: int = 2


@dataclass(frozen=True)
class KinematicsIntegralOptions:
    """Typed integral options for D3 kinematics temporal methods.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    method: Literal["cumulative_trapezoid", "simpson"] = "cumulative_trapezoid"
    initial_value: float = 0.0


@dataclass(frozen=True)
class KinematicsSmoothingOptions:
    """Typed smoothing options for D4 kinematics temporal methods.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    method: Literal["moving_average", "local_poly", "gaussian"] = "moving_average"
    window: int = 5
    poly_order: int = 2
    sigma: float = 1.0
    edge_mode: Literal["partial_renorm"] = "partial_renorm"


def _require_positive_odd_window(window: int, *, owner: str, field: str) -> None:
    if not isinstance(window, int) or window < 1 or window % 2 == 0:
        raise ValueError(f"{owner}: opts.{field} must be a positive odd integer.")


def _validate_rotation_temporal_options(opts: RotationTemporalOptions, *, owner: str) -> RotationTemporalOptions:
    validate_query_dim_name(opts.query_dim, owner=owner)
    if opts.method not in _ROTATION_METHODS:
        raise ValueError(f"{owner}: opts.method must be one of {sorted(_ROTATION_METHODS)!r}.")
    if opts.duplicate_policy not in _DUPLICATE_POLICIES:
        raise ValueError(
            f"{owner}: opts.duplicate_policy must be one of {sorted(_DUPLICATE_POLICIES)!r}."
        )
    return opts


def coerce_rotation_temporal_options(opts: object | None, *, owner: str) -> RotationTemporalOptions:
    """Normalize rotation temporal options at typed accessor boundaries.

    Parameters
    ----------
    opts : object | None
        Optional options controlling policy and numeric behavior for this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    RotationTemporalOptions
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if opts is None:
        out = RotationTemporalOptions()
    elif isinstance(opts, RotationTemporalOptions):
        out = opts
    elif isinstance(opts, ParamEvalOptions):
        out = RotationTemporalOptions(
            method=opts.method,
            duplicate_policy=opts.duplicate_policy,
            query_dim=opts.query_dim,
        )
    else:
        raise TypeError(f"{owner}: opts must be RotationTemporalOptions, ParamEvalOptions, or None.")
    return _validate_rotation_temporal_options(out, owner=owner)


def resolve_rotation_method(opts: RotationTemporalOptions) -> Literal["nearest", "linear", "slerp"]:
    if opts.method == "preferred":
        return "slerp"
    return opts.method


def as_rotation_method(
    opts: RotationTemporalOptions,
    *,
    method: Literal["nearest", "linear", "slerp"],
) -> RotationTemporalOptions:
    return replace(opts, method=method)


def _pose_options_from_method(
    *,
    method: str,
    duplicate_policy: Literal["invalid", "left", "right", "raise"],
    query_dim: str,
    owner: str,
) -> PoseTemporalOptions:
    if method not in _ROTATION_METHODS:
        raise ValueError(f"{owner}: opts.method must be one of {sorted(_ROTATION_METHODS)!r}.")
    if method == "nearest":
        position_opts = ParamEvalOptions(
            method="nearest",
            duplicate_policy=duplicate_policy,
            query_dim=query_dim,
        )
        rotation_opts = RotationTemporalOptions(
            method="nearest",
            duplicate_policy=duplicate_policy,
            query_dim=query_dim,
        )
        return PoseTemporalOptions(position_opts=position_opts, rotation_opts=rotation_opts)
    if method == "linear":
        position_opts = ParamEvalOptions(
            method="linear",
            duplicate_policy=duplicate_policy,
            query_dim=query_dim,
        )
        rotation_opts = RotationTemporalOptions(
            method="linear",
            duplicate_policy=duplicate_policy,
            query_dim=query_dim,
        )
        return PoseTemporalOptions(position_opts=position_opts, rotation_opts=rotation_opts)
    position_opts = ParamEvalOptions(
        method="linear",
        duplicate_policy=duplicate_policy,
        query_dim=query_dim,
    )
    rotation_opts = RotationTemporalOptions(
        method="slerp",
        duplicate_policy=duplicate_policy,
        query_dim=query_dim,
    )
    return PoseTemporalOptions(position_opts=position_opts, rotation_opts=rotation_opts)


def _validate_pose_temporal_options(opts: PoseTemporalOptions, *, owner: str) -> PoseTemporalOptions:
    validate_eval_options(opts.position_opts, owner=owner)
    _validate_rotation_temporal_options(opts.rotation_opts, owner=owner)
    if opts.position_opts.query_dim != opts.rotation_opts.query_dim:
        raise ValueError(
            f"{owner}: position_opts.query_dim and rotation_opts.query_dim must match; "
            f"got {opts.position_opts.query_dim!r} vs {opts.rotation_opts.query_dim!r}."
        )
    if opts.on is not None and (not isinstance(opts.on, str) or not opts.on):
        raise ValueError(f"{owner}: opts.on must be a non-empty string or None.")
    return opts


def coerce_pose_temporal_options(opts: object | None, *, owner: str) -> PoseTemporalOptions:
    """Normalize pose temporal options with shorthand support.

    Parameters
    ----------
    opts : object | None
        Optional options controlling policy and numeric behavior for this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    PoseTemporalOptions
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if opts is None:
        out = PoseTemporalOptions()
    elif isinstance(opts, PoseTemporalOptions):
        out = opts
    elif isinstance(opts, ParamEvalOptions):
        out = _pose_options_from_method(
            method=opts.method,
            duplicate_policy=opts.duplicate_policy,
            query_dim=opts.query_dim,
            owner=owner,
        )
    elif isinstance(opts, RotationTemporalOptions):
        out = _pose_options_from_method(
            method=resolve_rotation_method(opts),
            duplicate_policy=opts.duplicate_policy,
            query_dim=opts.query_dim,
            owner=owner,
        )
    else:
        raise TypeError(
            f"{owner}: opts must be PoseTemporalOptions, ParamEvalOptions, RotationTemporalOptions, or None."
        )
    return _validate_pose_temporal_options(out, owner=owner)


def _validate_derivative_options(
    opts: KinematicsDerivativeOptions,
    *,
    owner: str,
) -> KinematicsDerivativeOptions:
    if opts.method not in _DERIVATIVE_METHODS:
        raise ValueError(f"{owner}: opts.method must be one of {sorted(_DERIVATIVE_METHODS)!r}.")
    if not isinstance(opts.order, int) or opts.order <= 0:
        raise ValueError(f"{owner}: opts.order must be a positive integer.")
    _require_positive_odd_window(opts.window, owner=owner, field="window")
    if not isinstance(opts.poly_order, int) or opts.poly_order < 1:
        raise ValueError(f"{owner}: opts.poly_order must be a positive integer.")
    if opts.poly_order >= opts.window:
        raise ValueError(f"{owner}: opts.poly_order must be less than opts.window.")
    if opts.method == "finite_difference" and opts.edge_mode != "one_sided":
        raise ValueError(f"{owner}: opts.edge_mode must be 'one_sided' for finite_difference.")
    if opts.method == "local_poly" and opts.edge_mode != "partial_renorm":
        raise ValueError(f"{owner}: opts.edge_mode must be 'partial_renorm' for local_poly.")
    return opts


def _validate_integral_options(
    opts: KinematicsIntegralOptions,
    *,
    owner: str,
) -> KinematicsIntegralOptions:
    if opts.method not in _INTEGRAL_METHODS:
        raise ValueError(f"{owner}: opts.method must be one of {sorted(_INTEGRAL_METHODS)!r}.")
    if not isinstance(opts.initial_value, Real):
        raise TypeError(f"{owner}: opts.initial_value must be a real scalar.")
    value = float(opts.initial_value)
    if not (value == value and abs(value) != float("inf")):
        raise ValueError(f"{owner}: opts.initial_value must be finite.")
    return opts


def coerce_kinematics_derivative_options(
    opts: object | None,
    *,
    owner: str,
) -> KinematicsDerivativeOptions:
    if opts is None:
        out = KinematicsDerivativeOptions()
    elif isinstance(opts, KinematicsDerivativeOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be KinematicsDerivativeOptions or None.")
    return _validate_derivative_options(out, owner=owner)


def coerce_kinematics_integral_options(
    opts: object | None,
    *,
    owner: str,
) -> KinematicsIntegralOptions:
    if opts is None:
        out = KinematicsIntegralOptions()
    elif isinstance(opts, KinematicsIntegralOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be KinematicsIntegralOptions or None.")
    return _validate_integral_options(out, owner=owner)


def _validate_smoothing_options(
    opts: KinematicsSmoothingOptions,
    *,
    owner: str,
) -> KinematicsSmoothingOptions:
    if opts.method not in _SMOOTHING_METHODS:
        raise ValueError(f"{owner}: opts.method must be one of {sorted(_SMOOTHING_METHODS)!r}.")
    _require_positive_odd_window(opts.window, owner=owner, field="window")
    if opts.method == "local_poly":
        if not isinstance(opts.poly_order, int) or opts.poly_order < 1:
            raise ValueError(f"{owner}: opts.poly_order must be a positive integer.")
        if opts.poly_order >= opts.window:
            raise ValueError(f"{owner}: opts.poly_order must be less than opts.window.")
    if opts.method == "gaussian":
        if not isinstance(opts.sigma, Real):
            raise TypeError(f"{owner}: opts.sigma must be a real scalar.")
        sigma = float(opts.sigma)
        if not (sigma > 0.0 and sigma == sigma and abs(sigma) != float("inf")):
            raise ValueError(f"{owner}: opts.sigma must be positive and finite.")
    if opts.edge_mode != "partial_renorm":
        raise ValueError(f"{owner}: opts.edge_mode must be 'partial_renorm'.")
    return opts


def coerce_kinematics_smoothing_options(
    opts: object | None,
    *,
    owner: str,
) -> KinematicsSmoothingOptions:
    if opts is None:
        out = KinematicsSmoothingOptions()
    elif isinstance(opts, KinematicsSmoothingOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be KinematicsSmoothingOptions or None.")
    return _validate_smoothing_options(out, owner=owner)


def require_supported_derivative_options(opts: KinematicsDerivativeOptions, *, owner: str) -> None:
    if opts.method not in _DERIVATIVE_METHODS:
        raise ValueError(f"{owner}: opts.method must be one of {sorted(_DERIVATIVE_METHODS)!r}.")
    if opts.order != 1:
        raise ValueError(f"{owner}: derivative order {opts.order!r} is not implemented; use order=1.")
    if opts.method == "finite_difference":
        return
    if opts.method == "local_poly":
        if opts.poly_order >= opts.window:
            raise ValueError(f"{owner}: local_poly requires opts.poly_order < opts.window.")
        return
    raise ValueError(f"{owner}: derivative method {opts.method!r} is not implemented.")


def require_supported_integral_options(opts: KinematicsIntegralOptions, *, owner: str) -> None:
    if opts.method not in _INTEGRAL_METHODS:
        raise ValueError(
            f"{owner}: integral method {opts.method!r} is not implemented. "
            "Supported methods are 'cumulative_trapezoid' and 'simpson'."
        )


def require_supported_smoothing_options(opts: KinematicsSmoothingOptions, *, owner: str) -> None:
    if opts.method not in _SMOOTHING_METHODS:
        raise ValueError(f"{owner}: opts.method must be one of {sorted(_SMOOTHING_METHODS)!r}.")
    if opts.method == "local_poly" and opts.poly_order >= opts.window:
        raise ValueError(f"{owner}: smoothing requires opts.poly_order < opts.window.")


__all__ = [
    "KinematicsDerivativeOptions",
    "KinematicsIntegralOptions",
    "KinematicsSmoothingOptions",
    "PoseTemporalOptions",
    "RotationTemporalOptions",
    "as_rotation_method",
    "coerce_kinematics_derivative_options",
    "coerce_kinematics_integral_options",
    "coerce_kinematics_smoothing_options",
    "coerce_pose_temporal_options",
    "coerce_rotation_temporal_options",
    "require_supported_derivative_options",
    "require_supported_integral_options",
    "require_supported_smoothing_options",
    "resolve_rotation_method",
]
