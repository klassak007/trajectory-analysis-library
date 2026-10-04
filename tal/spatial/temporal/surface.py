from __future__ import annotations

from collections.abc import Sequence

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.analysis_object import AnalysisObject
from tal.core.orchestration.inputs import coerce_analysis_object_input

from ..acceleration import Acceleration, AngularAcceleration, LinearAcceleration
from ..metadata import KINEMATICS_KIND_VALUES
from ..position import Position
from ..velocity import AngularVelocity, LinearVelocity, Velocity

AO_TEMPORAL_KIND_VALUES: tuple[str, ...] = ("position", *KINEMATICS_KIND_VALUES)

_KIND_TO_TYPED_CLASS: dict[str, type] = {
    "position": Position,
    "linear_velocity": LinearVelocity,
    "angular_velocity": AngularVelocity,
    "velocity": Velocity,
    "linear_acceleration": LinearAcceleration,
    "angular_acceleration": AngularAcceleration,
    "acceleration": Acceleration,
}

_DIFFERENTIATE_METHODS: dict[str, str] = {
    "position": "differentiate",
    "linear_velocity": "differentiate",
    "angular_velocity": "differentiate",
    "velocity": "differentiate",
}

_INTEGRATE_METHODS: dict[str, str] = {
    "linear_velocity": "integrate",
    "linear_acceleration": "integrate",
    "angular_acceleration": "integrate",
    "acceleration": "integrate",
}

_SMOOTH_METHODS: dict[str, str] = {kind: "smooth" for kind in AO_TEMPORAL_KIND_VALUES}


def require_supported_ao_temporal_kind(kind: object, *, owner: str) -> str:
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"{owner}: kind must be a non-empty string; allowed={list(AO_TEMPORAL_KIND_VALUES)!r}.")
    cleaned = kind.strip()
    if cleaned not in AO_TEMPORAL_KIND_VALUES:
        raise ValueError(f"{owner}: unsupported kind {cleaned!r}; allowed={list(AO_TEMPORAL_KIND_VALUES)!r}.")
    return cleaned


def _typed_kind(value: object) -> str | None:
    if isinstance(value, Position):
        return "position"
    if isinstance(value, LinearVelocity):
        return "linear_velocity"
    if isinstance(value, AngularVelocity):
        return "angular_velocity"
    if isinstance(value, Velocity):
        return "velocity"
    if isinstance(value, LinearAcceleration):
        return "linear_acceleration"
    if isinstance(value, AngularAcceleration):
        return "angular_acceleration"
    if isinstance(value, Acceleration):
        return "acceleration"
    return None


def _coerce_temporal_source(value: object, *, kind: str | None, owner: str):
    inferred_kind = _typed_kind(value)
    if inferred_kind is not None:
        if kind is not None and require_supported_ao_temporal_kind(kind, owner=owner) != inferred_kind:
            raise ValueError(f"{owner}: kind {kind!r} does not match typed source kind {inferred_kind!r}.")
        return value, inferred_kind
    if kind is None:
        raise ValueError(f"{owner}: kind is required for AO-like untyped inputs; allowed={list(AO_TEMPORAL_KIND_VALUES)!r}.")
    resolved_kind = require_supported_ao_temporal_kind(kind, owner=owner)
    cls = _KIND_TO_TYPED_CLASS[resolved_kind]
    source = coerce_analysis_object_input(value, owner=owner)
    return cls(source), resolved_kind


def _dispatch_temporal_kind(
    source,
    *,
    kind: str,
    method_table: dict[str, str],
    on: str | None,
    opts: object | None,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
):
    method_name = method_table.get(kind)
    if method_name is None:
        supported = sorted(method_table)
        raise ValueError(f"{owner}: operation is not supported for kind {kind!r}; supported={supported!r}.")
    method = getattr(source, method_name)
    return method(
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
    )


def _to_analysis_object(value, *, validate: bool) -> AnalysisObject:
    if validate:
        return AnalysisObject._from_validated(analysis_object_dataset(value))
    return AnalysisObject._from_unvalidated(analysis_object_dataset(value))


def _maybe_wrap_target(
    value,
    *,
    target_cls: type[AnalysisObject] | None,
    validate: bool,
    kind: str,
    owner: str,
):
    if target_cls is None or target_cls is AnalysisObject:
        return _to_analysis_object(value, validate=validate)
    if not isinstance(target_cls, type) or not issubclass(target_cls, AnalysisObject):
        raise TypeError(f"{owner}: target_cls must be an AnalysisObject subclass or None.")
    if value.__class__ is not target_cls:
        raise ValueError(
            f"{owner}: target_cls must be {value.__class__.__name__} for kind {kind!r}; got {target_cls.__name__}."
        )
    return value


def differentiate(
    source: object,
    *,
    kind: str | None = None,
    on: str | None = None,
    opts: object | None = None,
    target_cls: type[AnalysisObject] | None = None,
    validate: bool = True,
    sequence_dim: str | None = None,
    batch_dims: Sequence[str] | None = None,
    sequence_size_coord: str | None = None,
):
    """Differentiate a supported spatial type along sequence semantics.

    Parameters
    ----------
    source : object
        Input source value consumed by this operation.
    kind : str | None, optional
        Operation-family selector used to dispatch temporal behavior.
    on : str | None, optional
        Coordinate/dimension name used as the operation domain.
    opts : object | None, optional
        Temporal options for the selected ``kind``. When ``None``, defaults are
        used. Common fields include ``method`` (for derivative/integral/smoothing
        mode), ``window``/``poly_order`` for local-polynomial methods, and
        ``sigma`` for gaussian smoothing.
    target_cls : type[AnalysisObject] | None, optional
        Optional output wrapper class constraint for generic temporal APIs.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.
    sequence_dim : str | None, optional
        Optional override for the sequence dimension used by temporal semantics.
    batch_dims : Sequence[str] | None, optional
        Optional override for batch dimensions used by temporal semantics.
    sequence_size_coord : str | None, optional
        Optional sequence-size coordinate used for ragged validity handling.

    Returns
    -------
    object
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
    >>> from tal.spatial.temporal.options import KinematicsDerivativeOptions
    >>> opts = KinematicsDerivativeOptions(method='local_poly')
    >>> isinstance(opts, KinematicsDerivativeOptions)
    True
    """
    owner = "spatial.temporal.differentiate"
    typed_source, resolved_kind = _coerce_temporal_source(source, kind=kind, owner=owner)
    out = _dispatch_temporal_kind(
        typed_source,
        kind=resolved_kind,
        method_table=_DIFFERENTIATE_METHODS,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
    )
    return _maybe_wrap_target(out, target_cls=target_cls, validate=validate, kind=resolved_kind, owner=owner)


def integrate(
    source: object,
    *,
    kind: str | None = None,
    on: str | None = None,
    opts: object | None = None,
    target_cls: type[AnalysisObject] | None = None,
    validate: bool = True,
    sequence_dim: str | None = None,
    batch_dims: Sequence[str] | None = None,
    sequence_size_coord: str | None = None,
):
    """Integrate a supported spatial type along sequence semantics.

    Parameters
    ----------
    source : object
        Input source value consumed by this operation.
    kind : str | None, optional
        Operation-family selector used to dispatch temporal behavior.
    on : str | None, optional
        Coordinate/dimension name used as the operation domain.
    opts : object | None, optional
        Temporal options for the selected ``kind``. When ``None``, defaults are
        used. Common fields include ``method`` (for derivative/integral/smoothing
        mode), ``window``/``poly_order`` for local-polynomial methods, and
        ``sigma`` for gaussian smoothing.
    target_cls : type[AnalysisObject] | None, optional
        Optional output wrapper class constraint for generic temporal APIs.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.
    sequence_dim : str | None, optional
        Optional override for the sequence dimension used by temporal semantics.
    batch_dims : Sequence[str] | None, optional
        Optional override for batch dimensions used by temporal semantics.
    sequence_size_coord : str | None, optional
        Optional sequence-size coordinate used for ragged validity handling.

    Returns
    -------
    object
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
    >>> from tal.spatial.temporal.options import KinematicsIntegralOptions
    >>> opts = KinematicsIntegralOptions(method='simpson')
    >>> isinstance(opts, KinematicsIntegralOptions)
    True
    """
    owner = "spatial.temporal.integrate"
    typed_source, resolved_kind = _coerce_temporal_source(source, kind=kind, owner=owner)
    out = _dispatch_temporal_kind(
        typed_source,
        kind=resolved_kind,
        method_table=_INTEGRATE_METHODS,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
    )
    return _maybe_wrap_target(out, target_cls=target_cls, validate=validate, kind=resolved_kind, owner=owner)


def smooth(
    source: object,
    *,
    kind: str | None = None,
    on: str | None = None,
    opts: object | None = None,
    target_cls: type[AnalysisObject] | None = None,
    validate: bool = True,
    sequence_dim: str | None = None,
    batch_dims: Sequence[str] | None = None,
    sequence_size_coord: str | None = None,
):
    """Smooth a supported spatial type along sequence semantics.

    Parameters
    ----------
    source : object
        Input source value consumed by this operation.
    kind : str | None, optional
        Operation-family selector used to dispatch temporal behavior.
    on : str | None, optional
        Coordinate/dimension name used as the operation domain.
    opts : object | None, optional
        Temporal options for the selected ``kind``. When ``None``, defaults are
        used. Common fields include ``method`` (for derivative/integral/smoothing
        mode), ``window``/``poly_order`` for local-polynomial methods, and
        ``sigma`` for gaussian smoothing.
    target_cls : type[AnalysisObject] | None, optional
        Optional output wrapper class constraint for generic temporal APIs.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.
    sequence_dim : str | None, optional
        Optional override for the sequence dimension used by temporal semantics.
    batch_dims : Sequence[str] | None, optional
        Optional override for batch dimensions used by temporal semantics.
    sequence_size_coord : str | None, optional
        Optional sequence-size coordinate used for ragged validity handling.

    Returns
    -------
    object
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
    >>> from tal.spatial.temporal.options import KinematicsSmoothingOptions
    >>> opts = KinematicsSmoothingOptions(method='gaussian')
    >>> isinstance(opts, KinematicsSmoothingOptions)
    True
    """
    owner = "spatial.temporal.smooth"
    typed_source, resolved_kind = _coerce_temporal_source(source, kind=kind, owner=owner)
    out = _dispatch_temporal_kind(
        typed_source,
        kind=resolved_kind,
        method_table=_SMOOTH_METHODS,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
    )
    return _maybe_wrap_target(out, target_cls=target_cls, validate=validate, kind=resolved_kind, owner=owner)


__all__ = [
    "AO_TEMPORAL_KIND_VALUES",
    "differentiate",
    "integrate",
    "require_supported_ao_temporal_kind",
    "smooth",
]
