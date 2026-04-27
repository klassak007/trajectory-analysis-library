from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ..metadata import get_acceleration_rep, get_velocity_rep

if TYPE_CHECKING:
    from ..acceleration import Acceleration
    from ..temporal.options import KinematicsDerivativeOptions, KinematicsIntegralOptions, KinematicsSmoothingOptions
    from ..velocity import Velocity


def _wrap_owner_error(exc: Exception, *, owner: str) -> Exception:
    text = str(exc)
    if text.startswith(f"{owner}:"):
        return exc
    return type(exc)(f"{owner}: {text}")


@dataclass(frozen=True)
class FamilyTemporalRequest:
    source: object
    on: str | None
    opts: object | None
    validate: bool
    sequence_dim: str | None
    batch_dims: Sequence[str] | None
    sequence_size_coord: str | None
    owner: str


def _velocity_family_parts(source: Velocity, *, validate: bool, owner: str):
    rep = get_velocity_rep(source.unsafe_data, owner=owner)
    linear = source.linear(validate=validate)
    angular = source.angular(validate=validate)
    return rep, linear, angular


def _acceleration_family_parts(source: Acceleration, *, validate: bool, owner: str):
    rep = get_acceleration_rep(source.unsafe_data, owner=owner)
    linear = source.linear(validate=validate)
    angular = source.angular(validate=validate)
    return rep, linear, angular


def _restore_rep(value, *, source_rep: str, validate: bool):
    if source_rep == "vector6":
        return value.to_rep("vector6", validate=validate)
    return value


def _invoke_member_temporal(value, *, method_name: str, request: FamilyTemporalRequest):
    method = getattr(value, method_name)
    return method(
        on=request.on,
        opts=request.opts,
        validate=request.validate,
        sequence_dim=request.sequence_dim,
        batch_dims=request.batch_dims,
        sequence_size_coord=request.sequence_size_coord,
    )


def _run_family_pair_operation(
    request: FamilyTemporalRequest,
    *,
    parts_resolver,
    method_name: str,
    compose: Callable[..., object],
):
    rep, linear, angular = parts_resolver(request.source, validate=request.validate, owner=request.owner)
    linear_out = _invoke_member_temporal(linear, method_name=method_name, request=request)
    angular_out = _invoke_member_temporal(angular, method_name=method_name, request=request)
    out = compose(linear_out, angular_out, validate=request.validate)
    return _restore_rep(out, source_rep=rep, validate=request.validate)


def differentiate_velocity_family(
    source: Velocity,
    *,
    on: str | None,
    opts: KinematicsDerivativeOptions | None,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> Acceleration:
    from ..acceleration import Acceleration

    request = FamilyTemporalRequest(source, on, opts, validate, sequence_dim, batch_dims, sequence_size_coord, owner)
    try:
        return _run_family_pair_operation(
            request,
            parts_resolver=_velocity_family_parts,
            method_name="differentiate",
            compose=Acceleration.from_linear_angular,
        )
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def smooth_velocity_family(
    source: Velocity,
    *,
    on: str | None,
    opts: KinematicsSmoothingOptions | None,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> Velocity:
    from ..velocity import Velocity

    request = FamilyTemporalRequest(source, on, opts, validate, sequence_dim, batch_dims, sequence_size_coord, owner)
    try:
        return _run_family_pair_operation(
            request,
            parts_resolver=_velocity_family_parts,
            method_name="smooth",
            compose=Velocity.from_linear_angular,
        )
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def integrate_acceleration_family(
    source: Acceleration,
    *,
    on: str | None,
    opts: KinematicsIntegralOptions | None,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> Velocity:
    from ..velocity import Velocity

    request = FamilyTemporalRequest(source, on, opts, validate, sequence_dim, batch_dims, sequence_size_coord, owner)
    try:
        return _run_family_pair_operation(
            request,
            parts_resolver=_acceleration_family_parts,
            method_name="integrate",
            compose=Velocity.from_linear_angular,
        )
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def smooth_acceleration_family(
    source: Acceleration,
    *,
    on: str | None,
    opts: KinematicsSmoothingOptions | None,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> Acceleration:
    from ..acceleration import Acceleration

    request = FamilyTemporalRequest(source, on, opts, validate, sequence_dim, batch_dims, sequence_size_coord, owner)
    try:
        return _run_family_pair_operation(
            request,
            parts_resolver=_acceleration_family_parts,
            method_name="smooth",
            compose=Acceleration.from_linear_angular,
        )
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


__all__ = [
    "differentiate_velocity_family",
    "integrate_acceleration_family",
    "smooth_acceleration_family",
    "smooth_velocity_family",
]
