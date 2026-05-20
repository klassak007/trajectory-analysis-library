from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import xarray as xr

from tal.core.schema import merge_schema

from .common import _read_rep_block, _require_dataset


@dataclass(frozen=True)
class _RepresentationMetadataSpec:
    what: str
    allowed: frozenset[str]
    default: str | None


class _RepGetter(Protocol):
    def __call__(self, ds: xr.Dataset, *, owner: str) -> str: ...


class _RepSetter(Protocol):
    def __call__(
        self,
        ds: xr.Dataset,
        *,
        rep: str,
        validate: bool,
        owner: str,
    ) -> xr.Dataset: ...


_POSITION_REP_SPEC = _RepresentationMetadataSpec(
    what="position",
    allowed=frozenset({"cart"}),
    default="cart",
)
_ROTATION_REP_SPEC = _RepresentationMetadataSpec(
    what="rotation",
    allowed=frozenset({"quat", "matrix"}),
    default="quat",
)
_POSE_REP_SPEC = _RepresentationMetadataSpec(
    what="pose",
    allowed=frozenset({"components", "matrix"}),
    default=None,
)
_LINEAR_VELOCITY_REP_SPEC = _RepresentationMetadataSpec(
    what="linear velocity",
    allowed=frozenset({"cart"}),
    default="cart",
)
_ANGULAR_VELOCITY_REP_SPEC = _RepresentationMetadataSpec(
    what="angular velocity",
    allowed=frozenset({"cart"}),
    default="cart",
)
_VELOCITY_REP_SPEC = _RepresentationMetadataSpec(
    what="velocity",
    allowed=frozenset({"components", "vector6"}),
    default="components",
)
_LINEAR_ACCELERATION_REP_SPEC = _RepresentationMetadataSpec(
    what="linear acceleration",
    allowed=frozenset({"cart"}),
    default="cart",
)
_ANGULAR_ACCELERATION_REP_SPEC = _RepresentationMetadataSpec(
    what="angular acceleration",
    allowed=frozenset({"cart"}),
    default="cart",
)
_ACCELERATION_REP_SPEC = _RepresentationMetadataSpec(
    what="acceleration",
    allowed=frozenset({"components", "vector6"}),
    default="components",
)


def _normalize_non_empty_rep(rep: object, *, owner: str) -> str:
    if not isinstance(rep, str) or not rep.strip():
        raise ValueError(f"{owner}: tal.ext.spatial.representation.rep must be a non-empty string.")
    return rep.strip()


def _normalize_rep(rep: object, *, allowed: frozenset[str], what: str, owner: str) -> str:
    cleaned = _normalize_non_empty_rep(rep, owner=owner)
    if cleaned not in allowed:
        raise ValueError(f"{owner}: unsupported {what} representation {cleaned!r}; allowed={sorted(allowed)!r}.")
    return cleaned


def _read_rep_or_default(
    ds: xr.Dataset,
    *,
    default: str,
    allowed: frozenset[str],
    what: str,
    owner: str,
) -> str:
    ds = _require_dataset(ds, owner=owner)
    block = _read_rep_block(ds, owner=owner)
    if block is None:
        return default
    if "rep" not in block:
        raise ValueError(f"{owner}: tal.ext.spatial.representation.rep must be set when representation block exists.")
    return _normalize_rep(block["rep"], allowed=allowed, what=what, owner=owner)


def _read_required_rep(ds: xr.Dataset, *, allowed: frozenset[str], what: str, owner: str) -> str:
    ds = _require_dataset(ds, owner=owner)
    block = _read_rep_block(ds, owner=owner)
    if block is None or "rep" not in block:
        raise ValueError(
            f"{owner}: tal.ext.spatial.representation.rep must be explicitly set for {what} ({sorted(allowed)!r})."
        )
    return _normalize_rep(block["rep"], allowed=allowed, what=what, owner=owner)


def _set_rep(
    ds: xr.Dataset,
    *,
    rep: str,
    allowed: frozenset[str],
    what: str,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    ds = _require_dataset(ds, owner=owner)
    normalized = _normalize_rep(rep, allowed=allowed, what=what, owner=owner)
    return merge_schema(ds, {"ext": {"spatial": {"representation": {"rep": normalized}}}}, validate=validate)


def _make_rep_getter(name: str, spec: _RepresentationMetadataSpec) -> _RepGetter:
    def getter(ds: xr.Dataset, *, owner: str) -> str:
        if spec.default is None:
            return _read_required_rep(ds, allowed=spec.allowed, what=spec.what, owner=owner)
        return _read_rep_or_default(
            ds,
            default=spec.default,
            allowed=spec.allowed,
            what=spec.what,
            owner=owner,
        )

    getter.__name__ = name
    getter.__qualname__ = name
    getter.__module__ = __name__
    getter.__doc__ = f"Return the TAL spatial {spec.what} representation."
    return getter


def _make_rep_setter(name: str, spec: _RepresentationMetadataSpec) -> _RepSetter:
    def setter(ds: xr.Dataset, *, rep: str, validate: bool, owner: str) -> xr.Dataset:
        return _set_rep(
            ds,
            rep=rep,
            allowed=spec.allowed,
            what=spec.what,
            validate=validate,
            owner=owner,
        )

    setter.__name__ = name
    setter.__qualname__ = name
    setter.__module__ = __name__
    setter.__doc__ = f"Set the TAL spatial {spec.what} representation."
    return setter


get_position_rep = _make_rep_getter("get_position_rep", _POSITION_REP_SPEC)
get_rotation_rep = _make_rep_getter("get_rotation_rep", _ROTATION_REP_SPEC)
get_pose_rep = _make_rep_getter("get_pose_rep", _POSE_REP_SPEC)
get_linear_velocity_rep = _make_rep_getter(
    "get_linear_velocity_rep",
    _LINEAR_VELOCITY_REP_SPEC,
)
get_angular_velocity_rep = _make_rep_getter(
    "get_angular_velocity_rep",
    _ANGULAR_VELOCITY_REP_SPEC,
)
get_velocity_rep = _make_rep_getter("get_velocity_rep", _VELOCITY_REP_SPEC)
get_linear_acceleration_rep = _make_rep_getter(
    "get_linear_acceleration_rep",
    _LINEAR_ACCELERATION_REP_SPEC,
)
get_angular_acceleration_rep = _make_rep_getter(
    "get_angular_acceleration_rep",
    _ANGULAR_ACCELERATION_REP_SPEC,
)
get_acceleration_rep = _make_rep_getter("get_acceleration_rep", _ACCELERATION_REP_SPEC)
set_position_rep = _make_rep_setter("set_position_rep", _POSITION_REP_SPEC)
set_rotation_rep = _make_rep_setter("set_rotation_rep", _ROTATION_REP_SPEC)
set_pose_rep = _make_rep_setter("set_pose_rep", _POSE_REP_SPEC)
set_linear_velocity_rep = _make_rep_setter(
    "set_linear_velocity_rep",
    _LINEAR_VELOCITY_REP_SPEC,
)
set_angular_velocity_rep = _make_rep_setter(
    "set_angular_velocity_rep",
    _ANGULAR_VELOCITY_REP_SPEC,
)
set_velocity_rep = _make_rep_setter("set_velocity_rep", _VELOCITY_REP_SPEC)
set_linear_acceleration_rep = _make_rep_setter(
    "set_linear_acceleration_rep",
    _LINEAR_ACCELERATION_REP_SPEC,
)
set_angular_acceleration_rep = _make_rep_setter(
    "set_angular_acceleration_rep",
    _ANGULAR_ACCELERATION_REP_SPEC,
)
set_acceleration_rep = _make_rep_setter("set_acceleration_rep", _ACCELERATION_REP_SPEC)


__all__ = [
    "get_acceleration_rep",
    "get_angular_acceleration_rep",
    "get_angular_velocity_rep",
    "get_linear_acceleration_rep",
    "get_linear_velocity_rep",
    "get_pose_rep",
    "get_position_rep",
    "get_rotation_rep",
    "get_velocity_rep",
    "set_acceleration_rep",
    "set_angular_acceleration_rep",
    "set_angular_velocity_rep",
    "set_linear_acceleration_rep",
    "set_linear_velocity_rep",
    "set_pose_rep",
    "set_position_rep",
    "set_rotation_rep",
    "set_velocity_rep",
]
