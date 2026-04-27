from __future__ import annotations

import xarray as xr

from tal.core.schema import merge_schema

from .common import _read_rep_block, _require_dataset

_ALLOWED_POSITION_REPS = {"cart"}
_ALLOWED_ROTATION_REPS = {"quat", "matrix"}
_ALLOWED_POSE_REPS = {"components", "matrix"}
_ALLOWED_LINEAR_VELOCITY_REPS = {"cart"}
_ALLOWED_ANGULAR_VELOCITY_REPS = {"cart"}
_ALLOWED_VELOCITY_REPS = {"components", "vector6"}
_ALLOWED_LINEAR_ACCELERATION_REPS = {"cart"}
_ALLOWED_ANGULAR_ACCELERATION_REPS = {"cart"}
_ALLOWED_ACCELERATION_REPS = {"components", "vector6"}


def _normalize_non_empty_rep(rep: object, *, owner: str) -> str:
    if not isinstance(rep, str) or not rep.strip():
        raise ValueError(f"{owner}: tal.ext.spatial.representation.rep must be a non-empty string.")
    return rep.strip()


def _normalize_rep(rep: object, *, allowed: set[str], what: str, owner: str) -> str:
    cleaned = _normalize_non_empty_rep(rep, owner=owner)
    if cleaned not in allowed:
        raise ValueError(f"{owner}: unsupported {what} representation {cleaned!r}; allowed={sorted(allowed)!r}.")
    return cleaned


def _read_rep_or_default(ds: xr.Dataset, *, default: str, allowed: set[str], what: str, owner: str) -> str:
    ds = _require_dataset(ds, owner=owner)
    block = _read_rep_block(ds, owner=owner)
    if block is None:
        return default
    if "rep" not in block:
        raise ValueError(f"{owner}: tal.ext.spatial.representation.rep must be set when representation block exists.")
    return _normalize_rep(block["rep"], allowed=allowed, what=what, owner=owner)


def _read_required_rep(ds: xr.Dataset, *, allowed: set[str], what: str, owner: str) -> str:
    ds = _require_dataset(ds, owner=owner)
    block = _read_rep_block(ds, owner=owner)
    if block is None or "rep" not in block:
        raise ValueError(
            f"{owner}: tal.ext.spatial.representation.rep must be explicitly set for {what} ({sorted(allowed)!r})."
        )
    return _normalize_rep(block["rep"], allowed=allowed, what=what, owner=owner)


def _set_rep(ds: xr.Dataset, *, rep: str, allowed: set[str], what: str, validate: bool, owner: str) -> xr.Dataset:
    ds = _require_dataset(ds, owner=owner)
    normalized = _normalize_rep(rep, allowed=allowed, what=what, owner=owner)
    return merge_schema(ds, {"ext": {"spatial": {"representation": {"rep": normalized}}}}, validate=validate)


def get_position_rep(ds: xr.Dataset, *, owner: str) -> str:
    return _read_rep_or_default(
        ds,
        default="cart",
        allowed=_ALLOWED_POSITION_REPS,
        what="position",
        owner=owner,
    )


def get_rotation_rep(ds: xr.Dataset, *, owner: str) -> str:
    return _read_rep_or_default(
        ds,
        default="quat",
        allowed=_ALLOWED_ROTATION_REPS,
        what="rotation",
        owner=owner,
    )


def get_pose_rep(ds: xr.Dataset, *, owner: str) -> str:
    return _read_required_rep(ds, allowed=_ALLOWED_POSE_REPS, what="pose", owner=owner)


def get_linear_velocity_rep(ds: xr.Dataset, *, owner: str) -> str:
    return _read_rep_or_default(
        ds,
        default="cart",
        allowed=_ALLOWED_LINEAR_VELOCITY_REPS,
        what="linear velocity",
        owner=owner,
    )


def get_angular_velocity_rep(ds: xr.Dataset, *, owner: str) -> str:
    return _read_rep_or_default(
        ds,
        default="cart",
        allowed=_ALLOWED_ANGULAR_VELOCITY_REPS,
        what="angular velocity",
        owner=owner,
    )


def get_velocity_rep(ds: xr.Dataset, *, owner: str) -> str:
    return _read_rep_or_default(
        ds,
        default="components",
        allowed=_ALLOWED_VELOCITY_REPS,
        what="velocity",
        owner=owner,
    )


def get_linear_acceleration_rep(ds: xr.Dataset, *, owner: str) -> str:
    return _read_rep_or_default(
        ds,
        default="cart",
        allowed=_ALLOWED_LINEAR_ACCELERATION_REPS,
        what="linear acceleration",
        owner=owner,
    )


def get_angular_acceleration_rep(ds: xr.Dataset, *, owner: str) -> str:
    return _read_rep_or_default(
        ds,
        default="cart",
        allowed=_ALLOWED_ANGULAR_ACCELERATION_REPS,
        what="angular acceleration",
        owner=owner,
    )


def get_acceleration_rep(ds: xr.Dataset, *, owner: str) -> str:
    return _read_rep_or_default(
        ds,
        default="components",
        allowed=_ALLOWED_ACCELERATION_REPS,
        what="acceleration",
        owner=owner,
    )


def set_position_rep(
    ds: xr.Dataset,
    *,
    rep: str,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    return _set_rep(
        ds,
        rep=rep,
        allowed=_ALLOWED_POSITION_REPS,
        what="position",
        validate=validate,
        owner=owner,
    )


def set_rotation_rep(
    ds: xr.Dataset,
    *,
    rep: str,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    return _set_rep(
        ds,
        rep=rep,
        allowed=_ALLOWED_ROTATION_REPS,
        what="rotation",
        validate=validate,
        owner=owner,
    )


def set_pose_rep(
    ds: xr.Dataset,
    *,
    rep: str,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    return _set_rep(
        ds,
        rep=rep,
        allowed=_ALLOWED_POSE_REPS,
        what="pose",
        validate=validate,
        owner=owner,
    )


def set_linear_velocity_rep(
    ds: xr.Dataset,
    *,
    rep: str,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    return _set_rep(
        ds,
        rep=rep,
        allowed=_ALLOWED_LINEAR_VELOCITY_REPS,
        what="linear velocity",
        validate=validate,
        owner=owner,
    )


def set_angular_velocity_rep(
    ds: xr.Dataset,
    *,
    rep: str,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    return _set_rep(
        ds,
        rep=rep,
        allowed=_ALLOWED_ANGULAR_VELOCITY_REPS,
        what="angular velocity",
        validate=validate,
        owner=owner,
    )


def set_velocity_rep(
    ds: xr.Dataset,
    *,
    rep: str,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    return _set_rep(
        ds,
        rep=rep,
        allowed=_ALLOWED_VELOCITY_REPS,
        what="velocity",
        validate=validate,
        owner=owner,
    )


def set_linear_acceleration_rep(
    ds: xr.Dataset,
    *,
    rep: str,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    return _set_rep(
        ds,
        rep=rep,
        allowed=_ALLOWED_LINEAR_ACCELERATION_REPS,
        what="linear acceleration",
        validate=validate,
        owner=owner,
    )


def set_angular_acceleration_rep(
    ds: xr.Dataset,
    *,
    rep: str,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    return _set_rep(
        ds,
        rep=rep,
        allowed=_ALLOWED_ANGULAR_ACCELERATION_REPS,
        what="angular acceleration",
        validate=validate,
        owner=owner,
    )


def set_acceleration_rep(
    ds: xr.Dataset,
    *,
    rep: str,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    return _set_rep(
        ds,
        rep=rep,
        allowed=_ALLOWED_ACCELERATION_REPS,
        what="acceleration",
        validate=validate,
        owner=owner,
    )


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
