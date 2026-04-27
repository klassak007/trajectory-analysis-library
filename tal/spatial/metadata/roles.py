from __future__ import annotations

import xarray as xr

from tal.core.schema import merge_schema

from .common import _read_roles_block, _require_dataset

_ALLOWED_POSITION_INTENTS = {"delta"}
KINEMATICS_KIND_VALUES: tuple[str, ...] = (
    "linear_velocity",
    "angular_velocity",
    "velocity",
    "linear_acceleration",
    "angular_acceleration",
    "acceleration",
)
_ALLOWED_KINEMATICS_KINDS = frozenset(KINEMATICS_KIND_VALUES)


def _normalize_position_intent(intent: object | None, *, owner: str) -> str | None:
    if intent is None:
        return None
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError(f"{owner}: tal.ext.spatial.roles.position_intent must be 'delta' or null.")
    cleaned = intent.strip()
    if cleaned not in _ALLOWED_POSITION_INTENTS:
        raise ValueError(
            f"{owner}: unsupported position intent {cleaned!r}; allowed={sorted(_ALLOWED_POSITION_INTENTS)!r}."
        )
    return cleaned


def _normalize_kinematics_kind(kind: object | None, *, owner: str) -> str | None:
    if kind is None:
        return None
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"{owner}: tal.ext.spatial.roles.kinematics_kind must be a non-empty string when provided.")
    cleaned = kind.strip()
    if cleaned not in _ALLOWED_KINEMATICS_KINDS:
        raise ValueError(
            f"{owner}: unsupported kinematics kind {cleaned!r}; allowed={list(KINEMATICS_KIND_VALUES)!r}."
        )
    return cleaned


def get_position_intent(ds: xr.Dataset, *, owner: str) -> str | None:
    ds = _require_dataset(ds, owner=owner)
    block = _read_roles_block(ds, owner=owner)
    if block is None or "position_intent" not in block:
        return None
    return _normalize_position_intent(block["position_intent"], owner=owner)


def set_position_intent(
    ds: xr.Dataset,
    *,
    intent: str | None,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    ds = _require_dataset(ds, owner=owner)
    normalized = _normalize_position_intent(intent, owner=owner)
    return merge_schema(
        ds,
        {"ext": {"spatial": {"roles": {"position_intent": normalized}}}},
        validate=validate,
    )


def get_kinematics_kind(ds: xr.Dataset, *, owner: str) -> str | None:
    ds = _require_dataset(ds, owner=owner)
    block = _read_roles_block(ds, owner=owner)
    if block is None or "kinematics_kind" not in block:
        return None
    return _normalize_kinematics_kind(block["kinematics_kind"], owner=owner)


def set_kinematics_kind(
    ds: xr.Dataset,
    *,
    kind: str | None,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    ds = _require_dataset(ds, owner=owner)
    normalized = _normalize_kinematics_kind(kind, owner=owner)
    return merge_schema(ds, {"ext": {"spatial": {"roles": {"kinematics_kind": normalized}}}}, validate=validate)


def normalize_kinematics_kind(
    ds: xr.Dataset,
    *,
    expected_kind: str,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    ds = _require_dataset(ds, owner=owner)
    expected = _normalize_kinematics_kind(expected_kind, owner=owner)
    current = get_kinematics_kind(ds, owner=owner)
    if current is None:
        return set_kinematics_kind(ds, kind=expected, validate=validate, owner=owner)
    if current != expected:
        raise ValueError(
            f"{owner}: tal.ext.spatial.roles.kinematics_kind must be {expected!r}; got {current!r}."
        )
    return ds


def validate_spatial_roles(ds: xr.Dataset, *, owner: str) -> None:
    ds = _require_dataset(ds, owner=owner)
    _ = _read_roles_block(ds, owner=owner)


__all__ = [
    "KINEMATICS_KIND_VALUES",
    "get_kinematics_kind",
    "get_position_intent",
    "normalize_kinematics_kind",
    "set_kinematics_kind",
    "set_position_intent",
    "validate_spatial_roles",
]
