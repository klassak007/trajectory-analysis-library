from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import xarray as xr

from tal.core.schema import merge_schema
from tal.utils.frame_schema import get_frames

from .common import _read_relation_block, _require_dataset

_INERTIAL_ROLE_PARENT = "parent"
_INERTIAL_ROLE_CHILD = "child"
_ALLOWED_INERTIAL_ROLES = frozenset({_INERTIAL_ROLE_PARENT, _INERTIAL_ROLE_CHILD})
_InertialRole = Literal["parent", "child"]


def _normalize_expressed_in(value: object, *, owner: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{owner}: tal.ext.spatial.relation.expressed_in must be a non-empty string or null."
        )
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(
            f"{owner}: tal.ext.spatial.relation.expressed_in must be a non-empty string or null."
        )
    return cleaned


def _normalize_inertial_role(value: object, *, owner: str) -> _InertialRole:
    if not isinstance(value, str):
        raise ValueError(
            f"{owner}: tal.ext.spatial.relation.instantaneous_inertial roles must be non-empty strings."
        )
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(
            f"{owner}: tal.ext.spatial.relation.instantaneous_inertial roles must be non-empty strings."
        )
    if cleaned == "expressed_in":
        raise ValueError(
            f"{owner}: tal.ext.spatial.relation.expressed_in is not a valid instantaneous_inertial role."
        )
    if cleaned not in _ALLOWED_INERTIAL_ROLES:
        allowed = sorted(_ALLOWED_INERTIAL_ROLES)
        raise ValueError(
            f"{owner}: tal.ext.spatial.relation.instantaneous_inertial roles must be in {allowed!r}; got {cleaned!r}."
        )
    return cleaned


def _normalize_instantaneous_inertial(
    value: object,
    *,
    owner: str,
) -> frozenset[_InertialRole] | None:
    if value is None:
        return None
    if isinstance(value, str):
        roles: tuple[object, ...] = (value,)
    elif isinstance(value, (tuple, list, set, frozenset)):
        roles = tuple(value)
    else:
        raise ValueError(
            f"{owner}: tal.ext.spatial.relation.instantaneous_inertial must be a role-set over {{'parent','child'}} or null."
        )
    return frozenset(_normalize_inertial_role(role, owner=owner) for role in roles)


def _serialize_instantaneous_inertial(
    value: frozenset[_InertialRole] | None,
) -> list[_InertialRole] | None:
    if value is None:
        return None
    return sorted(value)


def _relation_patch(
    *,
    expressed_in: str | None,
    instantaneous_inertial: frozenset[_InertialRole] | None,
) -> Mapping[str, object]:
    return {
        "ext": {
            "spatial": {
                "relation": {
                    "expressed_in": expressed_in,
                    "instantaneous_inertial": _serialize_instantaneous_inertial(instantaneous_inertial),
                }
            }
        }
    }


def get_expressed_in(ds: xr.Dataset, *, owner: str) -> str | None:
    ds = _require_dataset(ds, owner=owner)
    block = _read_relation_block(ds, owner=owner)
    if block is not None and "expressed_in" in block:
        return _normalize_expressed_in(block["expressed_in"], owner=owner)
    parent, _ = get_frames(ds)
    return parent


def _get_explicit_expressed_in(
    ds: xr.Dataset,
    *,
    owner: str,
) -> str | None:
    block = _read_relation_block(ds, owner=owner)
    if block is None or "expressed_in" not in block:
        return None
    return _normalize_expressed_in(block["expressed_in"], owner=owner)


def set_expressed_in(
    ds: xr.Dataset,
    *,
    expressed_in: str | None,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    ds = _require_dataset(ds, owner=owner)
    normalized = _normalize_expressed_in(expressed_in, owner=owner)
    block = _read_relation_block(ds, owner=owner)
    current_inertial = None
    if block is not None and "instantaneous_inertial" in block:
        current_inertial = _normalize_instantaneous_inertial(
            block["instantaneous_inertial"],
            owner=owner,
        )
    return merge_schema(
        ds,
        _relation_patch(expressed_in=normalized, instantaneous_inertial=current_inertial),
        validate=validate,
    )


def clear_expressed_in(
    ds: xr.Dataset,
    *,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    """Clear an explicit representation-basis identifier functionally."""
    return set_expressed_in(
        ds,
        expressed_in=None,
        validate=validate,
        owner=owner,
    )


def get_instantaneous_inertial(
    ds: xr.Dataset,
    *,
    owner: str,
) -> frozenset[_InertialRole]:
    ds = _require_dataset(ds, owner=owner)
    block = _read_relation_block(ds, owner=owner)
    if block is not None and "instantaneous_inertial" in block:
        normalized = _normalize_instantaneous_inertial(
            block["instantaneous_inertial"],
            owner=owner,
        )
        return frozenset() if normalized is None else normalized
    return frozenset()


def set_instantaneous_inertial(
    ds: xr.Dataset,
    *,
    instantaneous_inertial: object | None,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    ds = _require_dataset(ds, owner=owner)
    normalized = _normalize_instantaneous_inertial(instantaneous_inertial, owner=owner)
    current_expressed_in = _get_explicit_expressed_in(ds, owner=owner)
    return merge_schema(
        ds,
        _relation_patch(expressed_in=current_expressed_in, instantaneous_inertial=normalized),
        validate=validate,
    )


def normalize_configuration_relation_semantics(
    ds: xr.Dataset,
    *,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    ds = _require_dataset(ds, owner=owner)
    block = _read_relation_block(ds, owner=owner)
    if block is not None and "instantaneous_inertial" in block:
        raise ValueError(
            f"{owner}: tal.ext.spatial.relation.instantaneous_inertial is kinematics-only metadata."
        )
    expressed_in = _get_explicit_expressed_in(ds, owner=owner)
    return merge_schema(
        ds,
        _relation_patch(expressed_in=expressed_in, instantaneous_inertial=None),
        validate=validate,
    )


def normalize_kinematic_relation_semantics(
    ds: xr.Dataset,
    *,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    ds = _require_dataset(ds, owner=owner)
    expressed_in = _get_explicit_expressed_in(ds, owner=owner)
    inertial = get_instantaneous_inertial(ds, owner=owner)
    return merge_schema(
        ds,
        _relation_patch(expressed_in=expressed_in, instantaneous_inertial=inertial),
        validate=validate,
    )


__all__ = [
    "clear_expressed_in",
    "get_expressed_in",
    "get_instantaneous_inertial",
    "normalize_configuration_relation_semantics",
    "normalize_kinematic_relation_semantics",
    "set_expressed_in",
    "set_instantaneous_inertial",
]
