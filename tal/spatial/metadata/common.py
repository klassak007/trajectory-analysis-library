from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

_ALLOWED_SPATIAL_KEYS = {"representation", "roles", "relation"}
_ALLOWED_REP_KEYS = {"rep"}
_ALLOWED_RELATION_KEYS = {"expressed_in", "instantaneous_inertial"}


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<unrepr:{type(value).__name__}>"


def _key_order(key: Any) -> tuple[str, str]:
    return type(key).__name__, _safe_repr(key)


def _iter_deterministic_keys(mapping: Mapping[Any, Any]) -> list[Any]:
    return sorted(mapping.keys(), key=_key_order)


def _require_dataset(ds: object, *, owner: str) -> xr.Dataset:
    if isinstance(ds, xr.Dataset):
        return ds
    raise TypeError(f"{owner}: expected xr.Dataset; got {type(ds).__name__}.")


def _validate_allowed_keys(
    block: Mapping[Any, Any],
    *,
    allowed: set[str],
    path: str,
    owner: str,
) -> None:
    for key in _iter_deterministic_keys(block):
        if not isinstance(key, str):
            raise ValueError(
                f"{owner}: {path} keys must be strings; got {_safe_repr(key)} ({type(key).__name__})."
            )
        if key in allowed:
            continue
        raise ValueError(
            f"{owner}: unknown key {key!r} at {path}; allowed keys are {sorted(allowed)!r}."
        )


def _validate_string_keys(
    block: Mapping[Any, Any],
    *,
    path: str,
    owner: str,
) -> None:
    for key in _iter_deterministic_keys(block):
        if isinstance(key, str):
            continue
        raise ValueError(
            f"{owner}: {path} keys must be strings; got {_safe_repr(key)} ({type(key).__name__})."
        )


def _read_tal(ds: xr.Dataset, *, owner: str) -> Mapping[str, Any]:
    tal = ds.attrs.get("tal")
    if tal is None:
        return {}
    if isinstance(tal, Mapping):
        return tal
    raise ValueError(f"{owner}: tal must be a mapping when present.")


def _read_ext(tal: Mapping[str, Any], *, owner: str) -> Mapping[str, Any] | None:
    ext = tal.get("ext")
    if ext is None:
        return None
    if isinstance(ext, Mapping):
        return ext
    raise ValueError(f"{owner}: tal.ext must be a mapping when present.")


def _read_spatial(ds: xr.Dataset, *, owner: str) -> Mapping[str, Any] | None:
    tal = _read_tal(ds, owner=owner)
    ext = _read_ext(tal, owner=owner)
    if ext is None:
        return None
    spatial = ext.get("spatial")
    if spatial is None:
        return None
    if not isinstance(spatial, Mapping):
        raise ValueError(f"{owner}: tal.ext.spatial must be a mapping when present.")
    _validate_allowed_keys(spatial, allowed=_ALLOWED_SPATIAL_KEYS, path="tal.ext.spatial", owner=owner)
    return spatial


def _read_rep_block(ds: xr.Dataset, *, owner: str) -> Mapping[str, Any] | None:
    spatial = _read_spatial(ds, owner=owner)
    if spatial is None:
        return None
    block = spatial.get("representation")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"{owner}: tal.ext.spatial.representation must be a mapping when present.")
    _validate_allowed_keys(
        block,
        allowed=_ALLOWED_REP_KEYS,
        path="tal.ext.spatial.representation",
        owner=owner,
    )
    return block


def _read_roles_block(ds: xr.Dataset, *, owner: str) -> Mapping[str, Any] | None:
    spatial = _read_spatial(ds, owner=owner)
    if spatial is None:
        return None
    block = spatial.get("roles")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"{owner}: tal.ext.spatial.roles must be a mapping when present.")
    _validate_string_keys(block, path="tal.ext.spatial.roles", owner=owner)
    return block


def _read_relation_block(ds: xr.Dataset, *, owner: str) -> Mapping[str, Any] | None:
    spatial = _read_spatial(ds, owner=owner)
    if spatial is None:
        return None
    block = spatial.get("relation")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"{owner}: tal.ext.spatial.relation must be a mapping when present.")
    _validate_allowed_keys(
        block,
        allowed=_ALLOWED_RELATION_KEYS,
        path="tal.ext.spatial.relation",
        owner=owner,
    )
    return block


__all__ = [
    "_read_relation_block",
    "_read_rep_block",
    "_read_roles_block",
    "_require_dataset",
]
