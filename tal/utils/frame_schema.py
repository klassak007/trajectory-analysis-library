from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from tal.core.schema import UNSET, UnsetType, merge_schema
from tal.core.schema_errors import schema_error

_ALLOWED_FRAME_KEYS = {"parent", "child"}


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<unrepr:{type(value).__name__}>"


def _key_order_value(key: Any) -> tuple[str, str]:
    return type(key).__name__, _safe_repr(key)


def _iter_deterministic_keys(mapping: Mapping[Any, Any]) -> list[Any]:
    return sorted(mapping.keys(), key=_key_order_value)


def _require_dataset(ds: Any, *, owner: str) -> xr.Dataset:
    if isinstance(ds, xr.Dataset):
        return ds
    raise TypeError(f"{owner} expects xr.Dataset, got {type(ds).__name__}.")


def _read_tal_mapping(ds: xr.Dataset, *, owner: str) -> Mapping[str, Any]:
    tal = ds.attrs.get("tal")
    if tal is None:
        return {}
    if isinstance(tal, Mapping):
        return tal
    raise schema_error(
        code="schema.not_mapping",
        path="tal",
        expected="mapping",
        actual=type(tal).__name__,
        hint=f"{owner}: set ds.attrs['tal'] to a mapping payload",
    )


def _normalize_frame_id(value: Any, *, field: str, owner: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise schema_error(
            code="schema.frames.id.invalid",
            path=f"tal.ext.frames.{field}",
            expected="non-empty string or null",
            actual=type(value).__name__,
            hint=f"{owner}: pass a non-empty frame id string or None",
        )
    cleaned = value.strip()
    if cleaned:
        return cleaned
    raise schema_error(
        code="schema.frames.id.invalid",
        path=f"tal.ext.frames.{field}",
        expected="non-empty string or null",
        actual=value,
        hint=f"{owner}: pass a non-empty frame id string or None",
    )


def _read_frames_block(
    tal: Mapping[str, Any],
    *,
    owner: str,
) -> Mapping[str, Any] | None:
    ext = tal.get("ext")
    if ext is None:
        return None
    if not isinstance(ext, Mapping):
        raise schema_error(
            code="schema.not_mapping",
            path="tal.ext",
            expected="mapping",
            actual=type(ext).__name__,
            hint=f"{owner}: set tal.ext to a mapping",
        )
    block = ext.get("frames")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise schema_error(
            code="schema.not_mapping",
            path="tal.ext.frames",
            expected="mapping",
            actual=type(block).__name__,
            hint=f"{owner}: set tal.ext.frames to a mapping",
        )
    for key in _iter_deterministic_keys(block):
        if not isinstance(key, str):
            raise schema_error(
                code="schema.frames.key.invalid",
                path="tal.ext.frames",
                expected=f"mapping with string keys in {sorted(_ALLOWED_FRAME_KEYS)}",
                actual={"key": _safe_repr(key), "key_type": type(key).__name__},
                hint=f"{owner}: set tal.ext.frames keys to canonical strings only",
            )
        if key not in _ALLOWED_FRAME_KEYS:
            raise schema_error(
                code="schema.frames.unknown_key",
                path=f"tal.ext.frames.{key}",
                expected=sorted(_ALLOWED_FRAME_KEYS),
                actual=key,
                hint=f"{owner}: remove unknown frame key",
            )
    return block


def _resolve_next_value(
    current: str | None,
    candidate: str | None | UnsetType,
    *,
    field: str,
    owner: str,
) -> str | None:
    if candidate is UNSET:
        return current
    return _normalize_frame_id(candidate, field=field, owner=owner)


def _merge_ext_frames_patch(
    *,
    current_parent: str | None,
    current_child: str | None,
    next_parent: str | None,
    next_child: str | None,
) -> Mapping[str, Any]:
    if next_parent is None and next_child is None:
        return {"ext": {"frames": None}}
    block: dict[str, Any] = {}
    if next_parent is None:
        if current_parent is not None:
            block["parent"] = None
    else:
        block["parent"] = next_parent
    if next_child is None:
        if current_child is not None:
            block["child"] = None
    else:
        block["child"] = next_child
    return {"ext": {"frames": block}}


def get_frames(ds: xr.Dataset) -> tuple[str | None, str | None]:
    """Read frame ids from TAL schema metadata.

    Parameters
    ----------
    ds : object
        Dataset-like value. Must be an ``xarray.Dataset`` with TAL schema metadata.

    Returns
    -------
    tuple[str | None, str | None]
        ``(parent, child)`` frame ids. Missing values are returned as ``None``.

    Notes
    -----
    This function reads only ``tal.ext.frames.parent`` and ``tal.ext.frames.child``.
    Unknown frame keys, non-string ids, or invalid schema block types raise schema errors.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.utils.frame_schema import get_frames, set_frames
    >>> ds = xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]})
    >>> tagged = set_frames(ds, parent="world", child="sensor", validate=False)
    >>> get_frames(tagged)
    ('world', 'sensor')
    """
    ds = _require_dataset(ds, owner="get_frames")
    tal = _read_tal_mapping(ds, owner="get_frames")
    block = _read_frames_block(tal, owner="get_frames")
    if block is None:
        return None, None
    parent = _normalize_frame_id(block.get("parent"), field="parent", owner="get_frames")
    child = _normalize_frame_id(block.get("child"), field="child", owner="get_frames")
    return parent, child


def set_frames(
    ds: xr.Dataset,
    *,
    parent: str | None | UnsetType = UNSET,
    child: str | None | UnsetType = UNSET,
    validate: bool = True,
) -> xr.Dataset:
    """Write parent/child frame ids into TAL schema metadata.

    Parameters
    ----------
    ds : object
        Dataset-like value. Must be an ``xarray.Dataset``.
    parent : str | None | UnsetType, optional
        Parent frame id. ``UNSET`` keeps the existing value.
    child : str | None | UnsetType, optional
        Child frame id. ``UNSET`` keeps the existing value.
    validate : bool, default=True
        Validate the patched schema before returning.

    Returns
    -------
    xr.Dataset
        ``xarray.Dataset`` with updated ``tal.ext.frames`` metadata.

    Notes
    -----
    ``UNSET`` leaves an existing id unchanged. Passing ``None`` clears that id.
    The returned dataset is a copy with merged TAL schema metadata.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.utils.frame_schema import get_frames, set_frames
    >>> ds = xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]})
    >>> tagged = set_frames(ds, parent="world", child="tool", validate=False)
    >>> get_frames(tagged)
    ('world', 'tool')
    >>> get_frames(set_frames(tagged, child=None, validate=False))
    ('world', None)
    """
    ds = _require_dataset(ds, owner="set_frames")
    tal = _read_tal_mapping(ds, owner="set_frames")
    block = _read_frames_block(tal, owner="set_frames")
    current_parent: str | None = None
    current_child: str | None = None
    if block is not None:
        current_parent = _normalize_frame_id(block.get("parent"), field="parent", owner="set_frames")
        current_child = _normalize_frame_id(block.get("child"), field="child", owner="set_frames")
    next_parent = _resolve_next_value(current_parent, parent, field="parent", owner="set_frames")
    next_child = _resolve_next_value(current_child, child, field="child", owner="set_frames")
    patch = _merge_ext_frames_patch(
        current_parent=current_parent,
        current_child=current_child,
        next_parent=next_parent,
        next_child=next_child,
    )
    return merge_schema(ds, patch, validate=validate)
