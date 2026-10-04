from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from .options import AdapterMetadataPromotionOptions


def normalize_metadata_scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _is_nan_like(value: object) -> bool:
    try:
        return bool(np.isnan(value))  # type: ignore[arg-type]
    except Exception:
        return False


def _equal_or_both_nan(left: object, right: object) -> bool:
    if _is_nan_like(left) and _is_nan_like(right):
        return True
    return left == right


def _is_scalar_metadata(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    if isinstance(value, (list, tuple, Mapping)):
        return False
    return True


def _require_attr_metadata_name(name: str, *, owner: str) -> None:
    if name == "tal":
        raise ValueError(
            f"{owner}: metadata promotion key {name!r} is reserved for schema namespace and "
            "cannot be promoted to attrs."
        )


def aggregate_batch_metadata(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    keys: set[str] = set().union(*(row.keys() for row in rows))
    out: dict[str, object] = {}
    for key in sorted(keys):
        values = [normalize_metadata_scalar(row.get(key)) for row in rows]
        first = values[0]
        if all(_equal_or_both_nan(first, value) for value in values[1:]):
            out[key] = first
        else:
            out[key] = list(values)
    return out


def require_generated_metadata_preflight(
    *,
    options: AdapterMetadataPromotionOptions,
    generated_names: Sequence[str],
    scalar_generated_names: Sequence[str],
    user_metadata_names: Sequence[str],
    occupied_names: Sequence[str],
    owner: str,
) -> None:
    attr_targets = {options.scalar_target, options.nonscalar_target}
    if "attrs" in attr_targets:
        for name in user_metadata_names:
            _require_attr_metadata_name(name, owner=owner)
    generated = set(generated_names)
    reserved = tuple(sorted(generated.intersection(user_metadata_names)))
    if reserved:
        raise ValueError(
            f"{owner}: metadata columns {reserved!r} are reserved for generated "
            "adapter metadata."
        )
    if options.scalar_target != "batch_coord":
        return
    scalar_generated = set(scalar_generated_names)
    potential_coords = scalar_generated.union(user_metadata_names)
    collisions = tuple(sorted(potential_coords.intersection(occupied_names)))
    if collisions:
        generated_collision = bool(scalar_generated.intersection(collisions))
        source = "generated adapter metadata" if generated_collision else "adapter metadata"
        raise ValueError(
            f"{owner}: {source} batch-coordinate names "
            f"collide with configured output names: {collisions!r}."
        )


def _require_metadata_name_available(
    ds: xr.Dataset,
    *,
    name: str,
    target: str,
    owner: str,
) -> None:
    if target not in {"attrs", "batch_coord"}:
        raise ValueError(f"{owner}: unsupported metadata promotion target {target!r}.")
    if target == "attrs":
        _require_attr_metadata_name(name, owner=owner)
        if name in ds.attrs:
            raise ValueError(
                f"{owner}: metadata promotion target name {name!r} collides with existing attrs."
            )
        return
    if name in ds.data_vars or name in ds.coords or name in ds.dims:
        raise ValueError(
            f"{owner}: metadata promotion target name {name!r} collides with existing "
            "dataset vars/coords/dims."
        )
    if name in ds.attrs:
        raise ValueError(
            f"{owner}: metadata promotion target name {name!r} collides with existing attrs."
        )


def promote_adapter_metadata(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    metadata: Mapping[str, object],
    options: AdapterMetadataPromotionOptions,
    owner: str,
) -> xr.Dataset:
    attr_updates: dict[str, object] = {}
    coord_updates: dict[str, xr.DataArray] = {}
    for name, value in metadata.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{owner}: metadata keys must be non-empty strings; got {name!r}.")
        scalar = _is_scalar_metadata(value)
        target = options.scalar_target if scalar else options.nonscalar_target
        if target == "none":
            continue
        _require_metadata_name_available(ds, name=name, target=target, owner=owner)
        if target == "attrs":
            attr_updates[name] = normalize_metadata_scalar(value) if scalar else value
            continue
        scalar_value = normalize_metadata_scalar(value)
        coord_updates[name] = xr.DataArray(
            np.full((ds.sizes[batch_dim],), scalar_value, dtype=object),
            dims=(batch_dim,),
        )
    out = ds.assign_coords(coord_updates) if coord_updates else ds
    if attr_updates:
        out = out.assign_attrs(attr_updates)
    return out


__all__ = [
    "aggregate_batch_metadata",
    "normalize_metadata_scalar",
    "promote_adapter_metadata",
    "require_generated_metadata_preflight",
]
