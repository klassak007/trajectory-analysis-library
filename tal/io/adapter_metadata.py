from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .options import AdapterMetadataPromotionOptions


def _python_scalar(value: Any) -> Any:
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


def collect_csv_scalar_metadata(
    frame: pd.DataFrame,
    *,
    metadata_columns: tuple[str, ...],
    owner: str,
    source_path: str,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in metadata_columns:
        if name not in frame.columns:
            raise ValueError(f"{owner}: metadata column {name!r} was not found in {source_path!r}.")
        values = frame[name].dropna().unique().tolist()
        if len(values) > 1:
            raise ValueError(
                f"{owner}: metadata column {name!r} must be scalar per input file; "
                f"found multiple values in {source_path!r}."
            )
        out[name] = _python_scalar(values[0]) if values else None
    return out


def aggregate_batch_metadata(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    keys: set[str] = set().union(*(row.keys() for row in rows))
    out: dict[str, object] = {}
    for key in sorted(keys):
        values = [_python_scalar(row.get(key)) for row in rows]
        first = values[0]
        if all(_equal_or_both_nan(first, value) for value in values[1:]):
            out[key] = first
        else:
            out[key] = list(values)
    return out


def _require_metadata_name_available(
    ds: xr.Dataset,
    *,
    name: str,
    target: str,
    owner: str,
) -> None:
    if target == "attrs" and name == "tal":
        raise ValueError(
            f"{owner}: metadata promotion key {name!r} is reserved for schema namespace and cannot "
            "be promoted to attrs."
        )
    if name in ds.data_vars or name in ds.coords or name in ds.dims:
        raise ValueError(
            f"{owner}: metadata promotion target name {name!r} collides with existing "
            "dataset vars/coords/dims."
        )
    if name in ds.attrs:
        raise ValueError(
            f"{owner}: metadata promotion target name {name!r} collides with existing attrs."
        )
    if target not in {"attrs", "batch_coord"}:
        raise ValueError(f"{owner}: unsupported metadata promotion target {target!r}.")


def promote_adapter_metadata(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    metadata: Mapping[str, object],
    options: AdapterMetadataPromotionOptions,
    owner: str,
) -> xr.Dataset:
    out = ds
    for name, value in metadata.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{owner}: metadata keys must be non-empty strings; got {name!r}.")
        scalar = _is_scalar_metadata(value)
        if scalar:
            target = options.scalar_target
            if target == "none":
                continue
            _require_metadata_name_available(out, name=name, target=target, owner=owner)
            if target == "attrs":
                attrs = dict(out.attrs)
                attrs[name] = _python_scalar(value)
                out = out.copy(deep=False)
                out.attrs = attrs
                continue
            out = out.assign_coords(
                {name: xr.DataArray(np.full((out.sizes[batch_dim],), _python_scalar(value), dtype=object), dims=(batch_dim,))}
            )
            continue
        if options.nonscalar_target == "none":
            continue
        _require_metadata_name_available(out, name=name, target="attrs", owner=owner)
        attrs = dict(out.attrs)
        attrs[name] = value
        out = out.copy(deep=False)
        out.attrs = attrs
    return out


__all__ = ["aggregate_batch_metadata", "collect_csv_scalar_metadata", "promote_adapter_metadata"]
