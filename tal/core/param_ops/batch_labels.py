from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr


def batch_index(ds: xr.Dataset, *, dim: str) -> pd.Index:
    if dim in ds.coords and ds.coords[dim].dims == (dim,):
        return ds.get_index(dim)
    size = int(ds.sizes.get(dim, 0))
    return pd.Index(np.arange(size, dtype="int64"), name=dim)


def join_batch_index(
    indices: Sequence[pd.Index],
    *,
    mode: Literal["inner", "outer", "exact"],
    owner: str,
) -> pd.Index:
    if not indices:
        return pd.Index([], dtype="int64")
    if mode == "exact":
        base = indices[0]
        for index in indices[1:]:
            if not base.equals(index):
                raise ValueError(f"{owner}: batch_join='exact' requires identical batch labels.")
        return base
    if mode == "inner":
        out = indices[0]
        for index in indices[1:]:
            out = out.intersection(index, sort=False)
        return out
    out = indices[0]
    for index in indices[1:]:
        out = out.union(index, sort=False)
    return out


def index_matches_labels(source: pd.Index, labels: pd.Index) -> bool:
    return source.equals(labels)


def index_equivalent_labels(source: pd.Index, labels: pd.Index) -> bool:
    if len(source) != len(labels):
        return False
    if not labels_selectable_from(source, labels=labels):
        return False
    return labels_selectable_from(labels, labels=source)


def labels_selectable_from(
    source: pd.Index,
    *,
    labels: pd.Index,
) -> bool:
    try:
        indexer = source.get_indexer(labels)
    except (TypeError, ValueError, KeyError):
        return False
    return bool(np.all(np.asarray(indexer, dtype="int64") >= 0))


def missing_label_mask(
    labels: pd.Index,
    *,
    source: pd.Index,
    dim: str,
) -> xr.DataArray:
    missing = ~labels.isin(source)
    return xr.DataArray(
        np.asarray(missing, dtype=bool),
        dims=[dim],
        coords={dim: labels},
    )


def has_mixed_null_domain(index: pd.Index) -> bool:
    mask = np.asarray(index.isna(), dtype=bool)
    return bool(mask.any() and (~mask).any())


def labels_collapse_under_dtype(labels: pd.Index, *, dtype: np.dtype) -> bool:
    try:
        coerced = np.asarray(labels.to_numpy()).astype(dtype, copy=False)
    except (TypeError, ValueError):
        return False
    return not bool(pd.Index(coerced).is_unique)


__all__ = [
    "batch_index",
    "index_equivalent_labels",
    "has_mixed_null_domain",
    "index_matches_labels",
    "join_batch_index",
    "labels_selectable_from",
    "labels_collapse_under_dtype",
    "missing_label_mask",
]
