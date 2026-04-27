from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.catalog import Catalog, CatalogQueryOptions


def test_cat_hard_p11b_003_no_hidden_eager_realization_in_default_query_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: CAT_HARD_P11B_003_no_hidden_eager_realization_in_default_query_paths."""
    dask_array = pytest.importorskip("dask.array")
    payload = dask_array.arange(40, chunks=10).reshape((10, 4))
    ds = xr.Dataset(
        {"value": (("trial", "sample"), payload)},
        coords={"trial": np.arange(10), "sample": np.arange(4)},
        attrs={"source": "sim"},
    )
    cat = Catalog(ds, batch_dim="trial")

    def _fail_compute(self: xr.DataArray, **kwargs: object) -> xr.DataArray:  # pragma: no cover - guard path
        _ = kwargs
        raise AssertionError("query unexpectedly forced eager compute")

    monkeypatch.setattr(xr.DataArray, "compute", _fail_compute)
    out = cat.query(where={"op": "==", "field": "attr.source", "value": "sim"})
    assert out.group_labels == tuple(np.arange(10).tolist())


def test_catalog_query_chunked_metadata_requires_explicit_eager_policy() -> None:
    dask_array = pytest.importorskip("dask.array")
    payload = dask_array.arange(24, chunks=6).reshape((6, 4))
    batch_meta = dask_array.from_array(np.arange(6), chunks=3)
    ds = xr.Dataset(
        {"value": (("trial", "sample"), payload)},
        coords={"trial": np.arange(6), "sample": np.arange(4), "batch_meta": ("trial", batch_meta)},
    )
    cat = Catalog(ds, batch_dim="trial")
    with pytest.raises(ValueError, match="metadata field 'batch.batch_meta' is chunked"):
        cat.query(where={"op": ">=", "field": "batch.batch_meta", "value": 3})
    out = cat.query(
        where={"op": ">=", "field": "batch.batch_meta", "value": 3},
        opts=CatalogQueryOptions(metadata_eager_policy="allow"),
    )
    assert tuple(out.group_labels) == (3, 4, 5)
