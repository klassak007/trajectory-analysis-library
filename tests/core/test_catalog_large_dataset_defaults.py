from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import tal.catalog.selection as catalog_selection
from tal.catalog import (
    Catalog,
    CatalogExtractOptions,
    CatalogMetadataPromotionOptions,
    CatalogQueryOptions,
)
from tal.catalog.backends import datatree_extract_template
from tal.catalog.extract_coords import canonicalize_selected_payload_vars


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


def test_cat_perf_p11a_001_datatree_selection_preserves_dask_laziness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: CAT_PERF_P11A_001_datatree_selection_preserves_dask_laziness."""
    dask_array = pytest.importorskip("dask.array")
    root = xr.Dataset(
        {"root_weight": ("group", dask_array.from_array([100, 200, 300], chunks=2))},
        coords={
            "group": ["root-a", "root-b", "root-c"],
            "rank": ("group", dask_array.from_array([10, 20, 30], chunks=2)),
        },
    )
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "/a": xr.Dataset({"value": ("sample", [1.0])}),
            "/b": xr.Dataset({"value": ("sample", [2.0])}),
            "/c": xr.Dataset({"value": ("sample", [3.0])}),
        }
    )
    original_compute = dask_array.Array.compute

    def _fail_compute(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("DataTree selection unexpectedly computed root payloads")

    monkeypatch.setattr(dask_array.Array, "compute", _fail_compute)
    catalog = Catalog(tree)
    selected = catalog.sel(["c", "a"]).data.to_dataset(inherit=False)
    assert getattr(selected["root_weight"].data, "chunks", None) is not None
    assert getattr(selected.coords["rank"].data, "chunks", None) is not None

    monkeypatch.setattr(dask_array.Array, "compute", original_compute)
    np.testing.assert_array_equal(selected["root_weight"].compute(), [300, 100])
    np.testing.assert_array_equal(selected.coords["rank"].compute(), [30, 10])


def test_cat_perf_p11a_002_datatree_selection_skips_redundant_label_position_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: CAT_PERF_P11A_002_datatree_selection_skips_redundant_label_position_scan."""
    children = {
        f"/{label}": xr.Dataset({"value": ("sample", [index])})
        for index, label in enumerate(("a", "b", "c"))
    }
    rootless = Catalog(xr.DataTree.from_dict({"/": xr.Dataset(), **children}))
    root_metadata = xr.Dataset(coords={"trial": [0, 1, 2]})
    rooted = Catalog(
        xr.DataTree.from_dict({"/": root_metadata, **children}),
        batch_dim="trial",
    )

    def _fail_scan(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("selection redundantly scanned labels for known positions")

    monkeypatch.setattr(catalog_selection, "_datatree_label_positions", _fail_scan)
    assert rootless.sel(["c"]).group_labels == ("c",)
    assert rooted.isel([2, 0]).group_labels == ("c", "a")


def test_cat_perf_p11a_003_datatree_extract_template_reuses_owned_payload_memory() -> None:
    """ID: CAT_PERF_P11A_003_datatree_extract_template_reuses_owned_payload_memory."""
    values = np.arange(1000.0)
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset({"value": ("sample", values)}),
        }
    ).copy(deep=True)

    template = datatree_extract_template(tree, batch_dim="trial", owner="test")
    assert template is not None
    child = tree.children["a"].to_dataset(inherit=False)
    assert np.shares_memory(template["value"].data, child["value"].data)


def test_cat_perf_p11b_001_datatree_extract_preserves_dask_coordinate_laziness() -> None:
    """ID: CAT_PERF_P11B_001_datatree_extract_preserves_dask_coordinate_laziness."""
    dask = pytest.importorskip("dask")
    dask_array = pytest.importorskip("dask.array")
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"sample": [10, 20]}),
            "/a": xr.Dataset(
                {"value": ("sample", dask_array.from_array([1.0, 2.0], chunks=1))},
                coords={"time": ("sample", dask_array.from_array([9.0, 8.0], chunks=1))},
            ),
            "/b": xr.Dataset(
                {"value": ("sample", dask_array.from_array([3.0, 4.0], chunks=1))},
                coords={"time": ("sample", dask_array.from_array([7.0, 6.0], chunks=1))},
            ),
        }
    )
    catalog = Catalog(tree, batch_dim="trial")

    def _fail_task(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("DataTree extract unexpectedly computed coordinate payloads")

    with dask.callbacks.Callback(pretask=_fail_task):
        out = catalog.extract("value").unsafe_data
    assert getattr(out["value"].data, "chunks", None) is not None
    assert getattr(out.coords["time"].data, "chunks", None) is not None
    assert out.coords["time"].dims == ("trial", "sample")
    np.testing.assert_array_equal(out.coords["time"].compute(), [[9.0, 8.0], [7.0, 6.0]])


def test_cat_perf_p11b_002_sparse_datatree_coordinates_synthesize_lazily() -> None:
    """ID: CAT_PERF_P11B_002_sparse_datatree_coordinates_synthesize_lazily."""
    dask = pytest.importorskip("dask")
    dask_array = pytest.importorskip("dask.array")
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"sample": [10, 20]}),
            "/a": xr.Dataset(
                {"value": ("sample", dask_array.from_array([1.0, 2.0], chunks=1))},
                coords={"quality": ("sample", dask_array.from_array([9.0, 8.0], chunks=1))},
            ),
            "/b": xr.Dataset(
                {"value": ("sample", dask_array.from_array([3.0, 4.0], chunks=1))},
            ),
        }
    )

    def _fail_task(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("sparse coordinate synthesis unexpectedly computed")

    with dask.callbacks.Callback(pretask=_fail_task):
        out = Catalog(tree, batch_dim="trial").extract("value").unsafe_data
    assert getattr(out.coords["quality"].data, "chunks", None) is not None
    expected = np.asarray([[9.0, 8.0], [np.nan, np.nan]])
    np.testing.assert_array_equal(out.coords["quality"].compute(), expected)


def test_cat_perf_p11b_003_root_batch_payload_coordinates_remain_lazy() -> None:
    """ID: CAT_PERF_P11B_003_root_batch_payload_coordinates_remain_lazy."""
    dask = pytest.importorskip("dask")
    dask_array = pytest.importorskip("dask.array")
    root = xr.Dataset(
        coords={
            "trial": ["root-a", "root-b"],
            "sample": [10, 20],
            "time": (
                ("trial", "sample"),
                dask_array.from_array([[0.0, 0.5], [1.0, 1.5]], chunks=(1, 2)),
            ),
        }
    )
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "/a": xr.Dataset({"value": ("sample", [1.0, 2.0])}),
            "/b": xr.Dataset({"value": ("sample", [3.0, 4.0])}),
        }
    )

    def _fail_task(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("root batch payload coordinate unexpectedly computed")

    with dask.callbacks.Callback(pretask=_fail_task):
        out = Catalog(tree, batch_dim="trial").extract("value").unsafe_data
    assert getattr(out.coords["time"].data, "chunks", None) is not None
    np.testing.assert_array_equal(out.coords["time"].compute(), [[0.0, 0.5], [1.0, 1.5]])


def test_cat_perf_p11b_004_empty_extract_isolation_preserves_dask_laziness() -> None:
    """ID: CAT_PERF_P11B_004_empty_extract_isolation_preserves_dask_laziness."""
    dask = pytest.importorskip("dask")
    dask_array = pytest.importorskip("dask.array")
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"sample": [10, 20]}),
            "/a": xr.Dataset(
                {"value": ("sample", dask_array.from_array([1.0, 2.0], chunks=1))},
                coords={"time": ("sample", dask_array.from_array([0.1, 0.2], chunks=1))},
            ),
        }
    )
    empty = Catalog(tree, batch_dim="trial").sel([])

    def _fail_task(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("empty extract isolation unexpectedly computed")

    with dask.callbacks.Callback(pretask=_fail_task):
        out = empty.extract("value").unsafe_data
    assert getattr(out.coords["time"].data, "chunks", None) is not None


def test_cat_perf_p11b_005_nonempty_coord_isolation_preserves_dask_laziness() -> None:
    """ID: CAT_PERF_P11B_005_nonempty_coord_isolation_preserves_dask_laziness."""
    dask = pytest.importorskip("dask")
    dask_array = pytest.importorskip("dask.array")
    root = xr.Dataset(
        coords={
            "sample": [10, 20],
            "time": ("sample", dask_array.from_array([0.1, 0.2], chunks=1)),
        }
    )
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "/a": xr.Dataset({"value": ("sample", [1.0, 2.0])}),
            "/b": xr.Dataset({"value": ("sample", [3.0, 4.0])}),
        }
    )

    def _fail_task(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("non-empty coordinate isolation unexpectedly computed")

    with dask.callbacks.Callback(pretask=_fail_task):
        out = Catalog(tree, batch_dim="trial").extract("value").unsafe_data
    assert getattr(out.coords["time"].data, "chunks", None) is not None
    np.testing.assert_array_equal(out.coords["time"].compute(), [0.1, 0.2])


def test_cat_perf_p11b_006_object_coord_isolation_preserves_dask_laziness() -> None:
    """ID: CAT_PERF_P11B_006_object_coord_isolation_preserves_dask_laziness."""
    dask = pytest.importorskip("dask")
    dask_array = pytest.importorskip("dask.array")
    tags = np.empty((2,), dtype=object)
    tags[0] = {"items": ["first"]}
    tags[1] = {"items": ["second"]}
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={
                    "sample": [0, 1],
                    "tags": ("sample", dask_array.from_array(tags, chunks=1)),
                },
            ),
        }
    )
    catalog = Catalog(tree, batch_dim="trial")

    def _fail_task(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("object-coordinate isolation unexpectedly computed")

    with dask.callbacks.Callback(pretask=_fail_task):
        out = catalog.extract("value").unsafe_data
    assert getattr(out.coords["tags"].data, "chunks", None) is not None

    computed = out.coords["tags"].compute()
    computed.data[0, 0]["items"].append("changed")
    again = catalog.extract("value").unsafe_data.coords["tags"].compute()
    assert again.data[0, 0] == {"items": ["first"]}


def test_cat_perf_p11b_007_object_payload_isolation_preserves_dask_laziness() -> None:
    """ID: CAT_PERF_P11B_007_object_payload_isolation_preserves_dask_laziness."""
    dask = pytest.importorskip("dask")
    dask_array = pytest.importorskip("dask.array")
    values = np.empty((2,), dtype=object)
    values[0] = {"items": ["first"]}
    values[1] = {"items": ["second"]}
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset(
                {"value": ("sample", dask_array.from_array(values, chunks=1))},
                coords={"sample": [0, 1]},
            ),
        }
    )
    catalog = Catalog(tree, batch_dim="trial")

    def _fail_task(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("object-payload isolation unexpectedly computed")

    with dask.callbacks.Callback(pretask=_fail_task):
        out = catalog.extract("value").unsafe_data
    assert getattr(out["value"].data, "chunks", None) is not None

    computed = out["value"].compute()
    computed.data[0, 0]["items"].append("changed")
    again = catalog.extract("value").unsafe_data["value"].compute()
    assert again.data[0, 0] == {"items": ["first"]}


def test_cat_perf_p11b_008_dataset_extract_ownership_preserves_dask_laziness() -> None:
    """ID: CAT_PERF_P11B_008_dataset_extract_ownership_preserves_dask_laziness."""
    dask = pytest.importorskip("dask")
    dask_array = pytest.importorskip("dask.array")
    objects = np.empty((2, 1), dtype=object)
    objects[0, 0] = {"items": ["first"]}
    objects[1, 0] = {"items": ["second"]}
    source = xr.Dataset(
        {
            "numeric": (
                ("trial", "sample"),
                dask_array.from_array([[1.0], [2.0]], chunks=(1, 1)),
            ),
            "objects": (
                ("trial", "sample"),
                dask_array.from_array(objects, chunks=(1, 1)),
            ),
        },
        coords={"trial": ["a", "b"], "sample": [0]},
    )
    catalog = Catalog(source, batch_dim="trial")

    def _fail_task(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("Dataset extraction ownership unexpectedly computed")

    with dask.callbacks.Callback(pretask=_fail_task):
        out = catalog.extract().unsafe_data
    assert getattr(out["numeric"].data, "chunks", None) is not None
    assert getattr(out["objects"].data, "chunks", None) is not None

    computed = out.compute()
    computed["objects"].data[0, 0]["items"].append("changed")
    again = catalog.extract().unsafe_data["objects"].compute()
    assert again.data[0, 0] == {"items": ["first"]}


@pytest.mark.parametrize("backend", ("dataset", "datatree"))
def test_cat_perf_p11a_004_catalog_data_object_isolation_preserves_dask_laziness(
    backend: str,
) -> None:
    """ID: CAT_PERF_P11A_004_catalog_data_object_isolation_preserves_dask_laziness."""
    dask = pytest.importorskip("dask")
    dask_array = pytest.importorskip("dask.array")
    shape = (1, 1) if backend == "dataset" else (1,)
    values = np.empty(shape, dtype=object)
    values.flat[0] = {"items": ["source"]}
    lazy = dask_array.from_array(values, chunks=shape)
    if backend == "dataset":
        payload: xr.Dataset | xr.DataTree = xr.Dataset(
            {"value": (("trial", "sample"), lazy)},
            coords={"trial": ["a"], "sample": [0]},
        )
    else:
        child = xr.Dataset({"value": ("sample", lazy)}, coords={"sample": [0]})
        payload = xr.DataTree.from_dict({"/": xr.Dataset(), "/a": child})
    catalog = Catalog(payload, batch_dim="trial")

    def _fail_task(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("Catalog.data unexpectedly computed")

    with dask.callbacks.Callback(pretask=_fail_task):
        public = catalog.data
    public_array = (
        public["value"]
        if isinstance(public, xr.Dataset)
        else public.children["a"].to_dataset(inherit=False)["value"]
    )
    assert getattr(public_array.data, "chunks", None) is not None

    computed = public.compute()
    computed_array = (
        computed["value"]
        if isinstance(computed, xr.Dataset)
        else computed.children["a"].to_dataset(inherit=False)["value"]
    )
    computed_array.data.flat[0]["items"].append("changed")
    again = catalog.extract("value").unsafe_data["value"].compute()
    assert again.data.flat[0] == {"items": ["source"]}


def test_cat_perf_p11b_009_variable_canonicalization_assigns_once_per_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: CAT_PERF_P11B_009_variable_canonicalization_assigns_once_per_payload."""
    first = xr.Dataset(
        {
            name: (("sample", "axis"), np.arange(4).reshape(2, 2))
            for name in ("a", "b", "c")
        }
    )
    second = xr.Dataset(
        {
            name: (("axis", "sample"), np.arange(4).reshape(2, 2))
            for name in ("a", "b", "c")
        }
    )
    original_assign = xr.Dataset.assign
    calls = 0

    def _count_assign(self: xr.Dataset, *args: object, **kwargs: object) -> xr.Dataset:
        nonlocal calls
        calls += 1
        return original_assign(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "assign", _count_assign)
    out = canonicalize_selected_payload_vars(
        (first, second),
        selected=("a", "b", "c"),
        labels=("first", "second"),
        owner="Catalog.extract",
    )

    assert calls == 1
    assert all(out[1][name].dims == ("sample", "axis") for name in ("a", "b", "c"))


@pytest.mark.parametrize("empty", (False, True))
def test_cat_perf_p11b_010_disabled_metadata_promotion_skips_dask_projection(
    empty: bool,
) -> None:
    """ID: CAT_PERF_P11B_010_disabled_metadata_promotion_skips_dask_projection."""
    dask = pytest.importorskip("dask")
    dask_array = pytest.importorskip("dask.array")
    site = xr.DataArray(
        dask_array.from_array(np.asarray("field", dtype=object), chunks=()),
    )
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0])},
                coords={"sample": [0], "site": site},
            ),
        }
    )
    catalog = Catalog(tree, batch_dim="trial")
    target = catalog.sel([]) if empty else catalog
    promotion = CatalogMetadataPromotionOptions(
        scalar_target="none",
        nonscalar_target="none",
    )

    def _fail_task(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("disabled metadata promotion unexpectedly computed")

    with dask.callbacks.Callback(pretask=_fail_task):
        out = target.extract(
            "value",
            opts=CatalogExtractOptions(
                metadata_promotion=promotion,
                validate=False,
            ),
        ).unsafe_data

    assert "site" not in out.coords


def test_cat_perf_p11b_011_empty_default_metadata_reconciliation_preserves_dask_laziness() -> None:
    """ID: CAT_PERF_P11B_011_empty_default_metadata_reconciliation_preserves_dask_laziness."""
    dask = pytest.importorskip("dask")
    dask_array = pytest.importorskip("dask.array")
    site = xr.DataArray(
        dask_array.from_array(np.asarray("field", dtype=object), chunks=()),
    )
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0])},
                coords={"sample": [0], "site": site},
            ),
        }
    )
    empty = Catalog(tree, batch_dim="trial").sel([])

    def _fail_task(*args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise AssertionError("empty metadata reconciliation unexpectedly computed")

    with dask.callbacks.Callback(pretask=_fail_task):
        out = empty.extract(
            "value",
            opts=CatalogExtractOptions(validate=False),
        ).unsafe_data

    assert out.coords["site"].dims == ("trial",)
    assert out.coords["site"].size == 0
    assert getattr(out.coords["site"].data, "chunks", None) is not None
