from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.catalog import Catalog


def _dataset_catalog() -> Catalog:
    ds = xr.Dataset(
        {"value": (("trial", "sample"), np.arange(9).reshape(3, 3))},
        coords={"trial": ["a", "b", "c"], "sample": [0, 1, 2]},
    )
    return Catalog(ds, batch_dim="trial")


def _datatree_catalog() -> Catalog:
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset({"value": ("sample", [1, 2])}, coords={"sample": [0, 1]}),
            "/b": xr.Dataset({"value": ("sample", [3, 4])}, coords={"sample": [0, 1]}),
            "/c": xr.Dataset({"value": ("sample", [5, 6])}, coords={"sample": [0, 1]}),
        }
    )
    return Catalog(tree, batch_dim="trial")


def test_cat_core_p11a_002_catalog_browse_selection_semantics_are_deterministic() -> None:
    """ID: CAT_CORE_P11A_002_catalog_browse_selection_semantics_are_deterministic."""
    ds_cat = _dataset_catalog()
    tree_cat = _datatree_catalog()
    assert ds_cat.sel(["c", "a"]).group_labels == ("c", "a")
    assert tree_cat.sel(["c", "a"]).group_labels == ("c", "a")
    assert ds_cat.isel([2, 0]).group_labels == ("c", "a")
    assert tree_cat.isel([2, 0]).group_labels == ("c", "a")
    assert ds_cat.head(2).group_labels == ("a", "b")
    assert tree_cat.tail(2).group_labels == ("b", "c")


def test_cat_core_p11a_003_catalog_empty_selection_returns_empty_catalog() -> None:
    """ID: CAT_CORE_P11A_003_catalog_empty_selection_returns_empty_catalog."""
    ds_cat = _dataset_catalog()
    tree_cat = _datatree_catalog()
    assert ds_cat.sel([]).group_labels == ()
    assert tree_cat.sel([]).group_labels == ()
    assert ds_cat.isel([]).group_labels == ()
    assert tree_cat.isel([]).group_labels == ()
    assert ds_cat.head(0).group_labels == ()
    assert tree_cat.tail(0).group_labels == ()


def test_cat_hard_p11a_003_unsupported_selector_kinds_fail_closed_in_browse_surface() -> None:
    """ID: CAT_HARD_P11A_003_unsupported_selector_kinds_fail_closed_in_browse_surface."""
    cat = _dataset_catalog()
    with pytest.raises(ValueError, match="Catalog.sel"):
        cat.sel({"wrong_key": ["a"]})
    with pytest.raises(TypeError, match="Catalog.sel"):
        cat.sel({"trial": {"nested": "value"}})
    with pytest.raises(TypeError, match="Catalog.isel"):
        cat.isel({"trial": [0, 1.5]})
    with pytest.raises(ValueError, match="duplicate"):
        cat.sel(["a", "a"])


def test_cat_hard_p11a_005_catalog_isel_normalized_duplicate_positional_selectors_fail_closed() -> None:
    """ID: CAT_HARD_P11A_005_catalog_isel_normalized_duplicate_positional_selectors_fail_closed."""
    ds_cat = _dataset_catalog()
    tree_cat = _datatree_catalog()
    with pytest.raises(ValueError, match="duplicate positional selectors"):
        ds_cat.isel([-1, 2])
    with pytest.raises(ValueError, match="duplicate positional selectors"):
        tree_cat.isel([-1, 2])


def test_cat_core_p11a_007_catalog_isel_negative_alias_duplicate_policy_is_cross_backend_deterministic() -> None:
    """ID: CAT_CORE_P11A_007_catalog_isel_negative_alias_duplicate_policy_is_cross_backend_deterministic."""
    ds_cat = _dataset_catalog()
    tree_cat = _datatree_catalog()
    assert ds_cat.isel([-1, 1]).group_labels == ("c", "b")
    assert tree_cat.isel([-1, 1]).group_labels == ("c", "b")
    with pytest.raises(IndexError, match="Catalog.isel"):
        ds_cat.isel([-4])
    with pytest.raises(IndexError, match="Catalog.isel"):
        tree_cat.isel([-4])
