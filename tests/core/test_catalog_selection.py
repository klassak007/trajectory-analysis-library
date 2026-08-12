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


def _datatree_with_root_batch_metadata(*, root_size: int = 3) -> xr.DataTree:
    root_labels = np.asarray(["root-a", "root-b", "root-c", "root-d"], dtype=object)[:root_size]
    ranks = np.arange(1, root_size + 1) * 10
    weights = np.arange(1, root_size + 1) * 100
    root = xr.Dataset(
        {
            "root_weight": ("trial", weights),
            "calibration": ("sensor", [0.25, 0.5]),
        },
        coords={
            "trial": root_labels,
            "rank": ("trial", ranks),
            "sensor": ["left", "right"],
            "site": xr.DataArray("lab"),
        },
        attrs={"source": "root"},
    )
    children = {
        f"/{label}": xr.Dataset(
            {"value": ("sample", [index, index + 0.5])},
            coords={"sample": [0, 1]},
            attrs={"child_id": label},
        )
        for index, label in enumerate(("a", "b", "c"), start=1)
    }
    return xr.DataTree.from_dict({"/": root, **children})


def _datatree_catalog_with_root_batch_metadata() -> Catalog:
    return Catalog(_datatree_with_root_batch_metadata(), batch_dim="trial")


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


@pytest.mark.parametrize(
    ("method", "selector"),
    (
        ("sel", ["b"]),
        ("isel", [1]),
        ("sel", []),
        ("isel", []),
    ),
)
def test_cat_core_p11a_011_datatree_selection_preserves_root_name(
    method: str,
    selector: object,
) -> None:
    """ID: CAT_CORE_P11A_011_datatree_selection_preserves_root_name."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset({"value": ("sample", [1.0])}),
            "/b": xr.Dataset({"value": ("sample", [2.0])}),
        },
        name="experiment",
    )

    selected = getattr(Catalog(tree, batch_dim="trial"), method)(selector)

    assert selected.data.name == "experiment"


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


@pytest.mark.parametrize(
    ("method", "selector", "expected_labels", "positions"),
    (
        ("sel", ["c", "a"], ("c", "a"), (2, 0)),
        ("sel", "b", ("b",), (1,)),
        ("isel", [2, 0], ("c", "a"), (2, 0)),
        ("isel", slice(1, None), ("b", "c"), (1, 2)),
        ("head", 2, ("a", "b"), (0, 1)),
        ("tail", 2, ("b", "c"), (1, 2)),
        ("sel", [], (), ()),
        ("isel", [], (), ()),
    ),
)
def test_cat_core_p11a_008_datatree_selection_projects_root_batch_metadata(
    method: str,
    selector: object,
    expected_labels: tuple[str, ...],
    positions: tuple[int, ...],
) -> None:
    """ID: CAT_CORE_P11A_008_datatree_selection_projects_root_batch_metadata."""
    source = _datatree_catalog_with_root_batch_metadata()
    selected = getattr(source, method)(selector)
    tree = selected.data
    root = tree.to_dataset(inherit=False)

    assert selected.group_labels == expected_labels
    assert root.sizes["trial"] == len(positions)
    np.testing.assert_array_equal(root.coords["trial"], np.asarray(["root-a", "root-b", "root-c"])[list(positions)])
    np.testing.assert_array_equal(root.coords["rank"], np.asarray([10, 20, 30])[list(positions)])
    np.testing.assert_array_equal(root["root_weight"], np.asarray([100, 200, 300])[list(positions)])
    np.testing.assert_array_equal(root["calibration"], [0.25, 0.5])
    assert root.coords["site"].item() == "lab"
    assert root.attrs["source"] == "root"
    for label in expected_labels:
        local = tree.children[label].to_dataset(inherit=False)
        assert local.attrs["child_id"] == label
        assert local.sizes == {"sample": 2}
    source_root = source.data.to_dataset(inherit=False)
    np.testing.assert_array_equal(source_root.coords["rank"], [10, 20, 30])


@pytest.mark.parametrize("root_size", (0, 1, 2, 4))
def test_cat_hard_p11a_006_datatree_root_group_count_mismatch_fails_at_construction(
    root_size: int,
) -> None:
    """ID: CAT_HARD_P11A_006_datatree_root_group_count_mismatch_fails_at_construction."""
    error = rf"root batch dimension 'trial' length {root_size} does not match DataTree group count 3"
    with pytest.raises(ValueError, match=rf"Catalog\.__init__: {error}"):
        Catalog(_datatree_with_root_batch_metadata(root_size=root_size), batch_dim="trial")
