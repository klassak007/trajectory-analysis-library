from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tal.catalog import Catalog
from tal.core import AnalysisObject


class _MutableIndexValue:
    def __init__(self, value: str) -> None:
        self.items = [value]

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other


def _grouped_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"value": (("trial", "sample"), np.arange(6).reshape(2, 3))},
        coords={"trial": ["a", "b"], "sample": [0, 1, 2]},
    )


def _grouped_datatree() -> xr.DataTree:
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
            "/b": xr.Dataset({"value": ("sample", [3.0, 4.0])}, coords={"sample": [0, 1]}),
        }
    )


def test_cat_core_p11a_001_catalog_canonical_location_is_tal_catalog() -> None:
    """ID: CAT_CORE_P11A_001_catalog_canonical_location_is_tal_catalog."""
    assert Catalog.__module__.startswith("tal.catalog")


def test_cat_core_p11a_004_catalog_is_optional_relative_to_ao_direct_io_paths() -> None:
    """ID: CAT_CORE_P11A_004_catalog_is_optional_relative_to_ao_direct_io_paths."""
    ao = AnalysisObject(_grouped_dataset())
    from_ao = Catalog(ao, batch_dim="trial")
    from_ds = Catalog(_grouped_dataset(), batch_dim="trial")
    assert from_ao.group_labels == from_ds.group_labels == ("a", "b")


def test_cat_core_p11a_005_catalog_p11a_surface_is_browse_only_and_returns_catalog_objects() -> None:
    """ID: CAT_CORE_P11A_005_catalog_p11a_surface_is_browse_only_and_returns_catalog_objects."""
    cat = Catalog(_grouped_datatree())
    out = cat.sel(["a"]).head(1).tail(1)
    assert isinstance(out, Catalog)
    assert not hasattr(cat, "set_group")


def test_cat_core_p11a_006_catalog_batch_axis_is_constructor_scoped_with_schema_role_precedence() -> None:
    """ID: CAT_CORE_P11A_006_catalog_batch_axis_is_constructor_scoped_with_schema_role_precedence."""
    ao = AnalysisObject(_grouped_dataset()).set_roles(
        sequence_dim="sample",
        batch_dims=["trial"],
        core_dims=[],
    )
    by_role = Catalog(ao)
    assert by_role.batch_dim == "trial"
    explicit = Catalog(ao, batch_dim="trial")
    assert explicit.batch_dim == "trial"
    with pytest.raises(ValueError, match="conflicts with declared sequence_dim"):
        Catalog(ao, batch_dim="sample")
    one_dim_default = Catalog(xr.Dataset({"x": ("k", [1, 2, 3])}))
    assert one_dim_default.batch_dim == "k"


@pytest.mark.parametrize("backend", ("dataset", "datatree"))
def test_cat_core_p11a_009_catalog_data_owns_mutable_object_values(backend: str) -> None:
    """ID: CAT_CORE_P11A_009_catalog_data_owns_mutable_object_values."""
    values = np.empty((1, 1) if backend == "dataset" else (1,), dtype=object)
    values.flat[0] = {"items": ["source"]}
    if backend == "dataset":
        payload: xr.Dataset | xr.DataTree = xr.Dataset(
            {"value": (("trial", "sample"), values)},
            coords={"trial": ["a"], "sample": [0]},
        )
    else:
        child = xr.Dataset({"value": ("sample", values)}, coords={"sample": [0]})
        payload = xr.DataTree.from_dict({"/": xr.Dataset(), "/a": child})
    catalog = Catalog(payload, batch_dim="trial")

    public = catalog.data
    public_values = (
        public["value"].data
        if isinstance(public, xr.Dataset)
        else public.children["a"].to_dataset(inherit=False)["value"].data
    )
    public_values.flat[0]["items"].append("changed")

    again = catalog.extract("value").unsafe_data["value"]
    assert again.data.flat[0] == {"items": ["source"]}


def test_cat_core_p11a_010_catalog_data_preserves_extension_dtype_indexes() -> None:
    """ID: CAT_CORE_P11A_010_catalog_data_preserves_extension_dtype_indexes."""
    labels = pd.CategoricalIndex(
        ["a", "b"],
        categories=["a", "b", "unused"],
        ordered=True,
        name="trial",
    )
    source = xr.Dataset(
        {"value": (("trial", "sample"), [[1.0], [2.0]])},
        coords={"trial": labels, "sample": [0]},
    )

    public = Catalog(source, batch_dim="trial").data

    assert isinstance(public.indexes["trial"], pd.CategoricalIndex)
    assert public.indexes["trial"].equals(labels)

    for categories in (
        pd.Index(["a", "b", "unused"], dtype="string"),
        pd.Index([1, 2, 3], dtype="Int64"),
    ):
        extension_labels = pd.CategoricalIndex(
            pd.Categorical.from_codes([0, 1], categories=categories, ordered=True),
            name="trial",
        )
        extension_source = xr.Dataset(
            {"value": ("trial", [1.0, 2.0])},
            coords={"trial": extension_labels},
        )
        extension_public = Catalog(extension_source, batch_dim="trial").data
        copied = extension_public.indexes["trial"]
        assert isinstance(copied, pd.CategoricalIndex)
        assert copied.categories.dtype == categories.dtype


@pytest.mark.parametrize("backend", ("dataset", "datatree"))
def test_cat_core_p11a_012_catalog_data_preserves_secondary_xindexes(
    backend: str,
) -> None:
    """ID: CAT_CORE_P11A_012_catalog_data_preserves_secondary_xindexes."""
    child = xr.Dataset(
        {"value": ("sample", [1.0, 2.0])},
        coords={"sample": [0, 1], "time": ("sample", [0.1, 0.2])},
    ).set_xindex("time")
    child.coords["sample"].encoding["codec_meta"] = {"items": ["primary"]}
    child.coords["time"].encoding["codec_meta"] = {"items": ["auxiliary"]}
    payload: xr.Dataset | xr.DataTree
    if backend == "dataset":
        payload = child.expand_dims({"trial": ["a"]})
    else:
        payload = xr.DataTree.from_dict({"/": xr.Dataset(), "/a": child})
    catalog = Catalog(payload, batch_dim="trial")

    public = catalog.data
    public_ds = (
        public
        if isinstance(public, xr.Dataset)
        else public.children["a"].to_dataset(inherit=False)
    )
    assert "time" in public_ds.xindexes
    assert public_ds.coords["sample"].encoding["codec_meta"] == {
        "items": ["primary"]
    }
    assert public_ds.coords["time"].encoding["codec_meta"] == {
        "items": ["auxiliary"]
    }
    public_ds.coords["time"].data[0] = 9.9
    public_ds.coords["sample"].encoding["codec_meta"]["items"].append("changed")
    public_ds.coords["time"].encoding["codec_meta"]["items"].append("changed")

    again = catalog.data
    again_ds = (
        again
        if isinstance(again, xr.Dataset)
        else again.children["a"].to_dataset(inherit=False)
    )
    assert "time" in again_ds.xindexes
    np.testing.assert_array_equal(again_ds.coords["time"], [0.1, 0.2])
    assert again_ds.coords["sample"].encoding["codec_meta"] == {
        "items": ["primary"]
    }
    assert again_ds.coords["time"].encoding["codec_meta"] == {
        "items": ["auxiliary"]
    }

    indexed_child = xr.Dataset(
        {"value": (("x", "y"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={
            "x": [0, 1],
            "y": [0, 1],
            "xx": (("x", "y"), [[0.0, 0.0], [1.0, 1.0]]),
            "yy": (("x", "y"), [[1.0, 0.0], [1.0, 0.0]]),
        },
    ).set_xindex(["xx", "yy"], xr.indexes.NDPointIndex)
    indexed_payload: xr.Dataset | xr.DataTree
    if backend == "dataset":
        indexed_payload = indexed_child.expand_dims({"trial": ["a"]})
    else:
        indexed_payload = xr.DataTree.from_dict(
            {"/": xr.Dataset(), "/a": indexed_child}
        )
    indexed_catalog = Catalog(indexed_payload, batch_dim="trial")

    indexed_public = indexed_catalog.data
    indexed_public_ds = (
        indexed_public
        if isinstance(indexed_public, xr.Dataset)
        else indexed_public.children["a"].to_dataset(inherit=False)
    )
    assert isinstance(indexed_public_ds.xindexes["xx"], xr.indexes.NDPointIndex)
    indexed_public_ds.coords["xx"].data[0, 0] = 99.0
    indexed_again = indexed_catalog.data
    indexed_again_ds = (
        indexed_again
        if isinstance(indexed_again, xr.Dataset)
        else indexed_again.children["a"].to_dataset(inherit=False)
    )
    np.testing.assert_array_equal(
        indexed_again_ds.coords["xx"],
        [[0.0, 0.0], [1.0, 1.0]],
    )


@pytest.mark.parametrize("index_kind", ("auxiliary", "categorical"))
def test_cat_core_p11a_013_catalog_data_owns_registered_object_xindex_values(
    index_kind: str,
) -> None:
    """ID: CAT_CORE_P11A_013_catalog_data_owns_registered_object_xindex_values."""
    first = _MutableIndexValue("first")
    second = _MutableIndexValue("second")
    if index_kind == "categorical":
        labels = pd.CategoricalIndex(
            [first, second],
            categories=[first, second],
            ordered=True,
            name="trial",
        )
        source = xr.Dataset(
            {"value": ("trial", [1.0, 2.0])},
            coords={"trial": labels},
        )
        coord_name = "trial"
    else:
        tags = np.empty((2,), dtype=object)
        tags[:] = [first, second]
        source = xr.Dataset(
            {"value": (("trial", "sample"), [[1.0, 2.0]])},
            coords={"trial": ["a"], "sample": [0, 1], "tag": ("sample", tags)},
        ).set_xindex("tag")
        coord_name = "tag"
    catalog = Catalog(source, batch_dim="trial")

    public = catalog.data
    public.coords[coord_name].data[0].items.append("changed")
    again = catalog.data

    assert coord_name in again.xindexes
    assert again.coords[coord_name].data[0].items == ["first"]
    if index_kind == "categorical":
        assert isinstance(again.indexes[coord_name], pd.CategoricalIndex)
        assert again.indexes[coord_name].ordered is True


@pytest.mark.parametrize("backend", ("dataset", "datatree"))
def test_cat_core_p11a_014_catalog_data_owns_nonindex_categorical_object_values(
    backend: str,
) -> None:
    """ID: CAT_CORE_P11A_014_catalog_data_owns_nonindex_categorical_object_values."""
    variable_values = [_MutableIndexValue("var-a"), _MutableIndexValue("var-b")]
    coord_values = [_MutableIndexValue("coord-a"), _MutableIndexValue("coord-b")]
    child = xr.Dataset(
        {
            "category": (
                "sample",
                pd.Categorical(
                    variable_values,
                    categories=variable_values,
                    ordered=True,
                ),
            )
        },
        coords={
            "sample": [0, 1],
            "tag": (
                "sample",
                pd.Categorical(
                    coord_values,
                    categories=coord_values,
                    ordered=True,
                ),
            ),
        },
    )
    payload: xr.Dataset | xr.DataTree
    if backend == "dataset":
        payload = child.expand_dims({"trial": ["a"]})
    else:
        payload = xr.DataTree.from_dict({"/": xr.Dataset(), "/a": child})
    catalog = Catalog(payload, batch_dim="trial")

    public = catalog.data
    public_ds = (
        public
        if isinstance(public, xr.Dataset)
        else public.children["a"].to_dataset(inherit=False)
    )
    assert isinstance(public_ds["category"].data, pd.Categorical)
    assert isinstance(public_ds.coords["tag"].data, pd.Categorical)
    np.asarray(public_ds["category"].data, dtype=object).flat[0].items.append(
        "changed"
    )
    np.asarray(public_ds.coords["tag"].data, dtype=object).flat[1].items.append(
        "changed"
    )

    again = catalog.data
    again_ds = (
        again
        if isinstance(again, xr.Dataset)
        else again.children["a"].to_dataset(inherit=False)
    )
    assert np.asarray(again_ds["category"].data, dtype=object).flat[0].items == [
        "var-a"
    ]
    assert np.asarray(again_ds.coords["tag"].data, dtype=object).flat[1].items == [
        "coord-b"
    ]
    assert isinstance(again_ds["category"].dtype, pd.CategoricalDtype)
    assert isinstance(again_ds.coords["tag"].dtype, pd.CategoricalDtype)
    assert again_ds["category"].dtype.ordered is True
    assert again_ds.coords["tag"].dtype.ordered is True


def test_cat_hard_p11a_001_catalog_blocks_arithmetic_and_numpy_interop() -> None:
    """ID: CAT_HARD_P11A_001_catalog_blocks_arithmetic_and_numpy_interop."""
    cat = Catalog(_grouped_datatree())
    with pytest.raises(TypeError, match="Catalog.__add__"):
        _ = cat + 1
    with pytest.raises(TypeError, match="Catalog.__array__"):
        _ = np.asarray(cat)
    with pytest.raises(TypeError, match="Catalog.__array_ufunc__"):
        _ = np.add(cat, 1)


def test_cat_hard_p11a_002_invalid_backend_or_group_configuration_fails_closed() -> None:
    """ID: CAT_HARD_P11A_002_invalid_backend_or_group_configuration_fails_closed."""
    with pytest.raises(TypeError, match="backend='datatree'"):
        Catalog(_grouped_dataset(), backend="datatree")
    with pytest.raises(ValueError, match="batch_dim 'missing'"):
        Catalog(_grouped_dataset(), batch_dim="missing")
    ao = AnalysisObject(
        xr.Dataset(
            {"value": (("subject", "trial", "sample"), np.arange(12).reshape(2, 2, 3))},
            coords={"subject": ["s0", "s1"], "trial": ["a", "b"], "sample": [0, 1, 2]},
        )
    ).set_roles(sequence_dim="sample", batch_dims=["subject", "trial"], core_dims=[])
    with pytest.raises(ValueError, match="ambiguous batch-axis resolution"):
        Catalog(ao)


def test_cat_hard_p11a_004_post_construction_regroup_mutators_are_absent_in_p11a() -> None:
    """ID: CAT_HARD_P11A_004_post_construction_regroup_mutators_are_absent_in_p11a."""
    cat = Catalog(_grouped_datatree(), batch_dim="trial")
    with pytest.raises(AttributeError):
        cat.batch_dim = "new_trial"  # type: ignore[misc]
