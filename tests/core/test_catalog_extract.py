from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.catalog import Catalog, CatalogExtractOptions
from tal.core import AnalysisObject
from tal.core.schema_read import read_param_coord_name, read_roles, read_validity


def _dataset_catalog_with_validity() -> Catalog:
    ds = xr.Dataset(
        {
            "value_a": (("trial", "sample"), np.arange(6).reshape(3, 2).astype(float)),
            "value_b": (("trial", "sample"), np.arange(10, 16).reshape(3, 2).astype(float)),
        },
        coords={
            "trial": ["a", "b", "c"],
            "sample": [0, 1],
            "time": (("trial", "sample"), np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=float)),
            "sequence_size": ("trial", np.array([2, 1, 2], dtype=np.int64)),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=["trial"],
        core_dims=[],
        param_coord="time",
        sequence_size_coord="sequence_size",
        validate=True,
    )
    return Catalog(ao)


def _unbatched_child(values: list[float], *, stamp_validity: bool = True) -> xr.Dataset:
    coords: dict[str, object] = {
        "sample": np.arange(len(values), dtype=np.int64),
        "time": ("sample", np.arange(len(values), dtype=float)),
    }
    if stamp_validity:
        coords["sequence_size"] = xr.DataArray(np.array(len(values), dtype=np.int64))
    ds = xr.Dataset({"value": ("sample", np.asarray(values, dtype=float))}, coords=coords)
    kwargs: dict[str, object] = {
        "sequence_dim": "sample",
        "batch_dims": [],
        "core_dims": [],
        "param_coord": "time",
        "validate": True,
    }
    if stamp_validity:
        kwargs["sequence_size_coord"] = "sequence_size"
    ao = AnalysisObject.from_data(ds, **kwargs)
    return ao.unsafe_data


def _flat_datatree_catalog(*, stamp_validity: bool = True) -> Catalog:
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"source_name": "demo"}),
            "/a": _unbatched_child([1.0, 2.0], stamp_validity=stamp_validity),
            "/b": _unbatched_child([3.0, 4.0], stamp_validity=stamp_validity),
        }
    )
    return Catalog(tree, batch_dim="trial")


def test_cat_core_p11b_003_extract_produces_analysisobject_with_truthful_schema() -> None:
    """ID: CAT_CORE_P11B_003_extract_produces_analysisobject_with_truthful_schema."""
    cat = _dataset_catalog_with_validity()
    out = cat.extract()
    assert isinstance(out, AnalysisObject)
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.unsafe_data)
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ()
    assert read_param_coord_name(out.unsafe_data) == "time"


def test_cat_core_p11b_004_extract_variable_subset_semantics_are_deterministic() -> None:
    """ID: CAT_CORE_P11B_004_extract_variable_subset_semantics_are_deterministic."""
    cat = _dataset_catalog_with_validity()
    subset = cat.extract(["value_b"])
    assert tuple(subset.unsafe_data.data_vars) == ("value_b",)
    empty = cat.extract([])
    assert tuple(empty.unsafe_data.data_vars) == ()
    full = cat.extract()
    assert tuple(full.unsafe_data.data_vars) == ("value_a", "value_b")


def test_cat_core_p11b_005_sequence_size_coord_derivation_policy_is_deterministic() -> None:
    """ID: CAT_CORE_P11B_005_sequence_size_coord_derivation_policy_is_deterministic."""
    cat = _dataset_catalog_with_validity()
    strict = cat.extract(opts=CatalogExtractOptions(require_sequence_size_coord=True))
    assert "sequence_size" in strict.unsafe_data.coords
    ds = xr.Dataset(
        {"value": (("trial", "sample"), np.arange(4).reshape(2, 2).astype(float))},
        coords={"trial": ["a", "b"], "sample": [0, 1], "time": (("trial", "sample"), np.array([[0.0, 1.0], [0.0, 1.0]]))},
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=["trial"],
        core_dims=[],
        param_coord="time",
        validate=True,
    )
    with pytest.raises(ValueError, match="require_sequence_size_coord"):
        Catalog(ao).extract(opts=CatalogExtractOptions(require_sequence_size_coord=True))


def test_cat_core_p11b_007_empty_extract_returns_truthful_empty_analysisobject_when_representable() -> None:
    """ID: CAT_CORE_P11B_007_empty_extract_returns_truthful_empty_analysisobject_when_representable."""
    cat = _flat_datatree_catalog()
    out = cat.sel([]).extract()
    assert out.unsafe_data.sizes["trial"] == 0
    assert "value" in out.unsafe_data.data_vars
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.unsafe_data)
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ()


def test_cat_core_p11b_008_variable_subset_unknown_names_fail_closed_by_default() -> None:
    """ID: CAT_CORE_P11B_008_variable_subset_unknown_names_fail_closed_by_default."""
    cat = _dataset_catalog_with_validity()
    with pytest.raises(ValueError, match="unknown extract variable"):
        cat.extract(["missing"])
    out = cat.extract(
        ["missing", "value_a"],
        opts=CatalogExtractOptions(ignore_missing_vars=True),
    )
    assert tuple(out.unsafe_data.data_vars) == ("value_a",)


def test_cat_core_p11b_012_extract_validity_stamps_canonical_sequence_size_coord_schema() -> None:
    """ID: CAT_CORE_P11B_012_extract_validity_stamps_canonical_sequence_size_coord_schema."""
    cat = _dataset_catalog_with_validity()
    out = cat.extract()
    has_validity, size_name, layout = read_validity(out.unsafe_data)
    assert has_validity is True
    assert size_name == "sequence_size"
    assert layout == "left_packed"


def test_cat_hard_p11b_002_metadata_promotion_name_collisions_fail_closed() -> None:
    """ID: CAT_HARD_P11B_002_metadata_promotion_name_collisions_fail_closed."""
    child = _unbatched_child([1.0, 2.0])
    child = child.copy(deep=True)
    child.attrs["value"] = "collision"
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "/a": child})
    cat = Catalog(tree, batch_dim="trial")
    with pytest.raises(ValueError, match="collides"):
        cat.extract()


def test_cat_hard_p11b_012_single_group_extract_query_attr_metadata_does_not_self_collide() -> None:
    """ID: CAT_HARD_P11B_012_single_group_extract_query_attr_metadata_does_not_self_collide."""
    root = xr.DataTree(name="root")
    ds_a = xr.Dataset({"y": ("time", np.sin(np.linspace(0.0, 1.0, 6)))}, coords={"time": np.linspace(0.0, 1.0, 6)})
    ds_a.attrs["status"] = "pass"
    ds_b = xr.Dataset({"y": ("time", np.sin(np.linspace(0.2, 1.2, 4)))}, coords={"time": np.linspace(0.2, 1.2, 4)})
    ds_b.attrs["status"] = "fail"
    root["trial_a"] = xr.DataTree(ds_a)
    root["trial_b"] = xr.DataTree(ds_b)
    out = Catalog(root).query(status="fail").extract("y").as_dataset()
    assert out.sizes["group"] == 1
    assert "status" in out.coords
    assert out.coords["status"].dims == ("group",)
    np.testing.assert_array_equal(out.coords["status"].values, np.asarray(["fail"], dtype=object))


def test_cat_hard_p11b_009_dataset_explicit_batch_dim_sequence_collision_fails_closed() -> None:
    """ID: CAT_HARD_P11B_009_dataset_explicit_batch_dim_sequence_collision_fails_closed."""
    source = _dataset_catalog_with_validity().data
    with pytest.raises(ValueError, match="Catalog.__init__: explicit batch_dim 'sample' conflicts with declared sequence_dim"):
        Catalog(source, batch_dim="sample").extract()


def test_cat_hard_p11b_010_datatree_explicit_batch_dim_child_dim_collision_fails_closed_owner_prefixed() -> None:
    """ID: CAT_HARD_P11B_010_datatree_explicit_batch_dim_child_dim_collision_fails_closed_owner_prefixed."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": _unbatched_child([1.0, 2.0]),
            "/b": _unbatched_child([3.0, 4.0]),
        }
    )
    with pytest.raises(ValueError, match="Catalog.__init__: explicit batch_dim 'sample' collides with child dataset dim") as exc:
        Catalog(tree, batch_dim="sample").extract()
    assert "Dimension sample already exists." not in str(exc.value)


def test_cat_hard_p11b_011_extract_override_conflict_does_not_silently_drop_schema_metadata() -> None:
    """ID: CAT_HARD_P11B_011_extract_override_conflict_does_not_silently_drop_schema_metadata."""
    source = _dataset_catalog_with_validity().data
    roles_before = read_roles(source)
    param_before = read_param_coord_name(source)
    validity_before = read_validity(source)
    with pytest.raises(ValueError, match="conflicts with declared sequence_dim"):
        Catalog(source, batch_dim="sample").extract()
    assert read_roles(source) == roles_before
    assert read_param_coord_name(source) == param_before
    assert read_validity(source) == validity_before
