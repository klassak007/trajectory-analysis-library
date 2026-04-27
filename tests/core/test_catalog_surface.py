from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.catalog import Catalog
from tal.core import AnalysisObject


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
