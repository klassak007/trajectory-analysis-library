from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tal.catalog import Catalog, CatalogExtractOptions, CatalogMetadataPromotionOptions
from tal.catalog.extract import extract_catalog_to_analysis_object
from tal.catalog.types import make_catalog_state
from tal.core import AnalysisObject
from tal.core.schema_read import read_param_coord_name, read_roles, read_validity


class _Uncopyable:
    def __init__(self, remaining_copies: int = 0) -> None:
        self.remaining_copies = remaining_copies

    def __deepcopy__(self, memo: object) -> object:
        _ = memo
        if self.remaining_copies:
            return type(self)(self.remaining_copies - 1)
        raise RuntimeError("cannot copy")


class _MutableIndexValue:
    def __init__(self, value: str) -> None:
        self.items = [value]

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other


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


def _flat_datatree_catalog_with_root_batch_metadata() -> Catalog:
    root = xr.Dataset(
        {"calibration": ("sensor", [0.25, 0.5])},
        coords={
            "trial": ["root-a", "root-b"],
            "rank": ("trial", [10, 20]),
            "sample": [10, 20],
            "time": ("sample", [0.0, 0.5]),
            "sensor": ["left", "right"],
        }
    ).set_xindex("time")
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "/a": xr.Dataset({"value": ("sample", [1.0, 2.0])}),
            "/b": xr.Dataset({"value": ("sample", [3.0, 4.0])}),
        }
    )
    return Catalog(tree, batch_dim="trial")


def _datatree_catalog_with_local_coord_overrides() -> Catalog:
    root = xr.Dataset(
        coords={
            "trial": ["root-a", "root-b"],
            "sample": [10, 20],
            "time": ("sample", [0.0, 0.5]),
            "site": xr.DataArray("lab"),
        }
    )
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={"time": ("sample", [9.0, 8.0])},
            ),
            "/b": xr.Dataset(
                {"value": ("sample", [3.0, 4.0])},
                coords={"time": ("sample", [7.0, 6.0])},
            ),
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


def test_cat_core_p11b_014_datatree_root_metadata_preserves_empty_variable_extracts() -> None:
    """ID: CAT_CORE_P11B_014_datatree_root_metadata_preserves_empty_variable_extracts."""
    catalog = _flat_datatree_catalog_with_root_batch_metadata()
    ignored = CatalogExtractOptions(ignore_missing_vars=True)
    outputs = (
        (catalog.extract([]), ("a", "b")),
        (catalog.extract(["missing"], opts=ignored), ("a", "b")),
        (catalog.sel(["b"]).extract([]), ("b",)),
        (catalog.sel([]).extract([]), ()),
    )

    for output, labels in outputs:
        ds = output.unsafe_data
        assert tuple(ds.data_vars) == ()
        assert ds.sizes["trial"] == len(labels)
        np.testing.assert_array_equal(ds.coords["trial"], labels)
        np.testing.assert_array_equal(ds.coords["sample"], [10, 20])
        np.testing.assert_array_equal(ds.coords["time"], [0.0, 0.5])
        assert "sensor" not in ds.dims


def test_cat_core_p11b_015_datatree_extract_preserves_inherited_payload_coordinates() -> None:
    """ID: CAT_CORE_P11B_015_datatree_extract_preserves_inherited_payload_coordinates."""
    catalog = _flat_datatree_catalog_with_root_batch_metadata()
    outputs = (catalog.extract("value"), catalog.sel(["b"]).extract("value"))

    for output in outputs:
        ds = output.unsafe_data
        np.testing.assert_array_equal(ds.coords["sample"], [10, 20])
        np.testing.assert_array_equal(ds.coords["time"], [0.0, 0.5])
        assert "time" in ds.xindexes
        np.testing.assert_array_equal(
            ds.sel(time=0.5)["value"].data,
            ds["value"].isel(sample=1).data,
        )
        assert "sensor" not in ds.dims


def test_cat_core_p11b_016_datatree_extract_preserves_local_coord_overrides() -> None:
    """ID: CAT_CORE_P11B_016_datatree_extract_preserves_local_coord_overrides."""
    catalog = _datatree_catalog_with_local_coord_overrides()

    out = catalog.extract("value").unsafe_data
    np.testing.assert_array_equal(out.coords["sample"], [10, 20])
    np.testing.assert_array_equal(out.coords["time"], [[9.0, 8.0], [7.0, 6.0]])
    assert out.coords["time"].dims == ("trial", "sample")
    assert out.coords["site"].dims == ("trial",)
    np.testing.assert_array_equal(out.coords["site"], ["lab", "lab"])

    empty = catalog.sel([]).extract("value").unsafe_data
    np.testing.assert_array_equal(empty.coords["sample"], [10, 20])
    np.testing.assert_array_equal(empty.coords["time"], [9.0, 8.0])


def test_cat_core_p11b_017_selected_empty_datatree_extract_uses_selected_template() -> None:
    """ID: CAT_CORE_P11B_017_selected_empty_datatree_extract_uses_selected_template."""
    catalog = _datatree_catalog_with_local_coord_overrides()
    emptied = (
        catalog.sel(["b"]).sel([]),
        catalog.isel([1]).isel([]),
        catalog.sel(["b", "a"]).tail(0),
    )

    for selected in emptied:
        out = selected.extract("value").unsafe_data
        assert out.sizes["trial"] == 0
        np.testing.assert_array_equal(out.coords["sample"], [10, 20])
        np.testing.assert_array_equal(out.coords["time"], [7.0, 6.0])


def test_cat_core_p11b_018_datatree_extract_completes_sparse_local_coordinates() -> None:
    """ID: CAT_CORE_P11B_018_datatree_extract_completes_sparse_local_coordinates."""
    root = xr.Dataset(
        coords={
            "sample": [10, 20],
            "site": xr.DataArray("lab"),
        }
    )
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={
                    "site": xr.DataArray("field"),
                    "quality": ("sample", [1, 2]),
                },
            ),
            "/b": xr.Dataset({"value": ("sample", [3.0, 4.0])}),
        }
    )

    out = Catalog(tree, batch_dim="trial").extract("value").unsafe_data
    np.testing.assert_array_equal(out.coords["site"], ["field", "lab"])
    np.testing.assert_array_equal(out.coords["quality"].isel(trial=0), [1.0, 2.0])
    assert out.coords["quality"].dims == ("trial", "sample")
    assert np.isnan(out.coords["quality"].isel(trial=1)).all()


def test_cat_core_p11b_024_constant_child_local_scalar_coord_is_not_repromoted() -> None:
    """ID: CAT_CORE_P11B_024_constant_child_local_scalar_coord_is_not_repromoted."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"sample": [0, 1]}),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={"site": xr.DataArray("field")},
            ),
            "/b": xr.Dataset(
                {"value": ("sample", [3.0, 4.0])},
                coords={"site": xr.DataArray("field")},
            ),
        }
    )

    out = Catalog(tree, batch_dim="trial").extract("value").unsafe_data
    assert out.coords["site"].dims == ("trial",)
    np.testing.assert_array_equal(out.coords["site"], ["field", "field"])


@pytest.mark.parametrize(
    ("promotion", "expected_attr"),
    (
        (CatalogMetadataPromotionOptions(scalar_target="attr"), "field"),
        (CatalogMetadataPromotionOptions(scalar_target="none"), None),
    ),
)
def test_cat_core_p11b_032_explicit_scalar_metadata_target_rewrites_structural_coord(
    promotion: CatalogMetadataPromotionOptions,
    expected_attr: object,
) -> None:
    """ID: CAT_CORE_P11B_032_explicit_scalar_metadata_target_rewrites_structural_coord."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"sample": [0, 1]}),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={"site": xr.DataArray("field")},
            ),
            "/b": xr.Dataset(
                {"value": ("sample", [3.0, 4.0])},
                coords={"site": xr.DataArray("field")},
            ),
        }
    )

    out = Catalog(tree, batch_dim="trial").extract(
        "value",
        opts=CatalogExtractOptions(metadata_promotion=promotion),
    ).unsafe_data

    assert "site" not in out.coords
    assert out.attrs.get("site") == expected_attr


def test_cat_core_p11b_033_explicit_nonscalar_metadata_target_rewrites_structural_coord() -> None:
    """ID: CAT_CORE_P11B_033_explicit_nonscalar_metadata_target_rewrites_structural_coord."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"sample": [0, 1]}),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={"site": xr.DataArray("field")},
            ),
            "/b": xr.Dataset(
                {"value": ("sample", [3.0, 4.0])},
                coords={"site": xr.DataArray("lab")},
            ),
        }
    )
    promotion = CatalogMetadataPromotionOptions(nonscalar_target="attr")

    out = Catalog(tree, batch_dim="trial").extract(
        "value",
        opts=CatalogExtractOptions(metadata_promotion=promotion),
    ).unsafe_data

    assert "site" not in out.coords
    assert out.attrs["site"] == ["field", "lab"]


@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("scalar_target", ("attr", "none"))
def test_cat_core_p11b_034_metadata_targets_preserve_schema_owned_validity(
    scalar_target: Literal["attr", "none"],
    empty: bool,
) -> None:
    """ID: CAT_CORE_P11B_034_metadata_targets_preserve_schema_owned_validity."""
    first = _unbatched_child([1.0, 2.0])
    second = AnalysisObject(
        _unbatched_child([3.0, 4.0]).assign_coords(
            sequence_size=xr.DataArray(np.int64(1))
        )
    ).unsafe_data
    tree = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "/a": first, "/b": second}
    )
    promotion = CatalogMetadataPromotionOptions(scalar_target=scalar_target)

    catalog = Catalog(tree, batch_dim="trial")
    target = catalog.sel([]) if empty else catalog
    out = target.extract(
        "value",
        opts=CatalogExtractOptions(
            metadata_promotion=promotion,
            require_sequence_size_coord=True,
        ),
    ).unsafe_data

    assert read_validity(out) == (True, "sequence_size", "left_packed")
    expected = np.asarray([], dtype=np.int64) if empty else np.asarray([2, 1])
    np.testing.assert_array_equal(out.coords["sequence_size"], expected)
    assert "sequence_size" not in out.attrs


@pytest.mark.parametrize("scalar_target", ("batch_coord", "attr", "none"))
def test_cat_core_p11b_036_empty_extract_metadata_targets_match_nonempty_policy(
    scalar_target: Literal["batch_coord", "attr", "none"],
) -> None:
    """ID: CAT_CORE_P11B_036_empty_extract_metadata_targets_match_nonempty_policy."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"sample": [0, 1]}),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={"site": xr.DataArray("field")},
            ),
        }
    )
    promotion = CatalogMetadataPromotionOptions(scalar_target=scalar_target)

    out = Catalog(tree, batch_dim="trial").sel([]).extract(
        "value",
        opts=CatalogExtractOptions(metadata_promotion=promotion),
    ).unsafe_data

    if scalar_target == "batch_coord":
        assert out.coords["site"].dims == ("trial",)
        assert out.coords["site"].size == 0
        assert "site" not in out.attrs
    elif scalar_target == "attr":
        assert "site" not in out.coords
        assert out.attrs["site"] == "field"
    else:
        assert "site" not in out.coords
        assert "site" not in out.attrs


def test_cat_hard_p11b_017_coord_dedup_preserves_true_namespace_collisions() -> None:
    """ID: CAT_HARD_P11B_017_coord_dedup_preserves_true_namespace_collisions."""
    child = xr.Dataset(
        {"value": ("sample", [1.0, 2.0])},
        coords={"site": xr.DataArray("field")},
        attrs={"site": "metadata"},
    )
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"sample": [0, 1]}),
            "/a": child,
            "/b": child.copy(deep=True),
        }
    )

    with pytest.raises(ValueError, match="metadata promotion target name 'site' collides"):
        Catalog(tree, batch_dim="trial").extract("value")


@pytest.mark.parametrize(
    "coord",
    (
        xr.DataArray("local"),
        xr.DataArray([10, 20], dims=("sample",)),
    ),
)
def test_cat_hard_p11b_022_child_coordinate_batch_name_collision_fails_closed(
    coord: xr.DataArray,
) -> None:
    """ID: CAT_HARD_P11B_022_child_coordinate_batch_name_collision_fails_closed."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={"trial": coord},
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Catalog\.extract: child-local coordinate 'trial'.*collides with resolved batch_dim 'trial'",
    ):
        Catalog(tree, batch_dim="trial").extract("value")


def test_cat_hard_p11b_023_root_scalar_metadata_does_not_mask_payload_coord_collision() -> None:
    """ID: CAT_HARD_P11B_023_root_scalar_metadata_does_not_mask_payload_coord_collision."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"site": xr.DataArray("root-site")}),
            "/a": xr.Dataset(
                {"value": ("site", [1.0, 2.0])},
                coords={"site": ["left", "right"]},
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Catalog\.extract: metadata promotion target name 'site' collides",
    ):
        Catalog(tree, batch_dim="trial").extract("value")


def test_cat_core_p11b_019_datatree_extract_preserves_root_batch_payload_coordinates() -> None:
    """ID: CAT_CORE_P11B_019_datatree_extract_preserves_root_batch_payload_coordinates."""
    root = xr.Dataset(
        coords={
            "trial": ["root-a", "root-b"],
            "sample": [10, 20],
            "time": (("trial", "sample"), [[0.0, 0.5], [1.0, 1.5]]),
        }
    )
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "/a": xr.Dataset({"value": ("sample", [1.0, 2.0])}),
            "/b": xr.Dataset({"value": ("sample", [3.0, 4.0])}),
        }
    )
    catalog = Catalog(tree, batch_dim="trial")

    out = catalog.extract("value").unsafe_data
    assert out.coords["time"].dims == ("trial", "sample")
    np.testing.assert_array_equal(out.coords["time"], [[0.0, 0.5], [1.0, 1.5]])

    selected = catalog.sel(["b"]).extract("value").unsafe_data
    np.testing.assert_array_equal(selected.coords["time"], [[1.0, 1.5]])


def test_cat_core_p11b_020_coordinate_dimension_permutations_are_canonicalized() -> None:
    """ID: CAT_CORE_P11B_020_coordinate_dimension_permutations_are_canonicalized."""
    root = xr.Dataset(coords={"sample": [0, 1], "axis": ["x", "y"]})
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "/a": xr.Dataset(
                {"value": (("sample", "axis"), [[1.0, 2.0], [3.0, 4.0]])},
                coords={"quality": (("sample", "axis"), [[10, 20], [30, 40]])},
            ),
            "/b": xr.Dataset(
                {"value": (("sample", "axis"), [[5.0, 6.0], [7.0, 8.0]])},
                coords={"quality": (("axis", "sample"), [[50, 70], [60, 80]])},
            ),
        }
    )

    out = Catalog(tree, batch_dim="trial").extract("value").unsafe_data
    assert out.coords["quality"].dims == ("trial", "sample", "axis")
    np.testing.assert_array_equal(
        out.coords["quality"],
        [[[10, 20], [30, 40]], [[50, 60], [70, 80]]],
    )


def test_cat_core_p11b_031_variable_dimension_permutations_are_canonicalized() -> None:
    """ID: CAT_CORE_P11B_031_variable_dimension_permutations_are_canonicalized."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"sample": [0, 1], "axis": ["x", "y"]}),
            "/a": xr.Dataset(
                {"value": (("sample", "axis"), [[1.0, 2.0], [3.0, 4.0]])},
            ),
            "/b": xr.Dataset(
                {"value": (("axis", "sample"), [[5.0, 7.0], [6.0, 8.0]])},
            ),
        }
    )

    out = Catalog(tree, batch_dim="trial").extract("value").unsafe_data
    assert out["value"].dims == ("trial", "sample", "axis")
    np.testing.assert_array_equal(
        out["value"],
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
    )


def test_cat_hard_p11b_019_incompatible_variable_topology_fails_before_concat() -> None:
    """ID: CAT_HARD_P11B_019_incompatible_variable_topology_fails_before_concat."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={"sample": [0, 1]},
            ),
            "/b": xr.Dataset(
                {"value": ("axis", [3.0, 4.0])},
                coords={"axis": ["x", "y"]},
            ),
        }
    )

    error = r"Catalog\.extract: variable 'value' has incompatible dimensions.*expected \('sample',\), got \('axis',\)"
    with pytest.raises(ValueError, match=error):
        Catalog(tree, batch_dim="trial").extract("value")


def test_cat_core_p11b_021_empty_extract_output_isolated_from_catalog_template() -> None:
    """ID: CAT_CORE_P11B_021_empty_extract_output_isolated_from_catalog_template."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={"sample": [10, 20], "time": ("sample", [0.1, 0.2])},
            ),
            "/b": xr.Dataset(
                {"value": ("sample", [3.0, 4.0])},
                coords={"sample": [10, 20], "time": ("sample", [0.3, 0.4])},
            ),
        }
    )
    selected = Catalog(tree, batch_dim="trial").sel(["b"])
    empty = selected.sel([])

    out = empty.extract("value")
    assert not np.shares_memory(
        out.unsafe_data.coords["time"].data,
        empty._state.template.coords["time"].data,
    )
    out.unsafe_data.coords["time"].data[0] = 9.9

    np.testing.assert_array_equal(empty.extract("value").unsafe_data.coords["time"], [0.3, 0.4])
    np.testing.assert_array_equal(selected.extract("value").unsafe_data.coords["time"], [[0.3, 0.4]])


@pytest.mark.parametrize("child_count", (1, 2))
@pytest.mark.parametrize("coord_owner", ("child", "root"))
def test_cat_core_p11b_022_nonempty_extract_owns_invariant_coordinate_buffers(
    child_count: int,
    coord_owner: str,
) -> None:
    """ID: CAT_CORE_P11B_022_nonempty_extract_owns_invariant_coordinate_buffers."""
    shared_coords = {"sample": [10, 20], "time": ("sample", [0.1, 0.2])}
    root = xr.Dataset(coords=shared_coords) if coord_owner == "root" else xr.Dataset()
    children = {
        f"/{label}": xr.Dataset(
            {"value": ("sample", [float(index), float(index + 1)])},
            coords=shared_coords if coord_owner == "child" else None,
        )
        for index, label in enumerate(("a", "b")[:child_count], start=1)
    }
    catalog = Catalog(xr.DataTree.from_dict({"/": root, **children}), batch_dim="trial")
    out = catalog.extract("value").unsafe_data
    source = (
        catalog._state.data.to_dataset(inherit=False)
        if coord_owner == "root"
        else catalog._state.data.children["a"].to_dataset(inherit=False)
    )

    assert not np.shares_memory(out.coords["sample"].data, source.coords["sample"].data)
    assert not np.shares_memory(out.coords["time"].data, source.coords["time"].data)
    out.coords["sample"].data[0] = 99
    out.coords["time"].data.reshape(-1)[0] = 9.9

    again = catalog.extract("value").unsafe_data
    np.testing.assert_array_equal(again.coords["sample"], [10, 20])
    expected_time = [0.1, 0.2] if coord_owner == "root" else [[0.1, 0.2]] * child_count
    np.testing.assert_array_equal(again.coords["time"], expected_time)


def test_cat_core_p11b_023_empty_extract_owns_mutable_template_metadata() -> None:
    """ID: CAT_CORE_P11B_023_empty_extract_owns_mutable_template_metadata."""
    child = xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]})
    child.attrs["dataset_meta"] = {"items": ["source"]}
    child["value"].attrs["variable_meta"] = {"items": ["source"]}
    child["value"].encoding["codec_meta"] = {"items": ["source"]}
    catalog = Catalog(
        xr.DataTree.from_dict({"/": xr.Dataset(), "/a": child}),
        batch_dim="trial",
    )
    empty = catalog.sel([])
    options = CatalogExtractOptions(
        metadata_promotion=CatalogMetadataPromotionOptions(scalar_target="attr")
    )

    out = empty.extract("value", opts=options).unsafe_data
    out.attrs["dataset_meta"]["items"].append("changed")
    out["value"].attrs["variable_meta"]["items"].append("changed")
    out["value"].encoding["codec_meta"]["items"].append("changed")

    again = empty.extract("value", opts=options).unsafe_data
    assert again.attrs["dataset_meta"] == {"items": ["source"]}
    assert again["value"].attrs["variable_meta"] == {"items": ["source"]}
    assert again["value"].encoding["codec_meta"] == {"items": ["source"]}
    source = catalog._state.data.children["a"].to_dataset(inherit=False)
    assert source.attrs["dataset_meta"] == {"items": ["source"]}
    assert source["value"].attrs["variable_meta"] == {"items": ["source"]}
    assert source["value"].encoding["codec_meta"] == {"items": ["source"]}


@pytest.mark.parametrize("empty", (False, True))
def test_cat_core_p11b_025_extract_owns_dataset_encoding(empty: bool) -> None:
    """ID: CAT_CORE_P11B_025_extract_owns_dataset_encoding."""
    child = xr.Dataset({"value": ("sample", [1.0, 2.0])})
    child.encoding["storage_meta"] = {"items": ["source"]}
    catalog = Catalog(
        xr.DataTree.from_dict({"/": xr.Dataset(), "/a": child}),
        batch_dim="trial",
    )
    target = catalog.sel([]) if empty else catalog

    out = target.extract("value").unsafe_data
    out.encoding["storage_meta"]["items"].append("changed")

    again = target.extract("value").unsafe_data
    assert again.encoding["storage_meta"] == {"items": ["source"]}
    source = catalog._state.data.children["a"].to_dataset(inherit=False)
    assert source.encoding["storage_meta"] == {"items": ["source"]}
    if empty:
        assert target._state.template is not None
        assert target._state.template.encoding["storage_meta"] == {"items": ["source"]}


def test_cat_core_p11b_026_extract_owns_mutable_batch_coordinate_values() -> None:
    """ID: CAT_CORE_P11B_026_extract_owns_mutable_batch_coordinate_values."""
    tags = np.empty((2,), dtype=object)
    tags[0] = {"items": ["first"]}
    tags[1] = {"items": ["second"]}
    child = xr.Dataset(
        {"value": ("sample", [1.0, 2.0])},
        coords={"sample": [0, 1], "tags": ("sample", tags)},
    )
    child.coords["tags"].attrs["coord_meta"] = {"items": ["source"]}
    child.coords["tags"].encoding["codec_meta"] = {"items": ["source"]}
    catalog = Catalog(
        xr.DataTree.from_dict({"/": xr.Dataset(), "/a": child}),
        batch_dim="trial",
    )

    out = catalog.extract("value").unsafe_data
    assert out.coords["tags"].attrs["coord_meta"] == {"items": ["source"]}
    assert out.coords["tags"].encoding["codec_meta"] == {"items": ["source"]}
    out.coords["tags"].data[0, 0]["items"].append("changed")
    out.coords["tags"].attrs["coord_meta"]["items"].append("changed")
    out.coords["tags"].encoding["codec_meta"]["items"].append("changed")

    again = catalog.extract("value").unsafe_data
    assert again.coords["tags"].data[0, 0] == {"items": ["first"]}
    assert again.coords["tags"].attrs["coord_meta"] == {"items": ["source"]}
    assert again.coords["tags"].encoding["codec_meta"] == {"items": ["source"]}
    source = catalog._state.data.children["a"].to_dataset(inherit=False)
    assert source.coords["tags"].data[0] == {"items": ["first"]}
    assert source.coords["tags"].attrs["coord_meta"] == {"items": ["source"]}
    assert source.coords["tags"].encoding["codec_meta"] == {"items": ["source"]}


def test_cat_core_p11b_027_extract_owns_mutable_object_payload_values() -> None:
    """ID: CAT_CORE_P11B_027_extract_owns_mutable_object_payload_values."""
    values = np.empty((2,), dtype=object)
    values[0] = {"items": ["first"]}
    values[1] = {"items": ["second"]}
    child = xr.Dataset({"value": ("sample", values)}, coords={"sample": [0, 1]})
    catalog = Catalog(
        xr.DataTree.from_dict({"/": xr.Dataset(), "/a": child}),
        batch_dim="trial",
    )

    out = catalog.extract("value").unsafe_data
    out["value"].data[0, 0]["items"].append("changed")

    again = catalog.extract("value").unsafe_data
    assert again["value"].data[0, 0] == {"items": ["first"]}
    source = catalog._state.data.children["a"].to_dataset(inherit=False)
    assert source["value"].data[0] == {"items": ["first"]}


def test_cat_core_p11b_028_extract_owns_structured_object_coordinate_values() -> None:
    """ID: CAT_CORE_P11B_028_extract_owns_structured_object_coordinate_values."""
    tags = np.empty((2,), dtype=[("payload", object), ("code", "i4")])
    tags["payload"][0] = {"items": ["first"]}
    tags["payload"][1] = {"items": ["second"]}
    tags["code"] = [1, 2]
    child = xr.Dataset(
        {"value": ("sample", [1.0, 2.0])},
        coords={"sample": [0, 1], "tags": ("sample", tags)},
    )
    catalog = Catalog(
        xr.DataTree.from_dict({"/": xr.Dataset(), "/a": child}),
        batch_dim="trial",
    )

    out = catalog.extract("value").unsafe_data
    out.coords["tags"].data[0, 0]["payload"]["items"].append("changed")

    again = catalog.extract("value").unsafe_data
    assert again.coords["tags"].data[0, 0]["payload"] == {"items": ["first"]}
    source = catalog._state.data.children["a"].to_dataset(inherit=False)
    assert source.coords["tags"].data[0]["payload"] == {"items": ["first"]}


@pytest.mark.parametrize("metadata_owner", ("root", "child"))
def test_cat_core_p11b_029_promoted_mutable_metadata_is_isolated(
    metadata_owner: str,
) -> None:
    """ID: CAT_CORE_P11B_029_promoted_mutable_metadata_is_isolated."""
    root = xr.Dataset()
    first = xr.Dataset({"value": ("sample", [1.0])})
    second = xr.Dataset({"value": ("sample", [2.0])})
    owners = (root,) if metadata_owner == "root" else (first, second)
    for ds in owners:
        ds.attrs["metadata"] = {"items": ["source"]}
    catalog = Catalog(
        xr.DataTree.from_dict({"/": root, "/a": first, "/b": second}),
        batch_dim="trial",
    )

    out = catalog.extract("value").unsafe_data
    out.coords["metadata"].data[0]["items"].append("changed")

    again = catalog.extract("value").unsafe_data
    assert again.coords["metadata"].data[0] == {"items": ["source"]}
    source = (
        catalog._state.data.to_dataset(inherit=False)
        if metadata_owner == "root"
        else catalog._state.data.children["a"].to_dataset(inherit=False)
    )
    assert source.attrs["metadata"] == {"items": ["source"]}


def test_cat_core_p11b_030_dataset_extract_owns_eager_payload_and_metadata() -> None:
    """ID: CAT_CORE_P11B_030_dataset_extract_owns_eager_payload_and_metadata."""
    objects = np.empty((2, 1), dtype=object)
    objects[0, 0] = {"items": ["first"]}
    objects[1, 0] = {"items": ["second"]}
    source = xr.Dataset(
        {
            "numeric": (("trial", "sample"), [[1.0], [2.0]]),
            "objects": (("trial", "sample"), objects),
        },
        coords={"trial": ["a", "b"], "sample": [0]},
    )
    source.encoding["storage_meta"] = {"items": ["source"]}
    source["numeric"].attrs["variable_meta"] = {"items": ["source"]}
    catalog = Catalog(source, batch_dim="trial")

    out = catalog.extract().unsafe_data
    out["numeric"].data[0, 0] = 99.0
    out["objects"].data[0, 0]["items"].append("changed")
    out.encoding["storage_meta"]["items"].append("changed")
    out["numeric"].attrs["variable_meta"]["items"].append("changed")

    again = catalog.extract().unsafe_data
    assert again["numeric"].data[0, 0] == 1.0
    assert again["objects"].data[0, 0] == {"items": ["first"]}
    assert again.encoding["storage_meta"] == {"items": ["source"]}
    assert again["numeric"].attrs["variable_meta"] == {"items": ["source"]}


def test_cat_core_p11b_035_dataset_extract_preserves_extension_dtype_indexes() -> None:
    """ID: CAT_CORE_P11B_035_dataset_extract_preserves_extension_dtype_indexes."""
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

    catalog = Catalog(source, batch_dim="trial")
    out = catalog.extract("value").unsafe_data

    assert isinstance(out.indexes["trial"], pd.CategoricalIndex)
    assert out.indexes["trial"].equals(labels)

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
        extension_out = Catalog(
            extension_source,
            batch_dim="trial",
        ).extract("value").unsafe_data
        copied = extension_out.indexes["trial"]
        assert isinstance(copied, pd.CategoricalIndex)
        assert copied.categories.dtype == categories.dtype


def test_cat_core_p11b_040_dataset_extract_preserves_secondary_xindexes() -> None:
    """ID: CAT_CORE_P11B_040_dataset_extract_preserves_secondary_xindexes."""
    source = xr.Dataset(
        {"value": (("trial", "sample"), [[1.0, 2.0]])},
        coords={
            "trial": ["a"],
            "sample": [0, 1],
            "time": ("sample", [0.1, 0.2]),
        },
    ).set_xindex("time")
    source.coords["sample"].encoding["codec_meta"] = {"items": ["primary"]}
    source.coords["time"].encoding["codec_meta"] = {"items": ["auxiliary"]}

    catalog = Catalog(source, batch_dim="trial")
    out = catalog.extract("value").unsafe_data

    assert "time" in out.xindexes
    np.testing.assert_array_equal(out.coords["time"], [0.1, 0.2])
    assert out.coords["sample"].encoding["codec_meta"] == {"items": ["primary"]}
    assert out.coords["time"].encoding["codec_meta"] == {
        "items": ["auxiliary"]
    }
    out.coords["sample"].encoding["codec_meta"]["items"].append("changed")
    out.coords["time"].encoding["codec_meta"]["items"].append("changed")

    again = catalog.extract("value").unsafe_data
    assert again.coords["sample"].encoding["codec_meta"] == {
        "items": ["primary"]
    }
    assert again.coords["time"].encoding["codec_meta"] == {
        "items": ["auxiliary"]
    }

    indexed_source = xr.Dataset(
        {
            "value": (
                ("trial", "x", "y"),
                [[[1.0, 2.0], [3.0, 4.0]]],
            )
        },
        coords={
            "trial": ["a"],
            "x": [0, 1],
            "y": [0, 1],
            "xx": (("x", "y"), [[0.0, 0.0], [1.0, 1.0]]),
            "yy": (("x", "y"), [[1.0, 0.0], [1.0, 0.0]]),
        },
    ).set_xindex(["xx", "yy"], xr.indexes.NDPointIndex)
    catalog = Catalog(indexed_source, batch_dim="trial")

    indexed_out = catalog.extract("value").unsafe_data
    assert isinstance(indexed_out.xindexes["xx"], xr.indexes.NDPointIndex)
    indexed_out.coords["xx"].data[0, 0] = 99.0
    indexed_again = catalog.extract("value").unsafe_data
    np.testing.assert_array_equal(
        indexed_again.coords["xx"],
        [[0.0, 0.0], [1.0, 1.0]],
    )


def test_cat_core_p11b_041_dataset_extract_owns_registered_object_xindex_values() -> None:
    """ID: CAT_CORE_P11B_041_dataset_extract_owns_registered_object_xindex_values."""
    tags = np.empty((2,), dtype=object)
    tags[:] = [_MutableIndexValue("first"), _MutableIndexValue("second")]
    source = xr.Dataset(
        {"value": (("trial", "sample"), [[1.0, 2.0]])},
        coords={"trial": ["a"], "sample": [0, 1], "tag": ("sample", tags)},
    ).set_xindex("tag")
    catalog = Catalog(source, batch_dim="trial")

    out = catalog.extract("value").unsafe_data
    out.coords["tag"].data[0].items.append("changed")
    again = catalog.extract("value").unsafe_data

    assert "tag" in again.xindexes
    assert again.coords["tag"].data[0].items == ["first"]


@pytest.mark.parametrize(
    "metadata",
    (pd.NA, pd.NaT, np.datetime64("NaT"), np.nan),
    ids=("pandas-na", "pandas-nat", "numpy-nat", "numpy-nan"),
)
def test_cat_core_p11b_042_missing_scalar_metadata_promotes_consistently(
    metadata: object,
) -> None:
    """ID: CAT_CORE_P11B_042_missing_scalar_metadata_promotes_consistently."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"status": metadata}),
            "/a": xr.Dataset({"value": ("sample", [1.0])}),
            "/b": xr.Dataset({"value": ("sample", [2.0])}),
        }
    )

    out = Catalog(tree, batch_dim="trial").extract("value").unsafe_data

    assert out.coords["status"].dims == ("trial",)
    assert all(bool(pd.isna(value)) for value in out.coords["status"].data)


def test_cat_core_p11b_043_unindexed_dimension_coordinate_remains_invariant() -> None:
    """ID: CAT_CORE_P11B_043_unindexed_dimension_coordinate_remains_invariant."""
    def _child(values: list[float]) -> xr.Dataset:
        return xr.Dataset(
            {"value": ("sample", values)},
            coords={"sample": [10, 20]},
        ).drop_indexes("sample")

    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": _child([1.0, 2.0]),
            "/b": _child([3.0, 4.0]),
        }
    )

    out = Catalog(tree, batch_dim="trial").extract("value").unsafe_data

    assert out.coords["sample"].dims == ("sample",)
    assert "sample" not in out.xindexes
    np.testing.assert_array_equal(out.coords["sample"], [10, 20])


def test_cat_core_p11b_044_dataset_extract_owns_nonindex_categorical_object_values() -> None:
    """ID: CAT_CORE_P11B_044_dataset_extract_owns_nonindex_categorical_object_values."""
    variable_values = [_MutableIndexValue("var-a"), _MutableIndexValue("var-b")]
    coord_values = [_MutableIndexValue("coord-a"), _MutableIndexValue("coord-b")]
    source = xr.Dataset(
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
    catalog = Catalog(source)

    out = catalog.extract().unsafe_data
    assert isinstance(out["category"].data, pd.Categorical)
    assert isinstance(out.coords["tag"].data, pd.Categorical)
    out["category"].data[0].items.append("changed")
    out.coords["tag"].data[1].items.append("changed")

    again = catalog.extract().unsafe_data
    assert again["category"].data[0].items == ["var-a"]
    assert again.coords["tag"].data[1].items == ["coord-b"]
    assert isinstance(again["category"].dtype, pd.CategoricalDtype)
    assert isinstance(again.coords["tag"].dtype, pd.CategoricalDtype)
    assert again["category"].dtype.ordered is True
    assert again.coords["tag"].dtype.ordered is True


def test_cat_core_p11b_037_sparse_extension_coordinate_uses_object_missing_values() -> None:
    """ID: CAT_CORE_P11B_037_sparse_extension_coordinate_uses_object_missing_values."""
    tags = pd.CategoricalIndex(
        ["first", "second"],
        categories=["first", "second", "unused"],
        ordered=True,
        name="tag",
    )
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={"tag": ("sample", tags)},
            ),
            "/b": xr.Dataset({"value": ("sample", [3.0, 4.0])}),
        }
    )

    out = Catalog(tree, batch_dim="trial").extract("value").unsafe_data

    assert out.coords["tag"].dims == ("trial", "sample")
    assert out.coords["tag"].dtype == np.dtype(object)
    np.testing.assert_array_equal(out.coords["tag"].isel(trial=0), ["first", "second"])
    assert bool(out.coords["tag"].isel(trial=1).isnull().all())


def test_cat_core_p11b_038_auxiliary_xindexes_are_planned_as_batch_coordinates() -> None:
    """ID: CAT_CORE_P11B_038_auxiliary_xindexes_are_planned_as_batch_coordinates."""
    def _child(values: list[float], times: list[float]) -> xr.Dataset:
        return xr.Dataset(
            {"value": ("sample", values)},
            coords={"sample": [0, 1], "time": ("sample", times)},
        ).set_xindex("time")

    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": _child([1.0, 2.0], [0.1, 0.2]),
            "/b": _child([3.0, 4.0], [0.3, 0.4]),
        }
    )

    out = Catalog(tree, batch_dim="trial").extract("value").unsafe_data

    assert out.coords["time"].dims == ("trial", "sample")
    np.testing.assert_array_equal(out.coords["time"], [[0.1, 0.2], [0.3, 0.4]])


@pytest.mark.parametrize(
    "metadata",
    (
        ["left", "right"],
        np.asarray([1.0, np.nan]),
        pytest.param([1, pd.NA], id="nested-pandas-na"),
    ),
)
def test_cat_core_p11b_039_nested_attr_metadata_remains_one_cell_per_group(
    metadata: object,
) -> None:
    """ID: CAT_CORE_P11B_039_nested_attr_metadata_remains_one_cell_per_group."""
    child = xr.Dataset(
        {"value": ("sample", [1.0])},
        coords={"sample": [0]},
        attrs={"tags": metadata},
    )
    catalog = Catalog(
        xr.DataTree.from_dict({"/": xr.Dataset(), "/a": child, "/b": child}),
        batch_dim="trial",
    )

    out = catalog.extract("value").unsafe_data
    assert out.coords["tags"].dims == ("trial",)
    assert out.coords["tags"].shape == (2,)
    for value in out.coords["tags"].data:
        if isinstance(metadata, list) and metadata[-1] is pd.NA:
            assert value[0] == 1
            assert value[1] is pd.NA
        else:
            np.testing.assert_equal(value, metadata)

    empty = catalog.sel([]).extract("value").unsafe_data
    assert empty.coords["tags"].dims == ("trial",)
    assert empty.coords["tags"].size == 0
    attr_options = CatalogExtractOptions(
        metadata_promotion=CatalogMetadataPromotionOptions(scalar_target="attr")
    )
    for target in (catalog, catalog.sel([])):
        promoted = target.extract("value", opts=attr_options).unsafe_data
        promoted_value = promoted.attrs["tags"]
        if isinstance(metadata, list) and metadata[-1] is pd.NA:
            assert promoted_value[0] == 1
            assert promoted_value[1] is pd.NA
        else:
            np.testing.assert_equal(promoted_value, metadata)


def test_cat_hard_p11b_018_sparse_structured_coordinate_fails_closed() -> None:
    """ID: CAT_HARD_P11B_018_sparse_structured_coordinate_missing_fails_closed."""
    tags = np.empty((2,), dtype=[("payload", object), ("code", "i4")])
    tags["payload"] = [{"items": ["first"]}, {"items": ["second"]}]
    tags["code"] = [1, 2]
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={"sample": [0, 1], "tags": ("sample", tags)},
            ),
            "/b": xr.Dataset(
                {"value": ("sample", [3.0, 4.0])},
                coords={"sample": [0, 1]},
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Catalog\.extract: coordinate 'tags' is missing from DataTree group 'b'.*structured dtype",
    ):
        Catalog(tree, batch_dim="trial").extract("value")


def test_cat_hard_p11b_014_incompatible_child_coordinate_topology_fails_closed() -> None:
    """ID: CAT_HARD_P11B_014_incompatible_child_coordinate_topology_fails_closed."""
    root = xr.Dataset(coords={"sample": [0, 1], "axis": ["x", "y"]})
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "/a": xr.Dataset(
                {"value": (("sample", "axis"), [[1.0, 2.0], [3.0, 4.0]])},
                coords={"quality": ("axis", [10, 20])},
            ),
            "/b": xr.Dataset(
                {"value": (("sample", "axis"), [[5.0, 6.0], [7.0, 8.0]])},
                coords={"quality": ("sample", [30, 40])},
            ),
        }
    )

    error = r"coordinate 'quality' has incompatible dimensions.*expected \('axis',\), got \('sample',\)"
    with pytest.raises(ValueError, match=error):
        Catalog(tree, batch_dim="trial").extract("value")


def test_cat_hard_p11b_016_filtered_coordinate_topology_conflicts_fail_before_concat() -> None:
    """ID: CAT_HARD_P11B_016_filtered_coordinate_topology_conflicts_fail_before_concat."""
    root = xr.Dataset(coords={"sample": [0, 1]})
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={"axis": ["x", "y"], "quality": ("axis", [10, 20])},
            ),
            "/b": xr.Dataset(
                {"value": ("sample", [3.0, 4.0])},
                coords={"quality": ("sample", [30, 40])},
            ),
        }
    )

    error = r"coordinate 'quality' has incompatible dimensions.*expected \('axis',\), got \('sample',\)"
    with pytest.raises(ValueError, match=error):
        Catalog(tree, batch_dim="trial").extract("value")


@pytest.mark.parametrize("root_size", (1, 3))
def test_cat_hard_p11b_015_extract_defensively_rejects_root_group_count_mismatch(
    root_size: int,
) -> None:
    """ID: CAT_HARD_P11B_015_extract_defensively_rejects_root_group_count_mismatch."""
    root = xr.Dataset(
        coords={
            "trial": np.arange(root_size),
            "sample": [0],
            "time": (("trial", "sample"), np.arange(root_size, dtype=float).reshape(root_size, 1)),
        }
    )
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "/a": xr.Dataset({"value": ("sample", [1.0])}),
            "/b": xr.Dataset({"value": ("sample", [2.0])}),
        }
    )
    state = make_catalog_state(
        backend="datatree",
        batch_dim="trial",
        data=tree,
    )
    error = rf"Catalog\.extract: root batch dimension 'trial' length {root_size} does not match DataTree group count 2"

    with pytest.raises(ValueError, match=error):
        extract_catalog_to_analysis_object(
            state,
            variables=("value",),
            options=CatalogExtractOptions(),
            owner="Catalog.extract",
        )


def test_cat_hard_p11b_013_coordinate_failure_is_not_reported_as_batch_collision() -> None:
    """ID: CAT_HARD_P11B_013_coordinate_failure_is_not_reported_as_batch_collision."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset(
                {"value": ("sample", [1.0, 2.0])},
                coords={"quality": ("sample", [1.0, 2.0])},
            ),
            "/b": xr.Dataset({"value": ("sample", [3.0])}),
        }
    )

    with pytest.raises(ValueError) as exc_info:
        Catalog(tree, batch_dim="trial").extract("value")

    assert "conflicting sizes for dimension 'sample'" in str(exc_info.value)
    assert "collides with child payload dims" not in str(exc_info.value)


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


@pytest.mark.parametrize("backend", ("dataset", "datatree"))
def test_cat_hard_p11b_020_object_copy_failures_preserve_operation_owner(
    backend: str,
) -> None:
    """ID: CAT_HARD_P11B_020_object_copy_failures_preserve_operation_owner."""
    values = np.empty((1, 1) if backend == "dataset" else (1,), dtype=object)
    values.flat[0] = _Uncopyable(2 if backend == "dataset" else 1)
    if backend == "dataset":
        payload: xr.Dataset | xr.DataTree = xr.Dataset(
            {"value": (("trial", "sample"), values)},
            coords={"trial": ["a"], "sample": [0]},
        )
    else:
        child = xr.Dataset({"value": ("sample", values)}, coords={"sample": [0]})
        payload = xr.DataTree.from_dict({"/": xr.Dataset(), "/a": child})
    catalog = Catalog(payload, batch_dim="trial")

    with pytest.raises(
        ValueError,
        match=r"Catalog\.extract: could not isolate object values for array 'value'",
    ) as caught:
        catalog.extract("value")
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_cat_hard_p11b_021_deferred_object_copy_failure_preserves_owner() -> None:
    """ID: CAT_HARD_P11B_021_deferred_object_copy_failure_preserves_owner."""
    dask_array = pytest.importorskip("dask.array")
    values = np.empty((1, 1), dtype=object)
    values[0, 0] = _Uncopyable()
    source = xr.Dataset(
        {
            "value": (
                ("trial", "sample"),
                dask_array.from_array(values, chunks=(1, 1)),
            )
        },
        coords={"trial": ["a"], "sample": [0]},
    )
    result = Catalog(source, batch_dim="trial").extract("value").unsafe_data

    with pytest.raises(
        ValueError,
        match=r"Catalog\.extract: could not isolate object values for array 'value'",
    ):
        result["value"].compute()
