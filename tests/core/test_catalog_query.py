from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.catalog import Catalog, CatalogQueryOptions
from tal.core import AnalysisObject


def _query_catalog() -> Catalog:
    ds = xr.Dataset(
        {"value": (("trial", "sample"), np.arange(6).reshape(3, 2))},
        coords={
            "trial": ["a", "b", "c"],
            "sample": [0, 1],
            "rank": ("trial", [1, 2, 3]),
            "sensor": xr.DataArray("front"),
        },
        attrs={
            "source": ["sim", "real", "sim"],
            "trial": "shadow",
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=["trial"],
        core_dims=[],
        validate=True,
    )
    return Catalog(ao)


def _datatree_query_catalog_with_root_batch_metadata() -> Catalog:
    root = xr.Dataset(
        coords={
            "group": ["root-a", "root-b", "root-c"],
            "rank": ("group", [10, 20, 30]),
        }
    )
    tree = xr.DataTree.from_dict(
        {
            "/": root,
            "/a": xr.Dataset({"value": ("sample", [1.0, 2.0])}),
            "/b": xr.Dataset({"value": ("sample", [3.0, 4.0])}),
            "/c": xr.Dataset({"value": ("sample", [5.0, 6.0])}),
        }
    )
    return Catalog(tree)


def test_cat_core_p11b_001_query_no_match_returns_empty_catalog() -> None:
    """ID: CAT_CORE_P11B_001_query_no_match_returns_empty_catalog."""
    cat = _query_catalog()
    out = cat.query(where={"op": "==", "field": "batch.trial", "value": "missing"})
    assert out.group_labels == ()


def test_cat_core_p11b_002_query_filters_metadata_with_deterministic_and_or_invert_behavior() -> None:
    """ID: CAT_CORE_P11B_002_query_filters_metadata_with_deterministic_and_or_invert_behavior."""
    cat = _query_catalog()
    expr = {
        "op": "and",
        "args": [
            {"op": "==", "field": "attr.source", "value": "sim"},
            {
                "op": "or",
                "args": [
                    {"op": "not", "arg": {"op": "in", "field": "batch.rank", "value": [1, 2]}},
                    {"op": "==", "field": "batch.trial", "value": "a"},
                ],
            },
        ],
    }
    out = cat.query(where=expr)
    assert out.group_labels == ("a", "c")


def test_cat_core_p11b_006_query_operator_surface_is_finite_and_contract_backed() -> None:
    """ID: CAT_CORE_P11B_006_query_operator_surface_is_finite_and_contract_backed."""
    cat = _query_catalog()
    assert cat.query(where={"op": "==", "field": "batch.rank", "value": 2}).group_labels == ("b",)
    assert cat.query(where={"op": "!=", "field": "batch.rank", "value": 2}).group_labels == ("a", "c")
    assert cat.query(where={"op": "<", "field": "batch.rank", "value": 2}).group_labels == ("a",)
    assert cat.query(where={"op": "<=", "field": "batch.rank", "value": 2}).group_labels == ("a", "b")
    assert cat.query(where={"op": ">", "field": "batch.rank", "value": 2}).group_labels == ("c",)
    assert cat.query(where={"op": ">=", "field": "batch.rank", "value": 2}).group_labels == ("b", "c")
    assert cat.query(where={"op": "in", "field": "batch.trial", "value": ["a", "c"]}).group_labels == ("a", "c")
    assert cat.query(where={"op": "not in", "field": "batch.trial", "value": ["a", "c"]}).group_labels == ("b",)


def test_cat_core_p11b_009_structured_predicate_dsl_is_canonical_query_representation() -> None:
    """ID: CAT_CORE_P11B_009_structured_predicate_dsl_is_canonical_query_representation."""
    cat = _query_catalog()
    explicit = cat.query(
        where={
            "op": "and",
            "args": [
                {"op": "==", "field": "attr.source", "value": "sim"},
                {"op": "==", "field": "batch.rank", "value": 3},
            ],
        }
    )
    shorthand = cat.query(where={"op": "==", "field": "attr.source", "value": "sim"}, rank=3)
    assert explicit.group_labels == shorthand.group_labels == ("c",)


def test_cat_core_p11b_010_metadata_namespace_disambiguation_is_deterministic() -> None:
    """ID: CAT_CORE_P11B_010_metadata_namespace_disambiguation_is_deterministic."""
    cat = _query_catalog()
    with pytest.raises(ValueError, match="ambiguous unqualified metadata field"):
        cat.query(where={"op": "==", "field": "trial", "value": "a"})
    out = cat.query(where={"op": "==", "field": "sensor", "value": "front"})
    assert out.group_labels == ("a", "b", "c")


def test_cat_core_p11b_011_query_extract_use_constructor_resolved_grouping_policy() -> None:
    """ID: CAT_CORE_P11B_011_query_extract_use_constructor_resolved_grouping_policy."""
    ds = xr.Dataset(
        {"value": (("trial", "sample"), np.arange(6).reshape(3, 2))},
        coords={"trial": ["a", "b", "c"], "sample": [0, 1]},
    )
    cat = Catalog(ds, batch_dim="sample")
    out = cat.query(where={"op": "==", "field": "batch.sample", "value": 1})
    assert out.group_labels == (1,)


def test_cat_hard_p11b_001_invalid_query_expression_or_unknown_column_fails_closed_by_default() -> None:
    """ID: CAT_HARD_P11B_001_invalid_query_expression_or_unknown_column_fails_closed_by_default."""
    cat = _query_catalog()
    with pytest.raises(ValueError, match="unknown metadata field"):
        cat.query(where={"op": "==", "field": "attr.missing", "value": 1})
    out = cat.query(
        where={"op": "==", "field": "attr.missing", "value": 1},
        opts=CatalogQueryOptions(unknown_field_policy="ignore"),
        source="sim",
    )
    assert out.group_labels == ("a", "c")


def test_cat_hard_p11b_004_nested_datatree_behavior_is_explicitly_rejected_in_p11b() -> None:
    """ID: CAT_HARD_P11B_004_nested_datatree_behavior_is_explicitly_rejected_in_p11b."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/a": xr.Dataset(),
            "/a/nested": xr.Dataset(),
        }
    )
    with pytest.raises(ValueError, match="nested DataTree"):
        Catalog(tree)


def test_cat_hard_p11b_005_no_python_eval_or_freeform_expression_execution_in_query_paths() -> None:
    """ID: CAT_HARD_P11B_005_no_python_eval_or_freeform_expression_execution_in_query_paths."""
    cat = _query_catalog()
    with pytest.raises(TypeError, match="string query expressions"):
        cat.query("rank > 1")
    with pytest.raises(TypeError, match="callable predicates"):
        cat.query(where=lambda _ctx: True)


def test_cat_hard_p11b_006_backend_specific_expression_languages_are_rejected_in_p11b() -> None:
    """ID: CAT_HARD_P11B_006_backend_specific_expression_languages_are_rejected_in_p11b."""
    cat = _query_catalog()
    with pytest.raises(ValueError, match="unsupported predicate operator"):
        cat.query(where={"op": "sql", "expr": "rank > 1"})


def test_cat_hard_p11b_007_unqualified_ambiguous_metadata_fields_fail_closed() -> None:
    """ID: CAT_HARD_P11B_007_unqualified_ambiguous_metadata_fields_fail_closed."""
    cat = _query_catalog()
    with pytest.raises(ValueError, match="ambiguous unqualified metadata field"):
        cat.query(where={"op": "==", "field": "trial", "value": "a"})


def test_cat_hard_p11b_008_ambiguous_grouping_context_for_query_extract_fails_closed_without_mutable_regroup_api() -> None:
    """ID: CAT_HARD_P11B_008_ambiguous_grouping_context_for_query_extract_fails_closed_without_mutable_regroup_api."""
    ds = xr.Dataset(
        {"value": (("subject", "trial", "sample"), np.arange(12).reshape(2, 2, 3))},
        coords={"subject": ["s0", "s1"], "trial": ["a", "b"], "sample": [0, 1, 2]},
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=["subject", "trial"],
        core_dims=[],
        validate=True,
    )
    cat = Catalog(ao, batch_dim="trial")
    with pytest.raises(ValueError, match="ambiguous grouping context"):
        cat.query(where={"op": "==", "field": "batch.trial", "value": "a"})

    single_batch = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), np.arange(6).reshape(3, 2))},
            coords={"trial": ["a", "b", "c"], "sample": [0, 1]},
        ),
        sequence_dim="sample",
        batch_dims=["trial"],
        core_dims=[],
        validate=True,
    )
    with pytest.raises(ValueError, match="conflicts with declared sequence_dim"):
        Catalog(single_batch, batch_dim="sample").query(
            where={"op": "==", "field": "batch.sample", "value": 1}
        )


def test_cat_core_p11b_013_chained_datatree_query_uses_selected_root_batch_metadata() -> None:
    """ID: CAT_CORE_P11B_013_chained_datatree_query_uses_selected_root_batch_metadata."""
    catalog = _datatree_query_catalog_with_root_batch_metadata()
    reordered = catalog.sel(["b", "a"])
    assert reordered.query(
        where={"op": "==", "field": "batch.rank", "value": 20}
    ).group_labels == ("b",)
    subset = catalog.sel(["b"])
    assert subset.query(
        where={"op": "==", "field": "batch.rank", "value": 20}
    ).group_labels == ("b",)
    empty = catalog.sel([])
    assert empty.query(
        where={"op": "==", "field": "batch.rank", "value": 20}
    ).group_labels == ()
